"""Stage 1 - Cleaning.

Deterministic cleaning of acquired source documents. No LLM calls anywhere.

Input : data/interim/<region>/<doc_id>.json          (najdi | southern | northern |
                                                      eastern | western)
Output: data/interim/<region>/<doc_id>_cleaned.json  (originals are never overwritten)
        data/processed/cleaning_report.json          (per-doc stats + review flags)

DIALECT PHASE NOTE
------------------
Regional spelling variation is the signal this corpus exists to capture, so the shipped
text is never "corrected". Two rules enforce that:

  * `paragraphs_original` (what Stage 3 writes into chunks.jsonl) keeps أ/إ/آ and ى
    exactly as written, and keeps tatweel unless --strip-tatweel is passed. Expressive
    elongation (يا هــــلا) is dialect signal, not typographic padding.
  * `paragraphs` is the folded matching key (alef/ya folded, tatweel always stripped).
    It exists only for dedup.py's SHA-256 and MinHash. It is never shipped.

Pipeline per document
  1. ftfy encoding repair on raw_text.
  2. Segment the body according to --format (see SEGMENTERS): blank-line paragraphs for
     prose, stanzas/verse lines for poetry, entry blocks/lines for dictionaries. The
     segment list is what Stage 3 packs into chunks, so getting this right per source
     type matters more than anything else in this stage.
  3. Detach any front-matter header block (title / تأليف / ترجمة / مراجعة ...) and keep
     it as structured metadata instead of throwing it away.
  4. Arabic normalization (pyarabic) into the two variants described above. Diacritic
     stripping is a flag, default OFF - poetry and dialect transcription often carry
     meaningful vocalization.
  5. Track a before/after character delta and flag anything that lost >30% of its
     characters, plus a few other review signals. Nothing is ever dropped silently.
     The delta is measured against the orthography-preserving text (+ the detached
     header), i.e. what actually ships downstream in chunks.jsonl; the folded body
     length is reported separately as `char_count_folded_body`.

شرح بالعربية
------------
المرحلة الأولى: التنظيف. معالجة حتمية بالكامل للوثائق المصدرية، بلا أي استدعاء لنموذج
لغوي. المبدأ الحاكم أن الاختلاف الإملائي بين اللهجات هو الإشارة المطلوب حفظها، ولذلك
لا يُصحَّح النص المُسلَّم أبدًا.

تُحفَظ نسختان من كل وثيقة:
  * paragraphs_original — النص الذي يُسلَّم فعليًا إلى chunks.jsonl، بإملائه كما ورد
    (أ/إ/آ و ى دون توحيد، والتطويل باقٍ ما لم يُمرَّر --strip-tatweel).
  * paragraphs — مفتاح المطابقة فقط، بعد توحيد الألف والياء وحذف التطويل. يُستخدم
    حصريًا داخل dedup.py لحساب SHA-256 و MinHash، ولا يُسلَّم إطلاقًا.

الخطوات: إصلاح الترميز عبر ftfy، ثم تقسيم النص إلى وحدات حسب --format، ثم فصل مقدمة
الكتاب وحفظها كبيانات وصفية بدل إهمالها، ثم التطبيع بنسختيه، ثم قياس فارق عدد الحروف
قبل وبعد ووضع علامة مراجعة يدوية عند فقد يتجاوز 30%. لا يُحذف أي محتوى بصمت.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata

import ftfy
from pyarabic import araby

# --- Paths -------------------------------------------------------------------
# المسارات: مجلد الإدخال/الإخراج وملف تقرير التنظيف.
INTERIM_DIR = "data/interim"
REPORT_PATH = "data/processed/cleaning_report.json"

# --- Cleaning parameters -----------------------------------------------------
# معاملات التنظيف: فاصل الفقرات، وعتبة الإبلاغ عن فقد الحروف، وسقف عدد فقرات المقدمة.
PARAGRAPH_SEPARATOR = "\n\n"      # blank line between blocks
CHAR_LOSS_FLAG_THRESHOLD = 0.30   # flag doc for manual review above this loss
MAX_HEADER_PARAGRAPHS = 12        # safety cap so we never eat real prose

FORMATS = ("prose", "dictionary", "verse")

# A glossary headword opening a line: a parenthesised term followed by a colon. All
# examples below are INVENTED placeholders that exercise the same shapes - no text from
# a rights-unverified source appears in this file. Matches:
#   (كلمة) :تعريفها        ( َمثَل) :تعريف بمدخل مشكول        (كلمتان معا) :تعريف مركب
# The opening delimiter is matched as [()] because bidi reordering in the extracted
# text sometimes renders it as ")". Requiring the closing ")" AND the colon is what
# keeps this off numbered feature lines like "( :)1ططط ظظظ" and off mid-entry
# (placeholder swapped 2026-09-09: the previous two-word example, though written as
# an invented illustration, occurs verbatim in the rights-pending corpus and the
# licence audit refuses it. The colliding phrase is deliberately not repeated here.
# It also made this block's own "no text from a rights-unverified source appears in
# this file" claim untrue, which is exactly what the audit is for.)
# parentheticals such as "(إشارة جانبية).وتكملة التعريف...".
HEADWORD_RE = re.compile(r"(?m)^[ \t]*[()][^)\n]{1,45}\)[ \t]*:")
MIN_HEADWORDS_FOR_ENTRY_SPLIT = 5

# A sub-section heading standing on its own line. Invented placeholders again:
#   ( الباب الثامن )      ( لهجة قبيلة فلان )      ( بعض السمات الافتراضية )
# These carry no headword, so without a secondary break the whole sub-section preamble
# (title + prose introduction + numbered feature list) attaches as a tail to whatever
# glossary entry happened to precede it. Headings are always short standalone lines and
# never carry the "…) :" of an entry, which is how they stay distinguishable.
#
# The line must be WHOLLY parenthesised (optionally prefixed by لهجة/لهجات, optionally
# followed by a footnote marker like "()1"). That is what separates a real heading from
# the cross-reference lines that run inside definitions - "لهجة قبيلة أخرى ص 12-14",
# "لهجات مجاورة .قال الشاعر فلان" - which open with the same words but are running prose
# and must not break an entry apart.
SECTION_HEADING_RE = re.compile(
    r"(?m)^[ \t]*(?:"
    # لهج(ة|ات?) - the singular ends in teh marbuta, so "لهجات?" alone would never
    # match it and this whole branch would be dead for the commonest heading form.
    r"لهج(?:ة|ات?)[ \t]*\([^)\n]{1,80}\)"              # لهجة ( اسم الموضع )
    r"|\([ \t]*(?:الباب|لهجات|لهجة|بعض|الصفات|الألفاظ|الالفاظ)[^)\n]{0,80}\)"
    r")[ \t]*(?:\([ \t]*\)[ \t]*\d+)?[ \t]*$"
)
MAX_SECTION_HEADING_WORDS = 12

# Front-matter role markers used by Arabic e-book front matter. The marker sits on its
# own line, the name(s) follow on the next line(s) inside the same paragraph.
ROLE_MARKERS = {
    "تأليف": "author_stated",
    "ترجمة": "translator",
    "مراجعة": "reviewer",
    "تحرير": "editor",
    "إعداد": "compiler",
    "تقديم": "introduction_by",
    "جمع": "compiler",
    "شرح": "annotator",
}

# Paragraphs that are pure media placeholders with no textual content.
MEDIA_PLACEHOLDER_RE = re.compile(r"^\s*\[\s*(figure|image|figcaption)?\s*\]\s*$", re.IGNORECASE)

# Arabic letter normalization tables (explicit, so the transform is auditable).
ALEF_VARIANTS = {
    "آ": araby.ALEF,  # آ  alef with madda
    "أ": araby.ALEF,  # أ  alef with hamza above
    "إ": araby.ALEF,  # إ  alef with hamza below
    "ٱ": araby.ALEF,  # ٱ  alef wasla
    "ٲ": araby.ALEF,  # ٲ
    "ٳ": araby.ALEF,  # ٳ
}
YA_VARIANTS = {
    "ى": araby.YEH,  # ى  alef maksura -> ya
    "ی": araby.YEH,  # ی  farsi ya
}
NORMALIZE_TABLE = str.maketrans({**ALEF_VARIANTS, **YA_VARIANTS})


# --- Text helpers ------------------------------------------------------------
# دوال مساعدة على مستوى النص: تنظيف المسافات، ونسختا التطبيع (مفتاح المطابقة
# مقابل النص المُسلَّم)، وعدّ الكلمات.
def normalize_whitespace(text: str) -> str:
    """NBSP/zero-width cleanup, collapse runs of spaces, keep newlines meaningful."""
    # توحيد المسافات: حذف المسافة غير الفاصلة والمحارف عديمة العرض وعلامات اتجاه
    # النص، ودمج المسافات المتكررة، مع الإبقاء على فواصل الأسطر لأنها تحمل معنى.
    text = text.replace(" ", " ")                    # NBSP
    text = re.sub(r"[​-‏‪-‮﻿]", "", text)  # zero-width / bidi marks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def normalize_arabic(text: str, strip_diacritics: bool = False) -> str:
    """Folded MATCHING KEY only - never shipped. Alef/ya folded, tatweel always stripped."""
    # نسخة المطابقة فقط: توحيد الألف والياء وحذف التطويل. عملية فاقدة للمعلومة
    # (نشأة تصير نشاة)، ولذلك تُستخدم للمقارنة داخل dedup.py ولا تُسلَّم أبدًا.
    text = unicodedata.normalize("NFC", text)
    text = araby.strip_tatweel(text)
    text = text.translate(NORMALIZE_TABLE)
    if strip_diacritics:
        text = araby.strip_tashkeel(text)
    return text


def preserve_orthography(text: str, strip_diacritics: bool = False,
                         strip_tatweel: bool = False) -> str:
    """Shipped text. Alef/ya spelling is left exactly as written; tatweel only on request."""
    # النص المُسلَّم: يُترك الإملاء كما ورد في المصدر. لا توحيد للألف أو الياء،
    # ولا حذف للتطويل إلا بطلب صريح، لأن هذا الاختلاف هو مادة الدراسة نفسها.
    text = unicodedata.normalize("NFC", text)
    if strip_tatweel:
        text = araby.strip_tatweel(text)
    if strip_diacritics:
        text = araby.strip_tashkeel(text)
    return text


def word_count(text: str) -> int:
    # عدّ تقريبي للكلمات بالفواصل البيضاء، يُستخدم في إحصاءات التقرير.
    return len(text.split())


# --- Segmentation ------------------------------------------------------------
# Each segmenter returns (units, unit_type). A "unit" is the smallest thing Stage 3 is
# allowed to treat as indivisible: a prose paragraph, a whole glossary entry, a whole
# stanza. Splitting rules differ per source type, and getting this wrong upstream is
# invisible downstream - a glossary segmented as prose becomes one giant blob.
# التقسيم: كل دالة تُعيد (units, unit_type). الوحدة هي أصغر جزء لا يجوز لمرحلة
# التقطيع أن تشطره: فقرة نثرية، أو مدخل معجمي كامل، أو مقطع شعري كامل. قواعد
# التقسيم تختلف باختلاف نوع المصدر، وخطأ هنا لا يظهر لاحقًا: معجم قُسِّم كنثر
# يتحول إلى كتلة واحدة ضخمة.
def segment_prose(text: str) -> tuple[list[str], str]:
    """Blank-line separated paragraphs; single newlines stay inside a paragraph."""
    # نثر: الفقرة تنتهي بسطر فارغ. فاصل السطر المفرد يبقى داخل الفقرة نفسها.
    return [p.strip() for p in text.split(PARAGRAPH_SEPARATOR) if p.strip()], "paragraph"


def segment_verse(text: str) -> tuple[list[str], str]:
    """Stanzas (blank-line separated), internal line breaks preserved.

    A poem written without blank lines has no stanza structure to recover, so each line
    becomes its own unit rather than silently gluing the whole poem into one blob.
    """
    # شِعر: الوحدة هي المقطع المفصول بسطر فارغ، مع حفظ فواصل الأسطر داخله. وإذا خلت
    # القصيدة من الأسطر الفارغة فلا بنية مقاطع يمكن استرجاعها، فيصير كل سطر وحدة
    # مستقلة بدل لصق القصيدة كلها في كتلة واحدة.
    blocks = [b.strip() for b in text.split(PARAGRAPH_SEPARATOR) if b.strip()]
    if len(blocks) > 1:
        return blocks, "stanza"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines, "verse_line"


def segment_dictionary(text: str) -> tuple[list[str], str]:
    """One unit per glossary entry (headword + definition + any example).

    Three conventions are handled, checked in this order:

      1. HEADWORD-ANCHORED (`entry_headword`) - an entry opens with a parenthesised
         headword followed by a colon, and its definition runs across as many lines as
         it needs until the next headword starts. Neither blank lines nor line breaks
         delimit entries. Shape (invented placeholder text):
             (كلمة) :تعريفها الأول، ثم مثال على استعمالها في جملة
             وتكملة التعريف تنساب على سطر ثانٍ وثالث بلا سطر فارغ يفصلها.
             (كلمة أخرى) :يبدأ المدخل التالي هنا.
         Splitting such a source on blank lines yields page-sized blobs; splitting on
         line breaks shreds every definition. Chosen whenever the text carries enough
         headword matches to be unambiguous.
      2. BLANK-LINE blocks (`entry_block`).
      3. ONE ENTRY PER LINE (`entry_line`), when the document has no blank lines.

    Text preceding the first headword (a section's prose introduction) is not forced
    into the first entry: it is split on blank lines so it stays independently packable.
    """
    # معجم: الوحدة هي المدخل الواحد (المدخل + شرحه + أمثلته). ثلاث اصطلاحات مدعومة
    # وتُجرَّب بهذا الترتيب:
    #   1. entry_headword — المدخل يبدأ بكلمة بين قوسين تليها نقطتان، ويمتد شرحه على
    #      أي عدد من الأسطر حتى يبدأ المدخل التالي. لا الأسطر الفارغة ولا فواصل
    #      الأسطر تفصل المداخل، وهذا هو اصطلاح المصدر الحالي. تقسيم مثل هذا المصدر
    #      بالأسطر الفارغة ينتج كتلًا بحجم الصفحة، وتقسيمه بفواصل الأسطر يمزّق كل شرح.
    #   2. entry_block — مداخل يفصل بينها سطر فارغ.
    #   3. entry_line — مدخل واحد في كل سطر، حين يخلو الملف من الأسطر الفارغة.
    # أما النص السابق لأول مدخل (مقدمة القسم) فيُقسَّم بالأسطر الفارغة حتى لا يُلحَق
    # بالمدخل الأول ويصير وحدة ضخمة.
    matches = list(HEADWORD_RE.finditer(text))
    if len(matches) >= MIN_HEADWORDS_FOR_ENTRY_SPLIT:
        starts = {m.start() for m in matches}
        # secondary break: sub-section headings also start a new unit, so a section's
        # title + prose introduction + feature list never trails off the last entry of
        # the section before it.
        for h in SECTION_HEADING_RE.finditer(text):
            line = h.group(0).strip()
            if HEADWORD_RE.match(line):
                continue                      # an entry that merely begins with "لهجة"
            if len(line.split()) > MAX_SECTION_HEADING_WORDS:
                continue                      # a wrapped prose line, not a heading
            starts.add(h.start())

        cuts = sorted(starts)
        units: list[str] = []
        preamble = text[: cuts[0]].strip()
        if preamble:
            units.extend(p.strip() for p in preamble.split(PARAGRAPH_SEPARATOR) if p.strip())
        for i, pos in enumerate(cuts):
            end = cuts[i + 1] if i + 1 < len(cuts) else len(text)
            unit = text[pos:end].strip()
            if not unit:
                continue
            if HEADWORD_RE.match(unit):
                units.append(unit)             # a glossary entry stays whole
            else:
                # a sub-section block (heading + prose introduction + feature list) is
                # ordinary prose and can run for pages, so split it at paragraph
                # boundaries rather than emitting one unpackable unit
                units.extend(p.strip() for p in unit.split(PARAGRAPH_SEPARATOR) if p.strip())
        return units, "entry_headword"

    blocks = [b.strip() for b in text.split(PARAGRAPH_SEPARATOR) if b.strip()]
    if len(blocks) > 1:
        return blocks, "entry_block"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines, "entry_line"


SEGMENTERS = {
    "prose": segment_prose,
    "verse": segment_verse,
    "dictionary": segment_dictionary,
}


# --- Header (front-matter) extraction ----------------------------------------
# فصل مقدمة الكتاب: العنوان وأسماء المؤلف/المترجم/المراجع تُنزع من متن النص
# وتُحفظ كحقول وصفية مستقلة، لا تُحذف.
def _titles_match(paragraph: str, title: str) -> bool:
    """Loose comparison so a diacritized/normalized repeat of the title still matches."""
    # مقارنة متساهلة للعناوين: تتجاهل التشكيل والتطويل وتوحّد الألف والياء، حتى
    # يُتعرَّف على تكرار العنوان في الصفحات الأولى ولو اختلف ضبطه.
    def key(s: str) -> str:
        s = araby.strip_tashkeel(araby.strip_tatweel(s))
        s = s.translate(NORMALIZE_TABLE)
        s = re.sub(r"[^\w\s]", " ", s)
        return " ".join(s.split())

    p, t = key(paragraph), key(title)
    if not p:
        return False
    # the corpus title sometimes carries a " : subtitle" tail
    return p == t or p in t or t.startswith(p)


def extract_header(paragraphs: list[str], title: str) -> tuple[dict, list[str], list[str]]:
    """Split leading front matter off the body.

    Returns (roles, header_paragraphs, body_paragraphs). A paragraph belongs to the
    header if it is a role block (تأليف/ترجمة/...), a repeat of the book title, or a
    short line appearing before the first role block (subtitle). Scanning stops at the
    first paragraph that fails all three tests.
    """
    # فصل المقدمة: تُعد الفقرة جزءًا من المقدمة إذا كانت كتلة دور (تأليف/ترجمة/مراجعة)،
    # أو تكرارًا للعنوان، أو سطرًا قصيرًا سابقًا لأول كتلة دور (عنوان فرعي). يتوقف
    # المسح عند أول فقرة تفشل في الاختبارات الثلاثة، وتُحفظ الأسماء في حقول مستقلة.
    roles: dict[str, list[str]] = {}
    header: list[str] = []
    seen_role = False
    idx = 0

    for idx, para in enumerate(paragraphs):
        if idx >= MAX_HEADER_PARAGRAPHS:
            break
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            header.append(para)
            continue

        marker = lines[0].strip(" :：")
        if marker in ROLE_MARKERS and len(lines) > 1:
            field = ROLE_MARKERS[marker]
            roles.setdefault(field, []).extend(lines[1:])
            header.append(para)
            seen_role = True
            continue

        if _titles_match(para, title):
            header.append(para)
            continue

        # subtitle line before any role block (e.g. "عن الحروب الأوروبية ماضيها وحاضرها")
        if not seen_role and len(para) <= 120 and "\n" not in para:
            header.append(para)
            continue

        break
    else:
        idx = len(paragraphs)

    if not seen_role:
        # No authorship block found -> be conservative, only strip leading title repeats.
        header, idx = [], 0
        for i, para in enumerate(paragraphs[:MAX_HEADER_PARAGRAPHS]):
            if _titles_match(para, title):
                header.append(para)
                idx = i + 1
            else:
                break

    flat = {k: " / ".join(v) for k, v in roles.items()}
    return flat, header, paragraphs[idx:]


# --- Per-document cleaning ---------------------------------------------------
# التنظيف على مستوى الوثيقة الواحدة: يجمع كل الخطوات السابقة ويُخرج السجل النهائي.
def clean_document(record: dict, strip_diacritics: bool = False, fmt: str = "prose",
                   strip_tatweel: bool = False) -> dict:
    raw_text = record["raw_text"]
    raw_chars = len(raw_text)

    # 1. encoding repair
    repaired = ftfy.fix_text(raw_text)
    ftfy_changed = repaired != raw_text

    # 2. format-aware segmentation
    normalized_all = normalize_whitespace(repaired)
    paragraphs, unit_type = SEGMENTERS[fmt](normalized_all)

    # 3. header detachment
    roles, header_paragraphs, body_paragraphs = extract_header(paragraphs, record.get("title", ""))

    # 4. normalization + placeholder accounting
    # Two parallel views of the body are kept:
    #   paragraphs           -> Arabic-normalized (alef/ya folded) - matching, dedup, default chunking
    #   paragraphs_original  -> encoding-repaired + whitespace-cleaned only, orthography intact
    # Alef/ya folding is lossy (نشأة -> نشاة, على -> علي). It is what you want as a
    # comparison key, but usually NOT what you want as generative training text, so the
    # unfolded form is preserved rather than thrown away.
    cleaned_paragraphs: list[str] = []
    original_paragraphs: list[str] = []
    placeholders: list[dict] = []
    for i, para in enumerate(body_paragraphs):
        if MEDIA_PLACEHOLDER_RE.match(para):
            placeholders.append({"body_index": i, "text": para})
            continue
        base = preserve_orthography(para, strip_diacritics=strip_diacritics,
                                    strip_tatweel=strip_tatweel).strip()
        norm = normalize_arabic(para, strip_diacritics=strip_diacritics).strip()
        if norm:
            cleaned_paragraphs.append(norm)
            original_paragraphs.append(base)

    cleaned_text = "\n\n".join(cleaned_paragraphs)
    original_text = "\n\n".join(original_paragraphs)
    header_text = "\n\n".join(header_paragraphs)
    roles = {k: normalize_arabic(v, strip_diacritics=strip_diacritics) for k, v in roles.items()}

    # 5. deltas + flags
    # Measured against the orthography-preserving text, since that is what ships in
    # chunks.jsonl. Separators are excluded: re-joining units with "\n\n" can ADD
    # characters relative to a single-newline source (a glossary, say), which would
    # otherwise show up as a nonsensical negative loss. Only content characters count.
    retained_chars = sum(len(u) for u in original_paragraphs) + sum(len(h) for h in header_paragraphs)
    loss_ratio = (raw_chars - retained_chars) / raw_chars if raw_chars else 0.0

    flags: list[str] = []
    if loss_ratio > CHAR_LOSS_FLAG_THRESHOLD:
        flags.append("high_char_loss")
    # Front matter is a book convention. Only prose sources are expected to carry it,
    # so flagging its absence on a glossary or a diwan would be pure noise.
    if fmt == "prose" and not roles:
        flags.append("no_header_detected")
    if placeholders:
        flags.append("media_placeholders_removed")
    if len(cleaned_paragraphs) < 5:
        flags.append("very_few_paragraphs")

    cleaned = dict(record)
    cleaned.pop("raw_text", None)
    region = record.get("region")
    cleaned.update(
        {
            "region": region,
            "source_format": fmt,
            "unit_type": unit_type,
            "cleaned_text": cleaned_text,
            "paragraphs": cleaned_paragraphs,
            "cleaned_text_original_orthography": original_text,
            "paragraphs_original": original_paragraphs,
            "header_block": header_text,
            "translator": roles.get("translator"),
            "reviewer": roles.get("reviewer"),
            "editor": roles.get("editor"),
            "author_stated": roles.get("author_stated"),
            "header_roles": roles,
            "removed_placeholders": placeholders,
            "cleaning_stats": {
                "char_count_raw": raw_chars,
                "char_count_clean_body": len(original_text),
                "char_count_header": len(header_text),
                "char_count_retained": retained_chars,
                "char_delta": raw_chars - retained_chars,
                "char_loss_ratio": round(loss_ratio, 4),
                "char_count_measured_against": "paragraphs_original + header units "
                                               "(orthography-preserving content, excluding "
                                               "structural separators)",
                "char_count_folded_body": len(cleaned_text),
                "unit_count_raw": len(paragraphs),
                "unit_count_body": len(original_paragraphs),
                "header_paragraph_count": len(header_paragraphs),
                "word_count_clean": word_count(original_text),
                "ftfy_changed_text": ftfy_changed,
            },
            "review_flags": flags,
            "cleaning_config": {
                "source_format": fmt,
                "unit_type": unit_type,
                "strip_diacritics": strip_diacritics,
                "strip_tatweel": strip_tatweel,
                "paragraph_separator": "\\n\\n",
                # applied to the SHIPPED text (paragraphs_original)
                "shipped_text_transforms": ["ftfy", "nfc", "whitespace"]
                + (["tatweel"] if strip_tatweel else [])
                + (["tashkeel"] if strip_diacritics else []),
                # applied to the folded MATCHING KEY only (paragraphs / cleaned_text)
                "matching_key_transforms": ["ftfy", "nfc", "tatweel", "alef_variants",
                                            "ya_alef_maksura", "whitespace"]
                + (["tashkeel"] if strip_diacritics else []),
                "char_loss_flag_threshold": CHAR_LOSS_FLAG_THRESHOLD,
            },
        }
    )
    return cleaned


# --- Corpus scoping ----------------------------------------------------------
# Cleaning settings (--format, --strip-tatweel, --strip-diacritics) apply to every
# document in a run, but data/interim/ can hold more than one corpus at a time - the
# dialect dictionary must KEEP tatweel while the classical lexicon must have it stripped.
# A run that silently spanned both would apply one corpus's settings to the other and
# report success, so a run is always scoped to exactly one corpus.
# تحديد نطاق المدونة: إعدادات التنظيف تسري على كل وثائق التشغيلة، بينما قد يحوي
# data/interim أكثر من مدونة في وقت واحد، ولكل مدونة إعداداتها. تشغيلة تشمل مدونتين
# ستطبّق إعدادات إحداهما على الأخرى وتُنهي عملها بنجاح ظاهري، ولذلك تُقصر كل تشغيلة
# على مدونة واحدة.
UNSPECIFIED_CORPUS = "(unspecified)"


def corpus_of(record: dict) -> str:
    """Corpus a record belongs to; records predating the field group under one name."""
    return record.get("corpus") or UNSPECIFIED_CORPUS


def scope_to_corpus(items: list, corpus: str | None, stage: str) -> list:
    """Filter (path, record) pairs to one corpus, or refuse if the scope is ambiguous.

    Never falls back to "process everything": that is the silent-failure this exists to
    prevent. With one corpus present and no --corpus given the run proceeds unchanged.

    UNLABELLED RECORDS ARE REFUSED, and that is the hard-won part of this function.
    ------------------------------------------------------------------------------
    This guard once passed a run that mixed two corpora into a single chunks.jsonl and
    overwrote another corpus's report, and it did so WITHOUT MALFUNCTIONING. The commit
    that added the guard also added the `corpus` field to the ingest scripts, because
    the guard is only as good as that field. A contributor branched from a commit that
    predated both, regenerated one corpus's interim with the older ingest, and wrote a
    new ingest for a third source modelled on the older ones - so neither document
    carried `corpus`. `corpus_of()` mapped both to UNSPECIFIED_CORPUS, the grouping saw
    ONE key, and the guard concluded the run was correctly scoped and let it through.

    The bug was not the bypass. It was that MISSING METADATA WAS READ AS AGREEMENT.
    UNSPECIFIED_CORPUS is not a corpus; it is the absence of an answer, and lumping
    every unlabelled record under one name makes distinct corpora indistinguishable
    exactly when there is least evidence that they match. A safety check whose failure
    mode is "no data, therefore consistent" fails open, and gets more dangerous the more
    unlabelled input it sees.

    So absence is now its own refusal, separate from the ambiguity refusal below. At the
    time this was written every record in data/interim carried `corpus`, making it a
    no-op on the existing corpora - it costs nothing today and only fires on the case
    that actually broke.

    السبب الجذري: غياب الحقل لا يعني الاتفاق. كانت السجلات غير الموسومة تُجمع تحت اسم
    واحد، فتبدو المدونتان مدونة واحدة، ويمر الفحص بنجاح ظاهري. الغياب الآن رفض مستقل.
    """
    unlabelled = [path for path, record in items if not record.get("corpus")]
    if unlabelled:
        listing = "\n".join(f"      {p}" for p in sorted(unlabelled)[:10])
        more = ("\n      ... and %d more" % (len(unlabelled) - 10)) if len(unlabelled) > 10 else ""
        raise SystemExit(
            f"[{stage}] REFUSING to run: {len(unlabelled)} of {len(items)} document(s)\n"
            f"    carry no `corpus` field. A missing corpus is NOT a corpus - grouping\n"
            f"    unlabelled records together would make two different corpora look like\n"
            f"    one and let a mixed run through as if it were correctly scoped.\n"
            f"    unlabelled:\n{listing}{more}\n"
            f"    Fix the ingest script that produced them so it writes `corpus`, then\n"
            f"    re-run that ingest. Do not pass --corpus to work around this: it would\n"
            f"    silently relabel documents whose real corpus is unknown."
        )

    groups: dict[str, list] = {}
    for path, record in items:
        groups.setdefault(corpus_of(record), []).append((path, record))

    if corpus is not None:
        if corpus not in groups:
            raise SystemExit(
                f"[{stage}] no documents for --corpus {corpus!r}. Found: "
                + ", ".join(f"{k} ({len(v)})" for k, v in sorted(groups.items()))
            )
        return groups[corpus]

    if len(groups) > 1:
        listing = "\n".join(f"      {k:<20} {len(v)} document(s)"
                             for k, v in sorted(groups.items()))
        raise SystemExit(
            f"[{stage}] REFUSING to run: data/interim holds more than one corpus and no\n"
            f"    --corpus was given. Settings apply to every document in a run, so one\n"
            f"    corpus's settings would be applied to the other with no error raised.\n"
            f"    corpora found:\n{listing}\n"
            f"    Re-run scoped, e.g. --corpus {sorted(groups)[0]}"
        )
    return items


def guard_report_overwrite(report_path: str, corpus: str | None, stage: str) -> None:
    """Refuse to overwrite a report that describes a DIFFERENT corpus.

    Defence in depth, and it exists because the scoping guard above was defeated once
    and nothing downstream noticed. Stage reports default to a single shared path
    (data/processed/chunking_report.json), so a run for one corpus silently replaces
    another corpus's provenance record - which is exactly what happened: a saudi_dialect
    report describing 337 chunks across 5 documents was replaced by an unscoped
    2-document run, and the loss was invisible until someone read the file weeks later.

    This check does not care WHY the corpus differs. Whether the scoping guard was
    bypassed, defeated, or simply not applicable, writing a report whose corpus does not
    match the one already on disk destroys information, so it stops. That independence
    is the point of a second layer: it holds even when the first one has been fooled.

    A run with no corpus (corpus is None) is refused against ANY labelled report, since
    an unscoped run is precisely the thing that caused the loss. Pass --report with a
    per-corpus path, as data/eda already does for its reports.

    طبقة ثانية: لا تُستبدل تقرير مدونة بتقرير مدونة أخرى، أيًّا كان سبب الاختلاف.
    """
    if not os.path.exists(report_path):
        return
    try:
        with open(report_path, encoding="utf-8") as f:
            existing = json.load(f)
    except (ValueError, OSError):
        return                      # unreadable or not JSON: not ours to reason about
    if not isinstance(existing, dict):
        return
    prior = (existing.get("config") or {}).get("corpus", _MISSING)
    if prior is _MISSING:
        return                      # no corpus recorded: nothing to compare against
    if prior == corpus:
        return

    raise SystemExit(
        f"[{stage}] REFUSING to overwrite {report_path}:\n"
        f"    the existing report describes corpus {prior!r}\n"
        f"    this run would write corpus      {corpus!r}\n"
        f"    Overwriting would delete that corpus's chunking provenance, which has\n"
        f"    happened before and went unnoticed for days. Write this run to its own\n"
        f"    path instead, e.g. --report {_per_corpus_report_path(report_path, corpus)}"
    )


_MISSING = object()


def _per_corpus_report_path(report_path: str, corpus: str | None) -> str:
    """Suggest <report>_<corpus>.json, matching the data/eda naming already in use."""
    base, ext = os.path.splitext(report_path)
    return f"{base}_{corpus or 'UNSCOPED'}{ext or '.json'}"


# --- Runner ------------------------------------------------------------------
# المشغِّل: يمر على كل ملفات الإدخال، ويكتب نسخة _cleaned.json لكل وثيقة،
# ثم تقرير تنظيف مجمَّع.
def iter_input_files(interim_dir: str) -> list[str]:
    return sorted(
        p for p in glob.glob(os.path.join(interim_dir, "*", "*.json"))
        if not p.endswith("_cleaned.json")
    )


def self_test() -> bool:
    """Exercise the two scoping guards, including the case that defeated one of them.

    A guard that has never been shown to refuse is not a guard. Each case below states
    what it would mean if it failed, because these particular checks are load-bearing:
    the corpus-mixing incident of 2026-09-09 got through a guard that was present,
    correct, and silent.
    """
    ok = True
    import tempfile

    def rec(doc_id, corpus=None):
        d = {"doc_id": doc_id, "raw_text": "x"}
        if corpus is not None:
            d["corpus"] = corpus
        return d

    def refuses(fn, *a):
        try:
            fn(*a)
            return None
        except SystemExit as e:
            return str(e)

    # ---- (a) absence of the field is refused, not treated as agreement --------------
    # THE REGRESSION THAT MATTERS. Two unlabelled records once grouped under one
    # synthetic key and the run was allowed through as "correctly scoped".
    msg = refuses(scope_to_corpus,
                  [("a.json", rec("asas_albalagha")),
                   ("n.json", rec("majam_alkalimat_alshaabia_najd"))], None, "chunk")
    if msg is None:
        print("  [FAIL] two UNLABELLED records were accepted - missing metadata is "
              "being read as agreement again"); ok = False
    elif "no `corpus` field" not in msg:
        print("  [FAIL] unlabelled records refused for the wrong reason: %s" % msg[:90])
        ok = False
    # a single unlabelled record among labelled ones is refused too: the mix is what
    # matters, not whether the unlabelled ones happen to agree with each other
    if refuses(scope_to_corpus,
               [("a.json", rec("asas_albalagha", "classical_lexicon")),
                ("n.json", rec("najdi"))], None, "chunk") is None:
        print("  [FAIL] one unlabelled record among labelled ones was accepted")
        ok = False
    # ...and --corpus must NOT be a way around it: relabelling a document whose real
    # corpus is unknown is exactly the silent mislabel this prevents
    if refuses(scope_to_corpus, [("n.json", rec("najdi"))],
               "classical_lexicon", "chunk") is None:
        print("  [FAIL] --corpus bypassed the unlabelled refusal"); ok = False

    # ---- regression: the ORIGINAL guard still behaves exactly as before -------------
    msg = refuses(scope_to_corpus,
                  [("a.json", rec("asas_albalagha", "classical_lexicon")),
                   ("d.json", rec("dialect_dict_najdi", "saudi_dialect"))], None, "chunk")
    if msg is None:
        print("  [FAIL] two genuinely different corpora were allowed to mix"); ok = False
    elif "more than one corpus" not in msg:
        print("  [FAIL] mixed corpora refused for the wrong reason: %s" % msg[:90])
        ok = False
    # one corpus, unscoped -> proceeds unchanged, as it always has
    one = [("a.json", rec("asas_albalagha", "classical_lexicon")),
           ("b.json", rec("asas_albalagha", "classical_lexicon"))]
    if len(scope_to_corpus(one, None, "chunk")) != 2:
        print("  [FAIL] a correctly scoped single-corpus run no longer passes"); ok = False
    # explicit --corpus still selects its subset
    two = one + [("d.json", rec("dialect_dict_najdi", "saudi_dialect"))]
    if len(scope_to_corpus(two, "classical_lexicon", "chunk")) != 2:
        print("  [FAIL] --corpus no longer selects its subset"); ok = False
    if refuses(scope_to_corpus, two, "no_such_corpus", "chunk") is None:
        print("  [FAIL] an unknown --corpus was accepted"); ok = False

    # ---- (b) a report describing another corpus is never overwritten ----------------
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "chunking_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"config": {"corpus": "saudi_dialect"}, "total_chunks": 337}, f)

        msg = refuses(guard_report_overwrite, path, "classical_lexicon", "chunk")
        if msg is None:
            print("  [FAIL] overwrote a report belonging to another corpus"); ok = False
        else:
            for needle in ("saudi_dialect", "classical_lexicon"):
                if needle not in msg:
                    print("  [FAIL] refusal does not name %r: %s" % (needle, msg[:90]))
                    ok = False
        # an UNSCOPED run against a labelled report is the exact incident shape
        if refuses(guard_report_overwrite, path, None, "chunk") is None:
            print("  [FAIL] an unscoped run overwrote a labelled report"); ok = False
        # same corpus -> allowed, or the tool could never update its own report
        if refuses(guard_report_overwrite, path, "saudi_dialect", "chunk") is not None:
            print("  [FAIL] refused to rewrite a report for the SAME corpus"); ok = False
        # no existing file -> nothing to protect
        if refuses(guard_report_overwrite, os.path.join(td, "new.json"),
                   "anything", "chunk") is not None:
            print("  [FAIL] refused to create a report that did not exist yet"); ok = False
        # unreadable / non-JSON -> not ours to reason about, must not crash the run
        bad = os.path.join(td, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("not json{")
        if refuses(guard_report_overwrite, bad, "x", "chunk") is not None:
            print("  [FAIL] a non-JSON report aborted the run"); ok = False
        # a report with no corpus recorded -> nothing to compare, must not block
        nc = os.path.join(td, "nocorpus.json")
        with open(nc, "w", encoding="utf-8") as f:
            json.dump({"config": {}, "total_chunks": 1}, f)
        if refuses(guard_report_overwrite, nc, "x", "chunk") is not None:
            print("  [FAIL] a report with no corpus field blocked the run"); ok = False

    # ---- no-op on the real tree: every current record already carries `corpus` ------
    real = []
    for p in sorted(glob.glob(os.path.join(INTERIM_DIR, "*", "*_cleaned.json"))):
        with open(p, encoding="utf-8") as f:
            real.append((p, json.load(f)))
    if real:
        missing = [p for p, r in real if not r.get("corpus")]
        if missing:
            print("  [FAIL] %d existing interim record(s) lack `corpus`, so this change "
                  "is NOT a no-op: %s" % (len(missing), missing[:3])); ok = False
        else:
            print("  no-op check: all %d interim record(s) carry `corpus`" % len(real))
    else:
        print("  no-op check: SKIPPED, no interim records on this machine")

    print("passed: %s" % ok)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: deterministic cleaning.")
    ap.add_argument("--self-test", action="store_true",
                    help="exercise the corpus-scoping and report-overwrite guards")
    ap.add_argument("--interim-dir", default=INTERIM_DIR)
    ap.add_argument("--report", default=REPORT_PATH)
    ap.add_argument("--corpus", default=None,
                    help="Process only documents whose `corpus` field matches. Required "
                         "whenever data/interim holds more than one corpus - the run is "
                         "refused rather than silently applying one corpus's settings to "
                         "another.")
    ap.add_argument("--format", choices=FORMATS, default="prose", dest="fmt",
                    help="Source structure of this batch (default: prose). Chosen explicitly - "
                         "no auto-detection. prose = blank-line paragraphs; dictionary = glossary "
                         "entries; verse = stanzas/verse lines.")
    ap.add_argument("--strip-diacritics", action="store_true",
                    help="Strip tashkeel (default OFF: dialect transcription and poetry often "
                         "carry meaningful vocalization).")
    ap.add_argument("--strip-tatweel", action="store_true",
                    help="Strip tatweel from the SHIPPED text (default OFF: expressive elongation "
                         "such as يا هــلا is dialect signal). The folded matching key always has "
                         "tatweel stripped regardless.")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    files = iter_input_files(args.interim_dir)
    if not files:
        raise SystemExit(f"No input JSON found under {args.interim_dir}")

    loaded = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            loaded.append((path, json.load(f)))
    loaded = scope_to_corpus(loaded, args.corpus, "clean")
    print(f"[clean] corpus: {corpus_of(loaded[0][1])} | {len(loaded)} document(s)")

    report = {
        "documents": [],
        "flagged_for_review": [],
        "config": {"corpus": args.corpus,
                   "source_format": args.fmt,
                   "strip_diacritics": args.strip_diacritics,
                   "strip_tatweel": args.strip_tatweel,
                   "char_loss_flag_threshold": CHAR_LOSS_FLAG_THRESHOLD},
    }

    for path, record in loaded:
        cleaned = clean_document(record, strip_diacritics=args.strip_diacritics,
                                 fmt=args.fmt, strip_tatweel=args.strip_tatweel)
        out_path = path[: -len(".json")] + "_cleaned.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)

        stats = cleaned["cleaning_stats"]
        entry = {
            "doc_id": cleaned["doc_id"],
            "region": cleaned["region"],
            "source_format": cleaned["source_format"],
            "unit_type": cleaned["unit_type"],
            "title": cleaned["title"],
            "output": out_path.replace("\\", "/"),
            "translator": cleaned.get("translator"),
            "reviewer": cleaned.get("reviewer"),
            **stats,
            "review_flags": cleaned["review_flags"],
        }
        report["documents"].append(entry)
        if cleaned["review_flags"]:
            report["flagged_for_review"].append(entry)

        print(
            f"[clean] {cleaned['doc_id']:>10} {str(cleaned['region']):<10} "
            f"chars {stats['char_count_raw']:>7} -> {stats['char_count_retained']:>7} "
            f"({stats['char_loss_ratio']*100:5.2f}% removed)  "
            f"{cleaned['unit_type']}s {stats['unit_count_body']:>5}  "
            f"flags: {','.join(cleaned['review_flags']) or '-'}"
        )

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    total_raw = sum(d["char_count_raw"] for d in report["documents"])
    total_keep = sum(d["char_count_retained"] for d in report["documents"])
    print(
        f"\n[clean] {len(report['documents'])} documents cleaned | "
        f"{total_raw:,} -> {total_keep:,} chars "
        f"({(total_raw-total_keep)/total_raw*100:.2f}% removed overall) | "
        f"{len(report['flagged_for_review'])} flagged for review"
    )
    print(f"[clean] report -> {args.report}")


if __name__ == "__main__":
    main()
