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
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import unicodedata

import ftfy
from pyarabic import araby

# --- Paths -------------------------------------------------------------------
INTERIM_DIR = "data/interim"
REPORT_PATH = "data/processed/cleaning_report.json"

# --- Cleaning parameters -----------------------------------------------------
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
# keeps this off numbered feature lines like "( :)1ظاهرة صوتية" and off mid-entry
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

# Front-matter role markers used by Hindawi e-books. The marker sits on its own
# line, the name(s) follow on the next line(s) inside the same paragraph.
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
def normalize_whitespace(text: str) -> str:
    """NBSP/zero-width cleanup, collapse runs of spaces, keep newlines meaningful."""
    text = text.replace(" ", " ")                    # NBSP
    text = re.sub(r"[​-‏‪-‮﻿]", "", text)  # zero-width / bidi marks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def normalize_arabic(text: str, strip_diacritics: bool = False) -> str:
    """Folded MATCHING KEY only - never shipped. Alef/ya folded, tatweel always stripped."""
    text = unicodedata.normalize("NFC", text)
    text = araby.strip_tatweel(text)
    text = text.translate(NORMALIZE_TABLE)
    if strip_diacritics:
        text = araby.strip_tashkeel(text)
    return text


def preserve_orthography(text: str, strip_diacritics: bool = False,
                         strip_tatweel: bool = False) -> str:
    """Shipped text. Alef/ya spelling is left exactly as written; tatweel only on request."""
    text = unicodedata.normalize("NFC", text)
    if strip_tatweel:
        text = araby.strip_tatweel(text)
    if strip_diacritics:
        text = araby.strip_tashkeel(text)
    return text


def word_count(text: str) -> int:
    return len(text.split())


# --- Segmentation ------------------------------------------------------------
# Each segmenter returns (units, unit_type). A "unit" is the smallest thing Stage 3 is
# allowed to treat as indivisible: a prose paragraph, a whole glossary entry, a whole
# stanza. Splitting rules differ per source type, and getting this wrong upstream is
# invisible downstream - a glossary segmented as prose becomes one giant blob.
def segment_prose(text: str) -> tuple[list[str], str]:
    """Blank-line separated paragraphs; single newlines stay inside a paragraph."""
    return [p.strip() for p in text.split(PARAGRAPH_SEPARATOR) if p.strip()], "paragraph"


def segment_verse(text: str) -> tuple[list[str], str]:
    """Stanzas (blank-line separated), internal line breaks preserved.

    A poem written without blank lines has no stanza structure to recover, so each line
    becomes its own unit rather than silently gluing the whole poem into one blob.
    """
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
def _titles_match(paragraph: str, title: str) -> bool:
    """Loose comparison so a diacritized/normalized repeat of the title still matches."""
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
    # `region` replaces the MSA phase's `domain` as the partition key.
    region = record.get("region") or record.get("domain")
    cleaned.pop("domain", None)
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


# --- Runner ------------------------------------------------------------------
def iter_input_files(interim_dir: str) -> list[str]:
    return sorted(
        p for p in glob.glob(os.path.join(interim_dir, "*", "*.json"))
        if not p.endswith("_cleaned.json")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: deterministic cleaning.")
    ap.add_argument("--interim-dir", default=INTERIM_DIR)
    ap.add_argument("--report", default=REPORT_PATH)
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

    files = iter_input_files(args.interim_dir)
    if not files:
        raise SystemExit(f"No input JSON found under {args.interim_dir}")

    report = {
        "documents": [],
        "flagged_for_review": [],
        "config": {"source_format": args.fmt,
                   "strip_diacritics": args.strip_diacritics,
                   "strip_tatweel": args.strip_tatweel,
                   "char_loss_flag_threshold": CHAR_LOSS_FLAG_THRESHOLD},
    }

    for path in files:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)

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
