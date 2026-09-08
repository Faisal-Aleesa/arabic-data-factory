"""Acquisition - معجم اللهجات المحكية -> per-region interim documents.

This is the ACQUISITION step for the regional dialect corpus, not a fourth preprocessing
stage. The cleaning / dedup / chunking stages are unchanged and still start from
data/interim/.

Input : data/raw/<region>/dialect_dictionary_<region>.txt   (pdftotext -layout output,
                                                              form-feed separated pages)
Output: data/interim/<region>/<doc_id>.json                  (clean.py's input schema)
        docs/damaged_pages_review.csv                        (pages held back, for review)
        docs/residue_tokens_review.csv                       (short fragments, for review)

Why a page-aware step exists at all
-----------------------------------
Page boundaries only survive in the raw extraction (\\f separators). Two things have to
happen while they are still visible:

  * PAGE FURNITURE - the printed folio ("- 26 -") sits on every page and would otherwise
    land inside a glossary entry.
  * DAMAGED PAGES - a minority of pages in the source PDF draw overlapping text runs that
    pdftotext interleaves into unreadable tokens
    (two columns colliding into one run of letters, e.g. "أبجدهـوزحطي" where two separate
    words were drawn at overlapping positions). Those pages are EXCLUDED from the
    document body and logged with their page number and a snippet. No automatic repair is
    attempted - the review file is the handle for redoing them later with a different
    extraction approach.

Nothing is deleted: excluded pages stay in data/raw/ and are enumerated in the CSV.

شرح بالعربية
------------
مرحلة الاقتناء: تحويل النص المستخرج من الكتاب إلى وثيقة لكل منطقة، بصيغة تقرأها
مرحلة التنظيف. ليست مرحلة معالجة رابعة؛ مراحل التنظيف وكشف التكرار والتقطيع لم تتغير
وما زالت تبدأ من data/interim/.

لماذا توجد مرحلة تعي حدود الصفحات أصلًا؟ لأن حدود الصفحات لا تنجو إلا في النص الخام
(بفواصل \f)، وأمران يجب أن يُنجزا قبل أن تختفي:
  * ترقيم الصفحات المطبوع، الذي سيقع داخل مدخل معجمي لو تُرك.
  * الصفحات التالفة، وهي أقلية رسم فيها الملف الأصلي نصوصًا متراكبة يستخرجها البرنامج
    حروفًا متشابكة غير مقروءة. تُستبعد هذه الصفحات من متن الوثيقة وتُسجَّل بأرقامها
    ومقتطف منها. لا تُحاوَل أي معالجة تلقائية لها؛ ملف المراجعة هو المدخل لإعادة
    استخراجها لاحقًا بطريقة أخرى.

لا يُحذف شيء: الصفحات المستبعدة تبقى في data/raw/ ومُحصاة في ملف المراجعة.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re

RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"
REVIEW_CSV = "docs/damaged_pages_review.csv"
RESIDUE_CSV = "docs/residue_tokens_review.csv"

# First printed page contained in each region's extract (see docs: verified against the
# book's five part-title pages).
REGION_FIRST_PAGE = {
    "najdi": 19,
    "northern": 325,
    "western": 451,
    "southern": 516,
    "eastern": 793,
}

TITLE = "معجم اللهجات المحكية في المملكة العربية السعودية"
AUTHOR = "سليمان بن ناصر الدرسوني"
SOURCE = "provided directly by the team, not from a public repo"
LICENSE = "unverified_pending_review"

# The corpus this acquisition belongs to. Cleaning settings (--format, --strip-tatweel,
# --strip-diacritics) are corpus-wide, so every stage scopes its run by this value rather
# than processing whatever happens to sit in data/interim/. Same vocabulary as the
# manifest's corpus_phase column.
# المدونة التي ينتمي إليها هذا الاقتناء. إعدادات التنظيف تسري على المدونة كلها، ولذلك
# تُقصر كل مرحلة تشغيلها على هذه القيمة بدل معالجة كل ما يصادفها في data/interim.
CORPUS = "saudi_dialect"

BIDI_RE = re.compile(r"[​-‏‪-‮﻿]")
FOLIO_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$")
ARABIC_TOKEN_RE = re.compile(r"[ء-ٰٟـ]+")

# Damage detector. Arabic words top out around 12-13 letters; interleaved text runs
# produce far longer pseudo-tokens. A page is held back when more than
# DAMAGE_TOKEN_RATIO of its Arabic tokens exceed DAMAGE_TOKEN_LEN characters.
# كاشف التلف: الكلمة العربية لا تكاد تتجاوز اثني عشر أو ثلاثة عشر حرفًا، أما النصوص
# المتراكبة فتنتج كلمات وهمية أطول من ذلك بكثير. تُستبعد الصفحة إذا تجاوزت نسبة هذه
# الكلمات الحدَّ المحدد.
DAMAGE_TOKEN_LEN = 16
DAMAGE_TOKEN_RATIO = 0.02

# --- Displaced-diacritic repair ----------------------------------------------
# The producer emits each vocalized word's diacritics as separate zero-width glyphs,
# out of sequence relative to their base letters, so every extractor breaks the word:
#     printed  "qlmk + kasra"   extracted  "qlm <space> kasra k"
# Glyph geometry showed a mark's x0 coincides with its base letter's x0 to ~0.06pt, and
# the base always FOLLOWS the mark. Where the mark is preceded by something that cannot
# carry it (space, punctuation, line start) the attachment is unambiguous, so the repair
# is a deterministic rewrite - no dictionary, no guessing.
#
# Deliberately NOT repaired: a mark already preceded by a letter or tatweel may be
# correctly placed, and geometric ordering of cursive Arabic proved unreliable (a
# geometry-driven extractor was built and validated against pdftotext -layout: it
# transposes letters inside words, so it was rejected). Those marks stay as they are.
# إصلاح التشكيل المُزاح — الخلفية بالعربية:
# البرنامج المنتِج للملف يرسم حركات كل كلمة مشكولة كرموز مستقلة عديمة العرض، ويضعها
# في ترتيب لا يطابق ترتيب حروفها. لذلك يكسر كل برنامج استخراج الكلمةَ عند هذا الموضع:
# ما يُطبع كلمة واحدة مشكولة يخرج نصًا ككلمتين بينهما مسافة، والحركة معلقة بلا حرف.
#
# فحص هندسة الرموز في الملف أظهر أن الإحداثي الأفقي لكل حركة يطابق إحداثي حرفها
# الأساس بفارق لا يتجاوز 0.06 نقطة، وأن الحرف الأساس يأتي دائمًا بعد الحركة. ومن ثم:
# إذا سبق الحركةَ شيء لا يصلح أن يحملها (مسافة أو علامة ترقيم أو بداية سطر) فإسنادها
# إلى الحرف التالي قطعي لا تخمين فيه، وهذا ما تفعله repair_diacritics.
#
# ما لا يُصلَح عمدًا: الحركة المسبوقة بحرف أو بتطويل قد تكون في موضعها الصحيح أصلًا،
# ولا سبيل نصيًّا للتمييز.
#
# تجربة مرفوضة (١): بُني مستخرِج قائم على هندسة الرموز ليحل محل pdftotext، وقُيس
# مخرجه سطرًا بسطر مقابل pdftotext -layout فلم يطابقه إلا في نحو 10% من السطور، لأنه
# يقلب ترتيب الحروف داخل الكلمة (تخرج "في" مثلًا "يف"). السبب أن صناديق الحروف في
# الخط العربي المتصل تتراكب، فترتيبها بالإحداثيات غير موثوق. رُفض المستخرِج وبقي
# pdftotext -layout. لا تُعَد التجربة مرة أخرى دون معالجة هذه النقطة.
LETTER_CLS = "ء-غف-ي"
MARK_CLS = "ً-ٰٕۖ-ۭ"
TATWEEL = "ـ"

DETACHED_RE = re.compile(
    r"(?P<pre>[" + LETTER_CLS + r"])[ ]+(?P<mk>[" + MARK_CLS + r"]+)(?P<base>[" + LETTER_CLS + r"])"
)
LEADING_MARK_RE = re.compile(
    r"(?P<pre>^|[(){}\[\]«».,:؛،\-\"]|\n)"
    r"(?P<mk>[" + MARK_CLS + r"]+)(?P<base>[" + LETTER_CLS + r"])",
    re.M,
)


def repair_diacritics(text):
    """Reattach unambiguously displaced marks. Returns (text, space_joins, mark_moves)."""
    # إعادة إسناد الحركات المُزاحة في الحالات القطعية وحدها: تُحذف المسافة الدخيلة
    # وتُنقل الحركة إلى ما بعد حرفها الأساس. تُكرَّر العملية لأن حركات عدة قد تتوالى
    # في الكلمة الواحدة.
    counts = {"join": 0, "move": 0}

    def join(m):
        counts["join"] += 1
        return m.group("pre") + m.group("base") + m.group("mk")

    def move(m):
        counts["move"] += 1
        return m.group("pre") + m.group("base") + m.group("mk")

    prev, out = None, text
    while prev != out:                      # marks can chain within one word
        prev = out
        out = DETACHED_RE.sub(join, out)
    out = LEADING_MARK_RE.sub(move, out)
    return out, counts["join"], counts["move"]


def detached_mark_count(text):
    """Marks with no valid base immediately before them (tatweel counts as a base)."""
    # عدّ الحركات غير المسندة، لقياس أثر الإصلاح قبله وبعده. التطويل يُعد حاملًا
    # مشروعًا للحركة فلا يُحسب ضمنها.
    mark = re.compile("[" + MARK_CLS + "]")
    letter = re.compile("[" + LETTER_CLS + "]")
    n = 0
    for i, ch in enumerate(text):
        if mark.match(ch):
            p = text[i - 1] if i else ""
            if not (letter.match(p) or mark.match(p) or p == TATWEEL):
                n += 1
    return n


# Short tokens that survive the repair and are neither Arabic particles nor the
# dictionary's own abbreviations. Logged for review, never auto-joined: a dictionary-
# backed rejoin (arramooz + clitic stripping) was tested and produced false
# confirmations, so this is a review list rather than a fix list.
# بقايا الكلمات القصيرة — الخلفية بالعربية:
# بعد إصلاح التشكيل تبقى كلمات من حرف أو حرفين ليست أدوات عربية معروفة ولا اختصارات
# معجمية (مثل ص للصفحة، وج للجمع). بعضها شظايا كلمات انقسمت فعلًا، وبعضها كلمات سليمة.
#
# تجربة مرفوضة (٢): جُرِّب وصلها آليًا بالتحقق من الشكل الموصول في معجم عربي حقيقي
# (arramooz مع تقشير السوابق واللواحق). أنتج الأسلوب تأكيدات كاذبة: يقبل وصل كلمتين
# سليمتين في كلمة لا وجود لها، لأن تقشير السوابق واللواحق متساهل بما يكفي لإيجاد
# جذر لأي تركيب تقريبًا. كما جُرِّب استعمال التطويل في آخر الشظية دليلًا على الانقسام،
# فتبيّن أن التطويل في هذا المصدر زخرفي داخل كلمات تامة، فسقط الدليل.
#
# لذلك: تُسجَّل هذه البقايا في ملف مراجعة بأرقام صفحاتها وسياقها، ولا تُوصل آليًا.
# القاعدة المتبعة أن يُعلَّم الملتبس للمراجعة اليدوية لا أن يُخمَّن.
PARTICLES = set(("و في من ما لا ان "
                 "أن إن ثم قد هو هي "
                 "به له بك لك لم لن "
                 "عن على إلى أو او "
                 "يا ها اي أي كل مع "
                 "بل هل كي إذ اذ "
                 "لو").split())
ABBREVIATIONS = set("ص ج د ع ح ش ق ط خ ض "
                    "ف ك ل م ن ه ب ت ث ر "
                    "ز س غ ي ا و ء".split())
LETTER_ONLY_RE = re.compile("[" + LETTER_CLS + "]")


def residue_tokens(text):
    """(token, context) for <=2-letter tokens that look like fragments, not words."""
    # استخراج البقايا مع سياقها لملف المراجعة. القائمة متساهلة عمدًا: قد تدخل فيها
    # كلمات سليمة، وهذا أفضل من إغفال شظية حقيقية في قائمة يراجعها إنسان.
    out = []
    toks = text.split()
    for i, t in enumerate(toks):
        bare = "".join(LETTER_ONLY_RE.findall(t))
        if not bare or len(bare) > 2 or bare in PARTICLES or bare in ABBREVIATIONS:
            continue
        out.append((t, " ".join(toks[max(0, i - 3):i + 4])[:120]))
    return out



def page_lines(page_text: str) -> list[str]:
    # سطور الصفحة بعد حذف علامات اتجاه النص وسطر ترقيم الصفحة المطبوع.
    lines = [l.strip() for l in BIDI_RE.sub("", page_text).split("\n")]
    return [l for l in lines if l and not FOLIO_RE.match(l)]


def damage_score(lines: list[str]) -> tuple[float, str]:
    """(ratio of over-long Arabic tokens, worst snippet)."""
    # درجة التلف: نسبة الكلمات المفرطة الطول، مع أسوأ مقتطف يوضح المشكلة في التقرير.
    toks = [t for l in lines for t in ARABIC_TOKEN_RE.findall(l)]
    if not toks:
        return 0.0, ""
    bad = [t for t in toks if len(t) > DAMAGE_TOKEN_LEN]
    if not bad:
        return 0.0, ""
    worst = max(bad, key=len)
    snippet = next((l for l in lines if worst in l), worst)
    return len(bad) / len(toks), snippet.strip()[:160]


def ingest_region(region: str, raw_path: str, first_page: int) -> tuple[dict, list[dict], list[dict]]:
    # اقتناء منطقة واحدة: تقسيم النص إلى صفحات، ثم لكل صفحة حذف الترقيم المطبوع،
    # وفحص التلف واستبعاد التالف، وإصلاح التشكيل، وجمع البقايا. ثم تُركَّب الصفحات
    # المُبقاة في متن واحد ويُبنى سجل الوثيقة بإحصاءاته.
    raw = io.open(raw_path, encoding="utf-8").read()
    pages = raw.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()

    kept_pages: list[str] = []
    damaged: list[dict] = []
    residue: list[dict] = []
    n_join = n_move = 0
    detached_before = detached_after = 0

    for i, page in enumerate(pages):
        page_no = first_page + i
        lines = page_lines(page)
        if not lines:
            continue
        ratio, snippet = damage_score(lines)
        if ratio > DAMAGE_TOKEN_RATIO:
            damaged.append({
                "region": region,
                "page": page_no,
                "garbled_token_ratio": round(ratio, 4),
                "garbled_snippet": snippet,
                "action": "excluded_from_chunking",
                "source_file": raw_path.replace("\\", "/"),
            })
            continue

        page_text = "\n".join(lines)
        detached_before += detached_mark_count(page_text)
        page_text, j, m = repair_diacritics(page_text)
        n_join += j
        n_move += m
        detached_after += detached_mark_count(page_text)

        # per-page so the review list carries a page number to look the token up on
        for tok, ctx in residue_tokens(page_text):
            residue.append({
                "region": region,
                "page": page_no,
                "token": tok,
                "context": ctx,
                "action": "flagged_for_manual_review",
                "note": "short fragment; no dictionary-safe rejoin available",
            })

        kept_pages.append(page_text)

    body = "\n\n".join(kept_pages)
    record = {
        "doc_id": f"dialect_dict_{region}",
        "corpus": CORPUS,
        "title": f"{TITLE} - {region}",
        "author": AUTHOR,
        "region": region,
        "source": SOURCE,
        "license": LICENSE,
        "word_count": len(body.split()),
        "page_range": [first_page, first_page + len(pages) - 1],
        "pages_total": len(pages),
        "pages_kept": len(kept_pages),
        "pages_excluded_damaged": len(damaged),
        "excluded_pages": [d["page"] for d in damaged],
        "diacritic_repair": {
            "space_joins": n_join,
            "mark_moves": n_move,
            "detached_marks_before": detached_before,
            "detached_marks_after": detached_after,
            "method": "deterministic reattachment of unambiguously displaced marks; "
                      "ambiguous marks left untouched",
        },
        "residue_tokens_flagged": len(residue),
        "raw_text": body,
    }
    return record, damaged, residue


def main() -> None:
    ap = argparse.ArgumentParser(description="Acquisition: dialect dictionary -> interim JSON.")
    # واجهة سطر الأوامر: يمر على مناطق REGION_FIRST_PAGE، ويكتب وثيقة لكل منطقة،
    # وملفَّي مراجعة: الصفحات التالفة، وبقايا الكلمات القصيرة.
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--interim-dir", default=INTERIM_DIR)
    ap.add_argument("--review-csv", default=REVIEW_CSV)
    ap.add_argument("--residue-csv", default=RESIDUE_CSV)
    args = ap.parse_args()

    all_damaged: list[dict] = []
    all_residue: list[dict] = []
    summary = []

    for region, first_page in REGION_FIRST_PAGE.items():
        raw_path = os.path.join(args.raw_dir, region, f"dialect_dictionary_{region}.txt")
        if not os.path.exists(raw_path):
            print(f"[ingest] MISSING {raw_path} - skipping {region}")
            continue

        record, damaged, residue = ingest_region(region, raw_path, first_page)
        all_damaged.extend(damaged)
        all_residue.extend(residue)

        out_dir = os.path.join(args.interim_dir, region)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, record["doc_id"] + ".json")
        with io.open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        summary.append(record)
        d = record["diacritic_repair"]
        print(f"[ingest] {region:<9} pages {record['pages_kept']:>4}/{record['pages_total']:<4} "
              f"(-{record['pages_excluded_damaged']} damaged)  "
              f"{record['word_count']:>8,} words | diacritics: {d['space_joins']:>5} joins, "
              f"{d['mark_moves']:>4} moves, detached {d['detached_marks_before']:>5}"
              f"->{d['detached_marks_after']:<5} | residue {record['residue_tokens_flagged']:>4}")

    os.makedirs(os.path.dirname(args.review_csv), exist_ok=True)
    with io.open(args.review_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["region", "page", "garbled_token_ratio",
                                          "garbled_snippet", "action", "source_file"])
        w.writeheader()
        w.writerows(sorted(all_damaged, key=lambda d: d["page"]))

    with io.open(args.residue_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["region", "page", "token", "context",
                                          "action", "note"])
        w.writeheader()
        w.writerows(sorted(all_residue, key=lambda d: (d["region"], d["page"])))

    tot_pages = sum(r["pages_total"] for r in summary)
    tot_kept = sum(r["pages_kept"] for r in summary)
    tot_words = sum(r["word_count"] for r in summary)
    print(f"\n[ingest] {len(summary)} regions | pages {tot_kept:,}/{tot_pages:,} kept "
          f"({len(all_damaged)} excluded as damaged) | {tot_words:,} usable words")
    jb = sum(r["diacritic_repair"]["space_joins"] for r in summary)
    mv = sum(r["diacritic_repair"]["mark_moves"] for r in summary)
    db = sum(r["diacritic_repair"]["detached_marks_before"] for r in summary)
    da = sum(r["diacritic_repair"]["detached_marks_after"] for r in summary)
    print(f"[ingest] diacritic repair: {jb:,} space-joins + {mv:,} mark-moves | "
          f"detached marks {db:,} -> {da:,} ({(db-da)/max(db,1)*100:.1f}% reattached)")
    print(f"[ingest] {len(all_residue):,} residue tokens flagged for manual review")
    print(f"[ingest] damaged-page review file -> {args.review_csv}")
    print(f"[ingest] residue review file      -> {args.residue_csv}")


if __name__ == "__main__":
    main()
