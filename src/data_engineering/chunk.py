"""Stage 3 - Chunking.

Input : data/interim/<region>/<doc_id>_cleaned.json  (Stage 1 output)
        data/processed/dedup_report.json             (Stage 2, optional - skips exact dupes)
Output: data/processed/chunks.jsonl                  (one JSON object per line)
        data/processed/chunking_report.json

Text variant
------------
`chunk_text` is built from Stage 1's `paragraphs_original` - the encoding-repaired,
whitespace-cleaned text with Arabic orthography INTACT (أ/إ/آ and ى preserved). Alef/Yaa
folding is lossy (نشأة -> نشاة, على -> علي) and exists only as a matching key: dedup.py
consumes the folded `cleaned_text` for SHA-256 and MinHash. Folded text is never written
into the final chunks. `--text-variant normalized` can emit the folded text for
inspection, but the default is and should stay `original`.

Token counting
--------------
`token_count` is a WORD-BASED APPROXIMATION: whitespace-delimited tokens
(`len(text.split())`). No tokenizer, no model call - the stage stays fully
deterministic and dependency-free. For Modern Standard Arabic, a subword tokenizer
(e.g. Llama/Qwen SentencePiece) typically emits ~1.5-2.5 subword tokens per whitespace
word, so a 200-800 word chunk lands roughly in the 300-2000 subword-token band. Retune
TARGET_MIN/TARGET_MAX once the training tokenizer is fixed.

Formats (--format, explicit - no auto-detection)
------------------------------------------------
Stage 1 has already segmented each document into indivisible UNITS according to its own
--format, and Stage 3 only ever packs whole units. What changes per format is the unit
being packed and the format_type tag emitted:

  prose       paragraphs -> "narrative_paragraph" / "prose" / "list" / "footnote_block"
  dictionary  glossary entries -> "dictionary_entry". Short entries are bundled together
              to approach the target band instead of emitting one tiny chunk per entry;
              an entry is never split across chunks.
  verse       stanzas (or verse lines) -> "verse". Stanzas are packed whole, never broken
              mid-stanza, and chunks never span a poem boundary.

Pass the same --format to clean.py and chunk.py for a given batch. A mismatch is caught
and refused rather than silently producing garbage.

Boundaries
----------
Chunks break ONLY at paragraph boundaries. The corpus separates paragraphs with a blank
line ("\\n\\n"); Stage 1 already materialized that as an index-aligned paragraph list
(`paragraphs` / `paragraphs_original` share indices, so `source_pointer` is valid for
either variant), so chunking just packs consecutive paragraphs greedily:

  - keep adding paragraphs while the running count stays <= TARGET_MAX
  - flush once adding the next paragraph would exceed TARGET_MAX
  - a single paragraph longer than TARGET_MAX becomes its own chunk and is flagged
    `oversize_paragraph` (never split mid-paragraph, never discarded)
  - a trailing chunk below TARGET_MIN is kept and flagged `below_target_min`

شرح بالعربية
------------
المرحلة الثالثة: التقطيع. تحزم الوحدات الكاملة الآتية من المرحلة الأولى في قطع
يستهدف حجمها 200 إلى 800 وحدة عدّ، وتكتبها في chunks.jsonl.

النص المُسلَّم
--------------
يُبنى chunk_text من paragraphs_original، أي النص محفوظ الإملاء (أ/إ/آ و ى كما وردت).
النسخة الموحَّدة مفتاح مطابقة يخص dedup.py وحده ولا تُكتب هنا إطلاقًا.

عدّ الوحدات
-----------
token_count تقدير بعدد الكلمات المفصولة بمسافات، بلا أي مُجزِّئ لغوي، حتى تبقى
المرحلة حتمية ومستقلة عن النموذج. في العربية الفصحى يُنتج مُجزِّئ من نوع
SentencePiece قرابة 1.5 إلى 2.5 رمز لكل كلمة، فتُعاد معايرة الحدين عند تثبيت مُجزِّئ
التدريب.

لماذا --format إلزامي ولا يُكتشف تلقائيًا؟
------------------------------------------
لأن الخطأ في تحديد نوع المصدر لا يُنتج خطأ ظاهرًا، بل مخرجات معقولة الشكل وفاسدة
المضمون: معجم عومل كنثر يتحول إلى كتل بحجم الصفحة، وقصيدة عوملت كنثر تفقد حدود
مقاطعها. الاكتشاف التلقائي يعني تخمينًا صامتًا في موضع لا يظهر فيه أثر التخمين إلا
بعد التدريب. لذلك يُطلب تحديد النوع صراحةً، ويجب أن يكون هو نفسه المُمرَّر إلى
clean.py، وأي اختلاف بينهما يُرفض التشغيل عنده بدل المضي فيه.

ما يتغير بتغير النوع هو الوحدة المحزومة والوسم المُخرَج:
  prose       فقرات   -> narrative_paragraph / prose / list / footnote_block
  dictionary  مداخل معجمية -> dictionary_entry، وتُجمَّع المداخل القصيرة معًا لبلوغ
              النطاق المستهدف بدل إخراج قطعة ضئيلة لكل مدخل، ولا يُشطر مدخل أبدًا.
  verse       مقاطع شعرية -> verse، تُحزم كاملة ولا تُكسر في وسطها، ولا تعبر القطعة
              حدود قصيدتين.

قاعدة الحدود
------------
الكسر عند حدود الوحدات فقط: تُضاف الوحدات ما دام المجموع لا يتجاوز الحد الأعلى، ثم
تُغلق القطعة. الوحدة الأطول من الحد الأعلى تصير قطعة مستقلة وتُعلَّم oversize، ولا
تُشطر ولا تُحذف. والقطعة الأخيرة إن نزلت عن الحد الأدنى تُحفظ وتُعلَّم below_target_min.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

INTERIM_DIR = "data/interim"
DEDUP_REPORT = "data/processed/dedup_report.json"
CHUNKS_PATH = "data/processed/chunks.jsonl"
CHUNK_REPORT = "data/processed/chunking_report.json"

TARGET_MIN = 200
TARGET_MAX = 800

FORMATS = ("prose", "dictionary", "verse")

# A line that separates one poem from the next in a verse source (a title/heading line
# sitting alone between stanzas). Chunks are never allowed to span one.
# حدّ فاصل بين قصيدة وأخرى (سطر عنوان منفرد بين المقاطع)؛ لا يُسمح للقطعة بتجاوزه.
POEM_BOUNDARY_RE = re.compile(r"^\s*(?:[*\-=~_]{3,}|#+\s|\[[^\]]+\])\s*$")

# format_type heuristics
# قواعد استدلالية لتحديد format_type في مصادر النثر: علامات القوائم، وسطور الحواشي،
# وطول العنوان.
LIST_MARKER_RE = re.compile(r"^\s*(?:[-•*]\s|\(?[٠-٩0-9]+\)|[٠-٩0-9]+[.)]\s)")
FOOTNOTE_RE = re.compile(r"^\s*\^\(")
HEADING_MAX_WORDS = 8


def token_count(text: str) -> int:
    """Word-based token approximation (see module docstring)."""
    # تقدير عدد الوحدات بعدّ الكلمات المفصولة بمسافات؛ لا مُجزِّئ لغوي هنا.
    return len(text.split())


def is_heading(paragraph: str) -> bool:
    # عنوان مُرجَّح: سطر قصير لا ينتهي بعلامة ترقيم تدل على جملة تامة.
    return (
        token_count(paragraph) <= HEADING_MAX_WORDS
        and not paragraph.rstrip().endswith((".", "؟", "!", "،", ":", "؛"))
    )


def classify_format(paragraphs: list[str], fmt: str = "prose") -> str:
    """format_type for a chunk, from the units it contains."""
    # وسم نوع المحتوى: يُحسم مباشرة في المعجم والشعر، أما النثر فيُستدل عليه من
    # نسبة السطور التي تبدأ بعلامة قائمة أو بعلامة حاشية.
    if fmt == "dictionary":
        return "dictionary_entry"
    if fmt == "verse":
        return "verse"
    n = len(paragraphs)
    if n == 0:
        return "empty"
    listy = sum(1 for p in paragraphs if LIST_MARKER_RE.match(p))
    notes = sum(1 for p in paragraphs if FOOTNOTE_RE.match(p))
    if notes / n >= 0.5:
        return "footnote_block"
    if listy / n >= 0.5:
        return "list"
    body = [p for p in paragraphs if not is_heading(p)]
    if len(body) <= 1:
        return "prose"
    return "narrative_paragraph"


def pack_units(units: list[str], target_max: int,
               hard_break: "callable | None" = None) -> list[dict]:
    """Greedily pack whole units into chunks. Units are never split.

    Identical mechanics for all three formats - only what counts as a unit differs, which
    Stage 1 already decided. `hard_break(unit) -> bool` marks a unit that must start a new
    chunk regardless of how empty the current one is (used for poem boundaries in verse
    sources, so one chunk never contains the tail of one poem and the head of the next).

    Bundling short units is what keeps dictionary sources from producing one tiny chunk
    per glossary entry: entries accumulate until the next one would overflow target_max.
    """
    # الحزم الجَشِع: تُضاف الوحدات واحدة تلو الأخرى حتى توشك الإضافة التالية على تجاوز
    # الحد الأعلى، فتُغلق القطعة. الوحدة لا تُشطر أبدًا مهما طالت. آلية واحدة تخدم
    # الأنواع الثلاثة، والمختلف بينها هو ما يُعدّ وحدة، وقد حُسم ذلك في المرحلة الأولى.
    # المعامل hard_break يفرض بدء قطعة جديدة عند حدّ معيّن مهما كانت القطعة الحالية
    # فارغة، ويُستعمل لحدود القصائد. وتجميع الوحدات القصيرة هو ما يمنع المعجم من
    # إنتاج قطعة ضئيلة لكل مدخل.
    chunks: list[dict] = []
    buf: list[str] = []
    buf_tokens = 0
    start_idx = 0

    def flush(end_idx: int) -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        chunks.append({"unit_start": start_idx, "unit_end": end_idx, "units": list(buf)})
        buf, buf_tokens = [], 0

    for i, unit in enumerate(units):
        n = token_count(unit)

        if hard_break is not None and hard_break(unit) and buf:
            flush(i - 1)
            start_idx = i

        if n > target_max:
            # oversize single unit: flush what we have, emit it alone, never split it
            flush(i - 1)
            chunks.append({"unit_start": i, "unit_end": i, "units": [unit]})
            start_idx = i + 1
            continue

        if buf and buf_tokens + n > target_max:
            flush(i - 1)
            start_idx = i

        if not buf:
            start_idx = i
        buf.append(unit)
        buf_tokens += n

    flush(len(units) - 1)
    return chunks


def load_kept_doc_ids(dedup_report_path: str) -> set[str] | None:
    # يقرأ من تقرير المرحلة الثانية قائمة الوثائق المُبقاة، لتخطي المتطابقات حرفيًا.
    if not os.path.exists(dedup_report_path):
        return None
    with open(dedup_report_path, encoding="utf-8") as f:
        rep = json.load(f)
    return set(rep.get("documents_kept", []))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3: unit-boundary chunking.")
    # واجهة سطر الأوامر. الافتراضات المهمة: --text-variant = original أي النص محفوظ
    # الإملاء، و--format إلزامي التطابق مع ما مُرِّر إلى clean.py.
    ap.add_argument("--interim-dir", default=INTERIM_DIR)
    ap.add_argument("--dedup-report", default=DEDUP_REPORT)
    ap.add_argument("--out", default=CHUNKS_PATH)
    ap.add_argument("--report", default=CHUNK_REPORT)
    ap.add_argument("--target-min", type=int, default=TARGET_MIN)
    ap.add_argument("--target-max", type=int, default=TARGET_MAX)
    ap.add_argument("--text-variant", choices=["original", "normalized"], default="original",
                    help="original = orthography-preserving text from Stage 1 (default, "
                         "correct for training); normalized = alef/ya-folded text, for "
                         "inspection only - folded text belongs in dedup matching, not in "
                         "chunk_text.")
    ap.add_argument("--format", choices=FORMATS, default="prose", dest="fmt",
                    help="Source structure of this batch (default: prose). Chosen explicitly - "
                         "no auto-detection. Must match the --format used in clean.py.")
    args = ap.parse_args()

    kept = load_kept_doc_ids(args.dedup_report)
    if kept is None:
        print(f"[chunk] no dedup report at {args.dedup_report}; chunking every cleaned doc")

    paths = sorted(glob.glob(os.path.join(args.interim_dir, "*", "*_cleaned.json")))
    if not paths:
        raise SystemExit(f"No *_cleaned.json under {args.interim_dir}. Run clean.py first.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    per_doc = []
    skipped = []
    total_chunks = 0
    flag_counts: dict[str, int] = {}

    with open(args.out, "w", encoding="utf-8") as out:
        for path in paths:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)

            if kept is not None and doc["doc_id"] not in kept:
                skipped.append({"doc_id": doc["doc_id"], "reason": "exact_duplicate"})
                print(f"[chunk] skipping {doc['doc_id']} (exact duplicate)")
                continue

            doc_fmt = doc.get("source_format", "prose")
            if doc_fmt != args.fmt:
                raise SystemExit(
                    f"[chunk] format mismatch on {doc['doc_id']}: cleaned as '{doc_fmt}' but "
                    f"--format is '{args.fmt}'. Re-run clean.py --format {args.fmt}, or chunk "
                    f"this batch with --format {doc_fmt}."
                )

            key = "paragraphs" if args.text_variant == "normalized" else "paragraphs_original"
            units = doc[key]
            hard_break = POEM_BOUNDARY_RE.match if args.fmt == "verse" else None
            raw_chunks = pack_units(units, args.target_max, hard_break)

            doc_tokens = 0
            for n, c in enumerate(raw_chunks):
                text = "\n\n".join(c["units"])
                tokens = token_count(text)
                doc_tokens += tokens

                flags = []
                if tokens > args.target_max:
                    flags.append(f"oversize_{doc.get('unit_type', 'paragraph')}")
                elif tokens < args.target_min:
                    flags.append("below_target_min")
                for fl in flags:
                    flag_counts[fl] = flag_counts.get(fl, 0) + 1

                record = {
                    "chunk_id": f"{doc['doc_id']}_c{n:04d}",
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "license": doc["license"],
                    "region": doc["region"],
                    "format_type": classify_format(c["units"], args.fmt),
                    "token_count": tokens,
                    "chunk_text": text,
                    "source_pointer": {
                        "unit_start": c["unit_start"],
                        "unit_end": c["unit_end"],
                        "unit_count": len(c["units"]),
                        "unit_type": doc.get("unit_type", "paragraph"),
                    },
                    "title": doc["title"],
                    "author": doc["author"],
                    "translator": doc.get("translator"),
                    "token_count_method": "whitespace_word_count",
                    "text_variant": args.text_variant,
                    "source_format": args.fmt,
                    "review_flags": flags,
                    "source_doc_flags": doc.get("review_flags", []),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_chunks += len(raw_chunks)
            per_doc.append({
                "doc_id": doc["doc_id"],
                "region": doc["region"],
                "title": doc["title"],
                "unit_type": doc.get("unit_type", "paragraph"),
                "units": len(units),
                "chunks": len(raw_chunks),
                "tokens": doc_tokens,
            })
            print(f"[chunk] {doc['doc_id']:>10} {str(doc.get('region')):<10} "
                  f"{len(units):>5} {doc.get('unit_type', 'paragraph')}s -> "
                  f"{len(raw_chunks):>4} chunks ({doc_tokens:,} tokens)")

    report = {
        "config": {
            "target_min_tokens": args.target_min,
            "target_max_tokens": args.target_max,
            "token_count_method": "whitespace_word_count (word-based approximation)",
            "source_format": args.fmt,
            "boundary_rule": "whole units only, as segmented by Stage 1's --format",
            "oversize_policy": "kept whole and flagged, never split mid-unit",
            "poem_boundary_rule": ("chunks never span a poem boundary"
                                   if args.fmt == "verse" else "n/a"),
            "text_variant": args.text_variant,
        },
        "documents_chunked": len(per_doc),
        "documents_skipped": skipped,
        "total_chunks": total_chunks,
        "total_tokens": sum(d["tokens"] for d in per_doc),
        "chunk_flag_counts": flag_counts,
        "per_document": per_doc,
        "output": args.out,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[chunk] {total_chunks:,} chunks from {len(per_doc)} documents "
          f"({report['total_tokens']:,} tokens)")
    print(f"[chunk] flags: {flag_counts or '-'}")
    print(f"[chunk] chunks -> {args.out}\n[chunk] report -> {args.report}")


if __name__ == "__main__":
    main()
