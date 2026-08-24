"""Stage 3 - Chunking.

Input : data/interim/<domain>/<doc_id>_cleaned.json  (Stage 1 output)
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

# format_type heuristics
LIST_MARKER_RE = re.compile(r"^\s*(?:[-•*]\s|\(?[٠-٩0-9]+\)|[٠-٩0-9]+[.)]\s)")
FOOTNOTE_RE = re.compile(r"^\s*\^\(")
HEADING_MAX_WORDS = 8


def token_count(text: str) -> int:
    """Word-based token approximation (see module docstring)."""
    return len(text.split())


def is_heading(paragraph: str) -> bool:
    return (
        token_count(paragraph) <= HEADING_MAX_WORDS
        and not paragraph.rstrip().endswith((".", "؟", "!", "،", ":", "؛"))
    )


def classify_format(paragraphs: list[str]) -> str:
    """format_type for a chunk, from the paragraphs it contains."""
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


def chunk_paragraphs(paragraphs: list[str], target_min: int, target_max: int) -> list[dict]:
    """Greedy paragraph packing. Returns dicts with para_start/para_end/paragraphs."""
    chunks: list[dict] = []
    buf: list[str] = []
    buf_tokens = 0
    start_idx = 0

    def flush(end_idx: int) -> None:
        nonlocal buf, buf_tokens, start_idx
        if not buf:
            return
        chunks.append({"para_start": start_idx, "para_end": end_idx, "paragraphs": list(buf)})
        buf, buf_tokens = [], 0

    for i, para in enumerate(paragraphs):
        n = token_count(para)

        if n > target_max:
            # oversize single paragraph: flush what we have, emit it alone
            flush(i - 1)
            chunks.append({"para_start": i, "para_end": i, "paragraphs": [para]})
            start_idx = i + 1
            continue

        if buf and buf_tokens + n > target_max:
            flush(i - 1)
            start_idx = i

        if not buf:
            start_idx = i
        buf.append(para)
        buf_tokens += n

    flush(len(paragraphs) - 1)
    return chunks


def load_kept_doc_ids(dedup_report_path: str) -> set[str] | None:
    if not os.path.exists(dedup_report_path):
        return None
    with open(dedup_report_path, encoding="utf-8") as f:
        rep = json.load(f)
    return set(rep.get("documents_kept", []))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3: paragraph-boundary chunking.")
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

            key = "paragraphs" if args.text_variant == "normalized" else "paragraphs_original"
            paragraphs = doc[key]
            raw_chunks = chunk_paragraphs(paragraphs, args.target_min, args.target_max)

            doc_tokens = 0
            for n, c in enumerate(raw_chunks):
                text = "\n\n".join(c["paragraphs"])
                tokens = token_count(text)
                doc_tokens += tokens

                flags = []
                if tokens > args.target_max:
                    flags.append("oversize_paragraph")
                elif tokens < args.target_min:
                    flags.append("below_target_min")
                for fl in flags:
                    flag_counts[fl] = flag_counts.get(fl, 0) + 1

                record = {
                    "chunk_id": f"{doc['doc_id']}_c{n:04d}",
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "license": doc["license"],
                    "domain": doc["domain"],
                    "format_type": classify_format(c["paragraphs"]),
                    "token_count": tokens,
                    "chunk_text": text,
                    "source_pointer": {
                        "para_start": c["para_start"],
                        "para_end": c["para_end"],
                        "paragraph_count": len(c["paragraphs"]),
                    },
                    "title": doc["title"],
                    "author": doc["author"],
                    "translator": doc.get("translator"),
                    "token_count_method": "whitespace_word_count",
                    "text_variant": args.text_variant,
                    "review_flags": flags,
                    "source_doc_flags": doc.get("review_flags", []),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_chunks += len(raw_chunks)
            per_doc.append({
                "doc_id": doc["doc_id"],
                "domain": doc["domain"],
                "title": doc["title"],
                "paragraphs": len(paragraphs),
                "chunks": len(raw_chunks),
                "tokens": doc_tokens,
            })
            print(f"[chunk] {doc['doc_id']:>10} {doc['domain']:<17} "
                  f"{len(paragraphs):>5} paras -> {len(raw_chunks):>4} chunks "
                  f"({doc_tokens:,} tokens)")

    report = {
        "config": {
            "target_min_tokens": args.target_min,
            "target_max_tokens": args.target_max,
            "token_count_method": "whitespace_word_count (word-based approximation)",
            "boundary_rule": "paragraph boundaries only; separator '\\n\\n' from Stage 1",
            "oversize_policy": "kept whole and flagged, never split mid-paragraph",
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
