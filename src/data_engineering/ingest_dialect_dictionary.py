"""Acquisition - معجم اللهجات المحكية -> per-region interim documents.

This is the dialect-phase counterpart of the archived download_hindawi_corpus.py: it is
an ACQUISITION step, not a fourth preprocessing stage. The cleaning / dedup / chunking
stages are unchanged and still start from data/interim/.

Input : data/raw/<region>/dialect_dictionary_<region>.txt   (pdftotext -layout output,
                                                              form-feed separated pages)
Output: data/interim/<region>/<doc_id>.json                  (clean.py's input schema)
        docs/damaged_pages_review.csv                        (pages held back, for review)

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

BIDI_RE = re.compile(r"[​-‏‪-‮﻿]")
FOLIO_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$")
ARABIC_TOKEN_RE = re.compile(r"[ء-ٰٟـ]+")

# Damage detector. Arabic words top out around 12-13 letters; interleaved text runs
# produce far longer pseudo-tokens. A page is held back when more than
# DAMAGE_TOKEN_RATIO of its Arabic tokens exceed DAMAGE_TOKEN_LEN characters.
DAMAGE_TOKEN_LEN = 16
DAMAGE_TOKEN_RATIO = 0.02


def page_lines(page_text: str) -> list[str]:
    lines = [l.strip() for l in BIDI_RE.sub("", page_text).split("\n")]
    return [l for l in lines if l and not FOLIO_RE.match(l)]


def damage_score(lines: list[str]) -> tuple[float, str]:
    """(ratio of over-long Arabic tokens, worst snippet)."""
    toks = [t for l in lines for t in ARABIC_TOKEN_RE.findall(l)]
    if not toks:
        return 0.0, ""
    bad = [t for t in toks if len(t) > DAMAGE_TOKEN_LEN]
    if not bad:
        return 0.0, ""
    worst = max(bad, key=len)
    snippet = next((l for l in lines if worst in l), worst)
    return len(bad) / len(toks), snippet.strip()[:160]


def ingest_region(region: str, raw_path: str, first_page: int) -> tuple[dict, list[dict]]:
    raw = io.open(raw_path, encoding="utf-8").read()
    pages = raw.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()

    kept_pages: list[str] = []
    damaged: list[dict] = []
    kept_page_numbers: list[int] = []

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
        kept_pages.append("\n".join(lines))
        kept_page_numbers.append(page_no)

    body = "\n\n".join(kept_pages)
    record = {
        "doc_id": f"dialect_dict_{region}",
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
        "raw_text": body,
    }
    return record, damaged


def main() -> None:
    ap = argparse.ArgumentParser(description="Acquisition: dialect dictionary -> interim JSON.")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--interim-dir", default=INTERIM_DIR)
    ap.add_argument("--review-csv", default=REVIEW_CSV)
    args = ap.parse_args()

    all_damaged: list[dict] = []
    summary = []

    for region, first_page in REGION_FIRST_PAGE.items():
        raw_path = os.path.join(args.raw_dir, region, f"dialect_dictionary_{region}.txt")
        if not os.path.exists(raw_path):
            print(f"[ingest] MISSING {raw_path} - skipping {region}")
            continue

        record, damaged = ingest_region(region, raw_path, first_page)
        all_damaged.extend(damaged)

        out_dir = os.path.join(args.interim_dir, region)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, record["doc_id"] + ".json")
        with io.open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        summary.append(record)
        print(f"[ingest] {region:<9} pages {record['pages_kept']:>4}/{record['pages_total']:<4} "
              f"(-{record['pages_excluded_damaged']} damaged)  "
              f"{record['word_count']:>8,} words  -> {out_path}")

    os.makedirs(os.path.dirname(args.review_csv), exist_ok=True)
    with io.open(args.review_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["region", "page", "garbled_token_ratio",
                                          "garbled_snippet", "action", "source_file"])
        w.writeheader()
        w.writerows(sorted(all_damaged, key=lambda d: d["page"]))

    tot_pages = sum(r["pages_total"] for r in summary)
    tot_kept = sum(r["pages_kept"] for r in summary)
    tot_words = sum(r["word_count"] for r in summary)
    print(f"\n[ingest] {len(summary)} regions | pages {tot_kept:,}/{tot_pages:,} kept "
          f"({len(all_damaged)} excluded as damaged) | {tot_words:,} usable words")
    print(f"[ingest] damaged-page review file -> {args.review_csv}")


if __name__ == "__main__":
    main()
