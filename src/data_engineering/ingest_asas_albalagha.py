"""Acquisition - أساس البلاغة -> interim document.

Reads the 18 EPUB parts of Asas Al-Balagha, extracts their digital XHTML text,
detects spaced triliteral root headings, and serializes the entries as
blank-line-separated dictionary blocks.

The output follows clean.py's expected interim-document schema.

This script performs source-specific acquisition only.
Generic cleaning, deduplication, chunking, and EDA remain the responsibility of:

    clean.py
    dedup.py
    chunk.py
    eda.py

No LLM calls are used.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub


RAW_DIR = Path("data/raw/classical/asas_albalagha")
INTERIM_DIR = Path("data/interim/classical")

DOC_ID = "asas_albalagha"
TITLE = "أساس البلاغة"
AUTHOR = "محمود بن عمر الزمخشري"
SOURCE = "Arabic Wikisource"
LICENSE = "unverified_pending_review"
REGION = "classical"


ARABIC_LETTER = r"[ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهويى]"

ROOT_PATTERN = re.compile(
    rf"""
    (?<![\u0600-\u06FF])
    (
        {ARABIC_LETTER}
        [\u064B-\u065F\u0670]*
        \s+
        {ARABIC_LETTER}
        [\u064B-\u065F\u0670]*
        \s+
        {ARABIC_LETTER}
        [\u064B-\u065F\u0670]*
    )
    (?=\s)
    """,
    re.VERBOSE,
)


def normalize_extracted_text(text: str) -> str:
    text = text.replace("\u200f", "")
    text = text.replace("\u200e", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\xa0", " ")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_root(root: str) -> str:
    return re.sub(
        r"[\s\u064B-\u065F\u0670]",
        "",
        root,
    )


def extract_entries(epub_path: Path) -> list[dict]:
    book = epub.read_epub(str(epub_path))
    entries: list[dict] = []

    for item in book.get_items():
        if item.get_type() != 9:
            continue

        soup = BeautifulSoup(
            item.get_content(),
            "html.parser",
        )

        text = soup.get_text(
            separator=" ",
            strip=True,
        )

        text = normalize_extracted_text(text)

        if not text:
            continue

        matches = list(ROOT_PATTERN.finditer(text))

        if not matches:
            continue

        for index, match in enumerate(matches):
            root = normalize_root(match.group(1))

            content_start = match.end()

            if index + 1 < len(matches):
                content_end = matches[index + 1].start()
            else:
                content_end = len(text)

            content = text[
                content_start:content_end
            ].strip()

            if not content:
                continue

            entries.append(
                {
                    "root": root,
                    "content": content,
                    "source_file": epub_path.name,
                }
            )

    return entries


def entry_to_block(entry: dict) -> str:
    root = entry["root"].strip()
    content = entry["content"].strip()

    return f"{root}: {content}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Acquisition: Asas Al-Balagha EPUB parts "
            "-> clean.py-compatible interim JSON."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
    )

    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=INTERIM_DIR,
    )

    args = parser.parse_args()

    epub_files = sorted(
        args.raw_dir.glob("Part*.epub")
    )

    if not epub_files:
        raise SystemExit(
            f"No EPUB files found under {args.raw_dir}"
        )

    print(
        f"[ingest-asas] EPUB files found: {len(epub_files)}"
    )

    all_entries: list[dict] = []
    entries_per_file: dict[str, int] = {}

    for epub_path in epub_files:
        entries = extract_entries(epub_path)

        entries_per_file[
            epub_path.name
        ] = len(entries)

        all_entries.extend(entries)

        print(
            f"[ingest-asas] "
            f"{epub_path.name}: "
            f"{len(entries)} entries"
        )

    if not all_entries:
        raise SystemExit(
            "No dictionary entries were extracted."
        )

    raw_text = "\n\n".join(
        entry_to_block(entry)
        for entry in all_entries
    )

    roots = [
        entry["root"]
        for entry in all_entries
    ]

    record = {
        "doc_id": DOC_ID,
        "title": TITLE,
        "author": AUTHOR,
        "source": SOURCE,
        "license": LICENSE,
        "region": REGION,
        "raw_text": raw_text,
        "source_type": "classical_dictionary",
        "input_format": "EPUB",
        "source_parts": len(epub_files),
        "entry_count_ingest": len(all_entries),
        "unique_roots_ingest": len(set(roots)),
        "entries_per_source_file": entries_per_file,
        "ocr_used": False,
        "acquisition_method": (
            "digital EPUB XHTML extraction; "
            "spaced-triliteral-root segmentation"
        ),
    }

    args.interim_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.interim_dir
        / f"{DOC_ID}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            record,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"[ingest-asas] total entries: "
        f"{len(all_entries)}"
    )
    print(
        f"[ingest-asas] unique roots: "
        f"{len(set(roots))}"
    )
    print(
        f"[ingest-asas] approximate words: "
        f"{len(raw_text.split()):,}"
    )
    print(
        f"[ingest-asas] output -> "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()