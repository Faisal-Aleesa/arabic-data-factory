from pathlib import Path
import json
import re


INPUT_PATH = Path(
    "data/interim/najdi/najdi_layout_multi_test_v7.json"
)

OUTPUT_PATH = Path(
    "data/interim/najdi/najdi_layout_multi_test_v8_cleaned.json"
)


TRAILING_DASH_RE = re.compile(
    r"\s*[-–—]+\s*$"
)

MULTIPLE_DASHES_RE = re.compile(
    r"\s*[-–—]{2,}\s*"
)

TRAILING_NUMBER_RE = re.compile(
    r"\s+[0-9٠-٩]+\s*$"
)

TRAILING_SINGLE_ARABIC_CHAR_RE = re.compile(
    r"\s+[ء-ي]\s*$"
)

EXTRA_SPACES_RE = re.compile(
    r"\s+"
)


def clean_meaning(text: str) -> tuple[str, list[str], list[str]]:
    text = text.strip()

    cleanup_flags = []
    review_flags = []

    # -----------------------------------
    # SAFE AUTO-CLEANUP
    # -----------------------------------

    cleaned = MULTIPLE_DASHES_RE.sub(
        " ",
        text,
    )

    if cleaned != text:
        cleanup_flags.append(
            "removed_repeated_dash_noise"
        )

    text = cleaned

    cleaned = TRAILING_DASH_RE.sub(
        "",
        text,
    )

    if cleaned != text:
        cleanup_flags.append(
            "removed_trailing_dash_noise"
        )

    text = cleaned

    # -----------------------------------
    # FLAG ONLY — DO NOT DELETE
    # -----------------------------------

    if TRAILING_NUMBER_RE.search(text):
        review_flags.append(
            "possible_trailing_number_noise"
        )

    if TRAILING_SINGLE_ARABIC_CHAR_RE.search(text):
        review_flags.append(
            "possible_trailing_single_char_noise"
        )

    # -----------------------------------
    # WHITESPACE NORMALIZATION
    # -----------------------------------

    text = EXTRA_SPACES_RE.sub(
        " ",
        text,
    ).strip()

    return (
        text,
        cleanup_flags,
        review_flags,
    )


def main() -> None:
    data = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    cleaned_entries = []

    changed_count = 0
    flagged_count = 0

    for entry in data:
        (
            cleaned_meaning,
            cleanup_flags,
            cleanup_review_flags,
        ) = clean_meaning(
            entry["meaning"]
        )

        changed = (
            cleaned_meaning
            != entry["meaning"]
        )

        if changed:
            changed_count += 1

        if cleanup_review_flags:
            flagged_count += 1

        cleaned_entry = {
            **entry,

            # الأصل محفوظ دائمًا
            "meaning_original": entry["meaning"],

            # النسخة المحافظة
            "meaning": cleaned_meaning,

            "cleanup_applied": changed,
            "cleanup_flags": cleanup_flags,

            "cleanup_review_flags": (
                cleanup_review_flags
            ),
        }

        cleaned_entries.append(
            cleaned_entry
        )

    OUTPUT_PATH.write_text(
        json.dumps(
            cleaned_entries,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[cleanup-v8] total entries: "
        f"{len(cleaned_entries)}"
    )

    print(
        f"[cleanup-v8] safely changed: "
        f"{changed_count}"
    )

    print(
        f"[cleanup-v8] flagged only: "
        f"{flagged_count}"
    )

    print(
        f"[cleanup-v8] output -> "
        f"{OUTPUT_PATH}"
    )

    print()

    for entry in cleaned_entries:

        if (
            entry["cleanup_applied"]
            or entry["cleanup_review_flags"]
        ):

            print(
                f"page={entry['page']} "
                f"| {entry['headword']}"
            )

            print(
                f"  cleanup: "
                f"{entry['cleanup_flags']}"
            )

            print(
                f"  review : "
                f"{entry['cleanup_review_flags']}"
            )

            print(
                f"  before : "
                f"{entry['meaning_original']}"
            )

            print(
                f"  after  : "
                f"{entry['meaning']}"
            )

            print()


if __name__ == "__main__":
    main()