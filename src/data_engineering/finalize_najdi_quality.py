from pathlib import Path
import json
import re


INPUT_PATH = Path(
    "data/interim/najdi/najdi_layout_multi_test_v8_cleaned.json"
)

OUTPUT_PATH = Path(
    "data/interim/najdi/najdi_layout_multi_test_v9_final.json"
)

ACCEPTED_PATH = Path(
    "data/interim/najdi/najdi_v9_high_quality.json"
)


# رقم منفرد وسط الكلام:
# مثال: "حيث ١ تقطع"
STANDALONE_NUMBER_RE = re.compile(
    r"(?<!\S)[0-9٠-٩]+(?!\S)"
)

# رموز واضحة تستحق مراجعة
SUSPICIOUS_SYMBOL_RE = re.compile(
    r"[#©]"
)


def finalize_entry(entry: dict) -> dict:
    final_flags = list(
        entry.get("review_flags", [])
    )

    cleanup_review_flags = entry.get(
        "cleanup_review_flags",
        [],
    )

    final_flags.extend(
        cleanup_review_flags
    )

    meaning = entry["meaning"]

    # أرقام منفردة داخل الجملة
    if STANDALONE_NUMBER_RE.search(meaning):
        if "possible_standalone_number_noise" not in final_flags:
            final_flags.append(
                "possible_standalone_number_noise"
            )

    # رموز OCR واضحة
    if SUSPICIOUS_SYMBOL_RE.search(meaning):
        if "suspicious_ocr_symbol" not in final_flags:
            final_flags.append(
                "suspicious_ocr_symbol"
            )

    original_status = entry.get(
        "quality_status",
        "review",
    )

    if original_status == "rejected":
        final_status = "rejected"

    elif original_status == "accepted" and not final_flags:
        final_status = "accepted"

    else:
        final_status = "review"

    return {
        **entry,
        "final_status": final_status,
        "final_review_flags": final_flags,
    }


def main() -> None:
    data = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    final_entries = [
        finalize_entry(entry)
        for entry in data
    ]

    accepted = [
        entry
        for entry in final_entries
        if entry["final_status"] == "accepted"
    ]

    review = [
        entry
        for entry in final_entries
        if entry["final_status"] == "review"
    ]

    rejected = [
        entry
        for entry in final_entries
        if entry["final_status"] == "rejected"
    ]

    OUTPUT_PATH.write_text(
        json.dumps(
            final_entries,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    ACCEPTED_PATH.write_text(
        json.dumps(
            accepted,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total = len(final_entries)

    def pct(count: int) -> float:
        return round(
            count / total * 100,
            2,
        ) if total else 0.0

    print("[v9] FINAL QUALITY REPORT")
    print(f"total     : {total}")
    print(
        f"accepted  : {len(accepted)} "
        f"({pct(len(accepted))}%)"
    )
    print(
        f"review    : {len(review)} "
        f"({pct(len(review))}%)"
    )
    print(
        f"rejected  : {len(rejected)} "
        f"({pct(len(rejected))}%)"
    )

    print()
    print(
        f"[v9] all -> {OUTPUT_PATH}"
    )
    print(
        f"[v9] high quality -> {ACCEPTED_PATH}"
    )

    print()
    print("[v9] ACCEPTED ENTRIES")

    for entry in accepted:
        print(
            f"page={entry['page']} "
            f"| {entry['headword']} "
            f"| conf={entry['headword_conf']}"
        )


if __name__ == "__main__":
    main()