from pathlib import Path
import json


INPUT_PATH = Path(
    "data/interim/najdi/najdi_layout_multi_test_v9_final.json"
)

OUTPUT_PATH = Path(
    "data/interim/najdi/najdi_review_v10.json"
)


EASY_FLAGS = {
    "low_headword_confidence",
    "short_meaning",
    "removed_trailing_dash_noise",
    "removed_repeated_dash_noise",
    "possible_trailing_number_noise",
    "possible_trailing_single_char_noise",
}

HARD_FLAGS = {
    "very_low_headword_confidence",
    "headword_without_colon",
    "unexpected_symbol_in_headword",
    "unexpected_symbol_in_meaning",
    "very_short_meaning",
    "empty_meaning",
    "digit_in_headword",
    "suspicious_ocr_symbol",
    "possible_standalone_number_noise",
}


def classify_review(entry: dict) -> dict:
    flags = set(
        entry.get("final_review_flags", [])
    )

    # إذا فيه أي مشكلة قوية -> hard
    if flags & HARD_FLAGS:
        review_tier = "hard_review"

    # إذا كل المشاكل من النوع البسيط -> easy
    elif flags and flags.issubset(EASY_FLAGS):
        review_tier = "easy_review"

    # احتياط: أي حالة review ما عرفناها
    # ما نعتبرها سهلة تلقائيًا
    else:
        review_tier = "hard_review"

    return {
        **entry,
        "review_tier": review_tier,
    }


def main() -> None:
    data = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    review_entries = [
        entry
        for entry in data
        if entry.get("final_status") == "review"
    ]

    classified = [
        classify_review(entry)
        for entry in review_entries
    ]

    easy = [
        entry
        for entry in classified
        if entry["review_tier"] == "easy_review"
    ]

    hard = [
        entry
        for entry in classified
        if entry["review_tier"] == "hard_review"
    ]

    OUTPUT_PATH.write_text(
        json.dumps(
            classified,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total = len(classified)

    def pct(count: int) -> float:
        return round(
            count / total * 100,
            2,
        ) if total else 0.0

    print("[v10] REVIEW REPORT")
    print(f"total review : {total}")
    print(
        f"easy review  : {len(easy)} "
        f"({pct(len(easy))}%)"
    )
    print(
        f"hard review  : {len(hard)} "
        f"({pct(len(hard))}%)"
    )

    print()
    print("[v10] EASY REVIEW")

    for entry in easy:
        print(
            f"page={entry['page']} "
            f"| {entry['headword']} "
            f"| flags={entry['final_review_flags']}"
        )

    print()
    print(
        f"[v10] output -> {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()