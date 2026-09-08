from pathlib import Path
import json


FINAL_INPUT = Path(
    "data/interim/najdi/najdi_layout_multi_test_v9_final.json"
)

REVIEW_INPUT = Path(
    "data/interim/najdi/najdi_review_v10.json"
)

OUTPUT_PATH = Path(
    "data/processed/najdi_dictionary_final.json"
)

REPORT_PATH = Path(
    "data/eda/najdi_dictionary_final_report.json"
)


AUTO_APPROVE_FLAGS = {
    "low_headword_confidence",
    "short_meaning",
}


def load_json(path: Path) -> list[dict]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def is_auto_approvable_easy_review(
    entry: dict,
) -> bool:
    if entry.get("review_tier") != "easy_review":
        return False

    flags = set(
        entry.get("final_review_flags", [])
    )

    if not flags:
        return False

    return flags.issubset(
        AUTO_APPROVE_FLAGS
    )


def build_final_entry(
    entry: dict,
    approval_source: str,
) -> dict:
    return {
        "headword": entry["headword"],
        "meaning": entry["meaning"],
        "page": entry["page"],

        "headword_conf": entry.get(
            "headword_conf"
        ),

        "quality_score": entry.get(
            "quality_score"
        ),

        "approval_source": approval_source,

        "source": (
            "معجم الكلمات الشعبية في نجد"
        ),

        "region": "najdi",

        "format": "dictionary",

        "provenance": {
            "pdf_page": entry["page"],
            "ocr": "tesseract",
            "ocr_language": "ara",
            "psm": 6,
            "layout_source": "tsv",
        },

        "review_flags": entry.get(
            "final_review_flags",
            []
        ),

        "cleanup_flags": entry.get(
            "cleanup_flags",
            []
        ),

        "meaning_original": entry.get(
            "meaning_original",
            entry["meaning"],
        ),
    }


def main() -> None:
    final_data = load_json(
        FINAL_INPUT
    )

    review_data = load_json(
        REVIEW_INPUT
    )

    accepted_entries = [
        entry
        for entry in final_data
        if entry.get("final_status")
        == "accepted"
    ]

    auto_approved_easy = [
        entry
        for entry in review_data
        if is_auto_approvable_easy_review(
            entry
        )
    ]

    output_entries = []

    for entry in accepted_entries:
        output_entries.append(
            build_final_entry(
                entry,
                approval_source=(
                    "high_confidence"
                ),
            )
        )

    for entry in auto_approved_easy:
        output_entries.append(
            build_final_entry(
                entry,
                approval_source=(
                    "easy_review_auto_approved"
                ),
            )
        )

    # منع التكرار لو دخل نفس المدخل
    # من مسارين مختلفين لأي سبب.
    deduped = {}
    duplicates = 0

    for entry in output_entries:
        key = (
            entry["page"],
            entry["headword"].strip(),
        )

        if key in deduped:
            duplicates += 1
            continue

        deduped[key] = entry

    final_entries = list(
        deduped.values()
    )

    final_entries.sort(
        key=lambda item: (
            item["page"],
            item["headword"],
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            final_entries,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "source": (
            "معجم الكلمات الشعبية في نجد"
        ),

        "total_extracted_entries": len(
            final_data
        ),

        "high_confidence_accepted": len(
            accepted_entries
        ),

        "easy_review_total": len(
            [
                entry
                for entry in review_data
                if entry.get(
                    "review_tier"
                ) == "easy_review"
            ]
        ),

        "easy_review_auto_approved": len(
            auto_approved_easy
        ),

        "final_usable_entries": len(
            final_entries
        ),

        "duplicates_removed": duplicates,

        "selection_policy": {
            "direct_acceptance": (
                "final_status == accepted"
            ),

            "easy_review_auto_approval": (
                "review_tier == easy_review "
                "and all flags are limited to "
                "low_headword_confidence "
                "or short_meaning"
            ),

            "hard_review": "excluded",

            "rejected": "excluded",
        },
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[v12] FINAL NAJDI SUBSET")
    print(
        f"total extracted       : "
        f"{report['total_extracted_entries']}"
    )
    print(
        f"high-confidence       : "
        f"{report['high_confidence_accepted']}"
    )
    print(
        f"easy review total     : "
        f"{report['easy_review_total']}"
    )
    print(
        f"easy auto-approved    : "
        f"{report['easy_review_auto_approved']}"
    )
    print(
        f"final usable entries  : "
        f"{report['final_usable_entries']}"
    )
    print(
        f"duplicates removed    : "
        f"{report['duplicates_removed']}"
    )

    print()
    print(
        f"[v12] dataset -> "
        f"{OUTPUT_PATH}"
    )
    print(
        f"[v12] report  -> "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()