from pathlib import Path
import csv
import json
import re
from statistics import mean


OCR_DIR = Path("data/interim/najdi/ocr_pages")

START_PAGE = 27
END_PAGE = 321

TEST_PAGES = list(
    range(
        START_PAGE,
        END_PAGE + 1,
    )
)
OUTPUT_PATH = Path(
    "data/interim/najdi/najdi_layout_multi_test_v7.json"
)

REPORT_PATH = Path(
    "data/interim/najdi/najdi_layout_multi_test_v7_report.json"
)

HEADWORD_LEFT_MIN = 1350
MIN_WORD_CONFIDENCE = 0.0

ARABIC_LETTER_RE = re.compile(
    r"[ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهويىةؤئ]"
)

DIGIT_RE = re.compile(r"[0-9٠-٩]")
SYMBOL_RE = re.compile(
    r"[^\sء-يأإآؤئىةًٌٍَُِّْـ0-9٠-٩.,،؛:؟()\[\]«»\-]"
)

TRAILING_NUMBER_NOISE_RE = re.compile(
    r"\s+[0-9٠-٩]+\s*[-–—]?\s*$"
)

TRAILING_SINGLE_CHAR_RE = re.compile(
    r"\s+[ء-ي]\s*$"
)

CONTROL_CHARS = [
    "\u061c",
    "\u200e",
    "\u200f",
    "\u200c",
    "\u200d",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
    "\ufeff",
]


def clean_token(text: str) -> str:
    text = text.strip()

    for char in CONTROL_CHARS:
        text = text.replace(char, "")

    return text.strip()


def normalize_headword(text: str) -> str:
    text = (
        text
        .replace(":", "")
        .replace("：", "")
        .strip()
    )

    return text


def is_plausible_headword(text: str) -> bool:
    if not text:
        return False

    if not ARABIC_LETTER_RE.search(text):
        return False

    if len(text.split()) > 2:
        return False

    if len(text) > 25:
        return False

    letters_only = re.sub(
        r"[^ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهويىةؤئ]",
        "",
        text,
    )

    # نمنع أشياء مثل "ا"
    if len(letters_only) < 2:
        return False

    return True


def load_words(tsv_path: Path) -> list[dict]:
    words = []

    with tsv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:
            if row["level"] != "5":
                continue

            text = clean_token(
                row.get("text", "")
            )

            if not text:
                continue

            try:
                confidence = float(row["conf"])
            except ValueError:
                continue

            if confidence < MIN_WORD_CONFIDENCE:
                continue

            words.append(
                {
                    "text": text,
                    "left": int(row["left"]),
                    "top": int(row["top"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "conf": confidence,
                    "block_num": int(row["block_num"]),
                    "par_num": int(row["par_num"]),
                    "line_num": int(row["line_num"]),
                }
            )

    return words


def group_lines(words: list[dict]) -> list[dict]:
    grouped = {}

    for word in words:
        key = (
            word["block_num"],
            word["par_num"],
            word["line_num"],
        )

        grouped.setdefault(key, []).append(word)

    lines = []

    for key, line_words in grouped.items():
        line_words.sort(
            key=lambda w: w["left"],
            reverse=True,
        )

        lines.append(
            {
                "key": key,
                "top": min(
                    word["top"]
                    for word in line_words
                ),
                "words": line_words,
                "text": " ".join(
                    word["text"]
                    for word in line_words
                ),
            }
        )

    lines.sort(
        key=lambda line: line["top"]
    )

    return lines


def get_headword_candidate(
    line: dict,
) -> dict | None:
    if not line["words"]:
        return None

    rightmost = line["words"][0]
    raw_text = rightmost["text"]

    candidate = normalize_headword(raw_text)

    if not is_plausible_headword(candidate):
        return None

    if rightmost["left"] < HEADWORD_LEFT_MIN:
        return None

    return {
        "headword": candidate,
        "confidence": rightmost["conf"],
        "left": rightmost["left"],
        "top": rightmost["top"],
        "had_colon": (
            ":" in raw_text
            or "：" in raw_text
        ),
    }


def parse_entries(
    page_number: int,
    lines: list[dict],
) -> list[dict]:
    entries = []
    current = None

    for line in lines:
        headword_info = get_headword_candidate(
            line
        )

        if headword_info:
            if current:
                current["meaning"] = (
                    current["meaning"].strip()
                )
                entries.append(current)

            meaning_words = [
                word["text"]
                for word in line["words"][1:]
            ]

            current = {
                "headword": (
                    headword_info["headword"]
                ),
                "meaning": " ".join(
                    meaning_words
                ).strip(),
                "page": page_number,
                "headword_conf": round(
                    headword_info["confidence"],
                    2,
                ),
                "headword_left": (
                    headword_info["left"]
                ),
                "line_top": (
                    headword_info["top"]
                ),
                "had_colon": (
                    headword_info["had_colon"]
                ),
            }

            continue

        if current:
            if current["meaning"]:
                current["meaning"] += (
                    " " + line["text"]
                )
            else:
                current["meaning"] = (
                    line["text"]
                )

    if current:
        current["meaning"] = (
            current["meaning"].strip()
        )
        entries.append(current)

    return entries

def score_entry(entry: dict) -> dict:
    score = 100
    reasons = []

    headword = entry["headword"].strip()
    meaning = entry["meaning"].strip()
    confidence = entry["headword_conf"]

    # -------------------------
    # Headword confidence
    # -------------------------

    if confidence < 30:
        score -= 30
        reasons.append(
            "very_low_headword_confidence"
        )

    elif confidence < 60:
        score -= 15
        reasons.append(
            "low_headword_confidence"
        )

    # -------------------------
    # Colon / structure
    # -------------------------

    if not entry["had_colon"]:
        score -= 10
        reasons.append(
            "headword_without_colon"
        )

    # -------------------------
    # Meaning length
    # -------------------------

    if not meaning:
        score -= 60
        reasons.append(
            "empty_meaning"
        )

    elif len(meaning) < 8:
        score -= 25
        reasons.append(
            "very_short_meaning"
        )

    elif len(meaning) < 15:
        score -= 10
        reasons.append(
            "short_meaning"
        )

    # -------------------------
    # Headword checks
    # -------------------------

    if len(headword) <= 2:
        score -= 20
        reasons.append(
            "very_short_headword"
        )

    if DIGIT_RE.search(headword):
        score -= 40
        reasons.append(
            "digit_in_headword"
        )

    if SYMBOL_RE.search(headword):
        score -= 30
        reasons.append(
            "unexpected_symbol_in_headword"
        )

    # -------------------------
    # Meaning OCR noise
    # -------------------------

    if SYMBOL_RE.search(meaning):
        score -= 10
        reasons.append(
            "unexpected_symbol_in_meaning"
        )

    if TRAILING_NUMBER_NOISE_RE.search(meaning):
        score -= 10
        reasons.append(
            "trailing_number_noise"
        )

    if TRAILING_SINGLE_CHAR_RE.search(meaning):
        score -= 10
        reasons.append(
            "trailing_single_char_noise"
        )

    # -------------------------
    # Final score
    # -------------------------

    score = max(
        0,
        min(100, score),
    )

    # أي واحد من هذه الـflags
    # يمنع الدخول المباشر إلى accepted.
    critical_review_flags = {
        "very_low_headword_confidence",
        "low_headword_confidence",
        "headword_without_colon",
        "unexpected_symbol_in_headword",
        "unexpected_symbol_in_meaning",
        "very_short_meaning",
        "trailing_number_noise",
        "trailing_single_char_noise",
    }

    if (
        not is_plausible_headword(headword)
        or not meaning
        or score < 40
    ):
        quality_status = "rejected"

    elif (
        score < 85
        or any(
            flag in critical_review_flags
            for flag in reasons
        )
    ):
        quality_status = "review"

    else:
        quality_status = "accepted"

    return {
        **entry,
        "quality_score": score,
        "quality_status": quality_status,
        "review_flags": reasons,
    }

def build_report(
    entries: list[dict],
) -> dict:
    total = len(entries)

    accepted = [
        e for e in entries
        if e["quality_status"] == "accepted"
    ]

    review = [
        e for e in entries
        if e["quality_status"] == "review"
    ]

    rejected = [
        e for e in entries
        if e["quality_status"] == "rejected"
    ]

    confidences = [
        e["headword_conf"]
        for e in entries
        if e["headword_conf"] is not None
    ]

    return {
        "total_entries": total,
        "accepted": len(accepted),
        "review": len(review),
        "rejected": len(rejected),
        "accepted_pct": round(
            (len(accepted) / total * 100)
            if total
            else 0,
            2,
        ),
        "review_pct": round(
            (len(review) / total * 100)
            if total
            else 0,
            2,
        ),
        "rejected_pct": round(
            (len(rejected) / total * 100)
            if total
            else 0,
            2,
        ),
        "mean_headword_confidence": round(
            mean(confidences),
            2,
        ) if confidences else None,
    }


def main() -> None:
    all_entries = []

    for page_number in TEST_PAGES:
        tsv_path = (
            OCR_DIR
            / f"page_{page_number:04d}_layout.tsv"
        )

        if not tsv_path.exists():
            print(
                f"[v7] missing TSV: {tsv_path}"
            )
            continue

        words = load_words(tsv_path)
        lines = group_lines(words)

        raw_entries = parse_entries(
            page_number,
            lines,
        )

        scored_entries = [
            score_entry(entry)
            for entry in raw_entries
        ]

        all_entries.extend(
            scored_entries
        )

        print(
            f"[v7] page {page_number}: "
            f"{len(scored_entries)} entries"
        )

        for entry in scored_entries:
            print(
                f"  - {entry['headword']} "
                f"| score={entry['quality_score']} "
                f"| status={entry['quality_status']} "
                f"| flags={entry['review_flags']}"
            )

    report = build_report(
        all_entries
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            all_entries,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("[v7] QUALITY REPORT")
    print(
        f"total     : "
        f"{report['total_entries']}"
    )
    print(
        f"accepted  : "
        f"{report['accepted']} "
        f"({report['accepted_pct']}%)"
    )
    print(
        f"review    : "
        f"{report['review']} "
        f"({report['review_pct']}%)"
    )
    print(
        f"rejected  : "
        f"{report['rejected']} "
        f"({report['rejected_pct']}%)"
    )
    print(
        f"mean conf : "
        f"{report['mean_headword_confidence']}"
    )

    print()
    print(
        f"[v7] entries -> {OUTPUT_PATH}"
    )
    print(
        f"[v7] report  -> {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()