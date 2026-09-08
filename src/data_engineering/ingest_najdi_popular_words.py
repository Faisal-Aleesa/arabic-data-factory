from pathlib import Path
import json


INPUT_PATH = Path(
    "data/processed/najdi_dictionary_final.json"
)

OUTPUT_PATH = Path(
    "data/interim/najdi/majam_alkalimat_alshaabia_najd.json"
)


DOC_ID = "majam_alkalimat_alshaabia_najd"
TITLE = "معجم الكلمات الشعبية في نجد"

# حط هنا القيم نفسها المسجلة عندكم في license_manifest.csv
AUTHOR = "عبدالرحمن بن عبدالعزيز المانع"
LICENSE = "CC BY-SA 4.0"

SOURCE = "Local scanned PDF"
REGION = "najdi"


def main() -> None:
    entries = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(entries, list):
        raise ValueError(
            "Expected a list of dictionary entries."
        )

    blocks = []

    for entry in entries:
        headword = (
            entry.get("headword", "")
            .strip()
        )

        meaning = (
            entry.get("meaning", "")
            .strip()
        )

        if not headword or not meaning:
            continue

        blocks.append(
            f"{headword}: {meaning}"
        )

    raw_text = "\n\n".join(blocks)

    document = {
        "doc_id": DOC_ID,
        "title": TITLE,
        "author": AUTHOR,
        "source": SOURCE,
        "license": LICENSE,
        "region": REGION,
        "raw_text": raw_text,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[ingest-najdi] input entries : {len(entries)}"
    )
    print(
        f"[ingest-najdi] usable blocks : {len(blocks)}"
    )
    print(
        f"[ingest-najdi] raw chars     : {len(raw_text)}"
    )
    print(
        f"[ingest-najdi] output -> {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()