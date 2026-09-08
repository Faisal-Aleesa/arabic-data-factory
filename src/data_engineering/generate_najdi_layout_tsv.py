from pathlib import Path
import subprocess


OCR_DIR = Path("data/interim/najdi/ocr_pages")

TESSERACT_EXE = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

TESSDATA_DIR = Path(
    r"C:\Temp\tessdata_best"
)

START_PAGE = 27
END_PAGE = 321

PAGES = range(
    START_PAGE,
    END_PAGE + 1,
)

def main() -> None:
    for page_number in PAGES:
        image_path = (
            OCR_DIR
            / f"page_{page_number:04d}.png"
        )

        output_base = (
            OCR_DIR
            / f"page_{page_number:04d}_layout"
        )

        tsv_path = Path(
            str(output_base) + ".tsv"
        )

        if not image_path.exists():
            print(
                f"[layout] missing image: "
                f"{image_path}"
            )
            continue

        if tsv_path.exists():
            print(
                f"[layout] page {page_number}: "
                f"already exists"
            )
            continue

        command = [
            str(TESSERACT_EXE),
            str(image_path),
            str(output_base),
            "-l",
            "ara",
            "--psm",
            "6",
            "--oem",
            "1",
            "--tessdata-dir",
            str(TESSDATA_DIR),
            "-c",
            "tessedit_create_tsv=1",
        ]

        print(
            f"[layout] page {page_number}: running..."
        )

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            print(
                f"[layout] page {page_number}: FAILED"
            )
            print(result.stderr)
            continue

        if tsv_path.exists():
            print(
                f"[layout] page {page_number}: OK"
            )
        else:
            print(
                f"[layout] page {page_number}: "
                f"TSV not created"
            )


if __name__ == "__main__":
    main()