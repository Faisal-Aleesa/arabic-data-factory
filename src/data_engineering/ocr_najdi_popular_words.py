from pathlib import Path
import subprocess

from pdf2image import convert_from_path


PDF_PATH = Path("data/raw/najdi/majam_alkalimat_alshaabia_najd.pdf")
OUTPUT_DIR = Path("data/interim/najdi/ocr_pages")

TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_DIR = Path(r"C:\Temp\tessdata_best")
POPPLER_PATH = r"C:\poppler\poppler-26.07.0\Library\bin"

DPI = 350
LANG = "ara"
PSM = "6"
OEM = "1"

# Temporary test range
FIRST_PAGE = 27
LAST_PAGE = 321


def run_ocr_on_page(image_path: Path, output_base: Path) -> None:
    command = [
        str(TESSERACT_EXE),
        str(image_path),
        str(output_base),
        "-l",
        LANG,
        "--psm",
        PSM,
        "--oem",
        OEM,
        "--tessdata-dir",
        str(TESSDATA_DIR),
    ]

    subprocess.run(command, check=True)


def clean_ocr_text(text: str) -> str:
    """
    Remove invisible Unicode direction/control characters
    while preserving Arabic letters, spelling, punctuation,
    and source wording.
    """

    control_chars = [
        "\u061c",  # ARABIC LETTER MARK
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\u202a",  # LEFT-TO-RIGHT EMBEDDING
        "\u202b",  # RIGHT-TO-LEFT EMBEDDING
        "\u202c",  # POP DIRECTIONAL FORMATTING
        "\u202d",  # LEFT-TO-RIGHT OVERRIDE
        "\u202e",  # RIGHT-TO-LEFT OVERRIDE
        "\u2066",  # LEFT-TO-RIGHT ISOLATE
        "\u2067",  # RIGHT-TO-LEFT ISOLATE
        "\u2068",  # FIRST STRONG ISOLATE
        "\u2069",  # POP DIRECTIONAL ISOLATE
        "\ufeff",  # BOM / ZERO WIDTH NO-BREAK SPACE
    ]

    for char in control_chars:
        text = text.replace(char, "")

    return text


def save_cleaned_text(raw_text_path: Path) -> Path:
    raw_text = raw_text_path.read_text(encoding="utf-8")

    cleaned_text = clean_ocr_text(raw_text)

    cleaned_path = raw_text_path.with_name(
        f"{raw_text_path.stem}_cleaned.txt"
    )

    cleaned_path.write_text(
        cleaned_text,
        encoding="utf-8",
    )

    return cleaned_path


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    if not TESSERACT_EXE.exists():
        raise FileNotFoundError(
            f"Tesseract not found: {TESSERACT_EXE}"
        )

    arabic_model = TESSDATA_DIR / "ara.traineddata"

    if not arabic_model.exists():
        raise FileNotFoundError(
            f"Arabic tessdata_best model not found: {arabic_model}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ocr-najdi] PDF: {PDF_PATH}")
    print(f"[ocr-najdi] PDF pages: {FIRST_PAGE}-{LAST_PAGE}")
    print(f"[ocr-najdi] DPI: {DPI}")
    print(f"[ocr-najdi] language: {LANG}")
    print(f"[ocr-najdi] PSM: {PSM}")
    print(f"[ocr-najdi] OEM: {OEM}")
    print(f"[ocr-najdi] Poppler: {POPPLER_PATH}")

    pages = convert_from_path(
        PDF_PATH,
        dpi=DPI,
        poppler_path=POPPLER_PATH,
        first_page=FIRST_PAGE,
        last_page=LAST_PAGE,
    )

    print(f"[ocr-najdi] pages loaded: {len(pages)}")

    for pdf_page_number, page_image in enumerate(
        pages,
        start=FIRST_PAGE,
    ):
        image_path = (
            OUTPUT_DIR / f"page_{pdf_page_number:04d}.png"
        )

        output_base = (
            OUTPUT_DIR / f"page_{pdf_page_number:04d}"
        )

        raw_text_path = (
            OUTPUT_DIR / f"page_{pdf_page_number:04d}.txt"
        )

        page_image.save(image_path, "PNG")

        run_ocr_on_page(
            image_path=image_path,
            output_base=output_base,
        )

        cleaned_path = save_cleaned_text(raw_text_path)

        print(
            f"[ocr-najdi] PDF page "
            f"{pdf_page_number}/{LAST_PAGE} done "
            f"| cleaned -> {cleaned_path.name}"
        )

    print(f"[ocr-najdi] output -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()