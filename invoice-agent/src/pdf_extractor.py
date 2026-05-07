"""
PDF text extraction with automatic OCR fallback.

Strategy:
  1. Try pdfplumber (fast, lossless for digital PDFs).
  2. If extracted text is empty or suspiciously short (<50 chars), the PDF is
     likely a scanned image. Fall back to Tesseract OCR via pdf2image.

This two-stage approach avoids running OCR unnecessarily (it's 10-20× slower)
while still handling real-world scanned invoices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from config.settings import POPPLER_PATH, TESSERACT_PATH

logger = logging.getLogger(__name__)

# Point Tesseract at the Windows binary — harmless on Linux/Mac if the env var
# is not set (pytesseract will find the system install).
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Minimum character count below which we consider the digital extraction failed
_MIN_TEXT_LENGTH = 50


@dataclass
class ExtractionResult:
    text: str
    used_ocr: bool
    page_count: int
    source_path: str


def extract_text(pdf_path: str | Path) -> ExtractionResult:
    """
    Extract all text from a PDF file.

    Tries pdfplumber first; falls back to Tesseract OCR if the result is
    too short to be useful.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        ExtractionResult with the extracted text, OCR flag, and metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError: If both extraction methods fail.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Extracting text from %s", path.name)

    # --- Stage 1: digital extraction ---
    try:
        text, page_count = _extract_with_pdfplumber(path)
    except Exception as exc:
        logger.warning("pdfplumber failed (%s) — trying OCR", exc)
        text, page_count = "", 0

    if len(text.strip()) >= _MIN_TEXT_LENGTH:
        logger.info("Digital extraction succeeded (%d chars)", len(text))
        return ExtractionResult(
            text=text, used_ocr=False, page_count=page_count, source_path=str(path)
        )

    # --- Stage 2: OCR fallback ---
    logger.info(
        "Text too short (%d chars) — falling back to OCR", len(text.strip())
    )
    try:
        ocr_text, page_count = _extract_with_ocr(path)
    except Exception as exc:
        raise RuntimeError(
            f"Both pdfplumber and OCR failed for {path.name}"
        ) from exc

    if not ocr_text.strip():
        raise RuntimeError(f"OCR returned empty text for {path.name}")

    logger.info("OCR extraction succeeded (%d chars)", len(ocr_text))
    return ExtractionResult(
        text=ocr_text, used_ocr=True, page_count=page_count, source_path=str(path)
    )


def _extract_with_pdfplumber(path: Path) -> tuple[str, int]:
    """Return (full_text, page_count) using pdfplumber."""
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
    return "\n\n".join(pages), page_count


def _extract_with_ocr(path: Path) -> tuple[str, int]:
    """
    Convert each PDF page to an image and run Tesseract OCR.

    poppler_path is required on Windows; on Linux/Mac it can be None and
    pdf2image will find Poppler automatically.
    """
    kwargs: dict = {}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH

    images: list[Image.Image] = convert_from_path(str(path), dpi=300, **kwargs)
    page_texts: list[str] = []

    for i, img in enumerate(images):
        logger.debug("OCR processing page %d/%d", i + 1, len(images))
        text = pytesseract.image_to_string(img, lang="eng")
        page_texts.append(text)

    return "\n\n".join(page_texts), len(images)
