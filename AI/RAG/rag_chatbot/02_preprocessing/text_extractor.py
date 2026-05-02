"""
02_preprocessing/text_extractor.py
────────────────────────────────────
Extracts plain text from uploaded documents.

Supported formats:
    .pdf   → PyPDF2 (page-by-page; preserves page numbers in metadata)
    .docx  → python-docx (paragraph-by-paragraph)
    .txt   → plain UTF-8 / latin-1 read

Returns:
    A list of page/section dicts:
    [
        {"text": str, "page": int, "source": str},
        ...
    ]

Usage:
    from preprocessing.text_extractor import extract_text

    pages = extract_text(file_bytes=pdf_bytes, filename="report.pdf")
    for page in pages:
        print(page["page"], page["text"][:100])
"""

import io
import sys
import logging
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger(__name__)


def extract_text(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    Extract text from a file given its raw bytes and filename.

    Args:
        file_bytes: Raw bytes of the file.
        filename  : Original filename (used to determine the parser).

    Returns:
        List of dicts: [{"text": str, "page": int, "source": str}]

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(file_bytes, filename)
    elif ext == ".docx":
        return _extract_docx(file_bytes, filename)
    elif ext == ".txt":
        return _extract_txt(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'")


def _extract_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    """Extract text from a PDF file page-by-page using PyPDF2."""
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("Install PyPDF2: pip install PyPDF2")

    pages = []
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    total = len(reader.pages)
    log.info("PDF '%s': %d page(s)", filename, total)

    for page_num, page_obj in enumerate(reader.pages, start=1):
        raw = page_obj.extract_text() or ""
        if raw.strip():
            pages.append({
                "text": raw,
                "page": page_num,
                "source": filename,
            })

    if not pages:
        log.warning("PDF '%s' yielded no extractable text.", filename)
    return pages


def _extract_docx(file_bytes: bytes, filename: str) -> List[Dict]:
    """Extract text from a DOCX file, splitting on heading styles into sections."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

    doc = Document(io.BytesIO(file_bytes))
    sections = []
    current_text = []

    # Group paragraphs into logical sections (split on heading styles)
    section_index = 1
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading") and current_text:
            sections.append({
                "text": "\n".join(current_text),
                "page": section_index,
                "source": filename,
            })
            current_text = [text]
            section_index += 1
        else:
            current_text.append(text)

    if current_text:
        sections.append({
            "text": "\n".join(current_text),
            "page": section_index,
            "source": filename,
        })

    log.info("DOCX '%s': %d section(s)", filename, len(sections))
    return sections


def _extract_txt(file_bytes: bytes, filename: str) -> List[Dict]:
    """Decode a plain-text file, trying utf-8, latin-1, and cp1252 in order."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            text = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = file_bytes.decode("utf-8", errors="replace")

    log.info("TXT '%s': %d character(s)", filename, len(text))
    return [{"text": text, "page": 1, "source": filename}]
