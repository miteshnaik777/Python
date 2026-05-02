"""
02_preprocessing/cleaner.py
─────────────────────────────
Normalizes raw extracted text before chunking.

Operations performed (in order):
    1. Replace null bytes and non-printable control characters
    2. Normalize Unicode to NFC form
    3. Collapse multiple consecutive whitespace/newlines
    4. Strip leading/trailing whitespace per "page" entry

Usage:
    from preprocessing.cleaner import clean_pages, clean_text

    cleaned_pages = clean_pages(raw_pages)
    clean_str = clean_text(raw_string)
"""

import re
import sys
import unicodedata
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    Clean a single string of extracted text.

    Args:
        text: Raw text from a PDF page, DOCX section, or TXT file.

    Returns:
        Cleaned, normalized string.
    """
    # Remove null bytes and other control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Normalize Unicode (NFC: canonical decomposition then composition)
    text = unicodedata.normalize("NFC", text)

    # Replace Windows-style line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ consecutive newlines into a paragraph break (2 newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs on a single line into one space
    text = re.sub(r"[ \t]+", " ", text)

    # Strip each line individually, then rejoin
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Final strip
    return text.strip()


def clean_pages(pages: List[Dict]) -> List[Dict]:
    """
    Apply clean_text() to every page/section dict returned by text_extractor.

    Args:
        pages: List of dicts with at least a "text" key.

    Returns:
        New list of dicts with cleaned "text" values.
        Pages that become empty after cleaning are dropped.
    """
    cleaned = []
    for page in pages:
        text = clean_text(page.get("text", ""))
        if text:
            cleaned.append({**page, "text": text})
    return cleaned
