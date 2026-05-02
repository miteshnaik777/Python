"""
01_ingestion/file_validator.py
───────────────────────────────
Validates uploaded files before text extraction.

Checks performed:
    1. Extension must be in ALLOWED_EXTENSIONS (.pdf, .docx, .txt)
    2. File size must not exceed MAX_FILE_SIZE_MB
    3. File must not be empty (0 bytes)

Usage:
    from ingestion.file_validator import validate_file, ValidationError

    try:
        validate_file("report.pdf", file_bytes)
    except ValidationError as exc:
        print(exc)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg


class ValidationError(Exception):
    """Raised when an uploaded file fails validation."""


def validate_file(filename: str, content: bytes) -> None:
    """
    Validate a file by name and raw bytes.

    Args:
        filename: Original filename (used to extract the extension).
        content:  Raw file bytes.

    Raises:
        ValidationError: If any check fails.
    """
    _check_extension(filename)
    _check_not_empty(filename, content)
    _check_size(filename, content)


def _check_extension(filename: str) -> None:
    """Raise ValidationError if the file extension is not in ALLOWED_EXTENSIONS."""
    ext = Path(filename).suffix.lower()
    if ext not in cfg.ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(cfg.ALLOWED_EXTENSIONS)}"
        )


def _check_not_empty(filename: str, content: bytes) -> None:
    """Raise ValidationError if the file is 0 bytes."""
    if len(content) == 0:
        raise ValidationError(f"File '{filename}' is empty (0 bytes).")


def _check_size(filename: str, content: bytes) -> None:
    """Raise ValidationError if the file exceeds MAX_FILE_SIZE_MB."""
    size_mb = len(content) / (1024 * 1024)
    if size_mb > cfg.MAX_FILE_SIZE_MB:
        raise ValidationError(
            f"File '{filename}' is {size_mb:.1f} MB, "
            f"which exceeds the {cfg.MAX_FILE_SIZE_MB} MB limit."
        )


def get_file_info(filename: str, content: bytes) -> dict:
    """
    Return a metadata dict for a validated file.

    Returns:
        {
            "filename": str,
            "extension": str,
            "size_bytes": int,
            "size_mb": float,
        }
    """
    return {
        "filename": filename,
        "extension": Path(filename).suffix.lower(),
        "size_bytes": len(content),
        "size_mb": round(len(content) / (1024 * 1024), 4),
    }
