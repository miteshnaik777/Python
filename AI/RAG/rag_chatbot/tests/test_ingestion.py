"""
tests/test_ingestion.py
────────────────────────
Unit tests for the ingestion module (file_validator).

Run:
    python -m pytest tests/test_ingestion.py -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.file_validator import validate_file, ValidationError, get_file_info


# ── validate_file ─────────────────────────────────────────────────────────────

class TestValidateFile:

    def test_valid_pdf(self):
        """A 1-byte PDF-named file should pass extension and size checks."""
        validate_file("document.pdf", b"x")

    def test_valid_docx(self):
        validate_file("report.docx", b"x")

    def test_valid_txt(self):
        validate_file("notes.txt", b"hello world")

    def test_invalid_extension_raises(self):
        with pytest.raises(ValidationError, match="Unsupported file type"):
            validate_file("image.png", b"x")

    def test_empty_file_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_file("empty.pdf", b"")

    def test_uppercase_extension_accepted(self):
        """Extension check should be case-insensitive."""
        validate_file("REPORT.PDF", b"x")

    def test_file_too_large_raises(self):
        # Create a fake file > 20 MB
        huge_content = b"x" * (21 * 1024 * 1024)
        with pytest.raises(ValidationError, match="exceeds"):
            validate_file("big.pdf", huge_content)

    def test_exactly_at_limit_passes(self):
        """A file exactly at MAX_FILE_SIZE_MB should pass."""
        from config import cfg
        content = b"x" * (cfg.MAX_FILE_SIZE_MB * 1024 * 1024)
        validate_file("edge.pdf", content)


# ── get_file_info ─────────────────────────────────────────────────────────────

class TestGetFileInfo:

    def test_returns_correct_extension(self):
        info = get_file_info("Annual Report.PDF", b"data")
        assert info["extension"] == ".pdf"

    def test_returns_correct_size(self):
        content = b"hello" * 100
        info = get_file_info("test.txt", content)
        assert info["size_bytes"] == len(content)
        assert info["size_mb"] == round(len(content) / (1024 * 1024), 4)

    def test_filename_preserved(self):
        info = get_file_info("My Document.docx", b"x")
        assert info["filename"] == "My Document.docx"
