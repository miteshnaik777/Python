"""
tests/test_preprocessing.py
────────────────────────────
Unit tests for the preprocessing module (cleaner + chunker + extractor).

Run:
    python -m pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.cleaner import clean_text, clean_pages
from preprocessing.chunker import chunk_pages, chunk_text
from preprocessing.text_extractor import extract_text


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        result = clean_text("hello   world")
        assert result == "hello world"

    def test_collapses_many_newlines(self):
        result = clean_text("line1\n\n\n\nline2")
        assert result == "line1\n\nline2"

    def test_removes_null_bytes(self):
        result = clean_text("hello\x00world")
        assert "\x00" not in result

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_preserves_meaningful_newlines(self):
        text = "Paragraph one.\n\nParagraph two."
        result = clean_text(text)
        assert "Paragraph one." in result
        assert "Paragraph two." in result


class TestCleanPages:

    def test_drops_empty_pages_after_cleaning(self):
        pages = [
            {"text": "   \n\n  ", "source": "doc.pdf", "page": 1},
            {"text": "Real content here.", "source": "doc.pdf", "page": 2},
        ]
        result = clean_pages(pages)
        assert len(result) == 1
        assert result[0]["page"] == 2

    def test_preserves_metadata(self):
        pages = [{"text": "Some text.", "source": "test.pdf", "page": 3}]
        result = clean_pages(pages)
        assert result[0]["source"] == "test.pdf"
        assert result[0]["page"] == 3


# ── chunk_text / chunk_pages ──────────────────────────────────────────────────

class TestChunker:

    def _make_long_text(self, n_words: int = 300) -> str:
        return " ".join(["word"] * n_words)

    def test_chunk_text_returns_list(self):
        chunks = chunk_text("Hello world. This is a test.", source="test.txt")
        assert isinstance(chunks, list)
        assert all("text" in c for c in chunks)

    def test_chunks_have_required_keys(self):
        chunks = chunk_text("Sample text for chunking.", source="doc.pdf")
        required = {"text", "source", "page", "chunk_index"}
        for chunk in chunks:
            assert required.issubset(chunk.keys()), f"Missing keys in chunk: {chunk}"

    def test_long_text_produces_multiple_chunks(self):
        long_text = self._make_long_text(500)
        chunks = chunk_text(long_text, source="big.txt")
        assert len(chunks) > 1

    def test_chunk_index_increments(self):
        long_text = self._make_long_text(500)
        chunks = chunk_text(long_text, source="big.txt")
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_each_chunk_within_size_limit(self):
        from config import cfg
        long_text = self._make_long_text(1000)
        chunks = chunk_text(long_text, source="x.txt")
        for chunk in chunks:
            assert len(chunk["text"]) <= cfg.CHUNK_SIZE + cfg.CHUNK_OVERLAP + 50

    def test_source_metadata_propagated(self):
        pages = [{"text": "Content here.", "source": "annual.pdf", "page": 5}]
        chunks = chunk_pages(pages)
        for c in chunks:
            assert c["source"] == "annual.pdf"
            assert c["page"] == 5


# ── text_extractor ────────────────────────────────────────────────────────────

class TestTextExtractor:

    def test_txt_extraction(self):
        content = b"Hello, this is a plain text document."
        pages = extract_text(content, "test.txt")
        assert len(pages) == 1
        assert "Hello" in pages[0]["text"]

    def test_txt_latin1_encoding(self):
        content = "Caf\xe9 au lait".encode("latin-1")
        pages = extract_text(content, "menu.txt")
        assert len(pages) == 1

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            extract_text(b"data", "image.jpg")

    def test_pages_have_source_key(self):
        pages = extract_text(b"Some text", "doc.txt")
        assert pages[0]["source"] == "doc.txt"
