"""
02_preprocessing/chunker.py
─────────────────────────────
Splits cleaned text pages into overlapping chunks suitable for embedding.

Why chunking matters:
    - LLMs have a context window limit (e.g., 4096 tokens for Llama 2).
    - Smaller, focused chunks improve retrieval precision.
    - Overlap (50 chars) prevents a sentence from being split across chunks
      and losing its context at the boundary.

Strategy:
    LangChain's RecursiveCharacterTextSplitter splits on ["\n\n", "\n", ". ", " ", ""]
    in order, always preferring the largest natural break.

Chunk metadata:
    Each chunk dict carries:
    {
        "text"        : str   — the chunk content
        "source"      : str   — original filename
        "page"        : int   — page/section number in the source document
        "chunk_index" : int   — 0-based position within this document's chunks
    }

Usage:
    from preprocessing.chunker import chunk_pages

    chunks = chunk_pages(cleaned_pages)
    print(f"Total chunks: {len(chunks)}")
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg

log = logging.getLogger(__name__)


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Split a list of cleaned page dicts into overlapping text chunks.

    Args:
        pages: Output from cleaner.clean_pages() —
               list of {"text", "source", "page"} dicts.

    Returns:
        List of chunk dicts:
        [{"text", "source", "page", "chunk_index"}, ...]
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError("Install langchain: pip install langchain langchain-text-splitters")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Dict] = []
    chunk_index = 0

    for page in pages:
        raw_chunks = splitter.split_text(page["text"])
        for chunk_text in raw_chunks:
            if not chunk_text.strip():
                continue
            all_chunks.append({
                "text": chunk_text.strip(),
                "source": page["source"],
                "page": page["page"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    log.info(
        "Chunked %d page(s) → %d chunk(s) "
        "(chunk_size=%d, overlap=%d)",
        len(pages), len(all_chunks), cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP,
    )
    return all_chunks


def chunk_text(text: str, source: str = "unknown") -> List[Dict]:
    """
    Convenience wrapper: chunk a single string directly.

    Args:
        text  : Raw or cleaned text.
        source: Label to attach as metadata.

    Returns:
        List of chunk dicts.
    """
    page = {"text": text, "source": source, "page": 1}
    return chunk_pages([page])
