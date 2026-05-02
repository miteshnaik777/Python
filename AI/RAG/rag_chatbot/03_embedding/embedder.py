"""
03_embedding/embedder.py
─────────────────────────
Generates dense vector embeddings for text chunks.

Model: all-MiniLM-L6-v2 (SentenceTransformers)
    - Output dimension : 384
    - Fast and lightweight; ideal for semantic search
    - Runs on CPU (no GPU required in the classroom)

Why embeddings?
    Converting text to a fixed-length vector allows us to measure
    semantic similarity using cosine / L2 distance — so "refund policy"
    and "how to return an item" map to nearby vectors even though
    they share no words.

Usage:
    from embedding.embedder import get_embedder, embed_chunks, embed_query

    embedder = get_embedder()
    vectors  = embed_chunks(chunks, embedder)   # shape: (N, 384)
    q_vec    = embed_query("What is the return policy?", embedder)
"""

import sys
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder():
    """
    Load and cache the SentenceTransformer model.
    The model is downloaded on first call (~90 MB) and cached locally.

    Returns:
        A SentenceTransformer instance.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    log.info("Loading embedding model: %s", cfg.EMBEDDING_MODEL)
    model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    log.info("Embedding model loaded. Dimension: %d", cfg.EMBEDDING_DIM)
    return model


def embed_chunks(chunks: List[Dict], model=None) -> np.ndarray:
    """
    Generate embeddings for a list of chunk dicts.

    Args:
        chunks: List of dicts with at least a "text" key.
        model : Optional pre-loaded SentenceTransformer; loaded if None.

    Returns:
        numpy array of shape (len(chunks), EMBEDDING_DIM), dtype float32.
    """
    if model is None:
        model = get_embedder()

    texts = [chunk["text"] for chunk in chunks]
    log.info("Embedding %d chunk(s) ...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
    )
    log.info("Embeddings shape: %s", embeddings.shape)
    return embeddings.astype(np.float32)


def embed_query(query: str, model=None) -> np.ndarray:
    """
    Embed a single user query string.

    Args:
        query: Natural language question.
        model: Optional pre-loaded SentenceTransformer; loaded if None.

    Returns:
        numpy array of shape (1, EMBEDDING_DIM), dtype float32.
    """
    if model is None:
        model = get_embedder()

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.astype(np.float32)
