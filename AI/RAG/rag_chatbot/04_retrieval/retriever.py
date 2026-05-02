"""
04_retrieval/retriever.py
──────────────────────────
Retrieves the most relevant document chunks for a given user query.

How it works:
    1. Embed the user's query using the same model used at indexing time.
    2. Search the FAISS index for the top-k nearest vectors.
    3. Return the matching chunks (with text, source, page, score).

The retrieved chunks are then handed to the prompt builder (05_generation)
which constructs the context for the LLM.

Usage:
    from retrieval.retriever import Retriever

    retriever = Retriever(session_id="abc123")
    results   = retriever.retrieve("What is the refund policy?")
    for r in results:
        print(r["score"], r["source"], r["text"][:80])
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg
from embedding.embedder import embed_query, get_embedder
from embedding.vector_store import VectorStore

log = logging.getLogger(__name__)


class Retriever:
    """
    Wraps a VectorStore and exposes a simple retrieve() interface.

    Attributes:
        session_id: Identifies which FAISS index to load/use.
        top_k     : Default number of chunks to return.
    """

    def __init__(self, session_id: str, top_k: int = None):
        self.session_id = session_id
        self.top_k = top_k or cfg.TOP_K
        self._vector_store = VectorStore(session_id=session_id)
        self._embedder = None
        self._loaded = False

    def load_index(self) -> bool:
        """
        Load the FAISS index for this session from local disk.

        Returns:
            True if the index was found and loaded, False otherwise.
        """
        self._loaded = self._vector_store.load()
        if not self._loaded:
            log.warning(
                "No index found for session '%s'. "
                "Upload and process documents first.",
                self.session_id,
            )
        return self._loaded

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Embed the query and return the top-k most relevant chunks.

        Args:
            query: Natural language question from the user.
            top_k: Number of chunks to return (overrides instance default).

        Returns:
            List of chunk dicts sorted by relevance (highest score first):
            [
                {
                    "text"        : str,
                    "source"      : str,
                    "page"        : int,
                    "chunk_index" : int,
                    "score"       : float,   # cosine similarity 0-1
                },
                ...
            ]

        Raises:
            RuntimeError: If the index has not been loaded.
        """
        if not self._loaded:
            raise RuntimeError(
                f"Index not loaded for session '{self.session_id}'. "
                "Call load_index() first."
            )

        if self._embedder is None:
            self._embedder = get_embedder()

        query_vec = embed_query(query, self._embedder)
        k = top_k or self.top_k
        results = self._vector_store.search(query_vec, top_k=k)

        log.info(
            "Query: '%s...' → %d chunk(s) retrieved",
            query[:60], len(results),
        )
        return results

    def format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM.

        Args:
            results: Output of retrieve().

        Returns:
            A formatted string with source citations and chunk text.
        """
        lines = []
        for i, chunk in enumerate(results, start=1):
            source_label = f"[Source {i}: {chunk['source']}, p.{chunk['page']}]"
            lines.append(f"{source_label}\n{chunk['text']}")
        return "\n\n".join(lines)

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store
