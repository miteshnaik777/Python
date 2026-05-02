"""
03_embedding/vector_store.py
──────────────────────────────
Manages a FAISS vector index for document chunks.

Why FAISS?
    Facebook AI Similarity Search performs exact (IndexFlatL2) or
    approximate nearest-neighbor lookup in milliseconds over millions
    of vectors — far faster than a brute-force loop.

Persistence strategy:
    The index and its metadata are serialized to two files under
    tmp/index/<session_id>/:
        faiss.index   — binary FAISS index (numpy-compatible)
        metadata.json — list of chunk dicts (source, page, text, chunk_index)

Usage:
    from embedding.vector_store import VectorStore

    vs = VectorStore(session_id="abc123")
    vs.build(chunks, embeddings)      # first time
    vs.save()                         # persist locally
    vs.load()                         # restore from disk

    results = vs.search(query_vec, top_k=5)
"""

import json
import sys
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg

log = logging.getLogger(__name__)

_INDEX_FILE = "faiss.index"
_META_FILE = "metadata.json"


def _faiss_safe_path(path: Path) -> str:
    """
    Return a path string that FAISS's C++ I/O can open on Windows.
    Some FAISS builds raise Errno 22 (Invalid argument) for pathlib Paths
    or for long-path prefix (\\\\?\\).
    """
    s = os.path.normpath(str(path.resolve()))
    if sys.platform == "win32" and s.startswith("\\\\?\\"):
        s = s[4:]
    return s


class VectorStore:
    """FAISS-based vector store with on-disk persistence under LOCAL_INDEX_DIR."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.index = None
        self.metadata: List[Dict] = []
        self._local_dir = Path(cfg.LOCAL_INDEX_DIR) / session_id
        self._local_dir.mkdir(parents=True, exist_ok=True)

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        """
        Build a new FAISS IndexFlatIP (inner product = cosine for L2-normed vecs).

        Args:
            chunks    : List of chunk dicts (metadata stored in parallel).
            embeddings: Float32 array of shape (len(chunks), EMBEDDING_DIM).
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("Install faiss-cpu: pip install faiss-cpu")

        assert len(chunks) == len(embeddings), (
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match."
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner product (cosine for normalized vecs)
        self.index.add(embeddings)
        self.metadata = list(chunks)
        log.info(
            "FAISS index built: %d vector(s), dimension=%d",
            self.index.ntotal, dim,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int = None) -> List[Dict]:
        """
        Find the top-k most similar chunks for a query vector.

        Args:
            query_vector: Shape (1, EMBEDDING_DIM) float32 array.
            top_k       : Number of results (defaults to cfg.TOP_K).

        Returns:
            List of chunk dicts sorted by similarity (highest first),
            each augmented with a "score" key (cosine similarity, 0–1).
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call build() or load() first.")

        k = top_k or cfg.TOP_K
        k = min(k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        log.info("FAISS search returned %d result(s)", len(results))
        return results

    def get_vectors_and_metadata(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Return all stored vectors and their metadata for visualization.

        Returns:
            (vectors, metadata): vectors shape (n, EMBEDDING_DIM), metadata list of chunk dicts.

        Raises:
            RuntimeError: If index is not loaded.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")
        n = self.index.ntotal
        vectors = np.stack(
            [self.index.reconstruct(i) for i in range(n)],
            axis=0,
            dtype=np.float32,
        )
        return vectors, list(self.metadata)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save index + metadata to the session directory under LOCAL_INDEX_DIR."""
        self._save_local()

    def load(self) -> bool:
        """
        Load index + metadata from local disk.

        Returns:
            True if loaded successfully, False if no index exists.
        """
        return self._load_local()

    def _save_local(self) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("Install faiss-cpu: pip install faiss-cpu")

        index_path = self._local_dir / _INDEX_FILE
        meta_path = self._local_dir / _META_FILE

        faiss.write_index(self.index, _faiss_safe_path(index_path))
        meta_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Index saved locally to %s", self._local_dir)

    def _load_local(self) -> bool:
        try:
            import faiss
        except ImportError:
            raise ImportError("Install faiss-cpu: pip install faiss-cpu")

        index_path = self._local_dir / _INDEX_FILE
        meta_path = self._local_dir / _META_FILE

        if not index_path.exists() or not meta_path.exists():
            log.warning("No local index found at %s", self._local_dir)
            return False

        self.index = faiss.read_index(_faiss_safe_path(index_path))
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        log.info(
            "Index loaded from %s: %d vector(s)",
            self._local_dir, self.index.ntotal,
        )
        return True
