"""
tests/test_embedding.py
─────────────────────────
Unit tests for the embedding module (embedder + vector_store).

Note: These tests do NOT require AWS credentials.
      They run fully locally using in-memory FAISS.

Run:
    python -m pytest tests/test_embedding.py -v
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding.embedder import embed_chunks, embed_query, get_embedder
from embedding.vector_store import VectorStore
from config import cfg


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder_model():
    """Load the embedding model once for the whole test module."""
    return get_embedder()


@pytest.fixture
def sample_chunks():
    return [
        {"text": "The refund policy allows returns within 30 days.", "source": "policy.pdf", "page": 1, "chunk_index": 0},
        {"text": "Employees are entitled to 15 days of annual leave.", "source": "hr.pdf", "page": 2, "chunk_index": 1},
        {"text": "The quarterly revenue grew by 12% year-over-year.", "source": "finance.pdf", "page": 3, "chunk_index": 2},
        {"text": "Product installation requires a 64-bit operating system.", "source": "manual.pdf", "page": 1, "chunk_index": 3},
        {"text": "Customer complaints should be escalated within 24 hours.", "source": "policy.pdf", "page": 5, "chunk_index": 4},
    ]


# ── embedder ──────────────────────────────────────────────────────────────────

class TestEmbedder:

    def test_embed_chunks_shape(self, sample_chunks, embedder_model):
        embeddings = embed_chunks(sample_chunks, model=embedder_model)
        assert embeddings.shape == (len(sample_chunks), cfg.EMBEDDING_DIM)

    def test_embed_chunks_dtype_float32(self, sample_chunks, embedder_model):
        embeddings = embed_chunks(sample_chunks, model=embedder_model)
        assert embeddings.dtype == np.float32

    def test_embed_query_shape(self, embedder_model):
        vec = embed_query("What is the return policy?", model=embedder_model)
        assert vec.shape == (1, cfg.EMBEDDING_DIM)

    def test_embeddings_are_normalized(self, sample_chunks, embedder_model):
        """L2 norms should be ~1.0 since normalize_embeddings=True."""
        embeddings = embed_chunks(sample_chunks, model=embedder_model)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_different_queries_produce_different_vectors(self, embedder_model):
        v1 = embed_query("refund policy", model=embedder_model)
        v2 = embed_query("installation guide", model=embedder_model)
        assert not np.allclose(v1, v2)


# ── VectorStore ───────────────────────────────────────────────────────────────

class TestVectorStore:

    def _build_store(self, session_id, sample_chunks, embedder_model, tmp_path):
        """Helper: build a vector store with a custom local tmp dir."""
        import faiss
        from config import cfg

        embeddings = embed_chunks(sample_chunks, model=embedder_model)
        vs = VectorStore(session_id=session_id)
        # Override local dir to tmp_path for test isolation
        vs._local_dir = tmp_path / session_id
        vs._local_dir.mkdir(parents=True, exist_ok=True)
        vs.build(sample_chunks, embeddings)
        return vs, embeddings

    def test_build_creates_index(self, sample_chunks, embedder_model, tmp_path):
        vs, _ = self._build_store("test-build", sample_chunks, embedder_model, tmp_path)
        assert vs.index is not None
        assert vs.index.ntotal == len(sample_chunks)

    def test_search_returns_top_k(self, sample_chunks, embedder_model, tmp_path):
        vs, _ = self._build_store("test-search", sample_chunks, embedder_model, tmp_path)
        query_vec = embed_query("refund and returns", model=embedder_model)
        results = vs.search(query_vec, top_k=3)
        assert len(results) == 3

    def test_search_top_result_is_relevant(self, sample_chunks, embedder_model, tmp_path):
        """The top result for 'refund' should be the refund policy chunk."""
        vs, _ = self._build_store("test-relevance", sample_chunks, embedder_model, tmp_path)
        query_vec = embed_query("What is the refund policy?", model=embedder_model)
        results = vs.search(query_vec, top_k=1)
        assert "refund" in results[0]["text"].lower()

    def test_search_results_have_score(self, sample_chunks, embedder_model, tmp_path):
        vs, _ = self._build_store("test-score", sample_chunks, embedder_model, tmp_path)
        query_vec = embed_query("leave days", model=embedder_model)
        results = vs.search(query_vec, top_k=2)
        for r in results:
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.1

    def test_save_and_load_local(self, sample_chunks, embedder_model, tmp_path):
        """Saving then loading should reproduce the same index size."""
        vs, _ = self._build_store("test-persist", sample_chunks, embedder_model, tmp_path)
        vs._save_local()

        vs2 = VectorStore(session_id="test-persist")
        vs2._local_dir = tmp_path / "test-persist"
        loaded = vs2._load_local()

        assert loaded is True
        assert vs2.index.ntotal == len(sample_chunks)
        assert len(vs2.metadata) == len(sample_chunks)

    def test_metadata_preserved_after_load(self, sample_chunks, embedder_model, tmp_path):
        vs, _ = self._build_store("test-meta", sample_chunks, embedder_model, tmp_path)
        vs._save_local()

        vs2 = VectorStore(session_id="test-meta")
        vs2._local_dir = tmp_path / "test-meta"
        vs2._load_local()

        for original, loaded in zip(sample_chunks, vs2.metadata):
            assert original["source"] == loaded["source"]
            assert original["text"] == loaded["text"]
