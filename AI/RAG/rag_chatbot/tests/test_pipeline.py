"""
tests/test_pipeline.py
────────────────────────
Integration tests for the full RAG pipeline (local index; no network LLM calls).

Demo LLM mode is forced so query() never calls external APIs.

Run:
    python -m pytest tests/test_pipeline.py -v
"""

import sys
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_TXT_CONTENT = b"""
SevenMentor Refund Policy

Customers may request a full refund within 30 days of purchase.
To initiate a refund, please contact support@sevenmentor.com with
your order number and reason for return.

Refunds are processed within 5-7 business days after approval.
No refunds will be issued after 30 days from the date of purchase.
"""

SAMPLE_SESSION = "test-session-01"


@pytest.fixture(autouse=True)
def demo_llm_mode(monkeypatch):
    """No external LLM calls — use placeholder answers from the pipeline."""
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("RAG_USE_DEMO_LLM", "true")


# ── Ingestion tests ───────────────────────────────────────────────────────────

class TestPipelineIngest:

    def test_ingest_txt_file_returns_chunks(self, tmp_path):
        from pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        result = pipeline.ingest(
            files=[("policy.txt", SAMPLE_TXT_CONTENT)],
            session_id=SAMPLE_SESSION,
        )

        assert result["files_processed"] == 1
        assert result["total_chunks"] > 0
        assert result["session_id"] == SAMPLE_SESSION

    def test_ingest_invalid_extension_skipped(self):
        from pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        result = pipeline.ingest(
            files=[("image.png", b"fake png data")],
            session_id=SAMPLE_SESSION,
        )

        assert result["files_processed"] == 0
        assert len(result["skipped_files"]) == 1
        assert result["skipped_files"][0][0] == "image.png"

    def test_ingest_empty_file_skipped(self):
        from pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        result = pipeline.ingest(
            files=[("empty.txt", b"")],
            session_id=SAMPLE_SESSION,
        )

        assert result["files_processed"] == 0
        assert len(result["skipped_files"]) == 1

    def test_ingest_multiple_files(self):
        from pipeline.rag_pipeline import RAGPipeline

        second_doc = b"This is a second document about HR policies and annual leave."
        pipeline = RAGPipeline()
        result = pipeline.ingest(
            files=[
                ("policy.txt", SAMPLE_TXT_CONTENT),
                ("hr.txt", second_doc),
            ],
            session_id=SAMPLE_SESSION + "-multi",
        )

        assert result["files_processed"] == 2
        assert result["total_chunks"] > 1


# ── Query tests ───────────────────────────────────────────────────────────────

class TestPipelineQuery:

    @pytest.fixture(autouse=True)
    def setup_index(self):
        """Build an in-memory index before query tests."""
        from pipeline.rag_pipeline import RAGPipeline
        self.pipeline = RAGPipeline()
        self.pipeline.ingest(
            files=[("policy.txt", SAMPLE_TXT_CONTENT)],
            session_id=SAMPLE_SESSION,
        )

    def test_query_returns_answer(self):
        result = self.pipeline.query(
            question="What is the refund policy?",
            session_id=SAMPLE_SESSION,
        )
        assert "answer" in result
        assert "sources" in result
        assert "question" in result

    def test_query_answer_not_empty(self):
        result = self.pipeline.query(
            question="How many days for refund?",
            session_id=SAMPLE_SESSION,
        )
        assert result["answer"] != ""

    def test_query_sources_contain_metadata(self):
        result = self.pipeline.query(
            question="Tell me about refunds",
            session_id=SAMPLE_SESSION,
        )
        for source in result["sources"]:
            assert "text" in source
            assert "source" in source
            assert "page" in source

    def test_query_no_index_returns_guidance(self):
        from pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        result = pipeline.query(
            question="Anything",
            session_id="non-existent-session-xyz",
        )
        assert "answer" in result
        assert len(result["answer"]) > 0


# ── Prompt builder tests ──────────────────────────────────────────────────────

class TestPromptBuilder:

    def test_prompt_contains_system_marker(self):
        from generation.prompt_builder import build_rag_prompt

        chunks = [{"text": "Refunds within 30 days.", "source": "policy.pdf", "page": 1}]
        prompt = build_rag_prompt("What is the refund policy?", chunks)
        assert "[INST]" in prompt
        assert "<<SYS>>" in prompt

    def test_prompt_contains_query(self):
        from generation.prompt_builder import build_rag_prompt

        question = "What is the capital of France?"
        chunks = [{"text": "Paris is the capital.", "source": "geo.pdf", "page": 1}]
        prompt = build_rag_prompt(question, chunks)
        assert question in prompt

    def test_prompt_contains_chunk_text(self):
        from generation.prompt_builder import build_rag_prompt

        chunks = [{"text": "Unique sentinel text XYZ123.", "source": "doc.pdf", "page": 1}]
        prompt = build_rag_prompt("Any question", chunks)
        assert "Unique sentinel text XYZ123." in prompt
