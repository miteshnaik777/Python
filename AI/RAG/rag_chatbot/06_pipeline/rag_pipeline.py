"""
06_pipeline/rag_pipeline.py
─────────────────────────────
The central orchestrator for the RAG system.

Two public entry points:

    ingest(files, session_id)
        Validate → Extract → Clean → Chunk → Embed → Index → Save (local disk)

    query(question, session_id, history)
        Load Index → Retrieve Chunks → Build Prompt → LLM → Return Answer

The Streamlit app (07_frontend/app.py) calls only these two functions —
all internal complexity is hidden here.

Usage:
    from pipeline.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()

    # --- Ingestion phase ---
    result = pipeline.ingest(
        files=[("report.pdf", pdf_bytes), ("policy.docx", docx_bytes)],
        session_id="abc123",
    )
    print(result["total_chunks"])

    # --- Query phase ---
    answer = pipeline.query(
        question="What is the refund policy?",
        session_id="abc123",
    )
    print(answer["answer"])
    print(answer["sources"])
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.file_validator import validate_file, ValidationError
from preprocessing.text_extractor import extract_text
from preprocessing.cleaner import clean_pages
from preprocessing.chunker import chunk_pages
from embedding.embedder import embed_chunks, get_embedder
from embedding.vector_store import VectorStore
from retrieval.retriever import Retriever
from config import cfg
from generation.prompt_builder import build_rag_prompt_messages
from generation.external_llm import ExternalLLM

log = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the full ingestion and query pipeline for the RAG chatbot.

    The pipeline is stateless between requests — session_id is used to
    identify the correct FAISS index under tmp/index/.
    """

    def __init__(self):
        self._llm = None  # ExternalLLM (lazy)
        self._embedder = None

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        files: List[Tuple[str, bytes]],
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Full ingestion pipeline: validate → extract → clean → chunk → embed → index → save locally.

        Args:
            files       : List of (filename, file_bytes) tuples.
            session_id  : Unique identifier for this upload session.

        Returns:
            {
                "session_id"    : str,
                "files_processed": int,
                "total_chunks"  : int,
                "skipped_files" : list of (filename, error_message),
            }
        """
        log.info("=== Ingestion started for session: %s ===", session_id)

        all_chunks = []
        skipped = []

        for filename, content in files:
            log.info("Processing: %s", filename)
            try:
                # Step 1: Validate
                validate_file(filename, content)

                # Step 2: Extract text
                pages = extract_text(file_bytes=content, filename=filename)

                # Step 3: Clean
                pages = clean_pages(pages)

                # Step 4: Chunk
                chunks = chunk_pages(pages)
                all_chunks.extend(chunks)

                log.info("  → %d chunk(s) from '%s'", len(chunks), filename)

            except (ValidationError, ValueError, RuntimeError) as exc:
                log.warning("Skipping '%s': %s", filename, exc)
                skipped.append((filename, str(exc)))

        if not all_chunks:
            log.warning("No chunks generated — check that files contain extractable text.")
            return {
                "session_id": session_id,
                "files_processed": 0,
                "total_chunks": 0,
                "skipped_files": skipped,
            }

        # Step 5: Embed
        if self._embedder is None:
            self._embedder = get_embedder()
        embeddings = embed_chunks(all_chunks, self._embedder)

        # Step 6: Build FAISS index and persist locally
        vs = VectorStore(session_id=session_id)
        vs.build(all_chunks, embeddings)
        vs.save()

        result = {
            "session_id": session_id,
            "files_processed": len(files) - len(skipped),
            "total_chunks": len(all_chunks),
            "skipped_files": skipped,
        }
        log.info("=== Ingestion complete: %s ===", result)
        return result

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        session_id: str,
        conversation_history: List[Dict] = None,
        top_k: int = None,
    ) -> Dict[str, Any]:
        """
        Full query pipeline: load index → retrieve → prompt → LLM → answer.

        Args:
            question            : User's natural language question.
            session_id          : Session ID whose index to query.
            conversation_history: Optional list of prior {"role", "content"} turns.
            top_k               : Override default number of chunks to retrieve.

        Returns:
            {
                "answer"  : str   — the LLM-generated response,
                "sources" : list  — list of source chunk dicts used as context,
                "question": str   — the original question,
            }
        """
        log.info("=== Query: '%s...' (session: %s) ===", question[:60], session_id)

        # Step 1: Load retriever + index from local disk
        retriever = Retriever(session_id=session_id, top_k=top_k)
        loaded = retriever.load_index()
        if not loaded:
            return {
                "answer": (
                    "No documents have been indexed for this session. "
                    "Please upload and process your files first."
                ),
                "sources": [],
                "question": question,
            }

        # Step 2: Retrieve relevant chunks
        chunks = retriever.retrieve(question, top_k=top_k)

        # Step 3 & 4: Build messages and call LLM
        use_external = (cfg.LLM_PROVIDER or "").strip()
        if use_external:
            messages = build_rag_prompt_messages(
                query=question,
                context_chunks=chunks,
                conversation_history=conversation_history or [],
            )
            if self._llm is None:
                self._llm = ExternalLLM()
            try:
                answer = self._llm.generate_messages(messages)
            except (ValueError, RuntimeError) as e:
                answer = (
                    f"**LLM error:** {e}\n\n"
                    "Check `.env`: set `LLM_PROVIDER` (e.g. groq) and the matching API key. "
                    "See **EXTERNAL_LLM_SETUP.md** in the project folder."
                )
            log.info("Answer generated via external LLM (%d chars)", len(answer))
        elif cfg.USE_DEMO_LLM:
            answer = (
                "**[Demo mode]** Set `LLM_PROVIDER` and the right API key in `.env` for real answers. "
                "See **EXTERNAL_LLM_SETUP.md**."
            )
            log.info("Demo LLM: returning placeholder")
        else:
            answer = (
                "**No LLM configured.** Set `LLM_PROVIDER` (e.g. `groq`) and the matching API key in `.env`, "
                "or set `RAG_USE_DEMO_LLM=true` for a placeholder. See **EXTERNAL_LLM_SETUP.md**."
            )
            log.info("No LLM provider: returning configuration hint")

        return {
            "answer": answer,
            "sources": chunks,
            "question": question,
        }
