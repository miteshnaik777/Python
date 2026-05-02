"""
config.py
─────────
Central configuration for the RAG Chatbot project.
All modules import from here — never hardcode values elsewhere.

Usage:
    from config import cfg
    print(cfg.LLM_PROVIDER)
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Register numbered-directory aliases (01_ingestion → ingestion, etc.)
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))
import pathsetup  # noqa: F401

# Load .env from project root so it works regardless of working directory (e.g. Streamlit)
load_dotenv(_project_root / ".env")


@dataclass
class Config:
    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384            # dimension for all-MiniLM-L6-v2

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K: int = 5                      # number of chunks returned per query

    # ── Generation ────────────────────────────────────────────────────────────
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    # If True, return a short placeholder instead of calling an LLM
    USE_DEMO_LLM: bool = field(
        default_factory=lambda: os.getenv("RAG_USE_DEMO_LLM", "").lower() in ("1", "true", "yes")
    )

    # ── External LLM (OpenAI, Gemini, Groq, Ollama, OpenAI-compatible APIs) ──
    # Set LLM_PROVIDER to use: openai | gemini | groq | ollama | openai_compatible
    LLM_PROVIDER: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "").strip().lower()
    )
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    EXTERNAL_LLM_BASE_URL: str = field(
        default_factory=lambda: os.getenv("EXTERNAL_LLM_BASE_URL", "").strip()
    )
    EXTERNAL_LLM_API_KEY: str = field(
        default_factory=lambda: os.getenv("EXTERNAL_LLM_API_KEY", "").strip()
    )
    EXTERNAL_LLM_MODEL: str = field(
        default_factory=lambda: os.getenv("EXTERNAL_LLM_MODEL", "llama-3.3-70b-versatile")
    )

    # ── File Upload ───────────────────────────────────────────────────────────
    ALLOWED_EXTENSIONS: tuple = (".pdf", ".docx", ".txt")
    MAX_FILE_SIZE_MB: int = 20

    # ── Local temp paths ──────────────────────────────────────────────────────
    LOCAL_TMP_DIR: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "tmp")
    )
    LOCAL_INDEX_DIR: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "tmp", "index")
    )


# Singleton instance used across the project
cfg = Config()

# Ensure local temp directories exist at import time
os.makedirs(cfg.LOCAL_TMP_DIR, exist_ok=True)
os.makedirs(cfg.LOCAL_INDEX_DIR, exist_ok=True)
