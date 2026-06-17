"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Immutable, app-wide configuration."""

    youtube_api_key: str
    region_code: str = "IN"
    max_results: int = 20
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "youtube_videos"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from environment variables; fails fast if key is missing."""
        key = os.getenv("YOUTUBE_API_KEY")
        if not key:
            raise RuntimeError(
                "YOUTUBE_API_KEY is not set. "
                "Add it to your .env file (see .env.example)."
            )
        return cls(youtube_api_key=key)
