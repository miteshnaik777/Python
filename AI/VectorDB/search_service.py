"""Orchestrates embedding and vector storage for indexing and search."""

from __future__ import annotations

from embedder import Embedder
from vector_store import VectorStore
from youtube_client import Video


class SearchService:
    """Coordinates an Embedder and a VectorStore to index and search videos."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def index_videos(self, videos: list[Video]) -> None:
        """Embed and store each video as `"{title} {channel}"`."""
        for i, video in enumerate(videos):
            text = f"{video['title']} {video['channel']}"
            self._store.add(str(i), text, self._embedder.encode(text))

    def search(self, query: str, n_results: int = 5) -> list[str]:
        """Return the top-N matching documents for a free-text query."""
        return self._store.query(self._embedder.encode(query), n_results)
