"""ChromaDB-backed vector store for documents and embeddings."""

from __future__ import annotations

import chromadb


class VectorStore:
    """Wraps a single Chroma collection for add/query operations."""

    def __init__(self, collection_name: str) -> None:
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(collection_name)

    def add(self, doc_id: str, document: str, embedding: list[float]) -> None:
        """Insert a single document with its precomputed embedding."""
        self._collection.add(
            ids=[doc_id],
            documents=[document],
            embeddings=[embedding],
        )

    def query(self, embedding: list[float], n_results: int = 5) -> list[str]:
        """Return the top-N most similar documents for a query embedding."""
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )
        documents = results.get("documents") or [[]]
        return documents[0]
