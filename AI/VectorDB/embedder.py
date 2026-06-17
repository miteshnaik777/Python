"""Text embedding using a sentence-transformers model."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


class Embedder:
    """Encodes text into dense vector embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        """Encode a single string into a list of floats."""
        return self._model.encode(text).tolist()
