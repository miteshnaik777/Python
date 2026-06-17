"""Gradio UI for the semantic search service."""

from __future__ import annotations

import gradio as gr

from search_service import SearchService


class SearchUI:
    """Builds a Gradio Interface bound to a SearchService."""

    def __init__(self, service: SearchService) -> None:
        self._service = service

    def _handle(self, query: str) -> str:
        results = self._service.search(query)
        return "\n\n".join(results) if results else "No results."

    def build(self) -> gr.Interface:
        """Return a configured Gradio Interface ready to launch."""
        return gr.Interface(
            fn=self._handle,
            inputs="text",
            outputs="text",
            title="YouTube Semantic Search",
        )
