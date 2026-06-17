"""Entry point: wires Config, YouTube fetcher, embedder, store, and Gradio UI."""
###https://chatgpt.com/share/6a15bed7-4740-8323-a270-991c062b8f62
from __future__ import annotations

from config import Config
from embedder import Embedder
from search_service import SearchService
from ui import SearchUI
from vector_store import VectorStore
from youtube_client import YouTubeClient


def main() -> None:
    cfg = Config.from_env()

    videos = YouTubeClient(cfg.youtube_api_key).fetch_trending(
        region_code=cfg.region_code,
        max_results=cfg.max_results,
    )

    service = SearchService(
        embedder=Embedder(cfg.embedding_model),
        store=VectorStore(cfg.collection_name),
    )
    service.index_videos(videos)

    SearchUI(service).build().launch(share=True)


if __name__ == "__main__":
    main()
