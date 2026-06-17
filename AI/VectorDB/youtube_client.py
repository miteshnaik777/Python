"""Thin wrapper around the YouTube Data API v3."""

from __future__ import annotations

from typing import TypedDict

from googleapiclient.discovery import build


class Video(TypedDict):
    title: str
    channel: str


class YouTubeClient:
    """Fetches video metadata from the YouTube Data API."""

    def __init__(self, api_key: str) -> None:
        self._service = build("youtube", "v3", developerKey=api_key)

    def fetch_trending(
        self, region_code: str = "IN", max_results: int = 20
    ) -> list[Video]:
        """Return the most-popular videos in a region as (title, channel) dicts."""
        response = (
            self._service.videos()
            .list(
                part="snippet",
                chart="mostPopular",
                regionCode=region_code,
                maxResults=max_results,
            )
            .execute()
        )

        return [
            Video(
                title=item["snippet"]["title"],
                channel=item["snippet"]["channelTitle"],
            )
            for item in response.get("items", [])
        ]
