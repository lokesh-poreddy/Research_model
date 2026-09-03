"""researchforge/retrieval/semantic_scholar.py — Canonical Semantic Scholar Retriever.

RF-1.0.0-alpha.2.1: Canonical adapter for Semantic Scholar API.
Returns typed List[RetrievedItem].
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import requests

from .literature import LiteratureRetriever, RetrievedItem

logger = logging.getLogger(__name__)
S2_BASE = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarRetriever:
    """Canonical Semantic Scholar API retriever implementing LiteratureRetriever protocol."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 15.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> List[RetrievedItem]:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        params: Dict[str, Any] = {
            "query": query,
            "limit": max_results,
            "fields": "title,year,abstract,authors,externalIds,url",
        }
        try:
            resp = requests.get(
                f"{S2_BASE}/paper/search", params=params, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
            items: List[RetrievedItem] = []
            for item in resp.json().get("data", []):
                title = item.get("title", "")
                url = item.get("url", "") or ""
                abstract = item.get("abstract", "") or ""
                items.append(RetrievedItem(
                    source="semantic_scholar",
                    title=title,
                    url=url,
                    summary=abstract[:200],
                    score=1.0,
                ))
            return items
        except Exception as exc:
            logger.warning("Semantic Scholar search failed: %s", exc)
            return []
