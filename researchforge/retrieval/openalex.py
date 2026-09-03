"""researchforge/retrieval/openalex.py — Canonical OpenAlex Literature Retriever.

RF-1.0.0-alpha.2.1: Canonical adapter for OpenAlex API.
Returns typed List[RetrievedItem].
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import requests

from .literature import LiteratureRetriever, RetrievedItem

logger = logging.getLogger(__name__)
OPENALEX_BASE = "https://api.openalex.org"


class OpenAlexRetriever:
    """Canonical OpenAlex API retriever implementing LiteratureRetriever protocol."""

    def __init__(self, email: Optional[str] = None, timeout: float = 15.0) -> None:
        self.email = email
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> List[RetrievedItem]:
        params: Dict[str, Any] = {
            "filter": f"title.search:{query}",
            "per-page": max_results,
            "select": "title,doi,publication_year,abstract_inverted_index,authorships",
        }
        if self.email:
            params["mailto"] = self.email

        try:
            resp = requests.get(f"{OPENALEX_BASE}/works", params=params, timeout=self.timeout)
            resp.raise_for_status()
            items: List[RetrievedItem] = []
            for item in resp.json().get("results", []):
                title = item.get("title", "")
                doi = item.get("doi", "") or ""
                items.append(RetrievedItem(
                    source="openalex",
                    title=title,
                    url=doi,
                    summary=f"Published: {item.get('publication_year', 'unknown')}",
                    score=1.0,
                ))
            return items
        except Exception as exc:
            logger.warning("OpenAlex search failed: %s", exc)
            return []
