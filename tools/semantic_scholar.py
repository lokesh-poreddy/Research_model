"""
Semantic Scholar API client.
GET https://api.semanticscholar.org/graph/v1/paper/search?query=<query>
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)
S2_BASE = "https://api.semanticscholar.org/graph/v1"


def search_papers(
    query: str,
    max_results: int = 5,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search Semantic Scholar."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,year,abstract,authors,externalIds",
    }
    try:
        resp = requests.get(f"{S2_BASE}/paper/search", params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        results = []
        for item in resp.json().get("data", []):
            results.append({
                "title": item.get("title", ""),
                "year": item.get("year"),
                "abstract": item.get("abstract", ""),
                "doi": item.get("externalIds", {}).get("DOI", ""),
                "source": "semantic_scholar",
            })
        return results
    except Exception as exc:
        logger.warning("Semantic Scholar query failed: %s", exc)
        return []
