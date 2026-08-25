"""
OpenAlex API client for literature retrieval.
GET https://api.openalex.org/works?filter=title.search:<query>
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)
OPENALEX_BASE = "https://api.openalex.org"


def search_papers(
    query: str,
    max_results: int = 5,
    email: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search OpenAlex for papers matching query.
    Returns list of paper dicts: {title, doi, year, abstract, authors}.
    """
    params: Dict[str, Any] = {
        "filter": f"title.search:{query}",
        "per-page": max_results,
        "select": "title,doi,publication_year,abstract_inverted_index,authorships",
    }
    if email:
        params["mailto"] = email

    try:
        resp = requests.get(f"{OPENALEX_BASE}/works", params=params, timeout=15)
        resp.raise_for_status()
        results = []
        for item in resp.json().get("results", []):
            results.append({
                "title": item.get("title", ""),
                "doi": item.get("doi", ""),
                "year": item.get("publication_year"),
                "source": "openalex",
            })
        logger.debug("OpenAlex returned %d results for '%s'.", len(results), query)
        return results
    except Exception as exc:
        logger.warning("OpenAlex query failed: %s", exc)
        return []


def get_paper_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    """Retrieve a paper by DOI."""
    try:
        url = f"{OPENALEX_BASE}/works/doi:{doi}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("OpenAlex DOI lookup failed: %s", exc)
        return None
