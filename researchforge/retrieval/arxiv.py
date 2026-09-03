"""researchforge/retrieval/arxiv.py — Canonical arXiv Preprint Retriever.

RF-1.0.0-alpha.2.1: Canonical adapter for arXiv API.
Returns typed List[RetrievedItem].
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List
import requests

from .literature import LiteratureRetriever, RetrievedItem

logger = logging.getLogger(__name__)
ARXIV_BASE = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivRetriever:
    """Canonical arXiv API retriever implementing LiteratureRetriever protocol."""

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> List[RetrievedItem]:
        params = {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": max_results,
        }
        try:
            resp = requests.get(ARXIV_BASE, params=params, timeout=self.timeout)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            items: List[RetrievedItem] = []
            for entry in root.findall("atom:entry", NS):
                title_el = entry.find("atom:title", NS)
                summary_el = entry.find("atom:summary", NS)
                id_el = entry.find("atom:id", NS)
                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ""
                url = id_el.text.strip() if id_el is not None and id_el.text else ""
                items.append(RetrievedItem(
                    source="arxiv",
                    title=title,
                    url=url,
                    summary=summary[:200],
                    score=1.0,
                ))
            return items
        except Exception as exc:
            logger.warning("arXiv query failed: %s", exc)
            return []
