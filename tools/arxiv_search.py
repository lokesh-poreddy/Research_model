"""
arXiv API client.
http://export.arxiv.org/api/query?search_query=all:<query>
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)
ARXIV_BASE = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_papers(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search arXiv for preprints."""
    params = {
        "search_query": f'all:"{query}"',
        "start": 0,
        "max_results": max_results,
    }
    try:
        resp = requests.get(ARXIV_BASE, params=params, timeout=20)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        results = []
        for entry in root.findall("atom:entry", NS):
            title_el = entry.find("atom:title", NS)
            summary_el = entry.find("atom:summary", NS)
            id_el = entry.find("atom:id", NS)
            results.append({
                "title": title_el.text.strip() if title_el is not None else "",
                "abstract": summary_el.text.strip() if summary_el is not None else "",
                "arxiv_id": id_el.text.strip() if id_el is not None else "",
                "source": "arxiv",
            })
        return results
    except Exception as exc:
        logger.warning("arXiv query failed: %s", exc)
        return []
