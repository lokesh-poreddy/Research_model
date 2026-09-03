"""Retrieval package — Algorithm-Discovery Pipeline literature and code retrieval.

RF-1.0.0-alpha.2.1:
  - Canonical retrievers: GitHubRepositoryRetriever, OpenAlexRetriever,
    SemanticScholarRetriever, ArxivRetriever
  - Fan-out utility: retrieve_all() returning EvidenceCandidate instances
"""
from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional

from .literature import RetrievedItem, LiteratureRetriever, GitHubRepositoryRetriever
from .openalex import OpenAlexRetriever
from .semantic_scholar import SemanticScholarRetriever
from .arxiv import ArxivRetriever
from ..evidence import EvidenceCandidate


def retrieve_all(
    query: str,
    sources: Optional[List[str]] = None,
    max_results_per_source: int = 5,
    timeout: float = 10.0,
) -> List[EvidenceCandidate]:
    """Query multiple literature/code sources and package results into EvidenceCandidates."""
    active_sources = sources or ["github", "openalex", "semantic_scholar", "arxiv"]
    retrievers = {
        "github": GitHubRepositoryRetriever(),
        "openalex": OpenAlexRetriever(timeout=timeout),
        "semantic_scholar": SemanticScholarRetriever(timeout=timeout),
        "arxiv": ArxivRetriever(timeout=timeout),
    }

    candidates: List[EvidenceCandidate] = []
    now = time.time()

    for src in active_sources:
        retriever = retrievers.get(src)
        if not retriever:
            continue
        try:
            items = retriever.search(query, max_results=max_results_per_source)
            for item in items:
                cand_id = f"cand_{src}_{hashlib.sha256(f'{item.title}:{item.url}'.encode()).hexdigest()[:12]}"
                score = item.score if item.score is not None else 0.5
                candidates.append(EvidenceCandidate(
                    candidate_id=cand_id,
                    source=src,
                    relevance_score=float(score),
                    retrieval_query=query,
                    retrieval_timestamp=now,
                    retrieved_item={
                        "title": item.title,
                        "url": item.url,
                        "summary": item.summary,
                    },
                ))
        except Exception:
            # Source failures are caught to ensure robust fan-out
            continue

    return candidates


__all__ = [
    "RetrievedItem",
    "LiteratureRetriever",
    "GitHubRepositoryRetriever",
    "OpenAlexRetriever",
    "SemanticScholarRetriever",
    "ArxivRetriever",
    "retrieve_all",
]
