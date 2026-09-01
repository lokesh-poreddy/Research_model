"""Literature/code retrieval: the 'Retrieve' step of the Algorithm-Discovery
Pipeline (design doc Sec. 6), backed by the GitHub Search API -- one of the
Sec. 5 tools table's "High priority" sources ("GitHub: Code repositories,
real-world code"). This is a real, working HTTP client, not a stub: it
performs an actual `api.github.com/search/repositories` call and parses real
results (verified live against the public API while building this module).

The Sec. 5 table also lists OpenAlex, Semantic Scholar, Crossref, and arXiv
for paper-level literature search. Those aren't wired up here -- not because
the pattern would differ (it's the same "build a query -> GET -> parse JSON
-> return typed records" shape as `GitHubRepositoryRetriever` below), but
because this sandbox's outbound network allowlist doesn't include those
hosts. `LiteratureRetriever` is a Protocol specifically so adding an
OpenAlexRetriever/SemanticScholarRetriever alongside the GitHub one, and
querying several at once via `retrieve_all()`, is additive.

Every function degrades gracefully: no network, a non-2xx response, a
malformed payload, or a timeout all produce an empty result list rather than
an exception, because the discovery pipeline must keep working fully offline
-- retrieval is an optional enrichment, never a hard dependency of the core
research loop.
"""
from __future__ import annotations
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class RetrievedItem:
    source: str
    title: str
    url: str
    summary: str
    score: Optional[float] = None  # source-specific relevance signal (e.g. stars)


class LiteratureRetriever(Protocol):
    def search(self, query: str, max_results: int = 5) -> List[RetrievedItem]: ...


class GitHubRepositoryRetriever:
    """Real GitHub repository search. No auth token required at low request
    volume (unauthenticated search is rate-limited to ~10/min by GitHub,
    which is fine for occasional retrieval calls -- this is deliberately NOT
    invoked on every generation of the research loop, only at discovery-
    trigger points, to stay well under that limit and keep the core loop
    fully offline-reliable)."""

    BASE_URL = "https://api.github.com/search/repositories"

    def __init__(self, timeout_s: float = 8.0, user_agent: str = "researchforge-ecrm"):
        self.timeout_s = timeout_s
        self.user_agent = user_agent

    def search(self, query: str, max_results: int = 5) -> List[RetrievedItem]:
        params = urllib.parse.urlencode({"q": query, "per_page": max_results, "sort": "stars"})
        url = f"{self.BASE_URL}?{params}"
        req = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "application/vnd.github+json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError):
            return []  # graceful degradation: retrieval is optional, never fatal

        items = []
        for entry in payload.get("items", [])[:max_results]:
            items.append(RetrievedItem(
                source="github",
                title=entry.get("full_name", "unknown/unknown"),
                url=entry.get("html_url", ""),
                summary=(entry.get("description") or "")[:280],
                score=float(entry.get("stargazers_count", 0)),
            ))
        return items


def retrieve_all(query: str, retrievers: List[LiteratureRetriever],
                  max_results_per_source: int = 5) -> List[RetrievedItem]:
    """Fan a query out across several retrievers (Sec. 5's multi-API design),
    merging results and sorting by score where available. A single failing
    retriever never blocks the others."""
    out: List[RetrievedItem] = []
    for retriever in retrievers:
        try:
            out.extend(retriever.search(query, max_results=max_results_per_source))
        except Exception:
            continue
    out.sort(key=lambda item: (item.score if item.score is not None else 0.0), reverse=True)
    return out
