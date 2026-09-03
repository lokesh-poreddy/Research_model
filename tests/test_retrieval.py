"""tests/test_retrieval.py — Retrieval & Evidence Pipeline test suite.

Classification: CORE
Tests Phase 7:
  - Canonical literature adapters (GitHub, OpenAlex, Semantic Scholar, arXiv)
  - retrieve_all() fan-out
  - EvidenceCandidate creation and validation
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from researchforge.retrieval import (
    RetrievedItem,
    GitHubRepositoryRetriever,
    OpenAlexRetriever,
    SemanticScholarRetriever,
    ArxivRetriever,
    retrieve_all,
)
from researchforge.evidence import EvidenceCandidate


def test_retrieved_item_attributes():
    item = RetrievedItem(
        source="arxiv",
        title="Sample Paper",
        url="http://arxiv.org/abs/1234",
        summary="Summary text",
        score=0.95,
    )
    assert item.source == "arxiv"
    assert item.title == "Sample Paper"
    assert item.score == 0.95


def test_github_retriever_mock():
    import json
    from unittest.mock import MagicMock, patch
    fake_data = json.dumps({
        "items": [{
            "full_name": "org/repo",
            "html_url": "https://github.com/org/repo",
            "description": "repo description",
            "stargazers_count": 150,
        }]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_data
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        retriever = GitHubRepositoryRetriever()
        items = retriever.search("optimizer", max_results=2)
        assert len(items) == 1
        assert items[0].source == "github"
        assert items[0].title == "org/repo"
        assert items[0].score == 150.0


def test_openalex_retriever_mock():
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{
                "title": "OpenAlex Paper",
                "doi": "https://doi.org/10.1000/182",
                "publication_year": 2024,
            }]
        }
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        retriever = OpenAlexRetriever()
        items = retriever.search("neural networks", max_results=3)
        assert len(items) == 1
        assert items[0].source == "openalex"
        assert items[0].title == "OpenAlex Paper"
        assert items[0].url == "https://doi.org/10.1000/182"


def test_openalex_error_handling():
    with patch("requests.get", side_effect=Exception("Network error")):
        retriever = OpenAlexRetriever()
        items = retriever.search("test")
        assert items == []


def test_semantic_scholar_retriever_mock():
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{
                "title": "Semantic Paper",
                "year": 2023,
                "abstract": "Abstract of paper",
                "url": "https://www.semanticscholar.org/paper/123",
            }]
        }
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        retriever = SemanticScholarRetriever()
        items = retriever.search("graph learning", max_results=2)
        assert len(items) == 1
        assert items[0].source == "semantic_scholar"
        assert items[0].title == "Semantic Paper"


def test_semantic_scholar_error_handling():
    with patch("requests.get", side_effect=Exception("Timeout")):
        retriever = SemanticScholarRetriever()
        items = retriever.search("test")
        assert items == []


def test_arxiv_retriever_mock():
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2301.00001v1</id>
        <title>ArXiv Title</title>
        <summary>ArXiv Summary</summary>
      </entry>
    </feed>
    """
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.content = xml_content.encode("utf-8")
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        retriever = ArxivRetriever()
        items = retriever.search("quantum", max_results=2)
        assert len(items) == 1
        assert items[0].source == "arxiv"
        assert items[0].title == "ArXiv Title"
        assert items[0].summary == "ArXiv Summary"


def test_arxiv_error_handling():
    with patch("requests.get", side_effect=Exception("HTTP 500")):
        retriever = ArxivRetriever()
        items = retriever.search("test")
        assert items == []


def test_retrieve_all_fan_out():
    fake_item = RetrievedItem(source="mock", title="Mock Title", url="http://mock", summary="Mock")
    with patch.object(GitHubRepositoryRetriever, "search", return_value=[fake_item]), \
         patch.object(OpenAlexRetriever, "search", return_value=[fake_item]), \
         patch.object(SemanticScholarRetriever, "search", return_value=[fake_item]), \
         patch.object(ArxivRetriever, "search", return_value=[fake_item]):

        candidates = retrieve_all("reinforcement learning", sources=["github", "arxiv"])
        assert len(candidates) == 2
        sources = {c.source for c in candidates}
        assert sources == {"github", "arxiv"}


def test_retrieve_all_produces_valid_evidence_candidates():
    fake_item = RetrievedItem(source="openalex", title="Test Work", url="http://doi", summary="Summary", score=0.85)
    with patch.object(OpenAlexRetriever, "search", return_value=[fake_item]):
        candidates = retrieve_all("test query", sources=["openalex"])
        assert len(candidates) == 1
        cand = candidates[0]
        assert isinstance(cand, EvidenceCandidate)
        cand.validate()  # must pass strict JSON schema validation
        assert cand.source == "openalex"
        assert cand.retrieval_query == "test query"
