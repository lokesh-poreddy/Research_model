"""researchforge/evidence/evidence.py — Evidence and EvidenceCandidate domain objects.

RF-1.0.0-alpha.2.1: Formal separation between raw retrieved hits (EvidenceCandidate)
and adjudicated scientific evidence (Evidence).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from ..genome.schema import validate_genome

EVIDENCE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Evidence",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "evidence_id", "schema_version", "source", "title", "content",
        "relevance_score"
    ],
    "properties": {
        "evidence_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "source": {"type": "string"},
        "title": {"type": "string"},
        "content": {"type": "string"},
        "relevance_score": {"type": "number"},
        "candidate_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "citation_key": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "doi_or_url": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}

EVIDENCE_CANDIDATE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "EvidenceCandidate",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidate_id", "schema_version", "source", "relevance_score",
        "retrieval_query", "retrieval_timestamp", "retrieved_item"
    ],
    "properties": {
        "candidate_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "source": {"type": "string"},
        "relevance_score": {"type": "number"},
        "retrieval_query": {"type": "string"},
        "retrieval_timestamp": {"type": "number"},
        "retrieved_item": {"type": "object"},
        "evidence_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "adjudicated_at": {"oneOf": [{"type": "null"}, {"type": "number"}]},
        "adjudication_verdict": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}


@dataclass
class Evidence:
    """Adjudicated, fingerprinted, citable scientific evidence. Class A immutable."""
    evidence_id: str
    source: str
    title: str
    content: str
    relevance_score: float
    candidate_id: Optional[str] = None
    citation_key: Optional[str] = None
    doi_or_url: Optional[str] = None
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "schema_version": self.schema_version,
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "candidate_id": self.candidate_id,
            "citation_key": self.citation_key,
            "doi_or_url": self.doi_or_url,
            "researchforge_version": self.researchforge_version,
            "created_at": self.created_at,
        }

    def canonical_dict(self) -> Dict[str, Any]:
        d = self.to_dict()
        d.pop("created_at", None)
        return d

    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        validate_genome(self.to_dict(), EVIDENCE_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Evidence":
        return cls(
            evidence_id=d["evidence_id"],
            schema_version=d.get("schema_version", "1.0"),
            source=d["source"],
            title=d["title"],
            content=d["content"],
            relevance_score=float(d["relevance_score"]),
            candidate_id=d.get("candidate_id"),
            citation_key=d.get("citation_key"),
            doi_or_url=d.get("doi_or_url"),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )


@dataclass
class EvidenceCandidate:
    """A search hit that has NOT yet been adjudicated as Evidence."""
    candidate_id: str
    source: str
    relevance_score: float
    retrieval_query: str
    retrieval_timestamp: float
    retrieved_item: Dict[str, Any]
    evidence_id: Optional[str] = None
    adjudicated_at: Optional[float] = None
    adjudication_verdict: Optional[str] = None  # e.g. "accepted" | "rejected" | "duplicate"
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "schema_version": self.schema_version,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "retrieval_query": self.retrieval_query,
            "retrieval_timestamp": self.retrieval_timestamp,
            "retrieved_item": dict(self.retrieved_item),
            "evidence_id": self.evidence_id,
            "adjudicated_at": self.adjudicated_at,
            "adjudication_verdict": self.adjudication_verdict,
            "researchforge_version": self.researchforge_version,
            "created_at": self.created_at,
        }

    def canonical_dict(self) -> Dict[str, Any]:
        d = self.to_dict()
        d.pop("created_at", None)
        return d

    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        validate_genome(self.to_dict(), EVIDENCE_CANDIDATE_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceCandidate":
        return cls(
            candidate_id=d["candidate_id"],
            schema_version=d.get("schema_version", "1.0"),
            source=d["source"],
            relevance_score=float(d["relevance_score"]),
            retrieval_query=d["retrieval_query"],
            retrieval_timestamp=float(d["retrieval_timestamp"]),
            retrieved_item=dict(d["retrieved_item"]),
            evidence_id=d.get("evidence_id"),
            adjudicated_at=float(d["adjudicated_at"]) if d.get("adjudicated_at") is not None else None,
            adjudication_verdict=d.get("adjudication_verdict"),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )
