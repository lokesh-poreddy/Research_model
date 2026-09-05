from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
from .base import DomainObject


@dataclass(frozen=True)
class Evidence(DomainObject):
    source: str
    source_id: str | None = None
    retrieval_timestamp: str | None = None
    retriever: str | None = None
    source_url: str | None = None
    text_location: str | None = None
    quality_indicators: Dict[str, Any] | None = None
    claim_relationship: str | None = None
    confidence: float | None = None
    snippet: str | None = None
    metadata: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Evidence":
        return cls(**obj)
