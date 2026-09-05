from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from .base import DomainObject


@dataclass(frozen=True)
class Diagnosis(DomainObject):
    outcome_id: str
    interpretation: str | None = None
    probable_causes: List[str] | None = None
    confidence: float | None = None
    recovery_attempts: List[Dict[str, Any]] | None = None
    recovery_results: List[Dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Diagnosis":
        return cls(**obj)
