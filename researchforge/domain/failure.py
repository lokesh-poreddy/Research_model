from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from .base import DomainObject


@dataclass(frozen=True)
class Failure(DomainObject):
    experiment_id: str | None
    failure_category: str | None = None
    symptoms: Dict[str, Any] | None = None
    measured_conditions: Dict[str, Any] | None = None
    probable_causes: List[str] | None = None
    confidence: float | None = None
    recovery_attempts: List[Dict[str, Any]] | None = None
    recovery_results: List[Dict[str, Any]] | None = None
    transferable_lesson: str | None = None
    forbidden_future_actions: List[str] | None = None
    provenance_id: str | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Failure":
        return cls(**obj)
