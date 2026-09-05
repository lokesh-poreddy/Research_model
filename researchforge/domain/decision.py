from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from .base import DomainObject


@dataclass(frozen=True)
class Decision(DomainObject):
    research_state_fingerprint: str | None
    rsg_id: str | None
    hypothesis_id: str | None
    selected_tmg_id: str | None
    selected_operator: str | None
    decision_reason: str | None = None
    evidence_refs: List[str] | None = None
    memory_refs: List[str] | None = None
    expected_information_gain: float | None = None
    expected_performance_gain: float | None = None
    estimated_cost: float | None = None
    estimated_failure_risk: float | None = None
    exploration_score: float | None = None
    novelty_score: float | None = None
    confidence: float | None = None
    decision_timestamp: str | None = None
    policy_version: str | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Decision":
        return cls(**obj)
