from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from .base import DomainObject


@dataclass(frozen=True)
class ResearchState(DomainObject):
    # Contract-only representation of a ResearchState snapshot. No transition
    # logic is implemented here in Phase 1.
    problem_id: str | None = None
    active_question_id: str | None = None
    hypotheses: List[str] | None = None
    current_decision_id: str | None = None
    selected_tmg_id: str | None = None
    selected_rsg_id: str | None = None
    recent_experiment_refs: List[str] | None = None
    recent_evidence_refs: List[str] | None = None
    recent_failures: List[str] | None = None
    memory_context: Dict[str, Any] | None = None
    best_known_result: Dict[str, Any] | None = None
    budget_consumed: float | None = None
    budget_remaining: float | None = None
    research_phase: str | None = None
    unresolved_contradictions: List[str] | None = None
    policy_state: Dict[str, Any] | None = None
    provenance_id: str | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ResearchState":
        return cls(**obj)
