"""researchforge/state/research_state.py — ResearchState binding object.

RF-1.0.0-alpha.2.1: Class D canonical binding object.
Binds RSG + TMG + Evidence + Outcomes into a coherent view of the research process
at generation t. This is the object that VRDEG (alpha.3) stores as versioned graph nodes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..genome.schema import validate_genome

RESEARCH_STATE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ResearchState",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "state_id", "schema_version", "generation", "research_phase",
        "active_rsg_id", "candidate_tmg_ids", "best_metric", "budget_remaining"
    ],
    "properties": {
        "state_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "generation": {"type": "integer", "minimum": -1},
        "research_phase": {"type": "string"},
        "problem_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "question_ids": {
            "type": "array",
            "items": {"type": "string"}
        },
        "active_hypothesis_ids": {
            "type": "array",
            "items": {"type": "string"}
        },
        "active_rsg_id": {"type": "string"},
        "active_rsg_fingerprint": {"type": "string"},
        "candidate_tmg_ids": {
            "type": "array",
            "items": {"type": "string"}
        },
        "best_tmg_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "best_metric": {"type": "number"},
        "memory_record_ids": {
            "type": "array",
            "items": {"type": "string"}
        },
        "latest_evidence_ids": {
            "type": "array",
            "items": {"type": "string"}
        },
        "latest_outcome_ids": {
            "type": "array",
            "items": {"type": "string"}
        },
        "unresolved_failure_ids": {
            "type": "array",
            "items": {"type": "string"}
        },
        "budget_remaining": {"type": "integer"},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}


@dataclass
class ResearchState:
    """The complete, fingerprinted state of the research process at generation t."""
    state_id: str
    generation: int
    research_phase: str
    active_rsg_id: str
    candidate_tmg_ids: List[str]
    best_metric: float
    budget_remaining: int
    problem_id: Optional[str] = None
    question_ids: List[str] = field(default_factory=list)
    active_hypothesis_ids: List[str] = field(default_factory=list)
    active_rsg_fingerprint: str = ""
    best_tmg_id: Optional[str] = None
    memory_record_ids: List[str] = field(default_factory=list)
    latest_evidence_ids: List[str] = field(default_factory=list)
    latest_outcome_ids: List[str] = field(default_factory=list)
    unresolved_failure_ids: List[str] = field(default_factory=list)
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "schema_version": self.schema_version,
            "generation": self.generation,
            "research_phase": self.research_phase,
            "problem_id": self.problem_id,
            "question_ids": list(self.question_ids),
            "active_hypothesis_ids": list(self.active_hypothesis_ids),
            "active_rsg_id": self.active_rsg_id,
            "active_rsg_fingerprint": self.active_rsg_fingerprint,
            "candidate_tmg_ids": list(self.candidate_tmg_ids),
            "best_tmg_id": self.best_tmg_id,
            "best_metric": self.best_metric,
            "memory_record_ids": list(self.memory_record_ids),
            "latest_evidence_ids": list(self.latest_evidence_ids),
            "latest_outcome_ids": list(self.latest_outcome_ids),
            "unresolved_failure_ids": list(self.unresolved_failure_ids),
            "budget_remaining": self.budget_remaining,
            "researchforge_version": self.researchforge_version,
            "created_at": self.created_at,
        }

    def canonical_dict(self) -> Dict[str, Any]:
        d = self.to_dict()
        d.pop("created_at", None)
        d.pop("state_id", None)
        return d

    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        validate_genome(self.to_dict(), RESEARCH_STATE_SCHEMA)

    def evolve(self, **updates: Any) -> "ResearchState":
        """Return a new ResearchState with updated fields and a recomputed deterministic state_id."""
        d = copy.deepcopy(self.to_dict())
        d.update(updates)
        d["created_at"] = time.time()
        # Compute new state_id from updated canonical representation
        canon = copy.deepcopy(d)
        canon.pop("created_at", None)
        canon.pop("state_id", None)
        new_hash = hashlib.sha256(
            json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        d["state_id"] = f"state_{new_hash}"
        return self.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchState":
        return cls(
            state_id=d["state_id"],
            schema_version=d.get("schema_version", "1.0"),
            generation=int(d["generation"]),
            research_phase=d["research_phase"],
            problem_id=d.get("problem_id"),
            question_ids=list(d.get("question_ids", [])),
            active_hypothesis_ids=list(d.get("active_hypothesis_ids", [])),
            active_rsg_id=d.get("active_rsg_id", ""),
            active_rsg_fingerprint=d.get("active_rsg_fingerprint", ""),
            candidate_tmg_ids=list(d.get("candidate_tmg_ids", [])),
            best_tmg_id=d.get("best_tmg_id"),
            best_metric=float(d.get("best_metric", 0.0)),
            memory_record_ids=list(d.get("memory_record_ids", [])),
            latest_evidence_ids=list(d.get("latest_evidence_ids", [])),
            latest_outcome_ids=list(d.get("latest_outcome_ids", [])),
            unresolved_failure_ids=list(d.get("unresolved_failure_ids", [])),
            budget_remaining=int(d.get("budget_remaining", 0)),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )

    @classmethod
    def create(
        cls,
        generation: int,
        research_phase: str,
        active_rsg_id: str,
        candidate_tmg_ids: List[str],
        best_metric: float,
        budget_remaining: int,
        problem_id: Optional[str] = None,
        question_ids: Optional[List[str]] = None,
        active_hypothesis_ids: Optional[List[str]] = None,
        active_rsg_fingerprint: str = "",
        best_tmg_id: Optional[str] = None,
        memory_record_ids: Optional[List[str]] = None,
        latest_evidence_ids: Optional[List[str]] = None,
        latest_outcome_ids: Optional[List[str]] = None,
        unresolved_failure_ids: Optional[List[str]] = None,
    ) -> "ResearchState":
        payload = {
            "generation": generation,
            "research_phase": research_phase,
            "active_rsg_id": active_rsg_id,
            "active_rsg_fingerprint": active_rsg_fingerprint,
            "candidate_tmg_ids": list(candidate_tmg_ids),
            "best_tmg_id": best_tmg_id,
            "best_metric": best_metric,
            "budget_remaining": budget_remaining,
            "problem_id": problem_id,
            "question_ids": list(question_ids or []),
            "active_hypothesis_ids": list(active_hypothesis_ids or []),
            "memory_record_ids": list(memory_record_ids or []),
            "latest_evidence_ids": list(latest_evidence_ids or []),
            "latest_outcome_ids": list(latest_outcome_ids or []),
            "unresolved_failure_ids": list(unresolved_failure_ids or []),
        }
        state_id = f"state_{hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]}"
        return cls(
            state_id=state_id,
            generation=generation,
            research_phase=research_phase,
            active_rsg_id=active_rsg_id,
            active_rsg_fingerprint=active_rsg_fingerprint,
            candidate_tmg_ids=list(candidate_tmg_ids),
            best_tmg_id=best_tmg_id,
            best_metric=best_metric,
            budget_remaining=budget_remaining,
            problem_id=problem_id,
            question_ids=list(question_ids or []),
            active_hypothesis_ids=list(active_hypothesis_ids or []),
            memory_record_ids=list(memory_record_ids or []),
            latest_evidence_ids=list(latest_evidence_ids or []),
            latest_outcome_ids=list(latest_outcome_ids or []),
            unresolved_failure_ids=list(unresolved_failure_ids or []),
        )
