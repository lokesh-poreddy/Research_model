from __future__ import annotations

from typing import Dict, Any
from researchforge.state.events import EventType
from researchforge.domain.state import ResearchState


# Minimal legal phase graph. Allows branching by permitting multiple next phases.
LEGAL_PHASE_TRANSITIONS = {
    None: ["INITIALIZED", "QUESTIONING"],
    "INITIALIZED": ["QUESTIONING"],
    "QUESTIONING": ["HYPOTHESIS_FORMATION", "QUESTIONING"],
    "HYPOTHESIS_FORMATION": ["EXPERIMENT_PLANNING", "HYPOTHESIS_FORMATION"],
    "EXPERIMENT_PLANNING": ["EXPERIMENT_EXECUTION", "EXPERIMENT_PLANNING"],
    "EXPERIMENT_EXECUTION": ["EVALUATION", "EXPERIMENT_EXECUTION"],
    "EVALUATION": ["DIAGNOSIS", "DECISION"],
    "DIAGNOSIS": ["DECISION"],
    "DECISION": ["QUESTIONING", "HYPOTHESIS_FORMATION", "BRANCH"],
}


def validate_phase_transition(current_phase: str | None, next_phase: str) -> None:
    allowed = LEGAL_PHASE_TRANSITIONS.get(current_phase, [])
    if next_phase not in allowed:
        raise ValueError(f"Illegal phase transition from {current_phase} to {next_phase}")


def validate_event_for_state(state: ResearchState, event_type: EventType, payload: Dict[str, Any] | None) -> None:
    # Basic validation: ensure required references exist in payload for certain events
    if event_type == EventType.DECISION_MADE:
        if not payload or "decision_id" not in payload:
            raise ValueError("DECISION_MADE requires decision_id in payload")
    if event_type == EventType.EXPERIMENT_PLANNED:
        if not payload or "spec_id" not in payload:
            raise ValueError("EXPERIMENT_PLANNED requires spec_id in payload")
    if event_type == EventType.EXPERIMENT_COMPLETED:
        if not payload or "run_id" not in payload:
            raise ValueError("EXPERIMENT_COMPLETED requires run_id in payload")
    if event_type == EventType.OUTCOME_RECORDED:
        if not payload or "outcome_id" not in payload:
            raise ValueError("OUTCOME_RECORDED requires outcome_id in payload")
    # other validations can be added as needed
