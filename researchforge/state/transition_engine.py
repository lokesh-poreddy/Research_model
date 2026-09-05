from __future__ import annotations

from dataclasses import replace
from typing import List
from researchforge.domain.state import ResearchState
from researchforge.state.events import Event, EventType
from researchforge.domain.provenance import Provenance
from researchforge.state.validators import validate_event_for_state, validate_phase_transition


class ResearchStateTransitionEngine:
    """Deterministic transition engine producing new ResearchState snapshots.

    The engine does not mutate existing states; it returns a new ResearchState
    instance for each transition. Each transition must include provenance.
    """

    def __init__(self) -> None:
        pass

    def transition(self, state: ResearchState, event: Event, provenance: Provenance | None = None) -> ResearchState:
        # Validate event payload
        validate_event_for_state(state, EventType(event.event_type), event.payload)

        # Start from previous state's dict and produce a new state depending on event
        sdict = state.to_dict()
        next_state = dict(sdict)

        # Ensure we don't lose identity fields
        # Update recent_experiment_refs, evidence, failures as appropriate
        et = EventType(event.event_type)
        if et == EventType.DECISION_MADE:
            next_state["current_decision_id"] = event.payload.get("decision_id")
            next_state.setdefault("recent_experiment_refs", [])
        elif et == EventType.EXPERIMENT_PLANNED:
            spec_id = event.payload.get("spec_id")
            refs = list(next_state.get("recent_experiment_refs") or [])
            refs.append(spec_id)
            next_state["recent_experiment_refs"] = refs
            # advance phase if appropriate
            try:
                validate_phase_transition(next_state.get("research_phase"), "EXPERIMENT_PLANNING")
                next_state["research_phase"] = "EXPERIMENT_PLANNING"
            except Exception:
                pass
        elif et == EventType.EXPERIMENT_COMPLETED:
            run_id = event.payload.get("run_id")
            refs = list(next_state.get("recent_experiment_refs") or [])
            refs.append(run_id)
            next_state["recent_experiment_refs"] = refs
            try:
                validate_phase_transition(next_state.get("research_phase"), "EVALUATION")
                next_state["research_phase"] = "EVALUATION"
            except Exception:
                pass
        elif et == EventType.OUTCOME_RECORDED:
            outcome_id = event.payload.get("outcome_id")
            # attach to best_known_result for quick reference
            next_state["best_known_result"] = {"outcome_id": outcome_id}
        elif et == EventType.FAILURE_RECORDED:
            fid = event.payload.get("failure_id")
            failures = list(next_state.get("recent_failures") or [])
            failures.append(fid)
            next_state["recent_failures"] = failures
        elif et == EventType.EVIDENCE_ADDED:
            evid = event.payload.get("evidence_id")
            evids = list(next_state.get("recent_evidence_refs") or [])
            evids.append(evid)
            next_state["recent_evidence_refs"] = evids
        elif et == EventType.RESEARCH_INITIALIZED:
            next_state["research_phase"] = "INITIALIZED"

        # set provenance
        if provenance is not None:
            next_state["provenance_id"] = provenance.id

        # produce a new ResearchState keeping id uniqueness by deriving new id
        new_id = next_state.get("id")
        # derive new id as provided by caller or append transition marker
        if new_id is None:
            new_id = state.id + "-v1"

        # Create new ResearchState instance via from_dict to preserve types
        new_state = ResearchState.from_dict(dict({"id": new_id, **next_state}))
        return new_state


def apply_events(initial_state: ResearchState, events: List[Event], engine: ResearchStateTransitionEngine | None = None) -> ResearchState:
    engine = engine or ResearchStateTransitionEngine()
    state = initial_state
    for ev in events:
        state = engine.transition(state, ev)
    return state
