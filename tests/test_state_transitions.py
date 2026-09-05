import pytest
from researchforge.domain.state import ResearchState
from researchforge.state.events import Event, EventType
from researchforge.state.transition_engine import ResearchStateTransitionEngine, apply_events
from researchforge.domain.provenance import Provenance


def make_initial_state():
    return ResearchState(id="state0", schema_version="1", problem_id="prob0", research_phase=None)


def test_initialization_and_simple_transition():
    state0 = make_initial_state()
    engine = ResearchStateTransitionEngine()
    prov = Provenance(id="p0", schema_version="1", created_by="tester", created_at="2026-09-05T00:00:00Z")
    ev = Event.create(id="e1", schema_version="1", event_type=EventType.RESEARCH_INITIALIZED, payload=None, timestamp="2026-09-05T00:00:01Z", provenance_id=prov.id)
    s1 = engine.transition(state0, ev, provenance=prov)
    assert s1.research_phase == "INITIALIZED"
    assert s1.provenance_id == prov.id


def test_experiment_planned_and_completed_and_outcome():
    state0 = make_initial_state()
    engine = ResearchStateTransitionEngine()
    ev_plan = Event.create(id="e2", schema_version="1", event_type=EventType.EXPERIMENT_PLANNED, payload={"spec_id": "spec1"}, timestamp="2026-09-05T00:01:00Z")
    s1 = engine.transition(state0, ev_plan)
    assert s1.recent_experiment_refs and "spec1" in s1.recent_experiment_refs
    ev_completed = Event.create(id="e3", schema_version="1", event_type=EventType.EXPERIMENT_COMPLETED, payload={"run_id": "run1"})
    s2 = engine.transition(s1, ev_completed)
    assert "run1" in (s2.recent_experiment_refs or [])
    ev_outcome = Event.create(id="e4", schema_version="1", event_type=EventType.OUTCOME_RECORDED, payload={"outcome_id": "out1"})
    s3 = engine.transition(s2, ev_outcome)
    assert s3.best_known_result and s3.best_known_result.get("outcome_id") == "out1"


def test_failure_and_diagnosis_handling():
    state0 = make_initial_state()
    engine = ResearchStateTransitionEngine()
    ev_fail = Event.create(id="ef1", schema_version="1", event_type=EventType.FAILURE_RECORDED, payload={"failure_id": "f1"})
    s1 = engine.transition(state0, ev_fail)
    assert "f1" in (s1.recent_failures or [])


def test_apply_events_reconstruction_determinism():
    state0 = make_initial_state()
    events = [
        Event.create(id="e_init", schema_version="1", event_type=EventType.RESEARCH_INITIALIZED),
        Event.create(id="e_plan", schema_version="1", event_type=EventType.EXPERIMENT_PLANNED, payload={"spec_id": "sA"}),
        Event.create(id="e_comp", schema_version="1", event_type=EventType.EXPERIMENT_COMPLETED, payload={"run_id": "rA"}),
    ]
    s_final = apply_events(state0, events)
    s_final2 = apply_events(state0, events)
    assert s_final.fingerprint() == s_final2.fingerprint()


def test_invalid_event_payload_rejected():
    state0 = make_initial_state()
    engine = ResearchStateTransitionEngine()
    bad = Event.create(id="bad1", schema_version="1", event_type=EventType.EXPERIMENT_PLANNED, payload={})
    with pytest.raises(ValueError):
        engine.transition(state0, bad)
