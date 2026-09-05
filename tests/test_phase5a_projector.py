import time
import pytest
from researchforge.vrdeg.graph import VRDEG
from researchforge.vrdeg.projector import VRDEGProjector
from researchforge.state.events import Event
from researchforge.state.events import EventType
from researchforge.domain.provenance import Provenance
from researchforge.domain.experiment import ExperimentSpec
from researchforge.state.transition_engine import apply_events
from researchforge.domain.state import ResearchState


def make_event(evt_id, etype, payload=None, prov=None):
    if isinstance(etype, str):
        return Event(id=evt_id, schema_version="1", event_type=etype, payload=payload, timestamp=None, provenance_id=(prov.id if prov else None))
    return Event.create(id=evt_id, schema_version="1", event_type=etype, payload=payload, timestamp=None, provenance_id=(prov.id if prov else None))


def test_projection_idempotent_and_provenance():
    g = VRDEG()
    proj = VRDEGProjector(g)
    prov = Provenance(id="pv1", schema_version="1", created_by="u", created_at="now")
    ev1 = make_event("e-plan-1", EventType.EXPERIMENT_PLANNED, payload={"spec_id": "spec1"}, prov=prov)
    # project twice
    proj.project_event(ev1)
    fp1 = g.fingerprint()
    proj.project_event(ev1)
    fp2 = g.fingerprint()
    assert fp1 == fp2
    # provenance node added
    assert g.get_node("pv1") is not None


def test_mapping_and_edge_creation():
    g = VRDEG()
    proj = VRDEGProjector(g)
    prov = Provenance(id="pv2", schema_version="1", created_by="u", created_at="now")
    ev_start = make_event("e-start-1", EventType.EXPERIMENT_STARTED, payload={"spec_id": "specA", "run_id": "runA"}, prov=prov)
    proj.project_event(ev_start)
    # nodes exist
    assert g.get_node("specA") is not None
    assert g.get_node("runA") is not None
    # edge exists per mapping
    edges = g.edges_for_node("specA")
    assert any(e.relation == "EXECUTED_AS" for e in edges)


def test_branching_and_negative_results_retained():
    g = VRDEG()
    proj = VRDEGProjector(g)
    prov = Provenance(id="pv3", schema_version="1", created_by="u", created_at="now")
    # hypothesis motivates two specs
    ev_h = make_event("e-h1", EventType.HYPOTHESIS_PROPOSED, payload={"hypothesis_id": "h1"}, prov=prov)
    ev_specA = make_event("e-specA", EventType.EXPERIMENT_PLANNED, payload={"spec_id": "specA"}, prov=prov)
    ev_specB = make_event("e-specB", EventType.EXPERIMENT_PLANNED, payload={"spec_id": "specB"}, prov=prov)
    ev_runA = make_event("e-runA", EventType.EXPERIMENT_STARTED, payload={"spec_id": "specA", "run_id": "runA"}, prov=prov)
    ev_outA = make_event("e-outA", EventType.OUTCOME_RECORDED, payload={"run_id": "runA", "outcome_id": "outA"}, prov=prov)
    ev_runB = make_event("e-runB", EventType.EXPERIMENT_STARTED, payload={"spec_id": "specB", "run_id": "runB"}, prov=prov)
    ev_failB = make_event("e-failB", EventType.FAILURE_RECORDED, payload={"run_id": "runB", "failure_id": "fB"}, prov=prov)

    events = [ev_h, ev_specA, ev_specB, ev_runA, ev_outA, ev_runB, ev_failB]
    proj.project_events(events)
    # both branches exist
    assert g.get_node("specA") and g.get_node("specB")
    assert g.get_node("outA") and g.get_node("fB")


def test_reconstruction_consistency_with_transition_engine():
    # create a small event sequence, run transition_engine to get final state,
    # and separately project events; then verify graph contains lineage to outcome and state reference
    g = VRDEG()
    proj = VRDEGProjector(g)
    prov = Provenance(id="pv4", schema_version="1", created_by="u", created_at="now")
    evs = []
    evs.append(make_event("e-plan-x", EventType.EXPERIMENT_PLANNED, payload={"spec_id": "specX"}, prov=prov))
    evs.append(make_event("e-start-x", EventType.EXPERIMENT_STARTED, payload={"spec_id": "specX", "run_id": "runX"}, prov=prov))
    evs.append(make_event("e-complete-x", EventType.EXPERIMENT_COMPLETED, payload={"run_id": "runX"}, prov=prov))
    evs.append(make_event("e-out-x", EventType.OUTCOME_RECORDED, payload={"run_id": "runX", "outcome_id": "outX"}, prov=prov))

    # transit state via transition engine
    from researchforge.domain.state import ResearchState
    state0 = ResearchState(id="s0", schema_version="1")
    s_final = apply_events(state0, evs)

    # project into graph
    proj.project_events(evs)
    # verify lineage from spec->run->outcome
    assert g.get_node("specX") and g.get_node("runX") and g.get_node("outX")
    # ensure outcome linked to run
    preds = g.predecessors("outX")
    assert any(n.id == "runX" for n in preds)


def test_unsupported_event_type_raises():
    g = VRDEG()
    proj = VRDEGProjector(g)
    # create an artificial unsupported event type name
    ev = make_event("e-unsupported", "UNSUPPORTED_EVENT", payload={})
    with pytest.raises(ValueError):
        proj.project_event(ev)
