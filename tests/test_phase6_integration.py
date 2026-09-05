import pytest

from researchforge.domain.state import ResearchState
from researchforge.domain.provenance import Provenance
from researchforge.state.events import Event, EventType
from researchforge.integrator import ResearchHistoryService, IntegrationReport
from researchforge.vrdeg.queries import get_research_trajectory, get_experiment_lineage, get_failure_history, get_state_history, get_related_evidence


def make_prov(pid):
    return Provenance(id=pid, schema_version="1", created_by="tester", created_at="now")


def make_event(eid, etype, payload=None, prov=None):
    return Event.create(id=eid, schema_version="1", event_type=etype, payload=payload or {}, timestamp=None, provenance_id=(prov.id if prov else None))


def test_end_to_end_reconstruction_and_idempotency():
    svc = ResearchHistoryService()
    init = ResearchState(id="s0", schema_version="1")
    pv = make_prov("pv1")
    # deterministic representative trajectory
    evs = [
        make_event("e1", EventType.QUESTION_SELECTED, {"question_id": "q1", "problem_id": "p1"}, pv),
        make_event("e2", EventType.HYPOTHESIS_PROPOSED, {"hypothesis_id": "h1", "question_id": "q1"}, pv),
        make_event("e3", EventType.DECISION_MADE, {"decision_id": "d1", "hypothesis_id": "h1"}, pv),
        make_event("e4", EventType.EXPERIMENT_PLANNED, {"spec_id": "spec1", "decision_id": "d1"}, pv),
        make_event("e5", EventType.EXPERIMENT_STARTED, {"spec_id": "spec1", "run_id": "run1"}, pv),
        make_event("e6", EventType.OUTCOME_RECORDED, {"run_id": "run1", "outcome_id": "out1"}, pv),
        make_event("e7", EventType.VALIDITY_ASSESSED, {"outcome_id": "out1", "verdict": "VALID"}, pv),
        make_event("e8", EventType.DIAGNOSIS_RECORDED, {"diagnosis_id": "diag1", "outcome_id": "out1"}, pv),
        make_event("e9", EventType.DECISION_MADE, {"decision_id": "d2"}, pv),
    ]

    prov_map = {pv.id: pv}
    final_state_1 = svc.apply_events(init, evs, provenance_map=prov_map)
    graph1 = svc.graph
    fp_state_1 = final_state_1.fingerprint()
    fp_graph_1 = graph1.fingerprint()

    # reconstruct cleanly
    final_state_2, graph2 = svc.reconstruct(init, evs, provenance_map=prov_map)
    fp_state_2 = final_state_2.fingerprint()
    fp_graph_2 = graph2.fingerprint()

    assert fp_state_1 == fp_state_2
    assert fp_graph_1 == fp_graph_2

    # idempotent projection: projecting same events again on same graph doesn't duplicate
    svc.project_events(evs)
    assert fp_graph_1 == graph1.fingerprint()

    # queries return expected nodes
    traj = get_research_trajectory(graph1, "q1")
    assert any(n.id == "p1" for n in traj.nodes)
    lin = get_experiment_lineage(graph1, "spec1")
    assert lin.spec.id == "spec1"


def test_branching_and_negative_retention_and_provenance():
    svc = ResearchHistoryService()
    init = ResearchState(id="sB", schema_version="1")
    pv = make_prov("pvB")
    evs = [
        make_event("b1", EventType.HYPOTHESIS_PROPOSED, {"hypothesis_id": "hb1"}, pv),
        make_event("b2", EventType.EXPERIMENT_PLANNED, {"spec_id": "specA"}, pv),
        make_event("b3", EventType.EXPERIMENT_PLANNED, {"spec_id": "specB"}, pv),
        make_event("b4", EventType.EXPERIMENT_STARTED, {"spec_id": "specA", "run_id": "runA"}, pv),
        make_event("b5", EventType.OUTCOME_RECORDED, {"run_id": "runA", "outcome_id": "outA"}, pv),
        make_event("b6", EventType.EXPERIMENT_STARTED, {"spec_id": "specB", "run_id": "runB"}, pv),
        make_event("b7", EventType.FAILURE_RECORDED, {"run_id": "runB", "failure_id": "fB"}, pv),
    ]
    svc.apply_events(init, evs, provenance_map={pv.id: pv})
    g = svc.graph

    # both branches and negative result exist
    linA = get_experiment_lineage(g, "specA")
    linB = get_experiment_lineage(g, "specB")
    assert any(n.id == "runA" for n in linA.nodes)
    assert any(n.id == "fB" for n in linB.nodes) or any(n.id == "fB" for n in get_failure_history(g, "fB").related_runs)

    # provenance preserved
    assert g.get_node("pvB") is not None


def test_consistency_validator_detects_corruption():
    svc = ResearchHistoryService()
    init = ResearchState(id="sC", schema_version="1")
    pv = make_prov("pvC")
    evs = [
        make_event("c1", EventType.EXPERIMENT_PLANNED, {"spec_id": "specC"}, pv),
        make_event("c2", EventType.EXPERIMENT_STARTED, {"spec_id": "specC", "run_id": "runC"}, pv),
    ]
    svc.apply_events(init, evs, provenance_map={pv.id: pv})
    # validate consistency should be good
    rep = svc.validate_consistency()
    assert rep.consistency is True

    # simulate corruption: remove an expected edge
    # expected edge id from mapping of c2
    edge_id = f"edge:c2:EXECUTED_AS:specC->runC"
    if edge_id in svc.graph._edges:
        svc.graph._edges.pop(edge_id)
    rep2 = svc.validate_consistency()
    assert rep2.consistency is False
