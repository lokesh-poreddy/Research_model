import pytest

from researchforge.vrdeg.graph import VRDEG, GraphIntegrityError
from researchforge.vrdeg.node import GraphNode, NodeType
from researchforge.vrdeg.edge import GraphEdge, RelationType
from researchforge.vrdeg.queries import (
    get_research_trajectory,
    get_experiment_lineage,
    get_failure_history,
    get_state_history,
    get_related_evidence,
    TrajectoryRecord,
)


def make_node(nid, ntype, prov=None, parent=None, sv="1"):
    return GraphNode(id=nid, schema_version=sv, node_type=ntype, provenance_id=prov, parent_version_of=parent)


def make_edge(eid, src, tgt, relation, prov=None, sv="1"):
    return GraphEdge(id=eid, schema_version=sv, source_id=src, target_id=tgt, relation=relation, provenance_id=prov)


def test_get_research_trajectory_normal_empty_and_deterministic():
    g = VRDEG()
    nA = make_node("A", NodeType.RESEARCH_PROBLEM.value)
    nB = make_node("B", NodeType.RESEARCH_STATE.value)
    nC = make_node("C", NodeType.EXPERIMENT_SPEC.value)
    g.add_node(nA)
    g.add_node(nB)
    g.add_node(nC)
    e1 = make_edge("e1", "A", "B", RelationType.PRECEDES.value)
    e2 = make_edge("e2", "B", "C", RelationType.PRECEDES.value)
    g.add_edge(e1)
    g.add_edge(e2)

    # normal
    rec = get_research_trajectory(g, "B")
    assert isinstance(rec, TrajectoryRecord)
    assert any(n.id == "A" for n in rec.nodes)
    assert any(n.id == "C" for n in rec.nodes)

    # deterministic ordering: two calls equal
    rec2 = get_research_trajectory(g, "B")
    assert [n.id for n in rec.nodes] == [n.id for n in rec2.nodes]

    # empty result for missing
    rec_empty = get_research_trajectory(g, "missing")
    assert rec_empty.nodes == [] and rec_empty.edges == []


def test_branching_trajectory_and_negative_result_retention():
    g = VRDEG()
    h = make_node("H", NodeType.HYPOTHESIS.value)
    s1 = make_node("S1", NodeType.EXPERIMENT_SPEC.value)
    s2 = make_node("S2", NodeType.EXPERIMENT_SPEC.value)
    r1 = make_node("R1", NodeType.EXPERIMENT_RUN.value)
    f1 = make_node("F1", NodeType.FAILURE.value)
    g.add_node(h)
    g.add_node(s1)
    g.add_node(s2)
    g.add_node(r1)
    g.add_node(f1)
    g.add_edge(make_edge("e-h-s1", "H", "S1", RelationType.MOTIVATED_BY.value))
    g.add_edge(make_edge("e-h-s2", "H", "S2", RelationType.MOTIVATED_BY.value))
    g.add_edge(make_edge("e-s1-r1", "S1", "R1", RelationType.EXECUTED_AS.value))
    g.add_edge(make_edge("e-r1-f1", "R1", "F1", RelationType.FAILED_AS.value))

    rec = get_research_trajectory(g, "H")
    ids = {n.id for n in rec.nodes}
    assert "S1" in ids and "S2" in ids

    # negative result retained: failure is reachable from R1
    rec_r1 = get_research_trajectory(g, "R1")
    assert any(n.id == "F1" for n in rec_r1.nodes)


def test_provenance_references_and_multiple_versions_and_missing_refs():
    g = VRDEG()
    # provenance present
    p = make_node("P", NodeType.PROVENANCE.value)
    spec = make_node("specV1", NodeType.EXPERIMENT_SPEC.value, prov="P")
    g.add_node(p)
    g.add_node(spec)
    # versioned node
    spec2 = make_node("specV2", NodeType.EXPERIMENT_SPEC.value, prov="P", parent="specV1")
    g.add_versioned_node(spec2, previous_id="specV1")
    # link versions explicitly
    g.add_edge(make_edge("e-v", "specV2", "specV1", RelationType.BRANCH_OF.value))

    lin = get_experiment_lineage(g, "specV1")
    # contains both versions
    assert any(n.id == "specV1" for n in lin.nodes)
    assert any(n.id == "specV2" for n in lin.nodes)
    # provenance preserved in node
    found = [n for n in lin.nodes if n.id == "specV1"][0]
    assert found.provenance_id == "P"

    # missing referenced provenance: create node with missing prov
    orphan = make_node("orphan", NodeType.EXPERIMENT_SPEC.value, prov="MISSING")
    g.add_node(orphan)
    # query still returns orphan node
    traj = get_research_trajectory(g, "orphan")
    assert any(n.id == "orphan" for n in traj.nodes)
    # but validate_integrity should raise
    with pytest.raises(GraphIntegrityError):
        g.validate_integrity()


def test_failure_history_and_state_history_and_related_evidence():
    g = VRDEG()
    fail = make_node("failX", NodeType.FAILURE.value)
    run = make_node("runX", NodeType.EXPERIMENT_RUN.value)
    state1 = make_node("state1", NodeType.RESEARCH_STATE.value)
    state2 = make_node("state2", NodeType.RESEARCH_STATE.value)
    ev = make_node("evidence1", NodeType.EVIDENCE.value)
    g.add_node(fail)
    g.add_node(run)
    g.add_node(state1)
    g.add_node(state2)
    g.add_node(ev)
    g.add_edge(make_edge("ef", "runX", "failX", RelationType.FAILED_AS.value))
    g.add_edge(make_edge("snext", "state1", "state2", RelationType.NEXT_STATE.value))
    g.add_edge(make_edge("r_ev", "runX", "evidence1", RelationType.SUPPORTED_BY.value))

    fh = get_failure_history(g, "failX")
    assert fh.failure.id == "failX"
    assert any(r.id == "runX" for r in fh.related_runs)

    sh = get_state_history(g, "state1")
    assert any(n.id == "state2" for n in sh.linked_states)

    er = get_related_evidence(g, "runX")
    assert any(n.id == "evidence1" for n in er.evidences)
