import json
import pytest
from researchforge.vrdeg.node import GraphNode, NodeType
from researchforge.vrdeg.edge import GraphEdge, RelationType
from researchforge.vrdeg.graph import VRDEG, GraphIntegrityError
from researchforge.domain.provenance import Provenance
from researchforge.domain.experiment import ExperimentSpec, ExperimentRun, Outcome


def make_prov(pid="prov1"):
    return Provenance(id=pid, schema_version="1", created_by="u", created_at="now")


def test_node_and_edge_creation_and_serialization():
    g = VRDEG()
    n1 = GraphNode(id="n1", schema_version="1", node_type=NodeType.EXPERIMENT_SPEC.value, payload_ref="spec1", provenance_id="pv1")
    n2 = GraphNode(id="n2", schema_version="1", node_type=NodeType.EXPERIMENT_RUN.value, payload_ref="run1", provenance_id="pv2")
    g.add_node(n1)
    g.add_node(n2)
    e = GraphEdge(id="e1", schema_version="1", source_id="n1", target_id="n2", relation=RelationType.EXECUTED_AS.value, provenance_id="pv3")
    with pytest.raises(GraphIntegrityError):
        # missing provenance node references should be caught on validate
        g.validate_integrity()
    # add provenance nodes
    g.add_node(GraphNode(id="pv1", schema_version="1", node_type=NodeType.PROVENANCE.value))
    g.add_node(GraphNode(id="pv2", schema_version="1", node_type=NodeType.PROVENANCE.value))
    g.add_node(GraphNode(id="pv3", schema_version="1", node_type=NodeType.PROVENANCE.value))
    # revalidate
    g.validate_integrity()
    g.add_edge(e)
    assert g.get_node("n1").payload_ref == "spec1"
    assert g.get_edge("e1").relation == RelationType.EXECUTED_AS.value
    j = g.export_canonical()
    assert isinstance(j, str)
    # deterministic fingerprint
    fp1 = g.fingerprint()
    fp2 = g.fingerprint()
    assert fp1 == fp2


def test_duplicate_node_rejected_and_duplicate_edge_rejected():
    g = VRDEG()
    g.add_node(GraphNode(id="a", schema_version="1", node_type=NodeType.OUTCOME.value))
    with pytest.raises(GraphIntegrityError):
        g.add_node(GraphNode(id="a", schema_version="1", node_type=NodeType.OUTCOME.value))
    g.add_node(GraphNode(id="b", schema_version="1", node_type=NodeType.PROVENANCE.value))
    g.add_edge(GraphEdge(id="edge1", schema_version="1", source_id="a", target_id="b", relation=RelationType.HAS_PROVENANCE.value))
    with pytest.raises(GraphIntegrityError):
        g.add_edge(GraphEdge(id="edge1", schema_version="1", source_id="a", target_id="b", relation=RelationType.HAS_PROVENANCE.value))


def test_lineage_and_branching_and_reconstruction():
    g = VRDEG()
    # create hypothesis and two branches with outcomes
    g.add_node(GraphNode(id="h1", schema_version="1", node_type=NodeType.HYPOTHESIS.value))
    g.add_node(GraphNode(id="specA", schema_version="1", node_type=NodeType.EXPERIMENT_SPEC.value))
    g.add_node(GraphNode(id="runA", schema_version="1", node_type=NodeType.EXPERIMENT_RUN.value))
    g.add_node(GraphNode(id="outA", schema_version="1", node_type=NodeType.OUTCOME.value))
    g.add_node(GraphNode(id="specB", schema_version="1", node_type=NodeType.EXPERIMENT_SPEC.value))
    g.add_node(GraphNode(id="runB", schema_version="1", node_type=NodeType.EXPERIMENT_RUN.value))
    g.add_node(GraphNode(id="outB", schema_version="1", node_type=NodeType.OUTCOME.value))
    g.add_edge(GraphEdge(id="e1", schema_version="1", source_id="h1", target_id="specA", relation=RelationType.MOTIVATED_BY.value))
    g.add_edge(GraphEdge(id="e2", schema_version="1", source_id="specA", target_id="runA", relation=RelationType.EXECUTED_AS.value))
    g.add_edge(GraphEdge(id="e3", schema_version="1", source_id="runA", target_id="outA", relation=RelationType.PRODUCED.value))
    g.add_edge(GraphEdge(id="e4", schema_version="1", source_id="h1", target_id="specB", relation=RelationType.MOTIVATED_BY.value))
    g.add_edge(GraphEdge(id="e5", schema_version="1", source_id="specB", target_id="runB", relation=RelationType.EXECUTED_AS.value))
    g.add_edge(GraphEdge(id="e6", schema_version="1", source_id="runB", target_id="outB", relation=RelationType.PRODUCED.value))
    # lineage from outA should include runA and specA and h1
    lineage = g.trace_lineage("outA")
    ids = {n.id for n in lineage}
    assert "runA" in ids and "specA" in ids and "h1" in ids


def test_experiment_integration_and_failure_retention():
    g = VRDEG()
    spec = ExperimentSpec(id="sX", schema_version="1", metrics=["m"], provenance=Provenance(id="pvx", schema_version="1", created_by="u", created_at="now"))
    run = ExperimentRun(id="rX", schema_version="1", experiment_spec_id=spec.id, status=type("S", (), {"name": lambda: "FAILED"})())
    out = Outcome(id="oX", schema_version="1", run_id=run.id, metrics=None)
    # nodes
    g.add_node(GraphNode(id=spec.id, schema_version=spec.schema_version, node_type=NodeType.EXPERIMENT_SPEC.value, payload_ref=spec.id, provenance_id=spec.provenance.id))
    g.add_node(GraphNode(id=run.id, schema_version=run.schema_version, node_type=NodeType.EXPERIMENT_RUN.value, payload_ref=run.id, provenance_id=run.provenance.id if run.provenance else None))
    g.add_node(GraphNode(id=out.id, schema_version=out.schema_version, node_type=NodeType.OUTCOME.value, payload_ref=out.id, provenance_id=None))
    # edges
    g.add_edge(GraphEdge(id="er1", schema_version="1", source_id=spec.id, target_id=run.id, relation=RelationType.EXECUTED_AS.value))
    g.add_edge(GraphEdge(id="ro1", schema_version="1", source_id=run.id, target_id=out.id, relation=RelationType.PRODUCED.value))
    # ensure failed run is queryable and outcome exists (even if no metrics)
    assert g.get_node(run.id) is not None
    assert g.get_node(out.id) is not None
