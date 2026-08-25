"""
Unit tests for the Research Development Graph (RDG).
"""
import pytest

from rdg.graph import ResearchDevelopmentGraph
from rdg.nodes import NodeStatus, NodeType, RDGNode
from rdg.edges import EdgeRelation, RDGEdge
from rdg.consistency import ConsistencyError


class TestRDGNodes:
    def test_create_hypothesis_node(self):
        n = RDGNode.hypothesis("Some hypothesis")
        assert n.type == NodeType.HYPOTHESIS
        assert n.status == NodeStatus.PENDING
        assert len(n.id) == 36

    def test_factory_methods(self):
        assert RDGNode.problem("p").type == NodeType.PROBLEM
        assert RDGNode.gap("g").type == NodeType.GAP
        assert RDGNode.experiment("e").type == NodeType.EXPERIMENT
        assert RDGNode.finding("f").type == NodeType.FINDING
        assert RDGNode.claim("c").type == NodeType.CLAIM

    def test_roundtrip_serialization(self):
        n = RDGNode.hypothesis("Test hypothesis")
        d = n.to_dict()
        n2 = RDGNode.from_dict(d)
        assert n.id == n2.id
        assert n.type == n2.type
        assert n.content == n2.content


class TestRDGGraph:
    def setup_method(self):
        self.rdg = ResearchDevelopmentGraph()

    def test_add_and_get_node(self):
        n = RDGNode.hypothesis("Hyp 1")
        self.rdg.add_node(n)
        retrieved = self.rdg.get_node(n.id)
        assert retrieved is not None
        assert retrieved.content == "Hyp 1"

    def test_add_duplicate_raises(self):
        n = RDGNode.hypothesis("Hyp 1")
        self.rdg.add_node(n)
        with pytest.raises(ValueError):
            self.rdg.add_node(n)

    def test_connect_nodes_valid(self):
        gap = RDGNode.gap("Research gap about X")
        hyp = RDGNode.hypothesis("Hypothesis about X")
        self.rdg.add_node(gap)
        self.rdg.add_node(hyp)
        edge = self.rdg.connect(gap.id, hyp.id, EdgeRelation.MOTIVATES)
        assert edge.relation == EdgeRelation.MOTIVATES

    def test_connect_invalid_types_raises(self):
        hyp = RDGNode.hypothesis("Hyp")
        exp = RDGNode.experiment("Exp")
        self.rdg.add_node(hyp)
        self.rdg.add_node(exp)
        # MOTIVATES requires Gap → Hypothesis
        with pytest.raises(ValueError):
            self.rdg.connect(hyp.id, exp.id, EdgeRelation.MOTIVATES)

    def test_children_of(self):
        gap = RDGNode.gap("Gap")
        hyp = RDGNode.hypothesis("Hyp")
        self.rdg.add_node(gap)
        self.rdg.add_node(hyp)
        self.rdg.connect(gap.id, hyp.id, EdgeRelation.MOTIVATES)
        children = self.rdg.children_of(gap.id)
        assert hyp in children

    def test_stats(self):
        n1 = RDGNode.hypothesis("H1")
        n2 = RDGNode.gap("G1")
        self.rdg.add_node(n1)
        self.rdg.add_node(n2)
        s = self.rdg.stats()
        assert s["total_nodes"] == 2

    def test_save_load(self, tmp_path):
        n = RDGNode.hypothesis("Saved hypothesis")
        self.rdg.add_node(n)
        path = str(tmp_path / "rdg.json")
        self.rdg.save(path)
        rdg2 = ResearchDevelopmentGraph.load(path)
        assert len(rdg2) == 1
        assert rdg2.get_node(n.id).content == "Saved hypothesis"

    def test_merge_equivalent_nodes(self):
        for _ in range(3):
            self.rdg.add_node(RDGNode.hypothesis("Duplicate hypothesis"))
        merged = self.rdg.merge_equivalent_nodes()
        assert merged == 2
        assert len(self.rdg) == 1
