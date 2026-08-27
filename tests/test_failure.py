"""
Tests for v2 failure module: taxonomy, diagnosis, and repair handlers.
"""
from __future__ import annotations

import pytest

from failure.taxonomy import FailureCategory, REPAIR_ACTIONS, all_categories
from failure.diagnosis import diagnose_failure
from failure.repair import apply_repair, _note
from rdg.graph import ResearchDevelopmentGraph
from rdg.nodes import NodeStatus, NodeType, RDGNode
from ecrm.memory_store import ECRMMemoryStore


# ── Taxonomy tests ────────────────────────────────────────────────────────────

class TestTaxonomy:
    def test_all_categories_returns_strings(self):
        cats = all_categories()
        assert all(isinstance(c, str) for c in cats)
        assert "NegativeTransfer" in cats

    def test_all_categories_have_repair_action(self):
        for cat in FailureCategory:
            assert cat in REPAIR_ACTIONS, f"{cat} missing from REPAIR_ACTIONS"

    def test_repair_actions_are_strings(self):
        for cat, action in REPAIR_ACTIONS.items():
            assert isinstance(action, str), f"{cat} repair action is not a string"


# ── Diagnosis tests ───────────────────────────────────────────────────────────

class TestDiagnosis:
    def setup_method(self):
        self.rdg = ResearchDevelopmentGraph()

    def test_diagnose_timeout_from_error_message(self):
        node = RDGNode.experiment(content="timed-out experiment")
        node.status = NodeStatus.FAILED
        node.attributes["error"] = "subprocess timeout expired"
        self.rdg.add_node(node)
        cat, bad = diagnose_failure(self.rdg, node)
        assert cat == FailureCategory.TIMEOUT
        assert bad is node

    def test_diagnose_code_error(self):
        node = RDGNode.experiment(content="crashed experiment")
        node.status = NodeStatus.FAILED
        node.attributes["error"] = "NameError: undefined variable"
        self.rdg.add_node(node)
        cat, bad = diagnose_failure(self.rdg, node)
        assert cat == FailureCategory.CODE_ERROR

    def test_diagnose_divergence_from_failure_flags(self):
        node = RDGNode.finding(content="diverged run", score=0.1)
        node.attributes["failure_flags"] = ["Divergence"]
        self.rdg.add_node(node)
        cat, bad = diagnose_failure(self.rdg, node)
        assert cat == FailureCategory.DIVERGENCE
        assert bad is node

    def test_diagnose_divergence_case_insensitive(self):
        node = RDGNode.finding(content="diverged run 2", score=0.1)
        node.attributes["failure_flags"] = ["divergence"]
        self.rdg.add_node(node)
        cat, _ = diagnose_failure(self.rdg, node)
        assert cat == FailureCategory.DIVERGENCE

    def test_diagnose_low_performance(self):
        node = RDGNode.finding(content="low score", score=0.1)
        node.attributes["score"] = 0.1
        self.rdg.add_node(node)
        cat, bad = diagnose_failure(self.rdg, node, target_metric=0.5)
        assert cat == FailureCategory.LOW_PERFORMANCE

    def test_diagnose_unknown_for_healthy_node(self):
        node = RDGNode.finding(content="good result", score=0.9)
        node.attributes["score"] = 0.9
        self.rdg.add_node(node)
        cat, bad = diagnose_failure(self.rdg, node, target_metric=0.5)
        assert cat == FailureCategory.UNKNOWN
        assert bad is None


# ── Repair handler tests ──────────────────────────────────────────────────────

class TestRepairHandlers:
    def setup_method(self):
        self.rdg = ResearchDevelopmentGraph()
        self.memory = ECRMMemoryStore()

    def _make_node(self, content="test node") -> RDGNode:
        node = RDGNode.finding(content=content, score=0.0)
        self.rdg.add_node(node)
        return node

    def test_repair_divergence_adds_note(self):
        node = self._make_node("divergence node")
        msg = apply_repair(FailureCategory.DIVERGENCE, node, self.rdg, self.memory)
        assert "learning_rate" in msg.lower() or "gradient" in msg.lower()
        assert "repair" in node.attributes

    def test_repair_overfitting_adds_regularization_note(self):
        node = self._make_node("overfit node")
        msg = apply_repair(FailureCategory.OVERFITTING, node, self.rdg, self.memory)
        assert "regulariz" in msg.lower() or "dropout" in msg.lower()

    def test_repair_timeout_adds_note(self):
        node = self._make_node("timeout node")
        msg = apply_repair(FailureCategory.TIMEOUT, node, self.rdg, self.memory)
        assert "timeout" in msg.lower() or "complex" in msg.lower()

    def test_repair_negative_transfer_logs(self):
        node = self._make_node("ntr node")
        node.attributes["strategy_id"] = "param_mutation"
        msg = apply_repair(FailureCategory.NEGATIVE_TRANSFER, node, self.rdg, self.memory)
        assert "transfer" in msg.lower() or "retrieval" in msg.lower()

    def test_repair_stale_memory_calls_consolidate(self):
        node = self._make_node("stale node")
        msg = apply_repair(FailureCategory.STALE_MEMORY, node, self.rdg, self.memory)
        assert "consolidat" in msg.lower() or "stale" in msg.lower()

    def test_repair_code_error_includes_message(self):
        node = self._make_node("code error node")
        node.attributes["error"] = "SyntaxError line 42"
        msg = apply_repair(FailureCategory.CODE_ERROR, node, self.rdg, self.memory)
        assert "code" in msg.lower() or "error" in msg.lower()

    def test_repair_unknown_returns_fallback(self):
        node = self._make_node("unknown failure")
        msg = apply_repair(FailureCategory.UNKNOWN, node, self.rdg, self.memory)
        assert "ablation" in msg.lower() or "logged" in msg.lower()

    def test_repair_marks_node_failed(self):
        node = self._make_node("to fail")
        apply_repair(FailureCategory.LOW_PERFORMANCE, node, self.rdg)
        assert node.status == NodeStatus.FAILED
        assert node.failure_count == 1

    def test_note_helper_appends(self):
        node = self._make_node()
        node.attributes["repair"] = "first note"
        _note(node, "repair", "second note")
        assert "first note" in node.attributes["repair"]
        assert "second note" in node.attributes["repair"]
