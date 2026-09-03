"""
Tests for v2 policy modules: BudgetAllocator and select_branch.

Classification: LEGACY
Tests the legacy policy/ module. Retained as historical/archival evidence.
Canonical replacement: researchforge.policy.policy_learner
"""
from __future__ import annotations

import time
import pytest

pytestmark = pytest.mark.LEGACY

from policy.budget_allocator import BudgetAllocator
from policy.acquisition import select_branch
from rdg.nodes import RDGNode


# ── BudgetAllocator tests ─────────────────────────────────────────────────────

class TestBudgetAllocator:
    def setup_method(self):
        self.alloc = BudgetAllocator(budget_hours=1.0)

    def test_starts_empty(self):
        assert self.alloc.consumed_hours == 0.0
        assert self.alloc.remaining_fraction() == 1.0
        assert not self.alloc.is_over_budget()

    def test_record_accumulates(self):
        self.alloc.record("op_a", "optimization", 1800.0)  # 0.5 hours
        assert abs(self.alloc.consumed_hours - 0.5) < 1e-6
        assert abs(self.alloc.remaining_fraction() - 0.5) < 1e-6

    def test_over_budget_flag(self):
        self.alloc.record("op_a", "optimization", 4000.0)  # > 1 hour
        assert self.alloc.is_over_budget()

    def test_family_hours_breakdown(self):
        self.alloc.record("param_mutation", "optimization", 900.0)
        self.alloc.record("structure_add", "architecture", 900.0)
        fh = self.alloc.family_hours()
        assert "optimization" in fh and "architecture" in fh
        assert abs(fh["optimization"] - fh["architecture"]) < 1e-6

    def test_summary_keys(self):
        self.alloc.record("op_x", "data", 60.0)
        s = self.alloc.summary()
        assert "budget_hours" in s
        assert "consumed_hours" in s
        assert "remaining_fraction" in s
        assert "by_strategy" in s
        assert "op_x" in s["by_strategy"]

    def test_remaining_fraction_capped_at_zero(self):
        self.alloc.record("op_a", "optimization", 99_999.0)
        assert self.alloc.remaining_fraction() == 0.0

    def test_warn_fraction_triggers_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="policy.budget_allocator"):
            self.alloc.record("op_a", "optimization", 2_900.0)  # 80.6% of 1h
        assert any("80" in r.message or "budget" in r.message.lower()
                   for r in caplog.records)


# ── select_branch tests ───────────────────────────────────────────────────────

class TestSelectBranch:
    def test_empty_list_returns_none_with_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="policy.acquisition"):
            result = select_branch([], total_experiments=5)
        assert result is None
        assert any("empty" in r.message.lower() for r in caplog.records)

    def test_single_hypothesis_selected(self):
        h = RDGNode.hypothesis(content="test hypothesis")
        result = select_branch([h], total_experiments=1)
        assert result is h

    def test_ucb_prefers_unvisited(self):
        h_visited = RDGNode.hypothesis(content="tried many times")
        h_visited.times_tried = 10
        h_visited.best_metric = 0.8

        h_fresh = RDGNode.hypothesis(content="never tried")
        h_fresh.times_tried = 0

        # Fresh node should get higher UCB (high exploration bonus)
        result = select_branch([h_visited, h_fresh], total_experiments=20)
        assert result is h_fresh

    def test_thompson_returns_a_node(self):
        nodes = [RDGNode.hypothesis(content=f"hyp {i}") for i in range(3)]
        result = select_branch(nodes, total_experiments=5, policy_type="thompson")
        assert result in nodes

    def test_policy_type_rl_falls_through_to_ucb(self):
        """RL policy_type should still return a node via UCB path."""
        h = RDGNode.hypothesis(content="rl path test")
        result = select_branch([h], total_experiments=1, policy_type="rl")
        assert result is h
