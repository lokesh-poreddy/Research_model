"""
Integration test: full research loop.
"""
import pytest

from agents.controller_agent import ResearchController
from ecrm.memory_store import ECRMMemoryStore
from rdg.graph import ResearchDevelopmentGraph
from rdg.nodes import RDGNode
from rdg.edges import EdgeRelation
from benchmarks.metrics import compute_all_metrics
from failure.taxonomy import FailureCategory
from benchmarks.tasks import DigitsTask


class TestResearchLoop:
    def setup_method(self):
        self.rdg = ResearchDevelopmentGraph()
        self.memory = ECRMMemoryStore()

        problem = RDGNode.problem("Improve time-series forecasting accuracy")
        self.rdg.add_node(problem)
        gap = RDGNode.gap("Gap: Current models underfit on noisy series.")
        self.rdg.add_node(gap)
        self.rdg.connect(problem.id, gap.id, EdgeRelation.IDENTIFIES, validate=False)

        self.controller = ResearchController(
            rdg=self.rdg,
            memory=self.memory,
            problem_description="Improve time-series forecasting accuracy",
            use_mock_experiments=True,
        )

    def test_run_returns_summary(self):
        summary = self.controller.run(n_iterations=3)
        assert "total_experiments" in summary
        assert "best_score" in summary
        assert summary["total_experiments"] == 3

    def test_rdg_grows_during_run(self):
        initial_size = len(self.rdg)
        self.controller.run(n_iterations=3)
        assert len(self.rdg) > initial_size

    def test_memory_retains_reusable_records_only(self):
        self.controller.run(n_iterations=3)
        # ECRM retains improvements and diagnosed failures; neutral repeats do
        # not pollute the long-lived store.
        assert 0 < len(self.memory) <= 3

    def test_history_recorded(self):
        self.controller.run(n_iterations=5)
        assert len(self.controller.history) == 5
        for h in self.controller.history:
            assert "score" in h
            assert "success" in h
            assert "failure_category" in h

    def test_policy_updated(self):
        self.controller.run(n_iterations=5)
        # Policy Q-table should have some entries
        assert len(self.controller.policy.q) > 0


class TestBenchmarkMetrics:
    def test_compute_all_metrics(self):
        performance = [0.5, 0.55, 0.6, 0.58, 0.65]
        costs = [1.0, 1.0, 1.0, 1.0, 1.0]
        failures = [
            {"type": "CodeError", "context_hash": "abc"},
            {"type": "CodeError", "context_hash": "abc"},  # repeated
        ]
        mem_uses = [(True, 0.1), (True, -0.05)]
        claims = [{"supported_by": ["f1"]}, {"supported_by": []}]

        metrics = compute_all_metrics(performance, costs, failures, mem_uses, claims)
        assert metrics["failure_repetition_rate"] == pytest.approx(0.5, abs=0.01)
        assert metrics["negative_transfer_rate"] == pytest.approx(0.5, abs=0.01)
        assert metrics["research_reliability_score"] == pytest.approx(0.5, abs=0.01)
        assert "research_efficiency" in metrics
        assert "search_efficiency" in metrics


class TestRealOfflineTraining:
    def test_digits_task_trains_and_evaluates(self):
        task = DigitsTask(seed=4, n_train=100)
        rdg, memory = ResearchDevelopmentGraph(), ECRMMemoryStore()
        problem = RDGNode.problem(task.description())
        gap = RDGNode.gap("Improve the baseline.")
        rdg.add_node(problem)
        rdg.add_node(gap)
        rdg.connect(problem.id, gap.id, EdgeRelation.IDENTIFIES, validate=False)
        controller = ResearchController(
            rdg, memory, problem_description=task.description(),
            use_mock_experiments=False, task=task,
        )
        summary = controller.run(n_iterations=2)
        assert summary["best_score"] > 0.50
        assert all("score" in step for step in controller.history)


class TestFailureDiagnosis:
    def setup_method(self):
        from failure.diagnosis import diagnose_failure
        from failure.repair import apply_repair
        self.diagnose = diagnose_failure
        self.repair = apply_repair

    def test_diagnose_timeout(self):
        rdg = ResearchDevelopmentGraph()
        exp = RDGNode.experiment("Run experiment")
        exp.status = __import__("rdg.nodes", fromlist=["NodeStatus"]).NodeStatus.FAILED
        exp.attributes["error"] = "timeout exceeded"
        rdg.add_node(exp)
        cat, _ = self.diagnose(rdg, exp)
        assert cat == FailureCategory.TIMEOUT

    def test_diagnose_low_performance(self):
        rdg = ResearchDevelopmentGraph()
        finding = RDGNode.finding("Low accuracy result", score=0.3)
        rdg.add_node(finding)
        cat, node = self.diagnose(rdg, finding, target_metric=0.7)
        assert cat == FailureCategory.LOW_PERFORMANCE

    def test_repair_does_not_crash(self):
        rdg = ResearchDevelopmentGraph()
        node = RDGNode.finding("Failed finding", score=0.1)
        rdg.add_node(node)
        result = self.repair(FailureCategory.LOW_PERFORMANCE, node, rdg)
        assert isinstance(result, str)
