"""
RDE-Bench Evaluator.
Runs the full benchmark suite and returns a metrics report.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.controller_agent import ResearchController
from benchmarks.metrics import compute_all_metrics
from benchmarks.tasks.cifar10_task import CIFAR10Task
from benchmarks.tasks.ecg_task import ECGTask
from benchmarks.tasks.synthetic_task import SyntheticTimeSeriesTask
from ecrm.memory_store import ECRMMemoryStore
from rdg.graph import ResearchDevelopmentGraph
from rdg.nodes import RDGNode

logger = logging.getLogger(__name__)

TASKS = {
    "cifar10": CIFAR10Task,
    "ecg": ECGTask,
    "synthetic": SyntheticTimeSeriesTask,
}


class BenchmarkEvaluator:
    """
    Runs RDE-Bench tasks and collects comparative metrics.
    
    Usage:
        evaluator = BenchmarkEvaluator(tasks=["cifar10", "synthetic"])
        report = evaluator.run(n_iterations=20)
        print(report)
    """

    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        n_iterations: int = 20,
        mock: bool = True,
        output_dir: str = "./benchmark_results",
    ):
        self.task_names = tasks or list(TASKS.keys())
        self.n_iterations = n_iterations
        self.mock = mock
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        all_results = {}

        for task_name in self.task_names:
            logger.info("═══ Running task: %s ═══", task_name)
            task_cls = TASKS.get(task_name)
            if task_cls is None:
                logger.warning("Unknown task: %s", task_name)
                continue

            task = task_cls(mock=self.mock)
            result = self._run_single_task(task_name, task)
            all_results[task_name] = result

        # Save report
        report_path = self.output_dir / "benchmark_report.json"
        report_path.write_text(json.dumps(all_results, indent=2))
        logger.info("Benchmark report saved to %s", report_path)

        self._print_summary(all_results)
        return all_results

    def _run_single_task(self, task_name: str, task: Any) -> Dict[str, Any]:
        # Setup RDG + Memory
        rdg = ResearchDevelopmentGraph(graph_id=task_name)
        memory = ECRMMemoryStore()

        # Seed problem and gap
        problem_node = RDGNode.problem(content=task.description())
        rdg.add_node(problem_node)
        gap_node = RDGNode.gap(
            content=f"Gap: Existing baselines achieve ~{task.baseline_score:.0%}. How to improve?"
        )
        rdg.add_node(gap_node)
        from rdg.edges import EdgeRelation
        rdg.connect(problem_node.id, gap_node.id, EdgeRelation.IDENTIFIES, validate=False)

        # Run controller
        controller = ResearchController(
            rdg=rdg,
            memory=memory,
            problem_description=task.description(),
            use_mock_experiments=self.mock,
        )

        t0 = time.time()
        summary = controller.run(n_iterations=self.n_iterations)
        elapsed = time.time() - t0

        # Collect metrics
        perf_history = [h["score"] for h in controller.history]
        failure_log = [
            {"type": h["failure_category"], "context_hash": h["hypothesis"][:20]}
            for h in controller.history
            if not h["success"]
        ]
        claims_list = [
            {"supported_by": [n.id for n in rdg.findings[:1]]}
            for n in rdg.claims
        ]

        metrics = compute_all_metrics(
            performance_history=perf_history,
            compute_costs=[1.0] * len(perf_history),
            failure_log=failure_log,
            memory_uses=[(True, s - (task.baseline_score)) for s in perf_history],
            claims=claims_list,
        )
        metrics["best_score"] = controller.best_score
        metrics["baseline_score"] = task.baseline_score
        metrics["total_experiments"] = controller.total_experiments
        metrics["elapsed_seconds"] = elapsed

        logger.info(
            "[%s] best=%.4f | FRR=%.3f | RRS=%.3f",
            task_name, controller.best_score, metrics["failure_repetition_rate"],
            metrics["research_reliability_score"],
        )
        return metrics

    def _print_summary(self, results: Dict[str, Any]) -> None:
        print("\n" + "═" * 60)
        print("  RDE-Bench Results Summary")
        print("═" * 60)
        for task, metrics in results.items():
            print(f"\n  Task: {task}")
            print(f"    Best Score:      {metrics.get('best_score', 0):.4f} "
                  f"(baseline={metrics.get('baseline_score', 0):.4f})")
            print(f"    Search Eff (SE): {metrics.get('search_efficiency', 0)} evaluations")
            print(f"    Fail Repeat (FRR):{metrics.get('failure_repetition_rate', 0):.3f}")
            print(f"    NTR:             {metrics.get('negative_transfer_rate', 0):.3f}")
            print(f"    RRS:             {metrics.get('research_reliability_score', 0):.3f}")
        print("═" * 60 + "\n")
