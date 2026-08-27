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
from benchmarks.tasks.digits_task import DigitsTask
from ecrm.memory_store import ECRMMemoryStore
from rdg.graph import ResearchDevelopmentGraph
from rdg.nodes import RDGNode

logger = logging.getLogger(__name__)

TASKS = {
    "cifar10": CIFAR10Task,
    "ecg": ECGTask,
    "synthetic": SyntheticTimeSeriesTask,
    "digits": DigitsTask,
}


class BenchmarkEvaluator:
    """
    Runs RDE-Bench tasks and collects comparative metrics.

    Usage:
        evaluator = BenchmarkEvaluator(tasks=["digits", "synthetic"], n_seeds=5)
        report = evaluator.run(n_iterations=20)
        print(report)

    v2: ``n_seeds`` controls how many independent seeds are run per task.
    Results are aggregated (mean, std, min, max) for confidence intervals.
    The v2 ablation protocol requires n_seeds ≥ 5 before promotion.
    """

    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        n_iterations: int = 20,
        mock: bool = True,
        output_dir: str = "./benchmark_results",
        n_seeds: int = 1,          # v2: ≥ 5 for promotion-eligible results
    ):
        self.task_names = tasks or list(TASKS.keys())
        self.n_iterations = n_iterations
        self.mock = mock
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_seeds = max(1, n_seeds)

    def run(self) -> Dict[str, Any]:
        all_results = {}

        for task_name in self.task_names:
            logger.info("═══ Running task: %s (×%d seeds) ═══", task_name, self.n_seeds)
            task_cls = TASKS.get(task_name)
            if task_cls is None:
                logger.warning("Unknown task: %s", task_name)
                continue

            # Collect results across seeds
            seed_results = []
            for seed_idx in range(self.n_seeds):
                import random
                import numpy as np
                random.seed(seed_idx)
                np.random.seed(seed_idx)

                task = task_cls() if task_name == "digits" else task_cls(mock=self.mock)
                seed_result = self._run_single_task(task_name, task, seed=seed_idx)
                seed_results.append(seed_result)
                logger.info(
                    "[%s] seed=%d best=%.4f",
                    task_name, seed_idx, seed_result.get("best_score", 0.0)
                )

            # Aggregate across seeds
            result = self._aggregate_seeds(seed_results)
            all_results[task_name] = result

        # Save report
        report_path = self.output_dir / "benchmark_report.json"
        report_path.write_text(json.dumps(all_results, indent=2))
        logger.info("Benchmark report saved to %s", report_path)

        self._print_summary(all_results)
        return all_results

    @staticmethod
    def _aggregate_seeds(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-seed metrics into mean ± std for the report."""
        import statistics

        if not seed_results:
            return {}
        # Use the first seed's non-numeric fields as base
        aggregated = dict(seed_results[0])
        numeric_keys = [
            k for k, v in seed_results[0].items() if isinstance(v, (int, float))
        ]
        for key in numeric_keys:
            vals = [r[key] for r in seed_results if key in r]
            if len(vals) > 1:
                aggregated[key] = statistics.mean(vals)
                aggregated[f"{key}_std"] = statistics.stdev(vals)
                aggregated[f"{key}_min"] = min(vals)
                aggregated[f"{key}_max"] = max(vals)
            else:
                aggregated[key] = vals[0] if vals else 0.0
        aggregated["n_seeds"] = len(seed_results)
        return aggregated

    def _run_single_task(self, task_name: str, task: Any, seed: int = 0) -> Dict[str, Any]:
        """Run the controller on one task/seed combination."""
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
            task=task,
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
        print("  RDE-Bench Results Summary (v2)")
        print("═" * 60)
        for task, metrics in results.items():
            n_seeds = metrics.get("n_seeds", 1)
            best = metrics.get("best_score", 0)
            best_std = metrics.get("best_score_std", 0.0)
            print(f"\n  Task: {task}  (seeds={n_seeds})")
            seed_str = f" ±{best_std:.4f}" if n_seeds > 1 else ""
            print(f"    Best Score:      {best:.4f}{seed_str} "
                  f"(baseline={metrics.get('baseline_score', 0):.4f})")
            print(f"    Search Eff (SE): {metrics.get('search_efficiency', 0)} evaluations")
            print(f"    Fail Repeat (FRR):{metrics.get('failure_repetition_rate', 0):.3f}")
            print(f"    NTR:             {metrics.get('negative_transfer_rate', 0):.3f}")
            print(f"    RRS:             {metrics.get('research_reliability_score', 0):.3f}")
        print("═" * 60 + "\n")
