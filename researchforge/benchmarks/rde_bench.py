"""RDE-Bench harness: runs the ablation ladder (full system vs no-memory vs
random search) across tasks and seeds, and computes the metrics defined in
ResearchForge-ECRM Sec. 6 / Sec. 8 (RE, SE, FRR, MU, NTR, Memory Half-life).
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from typing import Dict, List

from ..pipeline.controller import ResearchController, RunResult, CONDITIONS
from .tasks import Task


@dataclass
class BenchSummary:
    task_name: str
    condition: str
    best_metric_mean: float
    best_metric_std: float
    research_efficiency: float       # RE: mean best metric per experiment
    search_efficiency: int           # SE: mean generation of first hit on target_metric
    failure_repetition_rate: float   # FRR
    negative_transfer_rate: float    # NTR
    memory_utility: float = 0.0      # MU: filled in for a memory condition, relative to 'no_memory'
    memory_half_life_days: float = float("nan")
    curves: List[List[float]] = field(default_factory=list)  # best-so-far per seed, incl. baseline


def run_condition(task: Task, condition: str, seeds: List[int], n_generations: int) -> BenchSummary:
    bests, curves, fr_rates, ntr_rates, se_list = [], [], [], [], []
    half_life = float("nan")

    for seed in seeds:
        ctrl = ResearchController(task, condition=condition, seed=seed)
        run: RunResult = ctrl.run(n_generations=n_generations)
        bests.append(run.best_metric)
        curves.append([t.best_so_far for t in run.trials])

        # Signature matches exactly what has_similar_failure keys on
        # (strategy + model_type -- see ResearchController._mem_key), so FRR
        # measures repetition of the *specific* failures the memory check is
        # designed to catch, not just "this strategy name was ever tried again."
        seen_bad = set()
        n_repeat = 0
        for t in run.trials:
            sig = (t.strategy, t.model_type)
            if t.failure != "None":
                if sig in seen_bad:
                    n_repeat += 1
                seen_bad.add(sig)
        fr_rates.append(n_repeat / len(run.trials) if run.trials else 0.0)

        used_mem = [t for t in run.trials if t.used_memory]
        neg = [t for t in used_mem if t.memory_negative_transfer]
        ntr_rates.append(len(neg) / len(used_mem) if used_mem else 0.0)

        hit = next((t.generation for t in run.trials if t.metric >= task.target_metric),
                   n_generations)
        se_list.append(hit)

        if condition == "full":
            half_life = run.memory_half_life_days

    mean_best = statistics.mean(bests)
    std_best = statistics.pstdev(bests) if len(bests) > 1 else 0.0
    re = mean_best / max(1, n_generations)
    return BenchSummary(
        task_name=task.name, condition=condition,
        best_metric_mean=mean_best, best_metric_std=std_best,
        research_efficiency=re, search_efficiency=int(round(statistics.mean(se_list))),
        failure_repetition_rate=statistics.mean(fr_rates),
        negative_transfer_rate=statistics.mean(ntr_rates),
        memory_half_life_days=half_life, curves=curves)


def run_rde_bench(tasks: List[Task], seeds: List[int] = (0, 1, 2),
                   n_generations: int = 25) -> Dict[str, Dict[str, BenchSummary]]:
    report: Dict[str, Dict[str, BenchSummary]] = {}
    for task in tasks:
        report[task.name] = {}
        for cond in CONDITIONS:
            report[task.name][cond] = run_condition(task, cond, list(seeds), n_generations)
        nomem = report[task.name]["no_memory"]
        for memory_cond in ("full", "trajectory_memory"):
            report[task.name][memory_cond].memory_utility = (
                report[task.name][memory_cond].best_metric_mean - nomem.best_metric_mean)
    return report


def print_report(report: Dict[str, Dict[str, BenchSummary]]) -> None:
    for task_name, conds in report.items():
        print(f"\n=== RDE-Bench: {task_name} ===")
        print(f"{'condition':<18}{'best':>8}{'RE':>8}{'SE':>6}{'FRR':>7}{'NTR':>7}{'MU':>9}")
        for cond_name in ("full", "trajectory_memory", "no_memory", "random"):
            s = conds[cond_name]
            mu = f"{s.memory_utility:+.4f}" if cond_name in ("full", "trajectory_memory") else "--"
            print(f"{cond_name:<18}{s.best_metric_mean:>8.4f}{s.research_efficiency:>8.4f}"
                  f"{s.search_efficiency:>6d}{s.failure_repetition_rate:>7.2f}"
                  f"{s.negative_transfer_rate:>7.2f}{mu:>9}")
        print("(best = mean best validation metric across seeds; RE = best/#experiments; "
              "SE = mean generation of first hit on target_metric; FRR/NTR are rates in [0,1]; "
              "MU is relative to no_memory)")
        print(f"analytic memory half-life (decay parameter): "
              f"{conds['full'].memory_half_life_days:.1f} days")
