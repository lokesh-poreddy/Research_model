"""
RDE-Bench Metrics.
Formally defined metrics for evaluating autonomous research agents.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


def research_efficiency(
    performance_gains: List[float],
    compute_costs: List[float],
) -> float:
    """
    RE = sum(performance_gain) / sum(compute_cost)
    Higher is better.
    """
    total_cost = sum(compute_costs)
    if total_cost == 0:
        return 0.0
    return sum(performance_gains) / total_cost


def search_efficiency(
    performance_history: List[float],
    target: Optional[float] = None,
) -> int:
    """
    SE = number of evaluations until first improvement above target.
    Returns total evaluations if never reached.
    """
    if not performance_history:
        return 0
    baseline = performance_history[0]
    threshold = target if target is not None else baseline * 1.01
    for i, score in enumerate(performance_history):
        if score >= threshold:
            return i + 1
    return len(performance_history)


def failure_repetition_rate(failure_log: List[Dict]) -> float:
    """
    FRR = #repeated_failures / #total_failures
    failure_log: list of {"type": str, "context_hash": str}
    Lower is better.
    """
    if not failure_log:
        return 0.0
    seen = set()
    repeated = 0
    for entry in failure_log:
        key = (entry.get("type", ""), entry.get("context_hash", ""))
        if key in seen:
            repeated += 1
        seen.add(key)
    return repeated / len(failure_log)


def memory_utility(
    scores_with_memory: List[float],
    scores_without_memory: List[float],
) -> float:
    """
    MU = mean(scores_with_memory) - mean(scores_without_memory)
    Positive = memory helps.
    """
    if not scores_with_memory or not scores_without_memory:
        return 0.0
    mean_with = sum(scores_with_memory) / len(scores_with_memory)
    mean_without = sum(scores_without_memory) / len(scores_without_memory)
    return mean_with - mean_without


def negative_transfer_rate(
    memory_uses: List[Tuple[bool, float]],  # (used_memory, improvement)
) -> float:
    """
    NTR = #harmful_memory_uses / #total_memory_uses
    harmful = improvement < 0 when memory was used.
    Lower is better.
    """
    used = [(u, imp) for u, imp in memory_uses if u]
    if not used:
        return 0.0
    harmful = sum(1 for _, imp in used if imp < 0)
    return harmful / len(used)


def memory_half_life(lambda_decay: float) -> float:
    """
    Return number of days until a memory record's retention
    falls to 50% (given exponential decay rate λ).
    """
    return math.log(2) / max(lambda_decay, 1e-9)


def research_reliability_score(claims: List[Dict]) -> float:
    """
    RRS = # claims with at least one supported finding / # total claims
    claims: list of {"supported_by": [finding_id, ...]}
    Higher is better.
    """
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if c.get("supported_by"))
    return supported / len(claims)


def compute_all_metrics(
    performance_history: List[float],
    compute_costs: List[float],
    failure_log: List[Dict],
    memory_uses: List[Tuple[bool, float]],
    claims: List[Dict],
    lambda_decay: float = 0.01,
) -> Dict[str, float]:
    """Compute all RDE-Bench metrics in one call."""
    gains = [max(0, b - a) for a, b in zip(performance_history, performance_history[1:])]
    return {
        "research_efficiency": research_efficiency(gains, compute_costs[1:] or [1.0] * len(gains)),
        "search_efficiency": search_efficiency(performance_history),
        "failure_repetition_rate": failure_repetition_rate(failure_log),
        "negative_transfer_rate": negative_transfer_rate(memory_uses),
        "memory_half_life_days": memory_half_life(lambda_decay),
        "research_reliability_score": research_reliability_score(claims),
    }
