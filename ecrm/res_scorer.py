"""
Research Experience Score (RES) computation.
RES combines reliability, negative transfer penalty, and uncertainty.
"""
from __future__ import annotations

import math
from typing import List, Optional


def compute_res(
    outcomes: List[float],
    ntr: float = 0.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
) -> float:
    """
    Research Experience Score:

    RES(h, G) = α · Rel(h, G)  +  β · 1/(1 + NTR(h, G))  -  γ · uncertainty

    where:
      Rel   = mean / (std + ε)       — repeatability-normalised performance
      NTR   = negative transfer rate  — penalises harmful memory reuse
      uncertainty = 1 / (1 + n)      — decreases as more trials accumulate

    Returns 0.0 when no outcomes are provided.
    """
    if not outcomes:
        return 0.0

    n = len(outcomes)
    mu = sum(outcomes) / n
    std = math.sqrt(sum((x - mu) ** 2 for x in outcomes) / n) if n > 1 else 0.0
    eps = 1e-6

    reliability = mu / (std + eps)
    ntr_penalty = 1.0 / (1.0 + max(0.0, ntr))
    uncertainty = 1.0 / (1.0 + n)

    return alpha * reliability + beta * ntr_penalty - gamma * uncertainty


def compute_reliability(outcomes: List[float]) -> float:
    """Return mean / (std + ε)."""
    if not outcomes:
        return 0.0
    n = len(outcomes)
    mu = sum(outcomes) / n
    std = math.sqrt(sum((x - mu) ** 2 for x in outcomes) / n) if n > 1 else 0.0
    return mu / (std + 1e-6)


def memory_utility(score_with_memory: float, score_without_memory: float) -> float:
    """
    Memory Utility (MU):
    MU = performance_with_memory - performance_without_memory
    """
    return score_with_memory - score_without_memory


def memory_weight(
    reliability: float,
    relevance: float,
    recency_weight: float = 1.0,
) -> float:
    """
    Combined memory usefulness score:
    MemoryUtility = Reliability × Relevance × RecencyWeight
    """
    return max(0.0, reliability * relevance * recency_weight)
