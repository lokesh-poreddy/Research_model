"""
Branch selection acquisition function.
Implements UCB-like scoring over candidate RDG hypothesis nodes.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from rdg.nodes import RDGNode
from ecrm.memory_store import ECRMMemoryStore


def ucb_score(
    node: RDGNode,
    total_experiments: int,
    c: float = 1.41,
    memory: Optional[ECRMMemoryStore] = None,
    failure_penalty: float = 0.5,
    policy_estimate: Optional[float] = None,
) -> float:
    """
    UCB acquisition score for a hypothesis node:

    score = reward + c * sqrt(log(N) / (1 + times_tried))
            * failure_penalty_if_similar_failure_exists

    Args:
        node: Candidate hypothesis node.
        total_experiments: Total experiments run so far (N).
        c: Exploration constant.
        memory: ECRM store for failure lookup.
        failure_penalty: Multiplier when a similar failure exists.
        policy_estimate: Optional override for expected reward (from RL policy).
    """
    reward = policy_estimate if policy_estimate is not None else (node.best_metric or 0.0)
    n_tried = max(0, node.times_tried)
    total = max(1, total_experiments)

    bonus = c * math.sqrt(math.log(total) / (1 + n_tried))
    score = reward + bonus

    # V2: ECRM changes the action ranking through context-sensitive empirical
    # support.  This makes memory part of the control policy rather than
    # passive text injected into a prompt.
    if memory is not None:
        score += memory.decision_support(node.content)["adjustment"]

    # Penalise if memory contains a similar failure
    if memory is not None and memory.has_similar_failure(node.content):
        score *= failure_penalty

    return score


def select_branch(
    hypotheses: List[RDGNode],
    total_experiments: int,
    memory: Optional[ECRMMemoryStore] = None,
    c: float = 1.41,
    failure_penalty: float = 0.5,
    q_values: Optional[dict] = None,
) -> Optional[RDGNode]:
    """
    Select the best hypothesis node to explore next.

    Args:
        hypotheses: List of candidate hypothesis nodes.
        total_experiments: Total number of experiments performed so far.
        memory: ECRM store (optional).
        c: UCB exploration constant.
        failure_penalty: Applied when memory flags a similar failure.
        q_values: Optional dict mapping node.id → Q-value from RL policy.

    Returns:
        The highest-scoring hypothesis node, or None if list is empty.
    """
    if not hypotheses:
        return None

    best_score = float("-inf")
    best_node = None

    for h in hypotheses:
        policy_est = q_values.get(h.id) if q_values else None
        score = ucb_score(h, total_experiments, c, memory, failure_penalty, policy_est)
        if score > best_score:
            best_score = score
            best_node = h

    return best_node
