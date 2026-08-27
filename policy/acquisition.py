"""
Branch selection acquisition function — v2.

v2 changes:
- Explicit logger.warning when ``hypotheses`` is empty (previously silent None).
- Thompson Sampling branch: when ``policy_type == "thompson"``,
  ``select_branch`` delegates to ``ThompsonBandit`` after updating rewards.
- ``ucb_score`` unchanged: still applies decision_support adjustment first,
  then failure-similarity penalty.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional

from rdg.nodes import RDGNode
from ecrm.memory_store import ECRMMemoryStore

logger = logging.getLogger(__name__)


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
            + ECRM decision_support adjustment
            * failure_penalty_if_similar_failure_exists

    Args:
        node: Candidate hypothesis node.
        total_experiments: Total experiments run so far (N).
        c: Exploration constant.
        memory: ECRM store for failure lookup and decision_support.
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
    policy_type: str = "ucb",
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
        policy_type: "ucb" (default) | "thompson" | "rl".

    Returns:
        The highest-scoring hypothesis node, or None if list is empty.
    """
    if not hypotheses:
        logger.warning(
            "select_branch called with empty hypothesis list "
            "(total_experiments=%d). Returning None.",
            total_experiments,
        )
        return None

    # ── Thompson Sampling path ─────────────────────────────────────────────
    if policy_type == "thompson":
        from policy.bandit import ThompsonBandit
        bandit = ThompsonBandit()
        # Seed bandit with any available Q-values as prior alpha/beta
        if q_values:
            for h in hypotheses:
                q = q_values.get(h.id)
                if q is not None:
                    # Use Q-value to bias the Beta prior: alpha proportional to Q
                    params = bandit._get_params(h.id)
                    params[0] = max(1.0, q * 10)
        selected = bandit.select(hypotheses)
        logger.debug("select_branch[thompson]: selected %s", selected.id[:8] if selected else None)
        return selected

    # ── UCB / RL path (default) ────────────────────────────────────────────
    best_score = float("-inf")
    best_node = None

    for h in hypotheses:
        policy_est = q_values.get(h.id) if q_values else None
        score = ucb_score(h, total_experiments, c, memory, failure_penalty, policy_est)
        if score > best_score:
            best_score = score
            best_node = h

    logger.debug(
        "select_branch[%s]: best=%s (score=%.4f) from %d candidates",
        policy_type,
        best_node.id[:8] if best_node else None,
        best_score,
        len(hypotheses),
    )
    return best_node
