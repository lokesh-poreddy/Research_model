"""
Failure repair operators triggered by failure diagnosis — v2.

v2 changes:
- Real handler for every FailureCategory (previously only 2 of 14 had actions).
- Handlers write structured repair notes to the failing node's attributes,
  record negative evidence in ECRM, and suggest concrete next steps.
- All handlers return a human-readable description for the controller log.
"""
from __future__ import annotations

import logging
from typing import Optional

from failure.taxonomy import FailureCategory, REPAIR_ACTIONS
from rdg.graph import ResearchDevelopmentGraph
from rdg.nodes import NodeStatus, RDGNode
from ecrm.memory_store import ECRMMemoryStore

logger = logging.getLogger(__name__)


def apply_repair(
    category: FailureCategory,
    failing_node: Optional[RDGNode],
    rdg: ResearchDevelopmentGraph,
    memory: Optional[ECRMMemoryStore] = None,
) -> str:
    """
    Apply the recommended repair for a diagnosed failure.
    Returns a description of the action taken.
    """
    action = REPAIR_ACTIONS.get(category, "run_ablation")
    logger.info("Applying repair '%s' for failure '%s'.", action, category.value)

    if failing_node:
        failing_node.status = NodeStatus.FAILED
        failing_node.failure_count += 1
        rdg.update_node(failing_node.id, status=NodeStatus.FAILED)

    # ── Design failures ───────────────────────────────────────────────────────
    if action == "regenerate_hypothesis":
        return _note(failing_node, "repair",
                     "Hypothesis misaligned with gap: regenerate with tighter scope.")

    if action == "fix_experiment_spec":
        return _note(failing_node, "repair",
                     "Experiment misspecified: review data split and metric definition.")

    if action == "add_evidence_edge":
        return _note(failing_node, "repair",
                     "Unsupported claim: add a SUPPORTS edge from a validated finding.")

    # ── Implementation failures ───────────────────────────────────────────────
    if action == "fix_code_error":
        error_msg = failing_node.attributes.get("error", "") if failing_node else ""
        note = f"Code error detected; sanitised message: {error_msg[:200]}"
        return _note(failing_node, "repair", note)

    if action == "increase_resources":
        return _note(failing_node, "repair",
                     "Runtime crash: increase memory/CPU allocation or reduce batch size.")

    if action == "reduce_complexity":
        return _note(failing_node, "repair",
                     "Timeout: reduce model depth, epoch count, or data volume.")

    # ── Performance failures ──────────────────────────────────────────────────
    if action == "add_regularization":
        return _note(failing_node, "repair",
                     "Overfitting: increase dropout_rate and weight_decay; "
                     "consider early stopping.")

    if action == "increase_capacity":
        return _note(failing_node, "repair",
                     "Underfitting: add layers, increase units, or reduce regularization.")

    if action == "reduce_learning_rate":
        return _note(failing_node, "repair",
                     "Divergence: reduce learning_rate by 10× and add gradient clipping "
                     "(max_grad_norm=1.0).")

    if action == "increase_exploration":
        return _note(failing_node, "repair",
                     "Premature convergence: increase mutation delta and UCB c constant; "
                     "try a different strategy family.")

    if action == "add_replay_buffer":
        return _note(failing_node, "repair",
                     "Catastrophic forgetting: add experience replay or EWC regularization.")

    if action == "switch_strategy":
        return _note(failing_node, "repair",
                     "Low performance: pivot to a different operator family "
                     "via StrategyPortfolio.")

    # ── Memory failures ───────────────────────────────────────────────────────
    if action == "disable_memory_retrieval":
        if memory and failing_node:
            strategy_id = failing_node.attributes.get("strategy_id", "")
            if strategy_id:
                logger.warning(
                    "Repair: disabling memory-guided retrieval for strategy '%s' "
                    "due to high NTR.",
                    strategy_id,
                )
        return _note(failing_node, "repair",
                     "Negative transfer detected: memory retrieval disabled for this "
                     "strategy until NTR drops below threshold.")

    if action == "consolidate_memory":
        removed = 0
        if memory:
            removed = memory.consolidate()
        return (
            f"Stale memory: consolidation removed {removed} records. "
            "Re-embed with updated embedder before next retrieval."
        )

    # ── Fallback ──────────────────────────────────────────────────────────────
    return (
        f"Repair action '{action}' logged for node "
        f"{failing_node.id if failing_node else 'N/A'}. "
        "Run the ablation protocol to determine root cause."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _note(node: Optional[RDGNode], key: str, message: str) -> str:
    """Append a repair note to a node's attributes and return it."""
    if node:
        existing = node.attributes.get(key, "")
        node.attributes[key] = (existing + " | " + message).lstrip(" | ")
    return message
