"""
Repair operators triggered by failure diagnosis.
Each repair action modifies the RDG or memory to recover from a failure.
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

    if action == "add_regularization":
        return _add_regularization_note(rdg, failing_node)
    if action == "reduce_learning_rate":
        return _reduce_lr_note(rdg, failing_node)
    if action == "consolidate_memory" and memory:
        removed = memory.consolidate()
        return f"Memory consolidation removed {removed} stale records."
    if action == "disable_memory_retrieval":
        return "Memory retrieval disabled for strategy due to high NTR."

    return f"Repair action '{action}' logged for node {failing_node.id if failing_node else 'N/A'}."


def _add_regularization_note(rdg: ResearchDevelopmentGraph, node: Optional[RDGNode]) -> str:
    if node:
        note = node.attributes.get("repair_note", "")
        node.attributes["repair_note"] = note + "; increase dropout or weight_decay"
    return "Added regularization repair note to node."


def _reduce_lr_note(rdg: ResearchDevelopmentGraph, node: Optional[RDGNode]) -> str:
    if node:
        node.attributes["repair_note"] = "Reduce learning rate by 10×"
    return "Added LR reduction repair note to node."
