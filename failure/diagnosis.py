"""
Failure Diagnosis Module.

Inspects RDG nodes and experiment results to classify failures
using the FailureTaxonomy, then returns a (category, node) tuple.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from rdg.nodes import NodeStatus, NodeType, RDGNode
from rdg.consistency import semantically_aligned
from rdg.graph import ResearchDevelopmentGraph
from failure.taxonomy import FailureCategory

logger = logging.getLogger(__name__)


def diagnose_failure(
    rdg: ResearchDevelopmentGraph,
    last_node: RDGNode,
    target_metric: float = 0.0,
) -> Tuple[FailureCategory, Optional[RDGNode]]:
    """
    Inspect the last RDG node and its evidence chain.
    Returns (FailureCategory, failing_node).

    Steps:
    1. Check for code/runtime errors (Experiment.status == "failed").
    2. Check semantic alignment of the evidence chain edges.
    3. Check if a Claim is unsupported.
    4. Check if a Finding's score is below target (LowPerformance).
    5. Fallback to UNKNOWN.
    """
    # ── Step 1: Execution error ───────────────────────────────────────────────
    if last_node.type == NodeType.EXPERIMENT:
        status = last_node.attributes.get("status", "")
        error_msg = last_node.attributes.get("error", "")
        if last_node.status == NodeStatus.FAILED or status in ("error", "failed"):
            if "timeout" in error_msg.lower():
                logger.warning("Diagnosed: TIMEOUT in %s", last_node.id)
                return (FailureCategory.TIMEOUT, last_node)
            logger.warning("Diagnosed: CODE_ERROR in %s", last_node.id)
            return (FailureCategory.CODE_ERROR, last_node)

    # ── Step 2: Semantic chain misalignment ───────────────────────────────────
    chain_edges = rdg.get_evidence_chain(last_node.id)
    for edge in chain_edges:
        src = rdg.get_node(edge.from_node)
        tgt = rdg.get_node(edge.to_node)
        if src and tgt and not semantically_aligned(src, tgt):
            logger.warning(
                "Semantic misalignment: %s <--> %s", src.id[:8], tgt.id[:8]
            )
            if src.type == NodeType.GAP and tgt.type == NodeType.HYPOTHESIS:
                return (FailureCategory.HYPOTHESIS_MISALIGNED, tgt)
            return (FailureCategory.EXPERIMENT_MISSPECIFIED, tgt)

    # ── Step 3: Unsupported Claim ─────────────────────────────────────────────
    if last_node.type == NodeType.CLAIM:
        supported_by = last_node.attributes.get("supported_by", [])
        if not supported_by:
            logger.warning("Diagnosed: UNSUPPORTED_CLAIM in %s", last_node.id)
            return (FailureCategory.UNSUPPORTED_CLAIM, last_node)

    # ── Step 4: Low performance ───────────────────────────────────────────────
    if last_node.type in (NodeType.FINDING, NodeType.CLAIM):
        score = last_node.attributes.get("score", None)
        if score is not None and float(score) < target_metric:
            logger.warning(
                "Diagnosed: LOW_PERFORMANCE in %s (score=%.4f < target=%.4f)",
                last_node.id, score, target_metric,
            )
            return (FailureCategory.LOW_PERFORMANCE, last_node)

    # ── Step 5: Check for overfitting / underfitting in attributes ────────────
    if last_node.type in (NodeType.FINDING,):
        train_loss = last_node.attributes.get("train_loss")
        val_loss = last_node.attributes.get("val_loss")
        if train_loss is not None and val_loss is not None:
            if val_loss > train_loss * 1.5:
                return (FailureCategory.OVERFITTING, last_node)
            if val_loss < train_loss * 0.5:
                return (FailureCategory.UNDERFITTING, last_node)

    return (FailureCategory.UNKNOWN, None)
