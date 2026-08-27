"""
AnalyzerAgent: interprets experiment results, creates Finding and
Claim nodes in the RDG, and updates the ECRM memory — v2.

v2 changes:
- ``memory.record_ntr()`` is called after every episode so the NTR
  detector accumulates real data (previously it was never called).
- ``strategy_id`` and baseline/achieved are forwarded from ``result``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from rdg.graph import ResearchDevelopmentGraph
from rdg.nodes import NodeStatus, RDGNode
from rdg.edges import EdgeRelation
from ecrm.memory_store import ECRMMemoryStore

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are a scientific analysis agent.
Given an experiment result, write a concise Finding and Claim.
Output JSON:
{
  "finding": "<observation about the result>",
  "claim": "<supported conclusion>",
  "supports_hypothesis": true/false,
  "failure_flags": ["Overfitting"|"Underfitting"|"Divergence"|"None"]
}"""


class AnalyzerAgent(BaseAgent):
    """Analyzes experiment outcomes and updates RDG + ECRM."""

    def __init__(self):
        super().__init__(name="AnalyzerAgent")

    def analyze(
        self,
        rdg: ResearchDevelopmentGraph,
        memory: ECRMMemoryStore,
        hypothesis_node: RDGNode,
        experiment_node: RDGNode,
        result: Dict[str, Any],
    ) -> Dict[str, RDGNode]:
        """
        Create Finding and Claim nodes, store in memory.
        Returns {"finding": <node>, "claim": <node>}.
        """
        score = result.get("score", 0.0)
        success = result.get("success", False)

        # LLM analysis
        user_prompt = (
            f"Hypothesis: {hypothesis_node.content}\n"
            f"Experiment: {experiment_node.content}\n"
            f"Result score: {score:.4f}, success: {success}\n"
            f"Train loss: {result.get('train_loss', '?')}, "
            f"Val loss: {result.get('val_loss', '?')}"
        )
        raw = self.llm_call(ANALYSIS_PROMPT, user_prompt, json_mode=True)
        parsed = self.parse_json_response(raw)

        finding_text = parsed.get("finding", f"Experiment scored {score:.4f}.")
        claim_text = parsed.get("claim", f"Model achieved {'good' if success else 'poor'} results.")
        failure_flags = [f for f in parsed.get("failure_flags", []) if f != "None"]

        # Create Finding node
        finding_node = RDGNode.finding(
            content=finding_text,
            score=score,
            attributes={
                "train_loss": result.get("train_loss", 0),
                "val_loss": result.get("val_loss", 0),
                "failure_flags": failure_flags,
            },
        )
        finding_node.status = NodeStatus.COMPLETED if success else NodeStatus.FAILED
        rdg.add_node(finding_node)
        rdg.connect(experiment_node.id, finding_node.id, EdgeRelation.PRODUCES)

        # Create Claim node
        claim_node = RDGNode.claim(
            content=claim_text,
            supported_by=[finding_node.id],
        )
        rdg.add_node(claim_node)
        rdg.connect(finding_node.id, claim_node.id, EdgeRelation.SUPPORTS)

        # Update hypothesis metrics
        if success:
            hypothesis_node.best_metric = max(
                hypothesis_node.best_metric or 0.0, score
            )
        hypothesis_node.times_tried += 1

        # Store in ECRM
        strategy_id = result.get("strategy_id", "")
        baseline = float(result.get("baseline", 0.0))
        memory.store(
            text=hypothesis_node.content,
            outcome={"score": score, "success": success,
                     "baseline": baseline,
                     "error": result.get("error", ""),
                     "strategy_id": strategy_id},
            link_node=hypothesis_node.id,
            failure_flags=failure_flags,
            context=result.get("memory_context", {}),
        )

        # v2: Record NTR so the NTR detector has real data.
        # ``used_memory`` is True when the hypothesis was informed by a
        # retrieved lesson (approximated by memory store being non-empty).
        if strategy_id:
            memory.record_ntr(
                strategy_id=strategy_id,
                used_memory=len(memory) > 0,
                baseline=baseline,
                achieved=score,
            )

        logger.info(
            "[AnalyzerAgent] Finding created (score=%.4f, flags=%s).",
            score, failure_flags,
        )
        return {"finding": finding_node, "claim": claim_node}
