"""
HypothesisAgent: generates new research hypotheses from Gap nodes — v2.

v2 changes:
- Retrieval uses ``memory.retrieve_for_context()`` (context-conditioned, NTR-weighted)
  instead of the flat similarity-only ``memory.retrieve()``.
- ``strategy_hint`` from the StrategyPortfolio is injected into the system prompt
  so the LLM receives operator-family guidance, not only raw text.
- A ``_build_evidence_block()`` helper formats retrieved lessons consistently.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from rdg.nodes import RDGNode
from ecrm.memory_store import ECRMMemoryStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are a rigorous scientific research assistant specializing in machine learning.
Your task is to generate a novel, testable hypothesis from a research gap.

Current strategy family: {strategy_hint}

The hypothesis must:
1. Be specific and directly address the research gap.
2. Be falsifiable through an experiment.
3. Be consistent with the strategy family above (prefer {strategy_hint} interventions).
4. Avoid ideas already in memory (negative transfer awareness).

Format your response as JSON:
{{"hypothesis": "<one-sentence hypothesis>", "rationale": "<brief justification>"}}
"""


class HypothesisAgent(BaseAgent):
    """
    Generates RDG Hypothesis nodes from Gap nodes.
    Uses ECRM context-conditioned memory to avoid repeating failed approaches
    and to surface compatible prior successes.
    """

    def __init__(self) -> None:
        super().__init__(name="HypothesisAgent")

    def generate(
        self,
        gap_node: RDGNode,
        memory: Optional[ECRMMemoryStore] = None,
        prior_hypotheses: Optional[List[str]] = None,
        strategy_hint: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a hypothesis for the given Gap node.

        Args:
            gap_node: The research Gap to address.
            memory: ECRM store for context-conditioned negative-evidence lookup.
            prior_hypotheses: Already-tried hypotheses to avoid.
            strategy_hint: Strategy family selected by StrategyPortfolio
                           (e.g. "optimization", "architecture", "data").
            context: Current task context dict for context-conditioned retrieval.

        Returns:
            A hypothesis string.
        """
        # Build system prompt with strategy guidance injected
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            strategy_hint=strategy_hint or "general"
        )

        # Retrieve context-conditioned evidence (v2 path: NTR-weighted + freshness)
        memory_block = ""
        if memory:
            evidence = memory.retrieve_for_context(
                gap_node.content, context=context, top_k=4
            )
            memory_block = self._build_evidence_block(evidence)

        prior_str = ""
        if prior_hypotheses:
            prior_str = (
                "Previously tried hypotheses (do not repeat):\n"
                + "\n".join(f"  - {h}" for h in prior_hypotheses)
            )

        user_prompt = (
            f"Research Gap: {gap_node.content}\n\n"
            f"{memory_block}\n"
            f"{prior_str}\n"
            "Generate a NEW hypothesis that addresses this gap, "
            f"preferably using {strategy_hint} techniques."
        )

        raw = self.llm_call(system_prompt, user_prompt, json_mode=True)
        parsed = self.parse_json_response(raw)
        hypothesis = parsed.get("hypothesis", raw)
        logger.info("[HypothesisAgent] Generated: %s", hypothesis[:80])
        return hypothesis

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_evidence_block(
        evidence: List,  # List[Tuple[MemoryRecord, float]]
    ) -> str:
        """Format context-conditioned memory evidence for the LLM prompt."""
        if not evidence:
            return ""
        lines = ["Context-conditioned memory evidence (use successes; avoid failures):"]
        for rec, weight in evidence:
            success_label = "✓ SUCCESS" if rec.outcome.get("success") else "✗ FAILURE"
            score = rec.outcome.get("score", "?")
            lesson = rec.lesson or rec.text
            lines.append(
                f"  [{success_label} | weight={weight:.3f}] {lesson} "
                f"→ score={score}"
            )
        return "\n".join(lines)
