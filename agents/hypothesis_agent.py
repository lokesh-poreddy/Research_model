"""
HypothesisAgent: generates new research hypotheses from Gap nodes,
incorporating retrieved literature and ECRM memory.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from agents.base_agent import BaseAgent
from rdg.nodes import RDGNode
from ecrm.memory_store import ECRMMemoryStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a rigorous scientific research assistant specializing in machine learning.
Your task is to generate a novel, testable hypothesis from a research gap.
The hypothesis must:
1. Be specific and directly address the research gap.
2. Be falsifiable through an experiment.
3. Avoid ideas already in memory (negative transfer awareness).
Format your response as JSON:
{"hypothesis": "<one-sentence hypothesis>", "rationale": "<brief justification>"}
"""


class HypothesisAgent(BaseAgent):
    """
    Generates RDG Hypothesis nodes from Gap nodes.
    Uses ECRM memory to avoid repeating failed approaches.
    """

    def __init__(self):
        super().__init__(name="HypothesisAgent")

    def generate(
        self,
        gap_node: RDGNode,
        memory: Optional[ECRMMemoryStore] = None,
        prior_hypotheses: Optional[List[str]] = None,
        strategy_hint: str = "",
    ) -> str:
        """
        Generate a hypothesis for the given Gap node.

        Args:
            gap_node: The research Gap to address.
            memory: ECRM store for negative evidence lookup.
            prior_hypotheses: Already-tried hypotheses to avoid.
            strategy_hint: Optional strategy description from SeaEvo-style memory.

        Returns:
            A hypothesis string.
        """
        # Retrieve similar past experiments from memory
        memory_context = ""
        if memory:
            results = memory.retrieve(gap_node.content, top_k=3)
            if results:
                memory_context = "Past experiments (avoid repeating failures):\n"
                for rec, sim in results:
                    success_label = "✓" if rec.outcome.get("success") else "✗"
                    memory_context += (
                        f"  {success_label} [{sim:.2f}] {rec.text} "
                        f"→ score={rec.outcome.get('score', '?')}\n"
                    )

        prior_str = ""
        if prior_hypotheses:
            prior_str = "Previously tried hypotheses (do not repeat):\n" + "\n".join(
                f"  - {h}" for h in prior_hypotheses
            )

        user_prompt = (
            f"Research Gap: {gap_node.content}\n\n"
            f"{memory_context}\n"
            f"{prior_str}\n"
            f"Strategy hint: {strategy_hint}\n\n"
            "Generate a NEW hypothesis that addresses this gap."
        )

        raw = self.llm_call(SYSTEM_PROMPT, user_prompt, json_mode=True)
        parsed = self.parse_json_response(raw)
        hypothesis = parsed.get("hypothesis", raw)
        logger.info("[HypothesisAgent] Generated: %s", hypothesis[:80])
        return hypothesis
