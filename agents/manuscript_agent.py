"""
ManuscriptAgent: generates a research summary / mini-paper
from RDG claims and memory evidence.
"""
from __future__ import annotations

import logging
from typing import Optional

from agents.base_agent import BaseAgent
from rdg.graph import ResearchDevelopmentGraph
from rdg.nodes import NodeType
from ecrm.memory_store import ECRMMemoryStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a scientific writing assistant.
Given the research findings from an autonomous agent's research loop,
write a concise research summary with sections:
- Abstract
- Key Findings
- Best Performing Configuration
- Conclusions and Future Work

Be precise and cite the evidence chains."""


class ManuscriptAgent(BaseAgent):
    """Generates research summaries from RDG claims."""

    def __init__(self):
        super().__init__(name="ManuscriptAgent")

    def write_summary(
        self,
        rdg: ResearchDevelopmentGraph,
        memory: ECRMMemoryStore,
        problem: str,
        best_score: float,
    ) -> str:
        """Generate a markdown research summary."""
        claims = rdg.claims
        findings = rdg.findings

        claim_text = "\n".join(
            f"- {c.content}" for c in claims[:10]
        ) or "No validated claims yet."

        finding_text = "\n".join(
            f"- {f.content} (score={f.attributes.get('score', '?'):.3f})"
            for f in sorted(findings, key=lambda n: n.attributes.get("score", 0), reverse=True)[:5]
        ) or "No findings yet."

        mem_stats = memory.stats()
        rdg_stats = rdg.stats()

        user_prompt = (
            f"Problem: {problem}\n"
            f"Best Score Achieved: {best_score:.4f}\n\n"
            f"Top Claims:\n{claim_text}\n\n"
            f"Top Findings:\n{finding_text}\n\n"
            f"Research Graph Stats: {rdg_stats}\n"
            f"Memory Stats: {mem_stats}\n\n"
            "Write the research summary."
        )
        return self.llm_call(SYSTEM_PROMPT, user_prompt)
