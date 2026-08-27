"""Procedural lesson extraction for ECRM.

The purpose is not to ask an LLM to invent a scientific explanation.  It
captures a compact, auditable statement of *what was tried, in what context,
and what happened*.  This is more transferable than keeping a verbose agent
trajectory and makes memory review possible for a human researcher.
"""
from __future__ import annotations

from typing import Any, Dict

from ecrm.context import canonical_context


def derive_lesson(text: str, outcome: Dict[str, Any], context: Dict[str, Any] | None) -> str:
    """Produce a deterministic, compact procedural lesson from an episode."""
    ctx = canonical_context(context)
    conditions = ", ".join(f"{key}={value}" for key, value in sorted(ctx.items())) or "context=unspecified"
    score = float(outcome.get("score", 0.0))
    baseline = float(outcome.get("baseline", 0.0))
    success = bool(outcome.get("success", False))
    outcome_label = "improved" if success and score > baseline else "did not improve"
    return f"Action: {text.strip()} | Conditions: {conditions} | Outcome: {outcome_label} ({score:.4f} vs {baseline:.4f})."
