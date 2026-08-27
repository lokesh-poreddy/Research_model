"""Context compatibility for outcome-conditioned research memory.

Similarity alone is unsafe: an intervention that worked for one model, data
regime, or objective can be harmful in another.  This module keeps the first
version intentionally inspectable and deterministic.  A learned compatibility
model can later replace ``context_compatibility`` without changing the memory
store API.
"""
from __future__ import annotations

from typing import Any, Dict


CONTEXT_KEYS = (
    "domain",
    "task_id",
    "objective",
    "data_regime",
    "model_family",
    "modality",
    "constraint_profile",
)


def canonical_context(context: Dict[str, Any] | None) -> Dict[str, str]:
    """Keep only stable, non-empty context features used in reuse decisions."""
    context = context or {}
    return {
        key: str(context[key]).strip().lower()
        for key in CONTEXT_KEYS
        if context.get(key) not in (None, "")
    }


def context_compatibility(source: Dict[str, Any] | None, target: Dict[str, Any] | None) -> float:
    """Return compatibility in ``[0, 1]`` for a stored and current context.

    Unknown context is assigned a cautious 0.5, matching facts increase the
    score, and explicit contradictions reduce it sharply.  It is deliberately
    conservative: an old success should not dominate a new decision simply
    because the natural-language summaries look similar.
    """
    source_norm = canonical_context(source)
    target_norm = canonical_context(target)
    shared = [key for key in CONTEXT_KEYS if key in source_norm and key in target_norm]
    if not shared:
        return 0.5
    matches = sum(source_norm[key] == target_norm[key] for key in shared)
    conflicts = len(shared) - matches
    agreement = matches / len(shared)
    conflict_penalty = 0.35 ** conflicts
    return max(0.05, min(1.0, (0.2 + 0.8 * agreement) * conflict_penalty))
