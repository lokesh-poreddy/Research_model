"""
BudgetAllocator — v2 compute-budget tracking and gating.

Tracks wall-clock seconds consumed per operator family and per strategy_id.
The controller queries ``remaining_fraction()`` to scale exploration:  when the
budget is nearly exhausted, the portfolio should favour exploitation.

Design rules (v2 protocol §"Promotion gates"):
- Budget is shared across every ablation rung under the same task.
- The allocator only *records* spend; the controller decides how to react.
- Blocking execution is a safety gate, not a policy.  ``is_over_budget()``
  returns a boolean; the caller logs and skips, it does not raise.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OperatorBudgetRecord:
    strategy_id: str
    family: str
    total_seconds: float = 0.0
    call_count: int = 0

    def add(self, seconds: float) -> None:
        self.total_seconds += seconds
        self.call_count += 1

    @property
    def hours(self) -> float:
        return self.total_seconds / 3600.0


class BudgetAllocator:
    """Track compute spend per operator and enforce the v2 budget ceiling.

    Args:
        budget_hours: Total allowed wall-clock hours for this experiment run.
        warn_fraction: Emit a warning when this fraction of budget is consumed.
    """

    def __init__(self, budget_hours: float = 10.0, warn_fraction: float = 0.8) -> None:
        self._budget_hours = budget_hours
        self._warn_fraction = warn_fraction
        self._records: Dict[str, OperatorBudgetRecord] = {}
        self._total_seconds: float = 0.0
        self._warned = False

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, strategy_id: str, family: str, seconds: float) -> None:
        """Record ``seconds`` of compute spent on ``strategy_id``."""
        if strategy_id not in self._records:
            self._records[strategy_id] = OperatorBudgetRecord(strategy_id, family)
        self._records[strategy_id].add(seconds)
        self._total_seconds += seconds
        self._check_warn()

    def _check_warn(self) -> None:
        fraction = self.consumed_fraction()
        if not self._warned and fraction >= self._warn_fraction:
            logger.warning(
                "BudgetAllocator: %.0f%% of compute budget consumed "
                "(%.2f / %.2f hours).  Switch to exploitation.",
                fraction * 100,
                self.consumed_hours,
                self._budget_hours,
            )
            self._warned = True

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def consumed_hours(self) -> float:
        return self._total_seconds / 3600.0

    def consumed_fraction(self) -> float:
        """Fraction of budget consumed in ``[0, 1]``."""
        if self._budget_hours <= 0:
            return 1.0
        return min(1.0, self.consumed_hours / self._budget_hours)

    def remaining_fraction(self) -> float:
        """Remaining budget as a fraction in ``[0, 1]``."""
        return max(0.0, 1.0 - self.consumed_fraction())

    def is_over_budget(self) -> bool:
        """Return True when the budget ceiling has been reached."""
        over = self.consumed_hours >= self._budget_hours
        if over:
            logger.warning(
                "BudgetAllocator: budget ceiling reached (%.2f hours). "
                "Sandbox execution will be skipped.",
                self._budget_hours,
            )
        return over

    def family_hours(self) -> Dict[str, float]:
        """Return total hours consumed per operator family."""
        totals: Dict[str, float] = {}
        for rec in self._records.values():
            totals[rec.family] = totals.get(rec.family, 0.0) + rec.hours
        return totals

    def summary(self) -> Dict[str, object]:
        return {
            "budget_hours": self._budget_hours,
            "consumed_hours": round(self.consumed_hours, 4),
            "remaining_fraction": round(self.remaining_fraction(), 4),
            "by_strategy": {
                sid: {"family": r.family, "hours": round(r.hours, 4), "calls": r.call_count}
                for sid, r in self._records.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"BudgetAllocator(consumed={self.consumed_hours:.3f}h / "
            f"{self._budget_hours:.1f}h, "
            f"remaining={self.remaining_fraction():.1%})"
        )
