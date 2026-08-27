"""A compact quality-diversity portfolio for evolution operators.

Scalar best-score archives often converge on one familiar mutation type.  This
portfolio tracks each strategy family separately and selects an underexplored,
promising operator.  It is deliberately lightweight so that a later learned
policy can replace it while preserving the same evidence interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable


DEFAULT_STRATEGIES = {
    "param_mutation": "optimization",
    "optimizer_mutation": "optimization",
    "structure_add": "architecture",
    "structure_remove": "architecture",
    "augmentation_mutation": "data",
}


@dataclass
class StrategyEvidence:
    strategy_id: str
    family: str
    trials: int = 0
    successes: int = 0
    mean_improvement: float = 0.0
    failure_streak: int = 0

    def update(self, improvement: float, success: bool) -> None:
        self.trials += 1
        self.mean_improvement += (improvement - self.mean_improvement) / self.trials
        if success:
            self.successes += 1
            self.failure_streak = 0
        else:
            self.failure_streak += 1


@dataclass
class StrategyPortfolio:
    """Select strategies by empirical value, uncertainty, and family diversity."""
    strategies: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_STRATEGIES))
    exploration: float = 0.35
    saturation_penalty: float = 0.08
    evidence: Dict[str, StrategyEvidence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for strategy_id, family in self.strategies.items():
            self.evidence.setdefault(strategy_id, StrategyEvidence(strategy_id, family))

    def select(self) -> str:
        total = sum(item.trials for item in self.evidence.values())
        family_trials = self._family_trials()
        ranked = []
        for strategy_id, item in self.evidence.items():
            uncertainty = self.exploration * math.sqrt(math.log(total + 2) / (item.trials + 1))
            diversity = self.exploration / (1 + family_trials[item.family])
            saturation = self.saturation_penalty * item.failure_streak
            ranked.append((item.mean_improvement + uncertainty + diversity - saturation, strategy_id))
        # A deterministic tie break keeps benchmarks reproducible.
        return max(ranked, key=lambda pair: (pair[0], pair[1]))[1]

    def record(self, strategy_id: str, improvement: float, success: bool) -> None:
        self.evidence[strategy_id].update(improvement, success)

    def summary(self) -> Dict[str, Dict[str, float | int | str]]:
        return {
            strategy_id: {
                "family": item.family,
                "trials": item.trials,
                "successes": item.successes,
                "mean_improvement": item.mean_improvement,
                "failure_streak": item.failure_streak,
            }
            for strategy_id, item in self.evidence.items()
        }

    def _family_trials(self) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        for item in self.evidence.values():
            totals[item.family] = totals.get(item.family, 0) + item.trials
        return totals
