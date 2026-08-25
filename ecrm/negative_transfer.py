"""
Negative Transfer Rate (NTR) detection.

NTR measures how often applying memory from past tasks
hurts performance on the current task.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NTRDetector:
    """
    Tracks outcomes of memory-retrieved experiments and detects
    when prior knowledge causes harm (negative transfer).
    """

    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: NTR above this value triggers a dampening flag.
        """
        self.threshold = threshold
        # strategy_id → list of (used_memory: bool, improvement: float)
        self._records: Dict[str, List[tuple]] = {}

    def record(
        self,
        strategy_id: str,
        used_memory: bool,
        baseline_score: float,
        achieved_score: float,
    ) -> None:
        """Log the outcome of a strategy that may have used memory."""
        improvement = achieved_score - baseline_score
        if strategy_id not in self._records:
            self._records[strategy_id] = []
        self._records[strategy_id].append((used_memory, improvement))

    def ntr_for_strategy(self, strategy_id: str) -> float:
        """
        NTR = (#memory_used AND worse) / (#memory_used total)
        """
        records = self._records.get(strategy_id, [])
        memory_used = [(used, imp) for used, imp in records if used]
        if not memory_used:
            return 0.0
        harmful = sum(1 for _, imp in memory_used if imp < 0)
        return harmful / len(memory_used)

    def global_ntr(self) -> float:
        """Aggregate NTR across all strategies."""
        all_memory_used = []
        for records in self._records.values():
            all_memory_used.extend([(u, i) for u, i in records if u])
        if not all_memory_used:
            return 0.0
        harmful = sum(1 for _, i in all_memory_used if i < 0)
        return harmful / len(all_memory_used)

    def is_harmful(self, strategy_id: str) -> bool:
        """Returns True if the strategy's NTR exceeds the threshold."""
        return self.ntr_for_strategy(strategy_id) > self.threshold

    def summary(self) -> Dict[str, float]:
        return {sid: self.ntr_for_strategy(sid) for sid in self._records}
