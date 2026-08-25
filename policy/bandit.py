"""
Multi-armed bandit policies for RDG branch selection.
Implements UCB1 and Thompson Sampling.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from rdg.nodes import RDGNode


class UCBBandit:
    """Upper Confidence Bound (UCB1) bandit over hypothesis nodes."""

    def __init__(self, c: float = 1.41):
        self.c = c
        self._counts: Dict[str, int] = {}
        self._values: Dict[str, float] = {}
        self._total: int = 0

    def select(self, nodes: List[RDGNode]) -> Optional[RDGNode]:
        if not nodes:
            return None
        # Explore any un-tried node first
        untried = [n for n in nodes if self._counts.get(n.id, 0) == 0]
        if untried:
            return random.choice(untried)

        scores = {
            n.id: (
                self._values.get(n.id, 0.0)
                + self.c * math.sqrt(math.log(self._total + 1) / (self._counts[n.id] + 1))
            )
            for n in nodes
        }
        best_id = max(scores, key=scores.__getitem__)
        return next(n for n in nodes if n.id == best_id)

    def update(self, node: RDGNode, reward: float) -> None:
        nid = node.id
        self._counts[nid] = self._counts.get(nid, 0) + 1
        old_val = self._values.get(nid, 0.0)
        self._values[nid] = old_val + (reward - old_val) / self._counts[nid]
        self._total += 1


class ThompsonBandit:
    """Thompson Sampling bandit (Beta distribution per arm)."""

    def __init__(self):
        # (alpha, beta) per hypothesis id
        self._params: Dict[str, List[float]] = {}

    def _get_params(self, node_id: str) -> List[float]:
        return self._params.setdefault(node_id, [1.0, 1.0])

    def select(self, nodes: List[RDGNode]) -> Optional[RDGNode]:
        if not nodes:
            return None
        samples = {}
        for n in nodes:
            a, b = self._get_params(n.id)
            samples[n.id] = random.betavariate(a, b)
        best_id = max(samples, key=samples.__getitem__)
        return next(n for n in nodes if n.id == best_id)

    def update(self, node: RDGNode, reward: float) -> None:
        """reward ∈ [0, 1] – treat as Bernoulli success probability."""
        reward_clipped = max(0.0, min(1.0, reward))
        params = self._get_params(node.id)
        params[0] += reward_clipped          # alpha += success
        params[1] += (1.0 - reward_clipped)  # beta  += failure
