"""
Q-learning policy for RDG branch selection.

State: (hypothesis_id, context_features)
Action: which hypothesis to expand next
Reward: improvement in validation metric
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from rdg.nodes import RDGNode

logger = logging.getLogger(__name__)


class QLearningPolicy:
    """
    Tabular Q-learning over hypothesis nodes.

    Q(h) ← Q(h) + α * (r + γ * max_Q(h') - Q(h))
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        default_q: float = 0.5,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.default_q = default_q
        self.q: Dict[str, float] = {}

    def estimate_reward(self, node: RDGNode) -> float:
        return self.q.get(node.id, self.default_q)

    def update(
        self,
        node: RDGNode,
        reward: float,
        next_nodes: Optional[List[RDGNode]] = None,
    ) -> None:
        """Temporal-difference update."""
        current_q = self.q.get(node.id, self.default_q)
        if next_nodes:
            max_future_q = max(self.q.get(n.id, self.default_q) for n in next_nodes)
        else:
            max_future_q = 0.0

        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q[node.id] = new_q
        logger.debug("Q[%s] %.4f → %.4f (reward=%.4f)", node.id[:8], current_q, new_q, reward)

    def best_node(self, nodes: List[RDGNode]) -> Optional[RDGNode]:
        if not nodes:
            return None
        return max(nodes, key=lambda n: self.q.get(n.id, self.default_q))

    def state_dict(self) -> Dict:
        return {"q": self.q, "alpha": self.alpha, "gamma": self.gamma}

    def load_state_dict(self, d: Dict) -> None:
        self.q = d.get("q", {})
        self.alpha = d.get("alpha", self.alpha)
        self.gamma = d.get("gamma", self.gamma)
