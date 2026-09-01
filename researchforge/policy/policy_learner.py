"""Policy Learner: chooses which evolution strategy ('branch') to try next.

Implements the UCB-style acquisition function from Sec. 4.1's
`select_branch` pseudocode:

    score = estimated_reward + c * sqrt(log(total_experiments) / (1 + times_tried))
    if memory.has_similar_failure(h): score *= 0.5

and an incremental action-value update in place of the design doc's
Sec. 4.4 Q-learning recursion (`current_q + alpha*(reward + gamma*max_future_q
- current_q)`). At the granularity this system operates at -- one strategy
choice per generation, with no meaningful "next state" beyond the choice
itself -- that recursion collapses to the standard multi-armed-bandit special
case of TD learning (drop the `gamma*max_future_q` term, since there is no
distinct future state to bootstrap from). That's a simplification of Sec.
4.4's general RL formulation, not a different algorithm: it's what Sec. 4.1's
own UCB formula already assumes.
"""
from __future__ import annotations
import math
import random
from typing import Callable, Dict, List, Optional


class PolicyLearner:
    def __init__(self, actions: List[str], alpha: float = 0.3, c: float = 1.0,
                 rng: Optional[random.Random] = None):
        self.actions = list(actions)
        self.alpha = alpha
        self.c = c
        self.q: Dict[str, float] = {a: 0.5 for a in actions}   # optimistic init
        self.times_tried: Dict[str, int] = {a: 0 for a in actions}
        self.total_trials = 0
        self.rng = rng or random.Random()

    def select_action(self, failure_check: Optional[Callable[[str], bool]] = None,
                       score_multiplier: Optional[Callable[[str], float]] = None) -> str:
        """failure_check(action) -> True if ECRM has seen a similar failure
        for this action; halves its acquisition score (Sec. 4.1). Separately,
        score_multiplier(action) -> a continuous [0,1]-ish factor (e.g. a
        context-conditioned success rate from memory.trajectory.TrajectoryMemory)
        multiplies the score directly -- a finer-grained alternative to the
        binary halving, used by the "trajectory_memory" RDE-Bench condition."""
        best_action, best_score = None, -math.inf
        for a in self.actions:
            reward = self.q[a]
            bonus = self.c * math.sqrt(math.log(self.total_trials + 1) / (1 + self.times_tried[a]))
            score = reward + bonus
            if failure_check is not None and failure_check(a):
                score *= 0.5
            if score_multiplier is not None:
                score *= score_multiplier(a)
            if score > best_score:
                best_score, best_action = score, a
        return best_action

    def select_random(self) -> str:
        """Uniform-random selection, used by the RDE-Bench random-search
        baseline -- no learning, no memory."""
        return self.rng.choice(self.actions)

    def update(self, action: str, reward: float) -> None:
        self.times_tried[action] += 1
        self.total_trials += 1
        self.q[action] += self.alpha * (reward - self.q[action])
