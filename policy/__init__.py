LEGACY_STATUS = {
    "canonical": False,
    "replacement": "researchforge.policy",
    "deprecated_since": "RF-1.0.0-alpha.2.1",
    "removal_target": None,  # preserved as historical/compatibility evidence
    "cross_imports_allowed": False,  # researchforge/ must never import from here
}

from policy.acquisition import select_branch, ucb_score
from policy.bandit import UCBBandit, ThompsonBandit
from policy.rl_policy import QLearningPolicy

__all__ = [
    "LEGACY_STATUS",
    "select_branch",
    "ucb_score",
    "UCBBandit",
    "ThompsonBandit",
    "QLearningPolicy",
]
