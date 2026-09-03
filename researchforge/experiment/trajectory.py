"""researchforge/experiment/trajectory.py — Trajectory fingerprinting.

RF-1.0.0-alpha.2.1: Introduced as a formal, reusable mechanism for
capturing the deterministic trajectory of a ResearchController run.

Purpose
-------
A TrajectoryFingerprint allows:
    1. Regression testing: same seed → same trajectory hash before and after
       architectural changes (e.g. ModelGenome → TMG migration in alpha.2.1).
    2. VRDEG provenance (alpha.3): each node in the persistent research graph
       will carry the trajectory hash up to that generation.
    3. Reproducibility claims: two independent runs with the same fingerprint
       produced identical decisions at every generation.

Usage
-----
    fingerprint = compute_trajectory_fingerprint(result)
    assert fingerprint.final_hash == expected_hash
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryFingerprint:
    """Complete deterministic fingerprint of a ResearchController run.

    Captures per-generation decisions, not just the final metric — so
    architectural changes that preserve the final result but alter the
    intermediate path will still be detected.

    Fields
    ------
    generation_hashes : list of str
        sha256 of the content-deterministic trial signatures up to each generation.
    operator_sequence : list of str
        The strategy/operator chosen at each generation (including baseline=gen -1).
    metric_sequence : list of float
        Experiment metric at each generation.
    best_metric : float
        Best metric at end of run.
    final_hash : str
        sha256 of all the above combined — the single canonical fingerprint.
    """
    generation_hashes: List[str] = field(default_factory=list)
    operator_sequence: List[str] = field(default_factory=list)
    metric_sequence: List[float] = field(default_factory=list)
    best_metric: float = 0.0
    final_hash: str = ""

    def __post_init__(self) -> None:
        if not self.final_hash and self.operator_sequence:
            self.final_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps({
            "generation_hashes": self.generation_hashes,
            "operator_sequence": self.operator_sequence,
            "metric_sequence": [round(m, 8) for m in self.metric_sequence],
            "best_metric": round(self.best_metric, 8),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def matches(self, other: "TrajectoryFingerprint") -> bool:
        """True iff the two trajectories are bitwise-equivalent."""
        return self.final_hash == other.final_hash


def compute_trajectory_fingerprint(result: Any) -> "TrajectoryFingerprint":
    """Compute a TrajectoryFingerprint from a RunResult.

    Parameters
    ----------
    result : RunResult
        Result object from ResearchController.run().

    Returns
    -------
    TrajectoryFingerprint
        Deterministic fingerprint of the run trajectory.
    """
    operator_sequence: List[str] = []
    metric_sequence: List[float] = []

    for trial in result.trials:
        operator_sequence.append(trial.strategy)
        metric_sequence.append(trial.metric)

    generation_hashes: List[str] = []
    trail_signatures: List[str] = []
    for trial in result.trials:
        # We hash the deterministic trial signature: (gen, strategy, model_type, metric, best_so_far, failure)
        sig = f"{trial.generation}:{trial.strategy}:{trial.model_type}:{round(trial.metric, 6)}:{round(trial.best_so_far, 6)}:{trial.failure}"
        trail_signatures.append(sig)
        gen_hash = hashlib.sha256(
            json.dumps(trail_signatures, sort_keys=True).encode()
        ).hexdigest()[:16]
        generation_hashes.append(gen_hash)

    fp = TrajectoryFingerprint(
        generation_hashes=generation_hashes,
        operator_sequence=operator_sequence,
        metric_sequence=metric_sequence,
        best_metric=result.best_metric,
    )
    fp.final_hash = fp._compute_hash()
    return fp
