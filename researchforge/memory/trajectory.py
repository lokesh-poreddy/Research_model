"""Trajectory-based contextual memory: an alternative to the flat, strategy-
keyed ECRM in memory/ecrm.py that instead conditions retrieval on the actual
state of the genome a strategy was applied to, not just its name and model
family.

This exists to genuinely test an idea (call it "does context-aware memory
retrieval beat flat memory retrieval") rather than to assert that it works.
The flat ECRM's has_similar_failure() already conditions on (strategy,
model_type, task) -- Section 4's `_mem_key()`. This module adds one further
axis: a coarse, model-type-specific *capacity regime* bucket ("low" /
"medium" / "high"), computed from the genome's actual hyperparameters/
architecture, so a record answers "did increase_capacity fail on an
MLPClassifier that was ALREADY high-capacity" rather than only "did
increase_capacity ever fail on an MLPClassifier."

Every record also carries the RDG node ids (hypothesis/experiment/finding)
it came from, so a trajectory's full provenance chain (Problem -> Gap ->
Hypothesis -> Experiment -> Finding[-> Claim]) can be reconstructed with
rdg.evidence_chain() -- the graph doesn't need a new schema for this, the
existing typed edges already encode the chain; this module just remembers
which finding a given piece of contextual experience came from.

Whether this actually beats the flat ECRM is an empirical question answered
in benchmarks/rde_bench.py's "trajectory_memory" condition, not assumed here.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..genome.model_genome import ModelGenome

CapacityBucket = str  # "low" | "medium" | "high" | "unknown"
Stage = str           # "baseline" | "early" | "mid" | "late"


def capacity_bucket(genome: ModelGenome) -> CapacityBucket:
    """A coarse, model-family-specific capacity regime for the genome's
    *current* architecture/hyperparameters. Thresholds are set relative to
    each family's default (genome.model_genome.DEFAULT_GENOMES) and the
    ranges the evolution operators in genome/operators.py actually reach."""
    if genome.model_type == "MLPClassifier":
        total_units = sum(genome.architecture.get("hidden_layer_sizes", [64]))
        if total_units < 64:
            return "low"
        if total_units <= 192:
            return "medium"
        return "high"
    if genome.model_type == "RandomForestClassifier":
        n_estimators = genome.architecture.get("n_estimators", 100)
        if n_estimators < 150:
            return "low"
        if n_estimators <= 350:
            return "medium"
        return "high"
    if genome.model_type in ("SVC", "LogisticRegression"):
        c = genome.hyperparameters.get("C", 1.0)
        if c < 0.5:
            return "low"
        if c <= 5.0:
            return "medium"
        return "high"
    return "unknown"


def generation_stage(generation: int, total_generations: int) -> Stage:
    if generation < 0:
        return "baseline"
    frac = generation / max(1, total_generations - 1)
    if frac < 0.34:
        return "early"
    if frac < 0.67:
        return "mid"
    return "late"


@dataclass
class TrajectoryRecord:
    id: str
    generation: int
    stage: Stage
    problem_context: str
    parent_model_type: str
    parent_capacity_bucket: CapacityBucket
    strategy: str
    child_model_type: str
    child_capacity_bucket: CapacityBucket
    metric: float
    success: bool
    failure: str
    hypothesis_id: str = ""
    experiment_id: str = ""
    finding_id: str = ""
    created_at: float = field(default_factory=time.time)


class TrajectoryMemory:
    """Stores TrajectoryRecords and answers context-conditioned queries.
    Deliberately a separate, independent memory implementation from
    memory.ecrm.ECRM (rather than a subclass) so the two can be benchmarked
    against each other cleanly as alternative "full-memory" conditions."""

    def __init__(self):
        self.records: List[TrajectoryRecord] = []

    def store(self, record: TrajectoryRecord) -> None:
        self.records.append(record)

    def _matches(self, strategy: str, parent_model_type: str,
                 parent_bucket: CapacityBucket) -> List[TrajectoryRecord]:
        return [r for r in self.records
                if r.strategy == strategy
                and r.parent_model_type == parent_model_type
                and r.parent_capacity_bucket == parent_bucket]

    def contextual_success_rate(self, strategy: str, parent_model_type: str,
                                 parent_bucket: CapacityBucket,
                                 min_samples: int = 2, default: float = 0.6) -> float:
        """P(success | strategy, model family, capacity regime). Falls back
        to `default` (a mildly optimistic prior, matching the policy
        learner's own optimistic Q-value initialisation) until enough
        context-matched evidence has accumulated to trust the empirical rate."""
        matches = self._matches(strategy, parent_model_type, parent_bucket)
        if len(matches) < min_samples:
            return default
        return sum(1 for m in matches if m.success) / len(matches)

    def contextual_mean_metric(self, strategy: str, parent_model_type: str,
                                parent_bucket: CapacityBucket,
                                min_samples: int = 2) -> Optional[float]:
        matches = self._matches(strategy, parent_model_type, parent_bucket)
        if len(matches) < min_samples:
            return None
        return sum(m.metric for m in matches) / len(matches)

    def similar_trajectory_recently_failed(self, strategy: str, parent_model_type: str,
                                            parent_bucket: CapacityBucket,
                                            window: int = 3) -> bool:
        """True if a majority of the most recent context-matched trajectories
        failed -- the trajectory-memory analogue of ECRM.has_similar_failure(),
        but conditioned on capacity regime as well as strategy/model family."""
        matches = self._matches(strategy, parent_model_type, parent_bucket)
        if not matches:
            return False
        recent = matches[-window:]
        return sum(1 for m in recent if not m.success) > len(recent) / 2

    def evidence_chain_for(self, trajectory_id: str, rdg) -> list:
        """Reconstructs the full RDG provenance chain (Problem -> ... ->
        Finding) a given trajectory came from, using the RDG's own
        evidence_chain() -- no separate graph schema needed."""
        rec = next((r for r in self.records if r.id == trajectory_id), None)
        if rec is None or not rec.finding_id:
            return []
        return rdg.evidence_chain(rec.finding_id)

    def stats(self) -> Dict[str, int]:
        contexts = {(r.strategy, r.parent_model_type, r.parent_capacity_bucket)
                    for r in self.records}
        return {"total_trajectories": len(self.records), "distinct_contexts": len(contexts)}


def new_trajectory_id() -> str:
    return f"traj_{uuid.uuid4().hex[:10]}"
