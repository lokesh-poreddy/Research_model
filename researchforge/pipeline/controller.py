"""ResearchController: ties the Research Development Graph, Model Genomes,
ECRM, Policy Learner, Failure Diagnosis, and Discovery Pipeline into the
research loop described in ResearchForge-ECRM Sec. 1 / Sec. 9's architecture
diagrams:

    select branch -> synthesize genome -> run experiment -> diagnose
    -> update memory -> update policy -> repeat

Four `condition`s implement the RDE-Bench ablation ladder:

  - "full":              policy learner (bandit) + flat ECRM (memory.ecrm) +
                          a strategy/model-family-conditioned failure check
                          -- the original complete system (Sec. 6/8).
  - "trajectory_memory":  policy learner + memory.trajectory.TrajectoryMemory
                          instead of the flat ECRM: retrieval is additionally
                          conditioned on the parent genome's actual capacity
                          regime (not just its model family), and the policy
                          score is scaled by a continuous contextual success
                          rate rather than halved by a binary failure flag.
                          A genuine alternative memory design, benchmarked
                          against "full" rather than assumed superior to it.
  - "no_memory":         policy learner still adapts within the run (so
                          short-horizon learning is not the thing being
                          measured), but no cross-experiment memory of any
                          kind -- isolates the *memory* contribution, full
                          stop, regardless of which memory design is used.
  - "random":            uniform-random strategy choice, no learning of any
                          kind -- the naive baseline ("LLM-only"/random-search
                          in the design doc's Sec. 6 baselines).
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..rdg.graph import ResearchDevelopmentGraph
from ..genome.model_genome import ModelGenome
from ..genome.operators import STRATEGIES
from ..memory.ecrm import ECRM
from ..memory.trajectory import (
    TrajectoryMemory, TrajectoryRecord, capacity_bucket, generation_stage, new_trajectory_id,
)
from ..policy.policy_learner import PolicyLearner
from ..diagnosis.failure_taxonomy import diagnose, ExperimentResult, FailureCategory
from ..evaluators.sklearn_evaluator import evaluate_genome
from .discovery import HeuristicSynthesizer, unit_test
from ..benchmarks.tasks import Task

CONDITIONS = ("full", "trajectory_memory", "no_memory", "random")


@dataclass
class TrialRecord:
    generation: int
    strategy: str
    model_type: str
    metric: float
    best_so_far: float
    failure: str
    used_memory: bool
    memory_negative_transfer: bool
    genome_id: str


@dataclass
class RunResult:
    task_name: str
    condition: str
    trials: List[TrialRecord] = field(default_factory=list)
    best_genome: Optional[ModelGenome] = None
    best_metric: float = 0.0
    rdg_stats: dict = field(default_factory=dict)
    memory_half_life_days: float = float("nan")
    trajectory_stats: Dict[str, int] = field(default_factory=dict)
    wall_time_s: float = 0.0


class ResearchController:
    def __init__(self, task: Task, condition: str = "full", seed: int = 0,
                 population_size: int = 6, initial_model_type: str = "LogisticRegression",
                 use_sandbox: bool = False, sandbox_timeout_s: float = 15.0):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        self.task = task
        self.condition = condition
        self.seed = seed
        self.rng = random.Random(seed)
        self.rdg = ResearchDevelopmentGraph()
        self.ecrm = ECRM(decay_lambda=0.08, retention_threshold=0.12)
        self.trajectory_memory = TrajectoryMemory()
        self.policy = PolicyLearner(STRATEGIES, rng=self.rng)
        self.synth = HeuristicSynthesizer()
        self.population_size = population_size
        # "memory_enabled" = uses SOME cross-experiment memory (either design);
        # the two flags below select WHICH design, mutually exclusive by construction.
        self.memory_enabled = condition in ("full", "trajectory_memory")
        self.use_flat_memory = condition == "full"
        self.use_trajectory_memory = condition == "trajectory_memory"
        self.use_policy = condition in ("full", "trajectory_memory", "no_memory")
        self._failed_signatures = set()
        self.use_sandbox = use_sandbox
        self._sandbox = None
        if use_sandbox:
            from ..safety.sandbox import SafeRunner, ResourceBudget
            self._sandbox = SafeRunner(ResourceBudget(per_experiment_timeout_s=sandbox_timeout_s))

        self.problem = self.rdg.add_node(
            "Problem", f"Improve {task.metric_fn.__name__} on {task.name}")
        self.gap = self.rdg.add_node(
            "Gap", f"No model yet reaches target {task.target_metric} on {task.name}")
        self.rdg.add_edge(self.problem.id, self.gap.id, "identifies")

        base = ModelGenome.default(initial_model_type, seed=seed)
        base._score = -1.0
        self.population: List[ModelGenome] = [base]

    # ------------------------------------------------------------------
    def _run_experiment(self, genome: ModelGenome) -> ExperimentResult:
        """Runs evaluate_genome() directly by default (fast, matches the
        RDE-Bench numbers this repo reports); set use_sandbox=True at
        construction to route every experiment through
        safety.sandbox.SafeRunner instead -- real process isolation, a hard
        per-experiment timeout, and a tracked budget, at the cost of one
        process fork per experiment. Recommended whenever genomes might come
        from a less-trusted source (e.g. a wired-up LLMSynthesizer) rather
        than this repo's own bounded evolution operators."""
        if self._sandbox is not None:
            from ..safety.sandbox import SafetyStatus
            outcome = self._sandbox.run(
                evaluate_genome, genome, self.task.X_train, self.task.y_train,
                self.task.X_val, self.task.y_val, self.task.metric_fn,
                target=self.task.target_metric)
            if outcome.status == SafetyStatus.OK:
                return outcome.value
            return ExperimentResult(metric=0.0, success=False,
                                     exception=f"{outcome.status.value}: {outcome.error}",
                                     target=self.task.target_metric)
        return evaluate_genome(genome, self.task.X_train, self.task.y_train,
                                self.task.X_val, self.task.y_val,
                                self.task.metric_fn, target=self.task.target_metric)

    def _mem_key(self, strategy: str, model_type: str) -> str:
        """Compact (strategy, model_type, task) descriptor used as the ECRM
        text key. Deliberately terse rather than a full sentence: with the
        offline hashed bag-of-words embedding (memory/embeddings.py), a
        template sentence like "Improve on genome derived from X for Y via Z"
        shares 6+ boilerplate words across *every* record, which swamps the
        2-3 words that actually distinguish one context from another (cosine
        similarity stays >=0.7 even for unrelated strategy/model pairs). A
        bare `"{strategy} {model_type} {task}"` key gives clean, interpretable
        similarity: 1.0 for an exact repeat, ~0.67 for a one-token difference,
        0.0 for no overlap -- so `has_similar_failure`'s threshold actually
        means something. A production build using a real semantic encoder
        would not need this workaround; free-text hypotheses would already
        separate cleanly in embedding space."""
        return f"{strategy} {model_type} {self.task.name}"

    def _select_strategy(self, parent: ModelGenome) -> Tuple[str, bool]:
        if self.condition == "random":
            return self.policy.select_random(), False
        if self.condition == "no_memory":
            return self.policy.select_action(), False
        if self.condition == "trajectory_memory":
            parent_bucket = capacity_bucket(parent)
            multiplier = lambda a: 0.3 + 0.7 * self.trajectory_memory.contextual_success_rate(
                a, parent.model_type, parent_bucket)
            return self.policy.select_action(score_multiplier=multiplier), True
        # condition == "full"
        failure_check = lambda a: self.ecrm.has_similar_failure(
            self._mem_key(a, parent.model_type), threshold=0.9)
        return self.policy.select_action(failure_check=failure_check), True

    def _record_trial(self, result: RunResult, generation: int, strategy: str,
                       model_type: str, exp_result: ExperimentResult, best_metric: float,
                       failure: FailureCategory, used_memory: bool,
                       neg_transfer: bool, genome_id: str) -> None:
        result.trials.append(TrialRecord(
            generation=generation, strategy=strategy, model_type=model_type,
            metric=exp_result.metric, best_so_far=best_metric, failure=failure.value,
            used_memory=used_memory, memory_negative_transfer=neg_transfer, genome_id=genome_id))

    # ------------------------------------------------------------------
    def run(self, n_generations: int = 25) -> RunResult:
        t0 = time.time()
        result = RunResult(task_name=self.task.name, condition=self.condition)

        # -- baseline: evaluate the seed genome before any search (generation -1)
        base = self.population[0]
        base_result = self._run_experiment(base)
        base_failure = diagnose(base_result)
        base._score = base_result.metric if base_result.success else -1.0
        best_metric = base._score
        best_genome = base if base_result.success else None

        hyp0 = self.rdg.add_node(
            "Hypothesis", f"Baseline {base.model_type} for {self.task.name}",
            attributes={"strategy": "baseline", "generation": -1})
        self.rdg.add_edge(self.gap.id, hyp0.id, "motivates")
        exp0 = self.rdg.add_node(
            "Experiment", f"Train/evaluate baseline {base.model_type}",
            attributes={"genome_id": base.model_id})
        self.rdg.add_edge(hyp0.id, exp0.id, "tested-by")
        finding0 = self.rdg.add_node(
            "Finding", f"metric={base_result.metric:.4f} failure={base_failure.value}",
            attributes={"metric": base_result.metric, "failure": base_failure.value})
        self.rdg.add_edge(exp0.id, finding0.id, "produces")
        if self.use_flat_memory:
            self.ecrm.store(text_summary=self._mem_key("baseline", base.model_type),
                             context={"task": self.task.name, "genome": base.to_dict()},
                             outcome={"metric": base_result.metric, "success": base_result.success,
                                      "failure": base_failure.value},
                             strategy="baseline")
        elif self.use_trajectory_memory:
            base_bucket = capacity_bucket(base)
            self.trajectory_memory.store(TrajectoryRecord(
                id=new_trajectory_id(), generation=-1, stage="baseline",
                problem_context=self.gap.content,
                parent_model_type=base.model_type, parent_capacity_bucket=base_bucket,
                strategy="baseline",
                child_model_type=base.model_type, child_capacity_bucket=base_bucket,
                metric=base_result.metric, success=base_result.success,
                failure=base_failure.value,
                hypothesis_id=hyp0.id, experiment_id=exp0.id, finding_id=finding0.id))
        self._record_trial(result, -1, "baseline", base.model_type, base_result,
                            best_metric, base_failure, False, False, base.model_id)

        # -- generational search loop
        for gen in range(n_generations):
            parent = self.rng.choice(self.population)
            hyp_text = f"Improve on genome derived from {parent.model_type} for {self.task.name}"
            strategy, used_memory = self._select_strategy(parent)

            hyp = self.rdg.add_node(
                "Hypothesis", f"{hyp_text} using strategy '{strategy}'",
                attributes={"strategy": strategy, "generation": gen})
            self.rdg.add_edge(self.gap.id, hyp.id, "motivates")

            child = self.synth.synthesize(strategy, parent, self.rng, self.population)
            child.generation = gen
            valid = unit_test(child)

            exp = self.rdg.add_node(
                "Experiment", f"Train/evaluate {child.model_type} (gen {gen})",
                attributes={"genome_id": child.model_id})
            self.rdg.add_edge(hyp.id, exp.id, "tested-by")

            if not valid:
                exp_result = ExperimentResult(metric=0.0, success=False,
                                               exception="invalid genome",
                                               target=self.task.target_metric)
            else:
                exp_result = self._run_experiment(child)

            failure = diagnose(exp_result)
            signature = (strategy, child.model_type)
            if failure != FailureCategory.NONE:
                self._failed_signatures.add(signature)

            finding = self.rdg.add_node(
                "Finding", f"metric={exp_result.metric:.4f} failure={failure.value}",
                attributes={"metric": exp_result.metric, "failure": failure.value})
            self.rdg.add_edge(exp.id, finding.id, "produces")
            if failure == FailureCategory.NONE:
                claim = self.rdg.add_node(
                    "Claim", f"Strategy '{strategy}' improved/held the {self.task.name} model")
                self.rdg.add_edge(finding.id, claim.id, "supports")

            neg_transfer = used_memory and exp_result.metric < best_metric - 0.05
            if neg_transfer and self.use_flat_memory:
                self.ecrm.flag_negative_transfer(strategy)

            if self.use_flat_memory:
                self.ecrm.store(
                    text_summary=self._mem_key(strategy, child.model_type),
                    context={"task": self.task.name, "genome": child.to_dict()},
                    outcome={"metric": exp_result.metric,
                             "success": failure == FailureCategory.NONE,
                             "failure": failure.value},
                    strategy=strategy)
                if gen % 8 == 7:
                    self.ecrm.consolidate()
            elif self.use_trajectory_memory:
                self.trajectory_memory.store(TrajectoryRecord(
                    id=new_trajectory_id(), generation=gen,
                    stage=generation_stage(gen, n_generations),
                    problem_context=self.gap.content,
                    parent_model_type=parent.model_type,
                    parent_capacity_bucket=capacity_bucket(parent),
                    strategy=strategy,
                    child_model_type=child.model_type,
                    child_capacity_bucket=capacity_bucket(child),
                    metric=exp_result.metric,
                    success=(failure == FailureCategory.NONE),
                    failure=failure.value,
                    hypothesis_id=hyp.id, experiment_id=exp.id, finding_id=finding.id))

            if self.use_policy:
                self.policy.update(strategy, reward=exp_result.metric)

            child._score = exp_result.metric if valid else -1.0
            if valid and exp_result.metric > best_metric:
                best_metric = exp_result.metric
                best_genome = child

            self.population.append(child)
            self.population.sort(key=lambda g: -getattr(g, "_score", -1.0))
            if len(self.population) > self.population_size:
                self.population = self.population[: self.population_size]

            self._record_trial(result, gen, strategy, child.model_type, exp_result,
                                best_metric, failure, used_memory, neg_transfer, child.model_id)

        result.best_genome = best_genome
        result.best_metric = best_metric
        result.rdg_stats = self.rdg.stats()
        result.memory_half_life_days = (
            self.ecrm.memory_half_life_days() if self.use_flat_memory else float("nan"))
        result.trajectory_stats = (
            self.trajectory_memory.stats() if self.use_trajectory_memory else {})
        result.wall_time_s = time.time() - t0
        return result
