"""Basic sanity tests for the ResearchForge-ECRM reference implementation.

Run with:   python tests/test_basic.py
       or:  python -m pytest tests/ -q   (if pytest is installed)
"""
from __future__ import annotations
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from researchforge.rdg.graph import ResearchDevelopmentGraph, EdgeConstraintError
from researchforge.genome.model_genome import ModelGenome
from researchforge.genome.operators import apply_strategy, STRATEGIES
from researchforge.memory.ecrm import ECRM
from researchforge.policy.policy_learner import PolicyLearner
from researchforge.diagnosis.failure_taxonomy import diagnose, ExperimentResult, FailureCategory
from researchforge.benchmarks.tasks import digits_task
from researchforge.pipeline.controller import ResearchController, CONDITIONS
from researchforge.memory.trajectory import (
    TrajectoryMemory, TrajectoryRecord, capacity_bucket, generation_stage, new_trajectory_id,
)


def test_rdg_typed_edges():
    g = ResearchDevelopmentGraph()
    p = g.add_node("Problem", "test problem")
    gap = g.add_node("Gap", "test gap")
    g.add_edge(p.id, gap.id, "identifies")
    try:
        g.add_edge(gap.id, p.id, "identifies")  # wrong direction/types
        raise AssertionError("expected EdgeConstraintError")
    except EdgeConstraintError:
        pass


def test_rdg_evidence_chain():
    g = ResearchDevelopmentGraph()
    p = g.add_node("Problem", "p")
    gap = g.add_node("Gap", "g")
    hyp = g.add_node("Hypothesis", "h")
    exp = g.add_node("Experiment", "e")
    finding = g.add_node("Finding", "f")
    claim = g.add_node("Claim", "c")
    g.add_edge(p.id, gap.id, "identifies")
    g.add_edge(gap.id, hyp.id, "motivates")
    g.add_edge(hyp.id, exp.id, "tested-by")
    g.add_edge(exp.id, finding.id, "produces")
    g.add_edge(finding.id, claim.id, "supports")
    chain = g.evidence_chain(claim.id)
    assert [n.type for n in chain] == ["Problem", "Gap", "Hypothesis", "Experiment", "Finding", "Claim"]


def test_genome_build_and_validate():
    g = ModelGenome.default("RandomForestClassifier", seed=1)
    g.validate()
    est = g.build_estimator()
    assert hasattr(est, "fit")


def test_operators_produce_valid_genomes():
    rng = random.Random(0)
    base = ModelGenome.default("MLPClassifier", seed=0)
    for strat in STRATEGIES:
        child = apply_strategy(strat, base, rng, population=[base])
        child.validate()
        child.build_estimator()


def test_ecrm_store_query_and_negative_transfer():
    mem = ECRM()
    mem.store("increase capacity on task X", {"task": "X"},
              {"metric": 0.9, "success": True}, "increase_capacity")
    results = mem.query("increase capacity on task X", k=1)
    assert len(results) == 1
    mem.flag_negative_transfer("increase_capacity")
    assert mem.negative_transfer_rate("increase_capacity") == 1.0


def test_ecrm_has_similar_failure_survives_archiving():
    mem = ECRM(decay_lambda=0.0, retention_threshold=0.9)  # force-archive everything
    mem.store("try change_family on task Y", {"task": "Y"},
              {"metric": 0.0, "success": False}, "change_family")
    archived = mem.consolidate()
    assert archived >= 1
    assert mem.has_similar_failure("try change_family on task Y", threshold=0.3) is True


def test_ecrm_tiering_and_reallocate():
    mem = ECRM(decay_lambda=0.0, retention_threshold=0.05, promotion_threshold=2)
    for _ in range(4):
        mem.store("reliable strategy on task Z", {"task": "Z"},
                  {"metric": 0.9, "success": True}, "increase_capacity")
    rec_ids = list(mem.records.keys())
    assert all(mem.records[rid].tier == "short_term" for rid in rec_ids)

    mem.consolidate()
    mem.consolidate()  # two survived passes -> promotion_threshold reached
    assert len(mem.long_term_memory()) == len(rec_ids)
    assert len(mem.working_memory()) == 0

    summary = mem.reallocate()
    assert summary["long_term_count"] == len(rec_ids)
    assert summary["demoted_to_short_term"] == 0


def test_failure_diagnosis():
    ok = ExperimentResult(metric=0.9, train_metric=0.91, success=True, target=0.5)
    assert diagnose(ok) == FailureCategory.NONE
    bad = ExperimentResult(metric=0.0, success=False, exception="boom", target=0.5)
    assert diagnose(bad) == FailureCategory.EXECUTION_ERROR
    low = ExperimentResult(metric=0.3, train_metric=0.35, success=True, target=0.5)
    assert diagnose(low) == FailureCategory.LOW_PERFORMANCE


def test_policy_learner_prefers_better_action():
    pl = PolicyLearner(["a", "b"], rng=random.Random(0))
    for _ in range(20):
        pl.update("a", reward=0.9)
        pl.update("b", reward=0.1)
    assert pl.q["a"] > pl.q["b"]
    assert pl.select_action() == "a"


def test_controller_end_to_end_smoke():
    task = digits_task(seed=0)
    ctrl = ResearchController(task, condition="full", seed=0)
    n_gen = 4
    result = ctrl.run(n_generations=n_gen)
    assert len(result.trials) == n_gen + 1  # +1 for the baseline trial
    assert result.best_metric > 0.0
    assert result.rdg_stats["_edges"] > 0
    assert result.rdg_stats.get("Hypothesis", 0) == n_gen + 1


def test_controller_all_conditions_run():
    task = digits_task(seed=0)
    for condition in CONDITIONS:
        ctrl = ResearchController(task, condition=condition, seed=1)
        result = ctrl.run(n_generations=3)
        assert result.best_metric > 0.0


def test_sandbox_kills_a_runaway_call_and_tracks_budget():
    import time
    from researchforge.safety.sandbox import SafeRunner, ResourceBudget, SafetyStatus

    def slow():
        time.sleep(5)
        return "unreachable"

    runner = SafeRunner(ResourceBudget(per_experiment_timeout_s=0.5))
    t0 = time.time()
    outcome = runner.run(slow)
    elapsed = time.time() - t0
    assert outcome.status == SafetyStatus.TIMEOUT
    assert elapsed < 2.0  # proves the process was actually killed, not just reported late
    assert runner.experiments_run == 1

    runner.kill()
    assert runner.run(slow).status == SafetyStatus.KILLED


def test_genome_safety_check_flags_unreasonable_architectures():
    from researchforge.genome.model_genome import ModelGenome

    reasonable = ModelGenome.default("RandomForestClassifier", seed=0)
    assert reasonable.safety_check() == []

    reckless = ModelGenome(model_type="RandomForestClassifier",
                            architecture={"n_estimators": 999_999, "max_depth": None},
                            hyperparameters={"min_samples_leaf": 1, "min_samples_split": 2})
    assert reckless.safety_check() != []


def test_capacity_bucket_matches_expected_regime():
    from researchforge.genome.model_genome import ModelGenome
    low_c = ModelGenome(model_type="SVC", architecture={"kernel": "rbf"},
                         hyperparameters={"C": 0.1, "gamma": "scale"})
    high_c = ModelGenome(model_type="SVC", architecture={"kernel": "rbf"},
                          hyperparameters={"C": 50.0, "gamma": "scale"})
    assert capacity_bucket(low_c) == "low"
    assert capacity_bucket(high_c) == "high"

    small_mlp = ModelGenome(model_type="MLPClassifier",
                             architecture={"hidden_layer_sizes": [16], "activation": "relu"},
                             hyperparameters={"alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 200})
    big_mlp = ModelGenome(model_type="MLPClassifier",
                           architecture={"hidden_layer_sizes": [256, 128], "activation": "relu"},
                           hyperparameters={"alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 200})
    assert capacity_bucket(small_mlp) == "low"
    assert capacity_bucket(big_mlp) == "high"


def test_generation_stage_buckets_span_a_run():
    assert generation_stage(-1, 25) == "baseline"
    assert generation_stage(0, 25) == "early"
    assert generation_stage(12, 25) == "mid"
    assert generation_stage(24, 25) == "late"


def test_trajectory_memory_contextual_success_rate_is_context_specific():
    """The whole point of trajectory memory over the flat ECRM: two contexts
    that share a strategy and model family but differ in capacity regime
    must be tracked -- and retrievable -- separately."""
    mem = TrajectoryMemory()

    def rec(bucket, success):
        return TrajectoryRecord(
            id=new_trajectory_id(), generation=1, stage="early", problem_context="p",
            parent_model_type="MLPClassifier", parent_capacity_bucket=bucket,
            strategy="increase_capacity", child_model_type="MLPClassifier",
            child_capacity_bucket=bucket, metric=0.9 if success else 0.1,
            success=success, failure="None" if success else "LowPerformance")

    for _ in range(3):
        mem.store(rec("low", True))       # succeeds when starting from low capacity
    for _ in range(3):
        mem.store(rec("high", False))     # fails when starting from already-high capacity

    rate_low = mem.contextual_success_rate("increase_capacity", "MLPClassifier", "low")
    rate_high = mem.contextual_success_rate("increase_capacity", "MLPClassifier", "high")
    assert rate_low == 1.0
    assert rate_high == 0.0
    # An unseen (strategy, model, bucket) combination falls back to the neutral default,
    # not to whatever the closest-but-different context happened to show.
    rate_unseen = mem.contextual_success_rate("increase_capacity", "SVC", "medium")
    assert 0.0 < rate_unseen < 1.0

    assert mem.similar_trajectory_recently_failed("increase_capacity", "MLPClassifier", "high") is True
    assert mem.similar_trajectory_recently_failed("increase_capacity", "MLPClassifier", "low") is False
    assert mem.stats()["total_trajectories"] == 6
    assert mem.stats()["distinct_contexts"] == 2


def test_trajectory_memory_evidence_chain_uses_real_rdg():
    from researchforge.rdg.graph import ResearchDevelopmentGraph
    g = ResearchDevelopmentGraph()
    problem = g.add_node("Problem", "p")
    gap = g.add_node("Gap", "g")
    hyp = g.add_node("Hypothesis", "h")
    exp = g.add_node("Experiment", "e")
    finding = g.add_node("Finding", "f")
    g.add_edge(problem.id, gap.id, "identifies")
    g.add_edge(gap.id, hyp.id, "motivates")
    g.add_edge(hyp.id, exp.id, "tested-by")
    g.add_edge(exp.id, finding.id, "produces")

    mem = TrajectoryMemory()
    tid = new_trajectory_id()
    mem.store(TrajectoryRecord(
        id=tid, generation=0, stage="early", problem_context="g",
        parent_model_type="SVC", parent_capacity_bucket="medium", strategy="crossover_top2",
        child_model_type="SVC", child_capacity_bucket="medium", metric=0.8, success=True,
        failure="None", hypothesis_id=hyp.id, experiment_id=exp.id, finding_id=finding.id))

    chain = mem.evidence_chain_for(tid, g)
    assert [n.type for n in chain] == ["Problem", "Gap", "Hypothesis", "Experiment", "Finding"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
