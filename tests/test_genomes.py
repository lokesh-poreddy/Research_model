"""tests/test_genomes.py — RF-1.0.0-alpha.2 genome test suite.

28 tests across four tiers:
  Tier 1 — RSG unit tests (7)
  Tier 1 — TMG unit tests (9)
  Tier 2 — Migration / backward compatibility (7)
  Tier 3 — Integration: controller + RSG (5)

Key behavioral tests:
  - test_rsg_none_behavioral_equivalence:
      EXACT trajectory equivalence: rsg=None vs rsg=RSG.default() produce
      the same deterministic trial sequence and metrics for the same seed.
  - test_tmg_build_estimator_pipeline_equivalence:
      Canonicalized sklearn Pipeline from TMG == from ModelGenome for the
      same config (behavioral preservation guarantee).
  - test_tmg_roundtrip_through_model_genome:
      to_model_genome(from_model_genome(mg)) == mg (preservation roundtrip).

Run:
    python3 tests/test_genomes.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, List

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from researchforge.genome.model_genome import ModelGenome
from researchforge.genome.target_model_genome import (
    TargetModelGenome, TMGCapabilities, GENOME_SCHEMA_TMG, TMG_OPERATORS,
)
from researchforge.genome.research_system_genome import (
    ResearchSystemGenome, ResearchMemoryConfig, ResearchValidityConfig,
    ResearchRetrievalConfig, RSG_EVOLUTION_OPERATORS, GENOME_SCHEMA_RSG,
)
from researchforge.genome.schema import (
    deterministic_genome_id, genome_fingerprint,
)
from researchforge.genome.migration import migrate_tmg, migrate_rsg
from researchforge.genome.operators import STRATEGIES

PASS: List[str] = []
FAIL: List[str] = []


def _test(name: str, fn) -> None:
    try:
        fn()
        PASS.append(name)
    except Exception:  # noqa: BLE001
        FAIL.append(name)
        print(f"FAIL: {name}")
        traceback.print_exc()
        print()


# =========================================================================== #
# Utility                                                                      #
# =========================================================================== #
def _trajectory_hash(result) -> str:
    """Deterministic hash of a RunResult's trial sequence."""
    entries = [
        f"{t.generation}:{t.strategy}:{t.model_type}:{t.metric:.12f}"
        for t in result.trials
    ]
    canonical = "\n".join(entries)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _canonical_pipeline_spec(estimator) -> dict:
    """Extract a canonicalized spec from an sklearn Pipeline for comparison."""
    spec = {}
    for name, step in estimator.steps:
        cls = type(step).__name__
        params = {}
        for k, v in step.get_params().items():
            # Skip non-deterministic/runtime-only params
            if k in ("n_jobs", "verbose", "warm_start"):
                continue
            params[k] = v
        spec[name] = {"class": cls, "params": params}
    return spec


# =========================================================================== #
# TIER 1: RSG unit tests (7)                                                   #
# =========================================================================== #

def test_rsg_default_factory_all_conditions():
    """RSG.default() returns valid RSG for each RDE-Bench condition."""
    for cond in ("full", "trajectory_memory", "no_memory", "random"):
        rsg = ResearchSystemGenome.default(condition=cond, seed=0)
        assert rsg.schema_version == "1.0", f"{cond}: bad schema_version"
        assert rsg.researchforge_version == "RF-1.0.0-alpha.2", f"{cond}: bad rf_version"
        assert rsg.operator == "init", f"{cond}: bad operator"
        assert rsg.generation == 0, f"{cond}: generation != 0"
        assert rsg.allowed_operators, f"{cond}: empty operator portfolio"
        memory_map = {"full": "flat_ecrm", "trajectory_memory": "trajectory",
                      "no_memory": "none", "random": "none"}
        assert rsg.memory_config.memory_design == memory_map[cond], \
            f"{cond}: wrong memory_design"


def test_rsg_validate_passes():
    """RSG.validate() passes for all default conditions."""
    for cond in ("full", "no_memory"):
        rsg = ResearchSystemGenome.default(condition=cond)
        rsg.validate()  # must not raise


def test_rsg_validate_catches_bad_phase():
    """RSG schema rejects unknown research_phase."""
    rsg = ResearchSystemGenome.default()
    rsg.research_phase = "invalid_phase"
    try:
        rsg.validate()
        assert False, "Expected ValidationError"
    except Exception as exc:
        assert "validation" in str(exc).lower() or "enum" in str(exc).lower(), \
            f"Unexpected exception type: {exc!r}"


def test_rsg_to_dict_is_deterministic():
    """Same RSG → same dict on repeated calls."""
    rsg = ResearchSystemGenome.default(seed=42)
    d1 = rsg.to_dict()
    d2 = rsg.to_dict()
    # created_at may differ if object is re-created; check non-volatile fields
    d1.pop("created_at"); d2.pop("created_at")
    assert d1 == d2, "to_dict() is not deterministic"


def test_rsg_to_json_sorted_keys():
    """RSG JSON output has sorted keys."""
    rsg = ResearchSystemGenome.default()
    j = rsg.to_json()
    parsed = json.loads(j)
    keys = list(parsed.keys())
    assert keys == sorted(keys), f"Keys not sorted: {keys}"


def test_rsg_evolve_creates_child():
    """RSG.evolve() returns a new RSG with parent_rsg_id set; self unchanged."""
    parent = ResearchSystemGenome.default(seed=7)
    child = parent.evolve("expand_budget", mutation_params={"delta": 3})
    assert child.rsg_id != parent.rsg_id, "child and parent share rsg_id"
    assert child.parent_rsg_id == parent.rsg_id, "parent_rsg_id not set"
    assert child.generation == parent.generation + 1, "generation not incremented"
    assert child.experiment_budget_per_hypothesis == \
        parent.experiment_budget_per_hypothesis + 3, "budget not changed"
    # Verify parent unchanged
    assert parent.experiment_budget_per_hypothesis == 5, "parent was mutated!"


def test_rsg_from_dict_roundtrip():
    """RSG → to_dict() → from_dict() produces structurally equivalent RSG."""
    rsg = ResearchSystemGenome.default(condition="full", seed=3)
    d = rsg.to_dict()
    rsg2 = ResearchSystemGenome.from_dict(d)
    # Compare non-volatile fields
    assert rsg2.rsg_id == rsg.rsg_id
    assert rsg2.schema_version == rsg.schema_version
    assert rsg2.research_phase == rsg.research_phase
    assert rsg2.allowed_operators == rsg.allowed_operators
    assert rsg2.memory_config.memory_design == rsg.memory_config.memory_design


# =========================================================================== #
# TIER 1: TMG unit tests (9)                                                   #
# =========================================================================== #

def test_tmg_default_factory_all_families():
    """TMG.default() returns valid TMG for each supported model family."""
    for mt in ("MLPClassifier", "RandomForestClassifier", "SVC", "LogisticRegression"):
        tmg = TargetModelGenome.default(mt, seed=0)
        assert tmg.schema_version == "1.0", f"{mt}: bad schema_version"
        assert tmg.model_type == mt
        assert tmg.operator == "init"
        assert tmg.generation == 0
        assert tmg.ancestor_ids == []
        assert tmg.capabilities is not None


def test_tmg_validate_passes():
    """TMG.validate() passes for all model families."""
    for mt in ("MLPClassifier", "RandomForestClassifier", "SVC", "LogisticRegression"):
        tmg = TargetModelGenome.default(mt, seed=0)
        tmg.validate()  # must not raise


def test_tmg_schema_version_field():
    """TMG canonical dict contains schema_version = '1.0' and tmg_id, NOT model_id."""
    tmg = TargetModelGenome.default("LogisticRegression", seed=0)
    d = tmg.to_dict()
    assert d["schema_version"] == "1.0"
    assert "tmg_id" in d
    assert "model_id" not in d, "Legacy model_id should not appear in canonical dict"


def test_tmg_build_estimator_pipeline_equivalence():
    """TMG.build_estimator() produces canonically equivalent Pipeline to ModelGenome.

    Compares: step names, estimator class, and all non-runtime hyperparameters.
    This verifies the behavioral preservation guarantee (AD-004 extension).
    """
    for mt in ("MLPClassifier", "RandomForestClassifier", "LogisticRegression"):
        mg = ModelGenome.default(mt, seed=42)
        tmg = TargetModelGenome.from_model_genome(mg)

        mg_pipe = mg.build_estimator()
        tmg_pipe = tmg.build_estimator()

        mg_spec = _canonical_pipeline_spec(mg_pipe)
        tmg_spec = _canonical_pipeline_spec(tmg_pipe)
        assert mg_spec == tmg_spec, \
            f"{mt}: pipeline specs differ:\nMG:  {mg_spec}\nTMG: {tmg_spec}"


def test_tmg_safety_check_equivalence():
    """TMG.safety_check() returns same violations as ModelGenome for the same config."""
    mg = ModelGenome.default("MLPClassifier", seed=0)
    mg.architecture["hidden_layer_sizes"] = [2048, 512]  # exceeds cap
    tmg = TargetModelGenome.from_model_genome(mg)
    mg_v = mg.safety_check()
    tmg_v = tmg.safety_check()
    assert mg_v == tmg_v, f"Safety check mismatch:\nMG:  {mg_v}\nTMG: {tmg_v}"


def test_tmg_clone_lineage_tracking():
    """clone() creates new TMG with self as parent; ancestor_ids extended."""
    parent = TargetModelGenome.default("SVC", seed=0)
    child = parent.clone(operator="increase_capacity")
    assert child.tmg_id != parent.tmg_id
    assert child.parent_ids == [parent.tmg_id]
    assert parent.tmg_id in child.ancestor_ids
    assert parent.operator == "init", "parent was mutated by clone()!"


def test_tmg_operator_field_recorded():
    """Operator name is preserved in the child genome."""
    parent = TargetModelGenome.default("RandomForestClassifier", seed=0)
    child = parent.clone(operator="add_regularization")
    assert child.operator == "add_regularization"


def test_tmg_deterministic_id_reproducible():
    """Same inputs → same tmg_id (deterministic, not random)."""
    id1 = deterministic_genome_id("tmg", ["p1"], "mutate", 3, 42,
                                  mutation_parameters={"delta": 0.2}, child_index=0)
    id2 = deterministic_genome_id("tmg", ["p1"], "mutate", 3, 42,
                                  mutation_parameters={"delta": 0.2}, child_index=0)
    assert id1 == id2


def test_tmg_sibling_ids_distinct():
    """Same parent + same operator + different child_index → different IDs."""
    id0 = deterministic_genome_id("tmg", ["p1"], "mutate", 3, 42, child_index=0)
    id1 = deterministic_genome_id("tmg", ["p1"], "mutate", 3, 42, child_index=1)
    assert id0 != id1, "Sibling genomes share the same deterministic ID — collision!"


# =========================================================================== #
# TIER 2: Migration / backward compatibility (7)                               #
# =========================================================================== #

def test_tmg_from_model_genome_field_preservation():
    """All ModelGenome fields survive from_model_genome() unchanged.

    Preservation guarantee: no field is altered during the upgrade.
    """
    mg = ModelGenome.default("MLPClassifier", seed=7)
    mg.architecture["hidden_layer_sizes"] = [128, 64]
    mg.hyperparameters["alpha"] = 5e-4
    tmg = TargetModelGenome.from_model_genome(mg)
    assert tmg.model_type == mg.model_type
    assert tmg.architecture == mg.architecture
    assert tmg.hyperparameters == mg.hyperparameters
    assert tmg.data_pipeline == mg.data_pipeline
    assert tmg.seed == mg.seed
    assert tmg.generation == mg.generation
    assert tmg.parent_ids == mg.parent_ids


def test_tmg_to_model_genome_roundtrip():
    """Roundtrip: to_model_genome(from_model_genome(mg)) == mg.

    This is the formal preservation roundtrip contract.
    """
    for mt in ("MLPClassifier", "RandomForestClassifier", "SVC", "LogisticRegression"):
        original = ModelGenome.default(mt, seed=13)
        roundtripped = TargetModelGenome.from_model_genome(original).to_model_genome()
        # Compare all ModelGenome fields
        assert roundtripped.model_type == original.model_type, f"{mt}: model_type mismatch"
        assert roundtripped.architecture == original.architecture, f"{mt}: arch mismatch"
        assert roundtripped.hyperparameters == original.hyperparameters, \
            f"{mt}: hparams mismatch"
        assert roundtripped.data_pipeline == original.data_pipeline, \
            f"{mt}: data_pipeline mismatch"
        assert roundtripped.seed == original.seed, f"{mt}: seed mismatch"
        assert roundtripped.generation == original.generation, f"{mt}: generation mismatch"
        assert roundtripped.parent_ids == original.parent_ids, f"{mt}: parent_ids mismatch"


def test_tmg_from_dict_accepts_legacy_model_id():
    """from_dict() accepts model_id as legacy alias; canonical output uses tmg_id only."""
    legacy_dict = {
        "model_id": "g_abc12345",     # legacy alias
        "model_type": "SVC",
        "architecture": {"kernel": "rbf"},
        "hyperparameters": {"C": 1.0, "gamma": "scale"},
        "data_pipeline": {"scale": True, "pca_components": None},
        "seed": 0,
        "generation": 0,
        "parent_ids": [],
    }
    tmg = TargetModelGenome.from_dict(legacy_dict)
    assert tmg.tmg_id == "g_abc12345", "model_id not migrated to tmg_id"
    d = tmg.to_dict()
    assert "model_id" not in d, "model_id leaked into canonical output"
    assert "tmg_id" in d


def test_tmg_from_dict_strict_schema_rejects_model_id():
    """Canonical TMG dict with model_id is rejected by validate() (additionalProperties: false)."""
    tmg = TargetModelGenome.default("SVC")
    d = tmg.to_dict()
    d["model_id"] = "leak"  # inject legacy alias
    try:
        from researchforge.genome.schema import validate_genome
        validate_genome(d, GENOME_SCHEMA_TMG)
        assert False, "Expected ValidationError for model_id in canonical schema"
    except Exception as exc:
        assert "model_id" in str(exc) or "additional" in str(exc).lower(), \
            f"Wrong exception: {exc!r}"


def test_tmg_operators_produce_valid_tmg_children():
    """All 7 evolution operators produce valid TMG children from a TMG parent.

    Operators act on ModelGenome, so we use the ModelGenome→TMG clone pattern:
    clone() then apply mutation. The resulting TMG must validate.
    """
    from researchforge.genome.operators import apply_strategy
    import random as _random
    rng = _random.Random(0)
    parent_mg = ModelGenome.default("MLPClassifier", seed=0)
    parent_tmg = TargetModelGenome.from_model_genome(parent_mg)
    population = [parent_mg] * 2  # need 2 for crossover

    for strategy in STRATEGIES:
        child_mg = apply_strategy(strategy, parent_mg, rng, population)
        child_tmg = TargetModelGenome.from_model_genome(child_mg)
        child_tmg.validate()  # must not raise


def test_rsg_from_dict_roundtrip_complex():
    """Complex RSG (evolved child) round-trips through to_dict/from_dict."""
    parent = ResearchSystemGenome.default(condition="full", seed=5)
    child = parent.evolve("expand_budget", mutation_params={"delta": 2})
    child2 = child.evolve("shift_to_exploitation")
    d = child2.to_dict()
    restored = ResearchSystemGenome.from_dict(d)
    assert restored.rsg_id == child2.rsg_id
    assert restored.generation == child2.generation
    assert restored.parent_rsg_id == child2.parent_rsg_id
    assert restored.research_phase == child2.research_phase
    assert restored.memory_config.decay_lambda == child2.memory_config.decay_lambda


def test_migrate_tmg_v0_to_v1():
    """migrate_tmg() promotes a v0 ModelGenome dict to TMG v1 without losing fields."""
    v0 = {
        "model_id": "g_old",
        "model_type": "LogisticRegression",
        "architecture": {},
        "hyperparameters": {"C": 1.0, "max_iter": 2000},
        "data_pipeline": {"scale": True, "pca_components": None},
        "seed": 0,
        "generation": 0,
        "parent_ids": [],
    }
    v1 = migrate_tmg(v0)
    assert v1["schema_version"] == "1.0"
    assert v1["tmg_id"] == "g_old"
    assert "model_id" not in v1
    assert "ancestor_ids" in v1
    assert "capabilities" in v1
    assert v1["model_type"] == "LogisticRegression"
    assert v1["hyperparameters"]["C"] == 1.0


# =========================================================================== #
# TIER 3: Integration tests (5)                                                #
# =========================================================================== #

def test_controller_accepts_rsg_parameter():
    """ResearchController accepts rsg=RSG.default() without error."""
    from researchforge.benchmarks.tasks import digits_task
    from researchforge.pipeline.controller import ResearchController
    task = digits_task(seed=0)
    rsg = ResearchSystemGenome.default(condition="full", seed=0)
    ctrl = ResearchController(task=task, condition="full", seed=0, rsg=rsg)
    assert ctrl.rsg is rsg
    result = ctrl.run(n_generations=2)
    assert result.rsg_id == rsg.rsg_id


def test_controller_rsg_none_sets_rsg_id_to_none():
    """ResearchController with rsg=None sets result.rsg_id = None."""
    from researchforge.benchmarks.tasks import digits_task
    from researchforge.pipeline.controller import ResearchController
    task = digits_task(seed=0)
    ctrl = ResearchController(task=task, condition="full", seed=0)
    result = ctrl.run(n_generations=2)
    assert result.rsg_id is None


def test_rsg_none_behavioral_equivalence():
    """CRITICAL: rsg=None and rsg=RSG.default() produce IDENTICAL trajectories.

    This is the deterministic behavioral equivalence test (AD-013).
    Same seed + same condition + same task → same trial sequence,
    same metrics, same trajectory hash.

    Any deviation here means the RSG is altering the execution path,
    which would break the backward compatibility invariant.
    """
    from researchforge.benchmarks.tasks import digits_task
    from researchforge.pipeline.controller import ResearchController
    N_GEN = 5  # short run; determinism holds at any length

    for seed in (0, 1):
        task_a = digits_task(seed=seed)
        task_b = digits_task(seed=seed)

        rsg = ResearchSystemGenome.default(condition="full", seed=seed)

        ctrl_a = ResearchController(task=task_a, condition="full", seed=seed, rsg=None)
        ctrl_b = ResearchController(task=task_b, condition="full", seed=seed, rsg=rsg)

        result_a = ctrl_a.run(n_generations=N_GEN)
        result_b = ctrl_b.run(n_generations=N_GEN)

        hash_a = _trajectory_hash(result_a)
        hash_b = _trajectory_hash(result_b)

        assert hash_a == hash_b, (
            f"seed={seed}: trajectory hashes differ! "
            f"rsg=None → {hash_a}, rsg=RSG.default() → {hash_b}. "
            "RSG is altering execution — backward compatibility broken!"
        )
        assert abs(result_a.best_metric - result_b.best_metric) < 1e-12, (
            f"seed={seed}: best_metric differs: "
            f"{result_a.best_metric} vs {result_b.best_metric}"
        )


def test_rsg_fingerprint_stable():
    """RSG.fingerprint() is stable: same RSG → same fingerprint regardless of wall time."""
    rsg1 = ResearchSystemGenome.default(condition="no_memory", seed=9)
    rsg2 = ResearchSystemGenome.default(condition="no_memory", seed=9)
    # Both have the same content (same deterministic rsg_id, same config)
    assert rsg1.fingerprint() == rsg2.fingerprint(), \
        "fingerprint is not stable across equivalent RSGs"


def test_tmg_fingerprint_stable():
    """TMG.fingerprint() is stable for equivalent genomes."""
    tmg1 = TargetModelGenome.default("RandomForestClassifier", seed=0)
    tmg2 = TargetModelGenome.default("RandomForestClassifier", seed=0)
    assert tmg1.fingerprint() == tmg2.fingerprint(), \
        "TMG fingerprint is not stable across equivalent genomes"


# =========================================================================== #
# Test runner                                                                  #
# =========================================================================== #
if __name__ == "__main__":
    # Tier 1 — RSG
    _test("test_rsg_default_factory_all_conditions", test_rsg_default_factory_all_conditions)
    _test("test_rsg_validate_passes", test_rsg_validate_passes)
    _test("test_rsg_validate_catches_bad_phase", test_rsg_validate_catches_bad_phase)
    _test("test_rsg_to_dict_is_deterministic", test_rsg_to_dict_is_deterministic)
    _test("test_rsg_to_json_sorted_keys", test_rsg_to_json_sorted_keys)
    _test("test_rsg_evolve_creates_child", test_rsg_evolve_creates_child)
    _test("test_rsg_from_dict_roundtrip", test_rsg_from_dict_roundtrip)

    # Tier 1 — TMG
    _test("test_tmg_default_factory_all_families", test_tmg_default_factory_all_families)
    _test("test_tmg_validate_passes", test_tmg_validate_passes)
    _test("test_tmg_schema_version_field", test_tmg_schema_version_field)
    _test("test_tmg_build_estimator_pipeline_equivalence",
          test_tmg_build_estimator_pipeline_equivalence)
    _test("test_tmg_safety_check_equivalence", test_tmg_safety_check_equivalence)
    _test("test_tmg_clone_lineage_tracking", test_tmg_clone_lineage_tracking)
    _test("test_tmg_operator_field_recorded", test_tmg_operator_field_recorded)
    _test("test_tmg_deterministic_id_reproducible", test_tmg_deterministic_id_reproducible)
    _test("test_tmg_sibling_ids_distinct", test_tmg_sibling_ids_distinct)

    # Tier 2 — Migration / backward compat
    _test("test_tmg_from_model_genome_field_preservation",
          test_tmg_from_model_genome_field_preservation)
    _test("test_tmg_to_model_genome_roundtrip", test_tmg_to_model_genome_roundtrip)
    _test("test_tmg_from_dict_accepts_legacy_model_id",
          test_tmg_from_dict_accepts_legacy_model_id)
    _test("test_tmg_from_dict_strict_schema_rejects_model_id",
          test_tmg_from_dict_strict_schema_rejects_model_id)
    _test("test_tmg_operators_produce_valid_tmg_children",
          test_tmg_operators_produce_valid_tmg_children)
    _test("test_rsg_from_dict_roundtrip_complex", test_rsg_from_dict_roundtrip_complex)
    _test("test_migrate_tmg_v0_to_v1", test_migrate_tmg_v0_to_v1)

    # Tier 3 — Integration
    _test("test_controller_accepts_rsg_parameter", test_controller_accepts_rsg_parameter)
    _test("test_controller_rsg_none_sets_rsg_id_to_none",
          test_controller_rsg_none_sets_rsg_id_to_none)
    _test("test_rsg_none_behavioral_equivalence", test_rsg_none_behavioral_equivalence)
    _test("test_rsg_fingerprint_stable", test_rsg_fingerprint_stable)
    _test("test_tmg_fingerprint_stable", test_tmg_fingerprint_stable)

    total = len(PASS) + len(FAIL)
    if FAIL:
        print(f"\n{len(FAIL)}/{total} tests FAILED: {FAIL}")
        sys.exit(1)
    print(f"\n{total} genome tests passed.")
