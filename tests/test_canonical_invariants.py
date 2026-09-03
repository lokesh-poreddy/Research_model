"""tests/test_canonical_invariants.py — Canonical Object Invariants test suite.

Classification: CORE
Verifies that all Class A and Class D canonical domain objects satisfy the
shared identity invariants defined in conftest.py:
  - serialize/deserialize roundtrip preserves canonical fingerprint
  - fingerprint() is deterministic
  - evolve()/clone() changes fingerprint
  - canonical dict has no legacy fields
  - schema_version is present
  - JSON schema rejects unexpected properties
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from conftest import CanonicalObjectInvariants

from researchforge.genome.research_system_genome import (
    ResearchSystemGenome,
    GENOME_SCHEMA_RSG,
)
from researchforge.genome.target_model_genome import (
    TargetModelGenome,
    GENOME_SCHEMA_TMG,
)
from researchforge.experiment import (
    ExperimentSpec,
    EXPERIMENT_SPEC_SCHEMA,
    ExperimentRun,
    EXPERIMENT_RUN_SCHEMA,
    ExperimentOutcome,
    EXPERIMENT_OUTCOME_SCHEMA,
)
from researchforge.state import (
    ResearchState,
    RESEARCH_STATE_SCHEMA,
)
from researchforge.research import (
    ResearchProblem,
    RESEARCH_PROBLEM_SCHEMA,
    Hypothesis,
    HYPOTHESIS_SCHEMA,
)


def test_rsg_invariants():
    rsg = ResearchSystemGenome.default("full", seed=42)
    inv = CanonicalObjectInvariants
    inv.assert_fingerprint_deterministic(rsg)
    inv.assert_serialize_roundtrip_fingerprint(rsg, ResearchSystemGenome)
    inv.assert_schema_version_present(rsg)
    inv.assert_no_legacy_fields(rsg, ["legacy_condition", "old_params"])

    evolved = rsg.evolve("expand_budget")
    inv.assert_evolve_changes_fingerprint(rsg, evolved)

    d_extra = rsg.to_dict()
    d_extra["_injected_garbage"] = 123
    inv.assert_additional_properties_rejected(d_extra, GENOME_SCHEMA_RSG)


def test_tmg_invariants():
    tmg = TargetModelGenome.default("RandomForestClassifier", seed=42)
    inv = CanonicalObjectInvariants
    inv.assert_fingerprint_deterministic(tmg)
    inv.assert_serialize_roundtrip_fingerprint(tmg, TargetModelGenome)
    inv.assert_schema_version_present(tmg)
    inv.assert_no_legacy_fields(tmg, ["model_id"])  # must use tmg_id in canonical dict

    child = tmg.clone("increase_capacity")
    inv.assert_evolve_changes_fingerprint(tmg, child)

    d_extra = tmg.to_dict()
    d_extra["_injected_garbage"] = "bad"
    inv.assert_additional_properties_rejected(d_extra, GENOME_SCHEMA_TMG)


def test_experiment_spec_invariants():
    spec = ExperimentSpec.create(
        tmg_id="tmg_inv",
        tmg_fingerprint="fp_tmg",
        dataset_name="digits",
        dataset_fingerprint="fp_d",
        data_pipeline_fingerprint="fp_p",
        evaluator="sklearn",
        metric_fn="accuracy",
        seed=1,
    )
    inv = CanonicalObjectInvariants
    inv.assert_fingerprint_deterministic(spec)
    inv.assert_serialize_roundtrip_fingerprint(spec, ExperimentSpec)
    inv.assert_schema_version_present(spec)
    inv.assert_no_legacy_fields(spec, ["model_id", "condition"])

    d_extra = spec.to_dict()
    d_extra["_injected_garbage"] = 1
    inv.assert_additional_properties_rejected(d_extra, EXPERIMENT_SPEC_SCHEMA)


def test_experiment_run_invariants():
    run = ExperimentRun(
        run_id="run_inv_1",
        spec_id="spec_inv_1",
        started_at=100.0,
        finished_at=101.5,
        wall_time_s=1.5,
        execution_mode="trusted_offline",
    )
    inv = CanonicalObjectInvariants
    inv.assert_fingerprint_deterministic(run)
    inv.assert_serialize_roundtrip_fingerprint(run, ExperimentRun)
    inv.assert_schema_version_present(run)

    d_extra = run.to_dict()
    d_extra["_bad_prop"] = True
    inv.assert_additional_properties_rejected(d_extra, EXPERIMENT_RUN_SCHEMA)


def test_experiment_outcome_invariants():
    out = ExperimentOutcome(
        outcome_id="out_inv_1",
        run_id="run_inv_1",
        spec_id="spec_inv_1",
        tmg_id="tmg_inv",
        metric=0.89,
        metric_fn="accuracy",
        svg_verdict="PASS",
        svg_report_fingerprint="fp_svg",
        success=True,
    )
    inv = CanonicalObjectInvariants
    inv.assert_fingerprint_deterministic(out)
    inv.assert_serialize_roundtrip_fingerprint(out, ExperimentOutcome)
    inv.assert_schema_version_present(out)

    d_extra = out.to_dict()
    d_extra["_bad_prop"] = 999
    inv.assert_additional_properties_rejected(d_extra, EXPERIMENT_OUTCOME_SCHEMA)


def test_research_state_invariants():
    state = ResearchState.create(
        generation=0,
        research_phase="exploration",
        active_rsg_id="rsg_inv",
        candidate_tmg_ids=["tmg_inv"],
        best_metric=0.88,
        budget_remaining=10,
    )
    inv = CanonicalObjectInvariants
    inv.assert_fingerprint_deterministic(state)
    inv.assert_serialize_roundtrip_fingerprint(state, ResearchState)
    inv.assert_schema_version_present(state)

    evolved = state.evolve(generation=1, best_metric=0.91)
    inv.assert_evolve_changes_fingerprint(state, evolved)

    d_extra = state.to_dict()
    d_extra["_bad"] = "field"
    inv.assert_additional_properties_rejected(d_extra, RESEARCH_STATE_SCHEMA)


def test_research_problem_invariants():
    prob = ResearchProblem(
        problem_id="prob_inv",
        title="Inv Problem",
        description="Inv Description",
        domain="General",
    )
    inv = CanonicalObjectInvariants
    inv.assert_fingerprint_deterministic(prob)
    inv.assert_serialize_roundtrip_fingerprint(prob, ResearchProblem)
    inv.assert_schema_version_present(prob)

    d_extra = prob.to_dict()
    d_extra["_bad"] = 1
    inv.assert_additional_properties_rejected(d_extra, RESEARCH_PROBLEM_SCHEMA)


def test_hypothesis_invariants():
    hyp = Hypothesis(
        hypothesis_id="hyp_inv",
        statement="Statement",
        predicted_outcome="Outcome",
    )
    inv = CanonicalObjectInvariants
    inv.assert_fingerprint_deterministic(hyp)
    inv.assert_serialize_roundtrip_fingerprint(hyp, Hypothesis)
    inv.assert_schema_version_present(hyp)

    d_extra = hyp.to_dict()
    d_extra["_bad"] = 2
    inv.assert_additional_properties_rejected(d_extra, HYPOTHESIS_SCHEMA)
