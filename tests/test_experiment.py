"""tests/test_experiment.py — RF-1.0.0-alpha.2.1 Canonical Contract Objects test suite.

Classification: CORE
Tests Phase 3 domain objects:
  - ExperimentSpec (Class A)
  - ExperimentRun (Class A)
  - ExperimentOutcome (Class A)
  - Evidence & EvidenceCandidate (Class A)
  - Artifact & Provenance (Class A)
  - ResearchDecision (Class A)
  - Failure (Class A)
  - MemoryRecord (Class B versioned mutable)
  - TrajectoryFingerprint
"""
from __future__ import annotations

import copy
import pytest

from researchforge.experiment import (
    ExperimentSpec,
    EXPERIMENT_SPEC_SCHEMA,
    ExperimentRun,
    EXPERIMENT_RUN_SCHEMA,
    ExperimentOutcome,
    EXPERIMENT_OUTCOME_SCHEMA,
    TrajectoryFingerprint,
    compute_trajectory_fingerprint,
)
from researchforge.evidence import (
    Evidence,
    EVIDENCE_SCHEMA,
    EvidenceCandidate,
    EVIDENCE_CANDIDATE_SCHEMA,
)
from researchforge.artifact import (
    Artifact,
    ARTIFACT_SCHEMA,
    Provenance,
    PROVENANCE_SCHEMA,
)
from researchforge.decision import (
    ResearchDecision,
    DECISION_SCHEMA,
)
from researchforge.diagnosis import (
    Failure,
    FAILURE_SCHEMA,
)
from researchforge.memory.record import (
    MemoryRecord,
    MEMORY_RECORD_SCHEMA,
)
from researchforge.genome.schema import validate_genome


# ─── ExperimentSpec tests ───────────────────────────────────────────────────

def test_experiment_spec_creation_and_validation():
    spec = ExperimentSpec.create(
        tmg_id="tmg_test123",
        tmg_fingerprint="fp_tmg_123",
        dataset_name="digits",
        dataset_fingerprint="fp_dataset_digits",
        data_pipeline_fingerprint="fp_pipe_scale",
        evaluator="sklearn_evaluator",
        metric_fn="accuracy",
        seed=42,
        execution_mode="trusted_offline",
        rsg_id="rsg_test456",
        validity_config_fingerprint="vc_fp_alpha005",
    )
    spec.validate()
    d = spec.to_dict()
    assert d["spec_id"].startswith("spec_")
    assert d["schema_version"] == "1.0"
    assert d["execution_mode"] == "trusted_offline"


def test_experiment_spec_roundtrip():
    spec = ExperimentSpec.create(
        tmg_id="tmg_test123",
        tmg_fingerprint="fp_tmg_123",
        dataset_name="digits",
        dataset_fingerprint="fp_dataset_digits",
        data_pipeline_fingerprint="fp_pipe_scale",
        evaluator="sklearn_evaluator",
        metric_fn="accuracy",
        seed=42,
    )
    d = spec.to_dict()
    restored = ExperimentSpec.from_dict(d)
    assert restored.spec_id == spec.spec_id
    assert restored.fingerprint() == spec.fingerprint()


def test_experiment_spec_validity_fingerprint_changes_spec_id():
    """Changing validity_config changes spec_id and fingerprint (prevents silent protocol changes)."""
    spec1 = ExperimentSpec.create(
        tmg_id="tmg_1",
        tmg_fingerprint="fp_tmg",
        dataset_name="digits",
        dataset_fingerprint="fp_d",
        data_pipeline_fingerprint="fp_p",
        evaluator="sklearn",
        metric_fn="accuracy",
        seed=0,
        validity_config_fingerprint="alpha_0.05",
    )
    spec2 = ExperimentSpec.create(
        tmg_id="tmg_1",
        tmg_fingerprint="fp_tmg",
        dataset_name="digits",
        dataset_fingerprint="fp_d",
        data_pipeline_fingerprint="fp_p",
        evaluator="sklearn",
        metric_fn="accuracy",
        seed=0,
        validity_config_fingerprint="alpha_0.01",  # altered validity protocol!
    )
    assert spec1.spec_id != spec2.spec_id
    assert spec1.fingerprint() != spec2.fingerprint()


def test_experiment_spec_rejects_additional_properties():
    spec = ExperimentSpec.create(
        tmg_id="tmg_1",
        tmg_fingerprint="fp_tmg",
        dataset_name="digits",
        dataset_fingerprint="fp_d",
        data_pipeline_fingerprint="fp_p",
        evaluator="sklearn",
        metric_fn="accuracy",
        seed=0,
    )
    d = spec.to_dict()
    d["unauthorized_field"] = "malicious"
    with pytest.raises(Exception):
        validate_genome(d, EXPERIMENT_SPEC_SCHEMA)


# ─── ExperimentRun tests ─────────────────────────────────────────────────────

def test_experiment_run_creation_and_validation():
    run = ExperimentRun(
        run_id="run_12345",
        spec_id="spec_67890",
        started_at=1000.0,
        finished_at=1002.5,
        wall_time_s=2.5,
        execution_mode="trusted_offline",
        runner_version="RF-1.0.0-alpha.2.1",
        exit_status="success",
        safety_verdict="pass",
        resource_usage={"peak_rss_mb": 120.0},
    )
    run.validate()
    d = run.to_dict()
    assert d["wall_time_s"] == 2.5
    assert d["resource_usage"]["peak_rss_mb"] == 120.0


def test_experiment_run_roundtrip():
    run = ExperimentRun(
        run_id="run_12345",
        spec_id="spec_67890",
        started_at=1000.0,
        finished_at=1002.5,
        wall_time_s=2.5,
        execution_mode="sandboxed",
        runner_version="RF-1.0.0-alpha.2.1",
        exit_status="timeout",
        safety_verdict="timeout",
        resource_usage={},
    )
    d = run.to_dict()
    restored = ExperimentRun.from_dict(d)
    assert restored.run_id == run.run_id
    assert restored.fingerprint() == run.fingerprint()


def test_experiment_run_rejects_additional_properties():
    run = ExperimentRun(
        run_id="run_1",
        spec_id="spec_1",
        started_at=100.0,
        finished_at=101.0,
        wall_time_s=1.0,
        execution_mode="trusted_offline",
        exit_status="success",
        safety_verdict="pass",
    )
    d = run.to_dict()
    d["garbage_property"] = True
    with pytest.raises(Exception):
        validate_genome(d, EXPERIMENT_RUN_SCHEMA)


# ─── ExperimentOutcome tests ─────────────────────────────────────────────────

def test_experiment_outcome_creation_and_validation():
    outcome = ExperimentOutcome(
        outcome_id="out_999",
        run_id="run_12345",
        spec_id="spec_67890",
        tmg_id="tmg_abc",
        metric=0.885,
        metric_fn="accuracy",
        svg_verdict="PASS",
        svg_report_fingerprint="svg_fp_report1",
        success=True,
    )
    outcome.validate()
    assert outcome.metric == 0.885
    assert outcome.svg_verdict == "PASS"


def test_experiment_outcome_roundtrip():
    outcome = ExperimentOutcome(
        outcome_id="out_999",
        run_id="run_12345",
        spec_id="spec_67890",
        tmg_id="tmg_abc",
        metric=0.885,
        metric_fn="accuracy",
        svg_verdict="PASS",
        svg_report_fingerprint="svg_fp_report1",
        success=True,
    )
    d = outcome.to_dict()
    restored = ExperimentOutcome.from_dict(d)
    assert restored.outcome_id == outcome.outcome_id
    assert restored.fingerprint() == outcome.fingerprint()


def test_experiment_outcome_rejects_additional_properties():
    outcome = ExperimentOutcome(
        outcome_id="out_1",
        run_id="run_1",
        spec_id="spec_1",
        tmg_id="tmg_1",
        metric=0.5,
        metric_fn="acc",
        svg_verdict="FAIL",
        svg_report_fingerprint="fp",
        success=False,
    )
    d = outcome.to_dict()
    d["extra"] = 123
    with pytest.raises(Exception):
        validate_genome(d, EXPERIMENT_OUTCOME_SCHEMA)


# ─── Evidence & EvidenceCandidate tests ─────────────────────────────────────

def test_evidence_creation_and_roundtrip():
    ev = Evidence(
        evidence_id="ev_001",
        source="arxiv",
        title="Attention Is All You Need",
        content="Transformer architecture based solely on attention mechanisms.",
        relevance_score=0.98,
        citation_key="vaswani2017attention",
        doi_or_url="https://arxiv.org/abs/1706.03762",
    )
    ev.validate()
    d = ev.to_dict()
    restored = Evidence.from_dict(d)
    assert restored.evidence_id == ev.evidence_id
    assert restored.fingerprint() == ev.fingerprint()


def test_evidence_candidate_and_adjudication():
    candidate = EvidenceCandidate(
        candidate_id="cand_101",
        source="github",
        relevance_score=0.75,
        retrieval_query="optimizer lr decay",
        retrieval_timestamp=1700000000.0,
        retrieved_item={"title": "repo/sgd", "url": "https://github.com/repo/sgd"},
    )
    candidate.validate()
    assert candidate.evidence_id is None
    # Adjudicate candidate
    candidate.evidence_id = "ev_001"
    candidate.adjudicated_at = 1700000050.0
    candidate.adjudication_verdict = "accepted"
    candidate.validate()
    restored = EvidenceCandidate.from_dict(candidate.to_dict())
    assert restored.evidence_id == "ev_001"
    assert restored.adjudication_verdict == "accepted"


def test_evidence_candidate_rejects_additional_properties():
    candidate = EvidenceCandidate(
        candidate_id="cand_102",
        source="arxiv",
        relevance_score=0.88,
        retrieval_query="transformer attention",
        retrieval_timestamp=1700000000.0,
        retrieved_item={"title": "paper"},
    )
    d = candidate.to_dict()
    d["extra_prop"] = "invalid"
    with pytest.raises(Exception):
        validate_genome(d, EVIDENCE_CANDIDATE_SCHEMA)


# ─── Artifact & Provenance tests ─────────────────────────────────────────────

def test_artifact_creation_and_validation():
    art = Artifact(
        artifact_id="art_model_w",
        artifact_type="model_weights",
        uri="file:///tmp/weights.bin",
        checksum_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        size_bytes=1048576,
    )
    art.validate()
    d = art.to_dict()
    restored = Artifact.from_dict(d)
    assert restored.artifact_id == art.artifact_id
    assert restored.fingerprint() == art.fingerprint()


def test_provenance_creation_and_validation():
    prov = Provenance(
        provenance_id="prov_001",
        entity_id="art_model_w",
        activity="training",
        agent_id="agent_rf1",
        input_artifact_ids=["art_data_train", "spec_001"],
        code_revision="commit_abc123",
        environment_fingerprint="env_fp_py313",
    )
    prov.validate()
    d = prov.to_dict()
    restored = Provenance.from_dict(d)
    assert restored.provenance_id == prov.provenance_id
    assert restored.fingerprint() == prov.fingerprint()


# ─── ResearchDecision tests ──────────────────────────────────────────────────

def test_research_decision_creation_and_validation():
    dec = ResearchDecision(
        decision_id="dec_001",
        decision_type="operator_selection",
        context_state_id="state_gen_3",
        rationale="Exploitation phase: increase capacity yielded positive transfer.",
        chosen_option="increase_capacity",
        candidate_options=["increase_capacity", "add_regularization", "change_family"],
        policy_confidence=0.82,
    )
    dec.validate()
    d = dec.to_dict()
    restored = ResearchDecision.from_dict(d)
    assert restored.decision_id == dec.decision_id
    assert restored.fingerprint() == dec.fingerprint()


# ─── Failure domain tests ────────────────────────────────────────────────────

def test_failure_creation_and_validation():
    fail = Failure(
        failure_id="fail_001",
        category="Overfitting",
        description="Train score 0.99 but validation score 0.65; gap exceeds 0.12",
        run_id="run_123",
        tmg_id="tmg_mlp",
    )
    fail.validate()
    d = fail.to_dict()
    restored = Failure.from_dict(d)
    assert restored.failure_id == fail.failure_id
    assert restored.fingerprint() == fail.fingerprint()


# ─── MemoryRecord (Class B) tests ────────────────────────────────────────────

def test_memory_record_immutable_core_fingerprint():
    """Mutable envelope changes (tier, count) do NOT alter the canonical core fingerprint."""
    rec = MemoryRecord(
        id="mem_1",
        text_summary="increase_capacity MLP digits",
        embedding=[0.1, 0.2, 0.3],
        context={"task": "digits"},
        outcome={"metric": 0.85},
        strategy="increase_capacity",
        tier="short_term",
        retrieval_count=0,
    )
    rec.validate()
    fp1 = rec.fingerprint()

    # Dynamic mutable envelope updates (in-place)
    rec.tier = "long_term"
    rec.retrieval_count = 5
    rec.negative_transfer_count = 1
    rec.consolidation_passes_survived = 2
    rec.validate()
    fp2 = rec.fingerprint()

    # Core fingerprint is preserved
    assert fp1 == fp2, "Mutable envelope altered canonical fingerprint of MemoryRecord"


def test_memory_record_roundtrip():
    rec = MemoryRecord(
        id="mem_2",
        text_summary="tune_learning_dynamics SVC digits",
        embedding=[0.4, 0.5],
        context={"task": "digits"},
        outcome={"metric": 0.90},
        strategy="tune_learning_dynamics",
    )
    d = rec.to_dict()
    restored = MemoryRecord.from_dict(d)
    assert restored.id == rec.id
    assert restored.fingerprint() == rec.fingerprint()


# ─── TrajectoryFingerprint tests ─────────────────────────────────────────────

def test_trajectory_fingerprint_deterministic_and_matching():
    fp1 = TrajectoryFingerprint(
        generation_hashes=["h1", "h2"],
        operator_sequence=["baseline", "increase_capacity"],
        metric_sequence=[0.8, 0.85],
        best_metric=0.85,
    )
    fp2 = TrajectoryFingerprint(
        generation_hashes=["h1", "h2"],
        operator_sequence=["baseline", "increase_capacity"],
        metric_sequence=[0.8, 0.85],
        best_metric=0.85,
    )
    assert fp1.final_hash == fp2.final_hash
    assert fp1.matches(fp2)

    # Any deviation changes final_hash
    fp3 = TrajectoryFingerprint(
        generation_hashes=["h1", "h2"],
        operator_sequence=["baseline", "add_regularization"],
        metric_sequence=[0.8, 0.85],
        best_metric=0.85,
    )
    assert not fp1.matches(fp3)
