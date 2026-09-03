"""tests/test_research.py — ResearchProblem and Hypothesis test suite.

Classification: CORE
Tests Phase 5 (Class C reserved positions):
  - ResearchProblem schema, round-trip, and strict rejection
  - Hypothesis schema, round-trip, and strict rejection
"""
from __future__ import annotations

import pytest

from researchforge.research import (
    ResearchProblem,
    RESEARCH_PROBLEM_SCHEMA,
    Hypothesis,
    HYPOTHESIS_SCHEMA,
)
from researchforge.genome.schema import validate_genome


def test_research_problem_creation_and_validation():
    prob = ResearchProblem(
        problem_id="prob_digits_acc",
        title="Improve Digit Classification Accuracy",
        description="Target > 0.95 accuracy under minimal budget on sklearn digits.",
        domain="computer_vision_tabular",
        success_criteria=["val_accuracy >= 0.95", "training_budget <= 25_generations"],
    )
    prob.validate()
    d = prob.to_dict()
    assert d["problem_id"] == "prob_digits_acc"
    assert len(d["success_criteria"]) == 2


def test_research_problem_roundtrip():
    prob = ResearchProblem(
        problem_id="prob_ecg_robustness",
        title="ECG Anomaly Detection",
        description="Noise-robust ECG classification",
        domain="biomedical",
    )
    d = prob.to_dict()
    restored = ResearchProblem.from_dict(d)
    assert restored.problem_id == prob.problem_id
    assert restored.fingerprint() == prob.fingerprint()


def test_research_problem_rejects_additional_properties():
    prob = ResearchProblem(
        problem_id="prob_1",
        title="Title",
        description="Desc",
        domain="General",
    )
    d = prob.to_dict()
    d["unrecognized_field"] = "bad"
    with pytest.raises(Exception):
        validate_genome(d, RESEARCH_PROBLEM_SCHEMA)


def test_hypothesis_creation_and_validation():
    hyp = Hypothesis(
        hypothesis_id="hyp_inc_cap_01",
        problem_id="prob_digits_acc",
        statement="Increasing MLP layer width from 64 to 128 will resolve underfitting.",
        predicted_outcome="val_accuracy improvement >= 0.05",
        target_tmg_id="tmg_mlp_128",
        status="active",
    )
    hyp.validate()
    d = hyp.to_dict()
    assert d["hypothesis_id"] == "hyp_inc_cap_01"
    assert d["status"] == "active"


def test_hypothesis_roundtrip():
    hyp = Hypothesis(
        hypothesis_id="hyp_rf_02",
        statement="Random forest will outperform SVC under label noise.",
        predicted_outcome="accuracy >= 0.88",
        status="confirmed",
    )
    d = hyp.to_dict()
    restored = Hypothesis.from_dict(d)
    assert restored.hypothesis_id == hyp.hypothesis_id
    assert restored.fingerprint() == hyp.fingerprint()
    assert restored.status == "confirmed"


def test_hypothesis_rejects_additional_properties():
    hyp = Hypothesis(
        hypothesis_id="hyp_3",
        statement="Statement",
        predicted_outcome="Outcome",
        status="inconclusive",
    )
    d = hyp.to_dict()
    d["garbage_key"] = 42
    with pytest.raises(Exception):
        validate_genome(d, HYPOTHESIS_SCHEMA)
