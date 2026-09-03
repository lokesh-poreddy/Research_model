"""tests/test_state.py — ResearchState test suite.

Classification: CORE
Tests Phase 4:
  - ResearchState schema validation and round-trip
  - Deterministic fingerprinting and evolve() transition
  - Controller integration and state sequence verification
"""
from __future__ import annotations

import pytest

from researchforge.state import ResearchState, RESEARCH_STATE_SCHEMA
from researchforge.pipeline.controller import ResearchController
from researchforge.benchmarks.tasks import digits_task
from researchforge.genome.schema import validate_genome


def test_research_state_creation_and_validation():
    state = ResearchState.create(
        generation=0,
        research_phase="exploration",
        active_rsg_id="rsg_123",
        active_rsg_fingerprint="fp_rsg_123",
        candidate_tmg_ids=["tmg_1", "tmg_2"],
        best_tmg_id="tmg_1",
        best_metric=0.85,
        budget_remaining=24,
        problem_id="prob_digits",
        active_hypothesis_ids=["hyp_0"],
    )
    state.validate()
    d = state.to_dict()
    assert d["state_id"].startswith("state_")
    assert d["generation"] == 0
    assert d["best_metric"] == 0.85
    assert len(d["candidate_tmg_ids"]) == 2


def test_research_state_roundtrip():
    state = ResearchState.create(
        generation=1,
        research_phase="exploitation",
        active_rsg_id="rsg_456",
        candidate_tmg_ids=["tmg_3"],
        best_metric=0.92,
        budget_remaining=20,
    )
    d = state.to_dict()
    restored = ResearchState.from_dict(d)
    assert restored.state_id == state.state_id
    assert restored.fingerprint() == state.fingerprint()
    assert restored.generation == 1
    assert restored.research_phase == "exploitation"


def test_research_state_fingerprint_deterministic():
    state1 = ResearchState.create(
        generation=2,
        research_phase="exploration",
        active_rsg_id="rsg_fixed",
        candidate_tmg_ids=["tmg_a", "tmg_b"],
        best_metric=0.88,
        budget_remaining=10,
    )
    state2 = ResearchState.create(
        generation=2,
        research_phase="exploration",
        active_rsg_id="rsg_fixed",
        candidate_tmg_ids=["tmg_a", "tmg_b"],
        best_metric=0.88,
        budget_remaining=10,
    )
    assert state1.fingerprint() == state2.fingerprint()
    assert state1.state_id == state2.state_id


def test_research_state_evolve_produces_new_state_id():
    state = ResearchState.create(
        generation=0,
        research_phase="exploration",
        active_rsg_id="rsg_1",
        candidate_tmg_ids=["tmg_1"],
        best_metric=0.80,
        budget_remaining=25,
    )
    evolved = state.evolve(
        generation=1,
        best_metric=0.87,
        budget_remaining=24,
        candidate_tmg_ids=["tmg_1", "tmg_2"],
    )
    assert evolved.state_id != state.state_id
    assert evolved.fingerprint() != state.fingerprint()
    assert evolved.generation == 1
    assert evolved.best_metric == 0.87
    assert len(evolved.candidate_tmg_ids) == 2


def test_research_state_rejects_additional_properties():
    state = ResearchState.create(
        generation=0,
        research_phase="exploration",
        active_rsg_id="rsg_1",
        candidate_tmg_ids=["tmg_1"],
        best_metric=0.80,
        budget_remaining=25,
    )
    d = state.to_dict()
    d["unauthorized_state_field"] = "bad"
    with pytest.raises(Exception):
        validate_genome(d, RESEARCH_STATE_SCHEMA)


def test_controller_produces_research_states():
    task = digits_task(seed=42)
    ctrl = ResearchController(task, condition="no_memory", seed=42)
    result = ctrl.run(n_generations=3)

    # 1 baseline (gen -1) + 3 generations = 4 states
    assert len(result.states) == 4, f"Expected 4 states, got {len(result.states)}"
    for st in result.states:
        st.validate()  # all generated states must pass strict schema validation


def test_controller_states_have_increasing_generations():
    task = digits_task(seed=42)
    ctrl = ResearchController(task, condition="full", seed=42)
    result = ctrl.run(n_generations=4)

    generations = [s.generation for s in result.states]
    assert generations == [-1, 0, 1, 2, 3]
    for s in result.states:
        assert s.state_id.startswith("state_")


def test_controller_states_capture_tmg_ids():
    task = digits_task(seed=42)
    ctrl = ResearchController(task, condition="full", seed=42)
    result = ctrl.run(n_generations=2)

    last_state = result.states[-1]
    assert last_state.candidate_tmg_ids
    assert all(tid.startswith("tmg_") or tid.startswith("g_") for tid in last_state.candidate_tmg_ids)
    assert last_state.best_metric == result.best_metric
