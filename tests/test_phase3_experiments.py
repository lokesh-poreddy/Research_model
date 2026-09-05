import time
import pytest
from researchforge.domain.experiment import ExperimentSpec
from researchforge.experiment.runner import CanonicalExperimentRunner
from researchforge.domain.execution import ExecutionPolicy
from researchforge.domain.provenance import Provenance
from researchforge.state.transition_engine import apply_events
from researchforge.domain.state import ResearchState


def sample_executor_success(spec):
    # return metrics and artifact refs
    return {"metrics": {m: 0.5 for m in (spec.metrics or [])}, "artifact_refs": ["art1"]}


def sample_executor_fail(spec):
    raise RuntimeError("executor crashed")


def sample_executor_no_metrics(spec):
    return {"artifact_refs": ["art2"]}


def test_valid_spec_construction_and_fingerprint():
    spec = ExperimentSpec(id="spec1", schema_version="1", metrics=["acc"], provenance=Provenance(id="p0", schema_version="1", created_by="u", created_at="now"))
    assert spec.metrics == ["acc"]
    fp = spec.fingerprint()
    assert isinstance(fp, str) and len(fp) == 64


def test_invalid_spec_rejected_by_runner():
    runner = CanonicalExperimentRunner(use_sandbox=False)
    spec = ExperimentSpec(id="spec2", schema_version="1", metrics=None)
    with pytest.raises(ValueError):
        runner.validate_spec(spec)


def test_spec_run_linkage_and_success_outcome():
    spec = ExperimentSpec(id="spec3", schema_version="1", metrics=["m1"], provenance=Provenance(id="p1", schema_version="1", created_by="u", created_at="now"))
    runner = CanonicalExperimentRunner(use_sandbox=False)
    policy = ExecutionPolicy(id="policy-default", schema_version="1", use_sandbox=False)
    run, outcome, events = runner.run(spec, sample_executor_success, execution_policy=policy)
    assert run.experiment_spec_id == spec.id
    assert run.status.name == "SUCCESS"
    assert outcome is not None and outcome.metrics
    # events include plan, start, complete, outcome, validity
    ev_types = [e.event_type for e in events]
    assert "EXPERIMENT_PLANNED" in ev_types
    assert "EXPERIMENT_STARTED" in ev_types
    assert "EXPERIMENT_COMPLETED" in ev_types
    assert "OUTCOME_RECORDED" in ev_types


def test_execution_failure_produces_failed_run_and_no_outcome():
    spec = ExperimentSpec(id="spec4", schema_version="1", metrics=["m1"], provenance=Provenance(id="p2", schema_version="1", created_by="u", created_at="now"))
    runner = CanonicalExperimentRunner(use_sandbox=False)
    policy = ExecutionPolicy(id="policy-default", schema_version="1", use_sandbox=False)
    run, outcome, events = runner.run(spec, sample_executor_fail, execution_policy=policy)
    assert run.status.name == "FAILED"
    assert outcome is None


def test_timeout_path_with_sandbox():
    # use a tiny timeout to force timeout
    from researchforge.safety.sandbox import ResourceBudget
    spec = ExperimentSpec(id="spec5", schema_version="1", metrics=["m1"], provenance=Provenance(id="p3", schema_version="1", created_by="u", created_at="now"))
    runner = CanonicalExperimentRunner(use_sandbox=True, budget=ResourceBudget(per_experiment_timeout_s=0.01))
    policy = ExecutionPolicy(id="policy-tiny", schema_version="1", use_sandbox=True, per_experiment_timeout_s=0.01)

    def long_executor(s):
        time.sleep(0.05)
        return {"metrics": {"m1": 0.1}}

    run, outcome, events = runner.run(spec, long_executor, execution_policy=policy)
    assert run.status.name in ("TIMEOUT", "FAILED")


def test_missing_measurement_path():
    spec = ExperimentSpec(id="spec6", schema_version="1", metrics=["m1"], provenance=Provenance(id="p4", schema_version="1", created_by="u", created_at="now"))
    runner = CanonicalExperimentRunner(use_sandbox=False)
    policy = ExecutionPolicy(id="policy-default", schema_version="1", use_sandbox=False)
    run, outcome, events = runner.run(spec, sample_executor_no_metrics, execution_policy=policy)
    assert run.status.name == "SUCCESS"
    assert outcome is None


def test_state_event_integration_and_reconstruction():
    spec = ExperimentSpec(id="spec7", schema_version="1", metrics=["m1"], provenance=Provenance(id="p5", schema_version="1", created_by="u", created_at="now"))
    runner = CanonicalExperimentRunner(use_sandbox=False)
    policy = ExecutionPolicy(id="policy-default", schema_version="1", use_sandbox=False)
    run, outcome, events = runner.run(spec, sample_executor_success, execution_policy=policy)
    # apply events to an initial ResearchState
    state0 = ResearchState(id="state0", schema_version="1")
    s_final = apply_events(state0, events)
    refs = s_final.recent_experiment_refs or []
    # should contain spec id and run id (transition engine appends both)
    assert any("spec7" in r for r in refs)
    assert any(run.id in r for r in refs)
