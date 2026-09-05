import time
import pytest
from researchforge.domain.experiment import ExperimentSpec
from researchforge.experiment.runner import CanonicalExperimentRunner
from researchforge.domain.execution import ExecutionPolicy
from researchforge.domain.provenance import Provenance
from researchforge.safety.sandbox import ResourceBudget
from researchforge.state.transition_engine import apply_events
from researchforge.domain.state import ResearchState


def test_execution_policy_serialization_and_fingerprint():
    p = ExecutionPolicy(id="p1", schema_version="1", use_sandbox=True, per_experiment_timeout_s=5.0)
    d = p.to_dict()
    p2 = ExecutionPolicy.from_dict(d)
    assert p.fingerprint() == p2.fingerprint()


def test_timeout_enforcement_and_timed_out_run():
    spec = ExperimentSpec(id="s_timeout", schema_version="1", metrics=["m"], provenance=Provenance(id="p1", schema_version="1", created_by="u", created_at="now"))
    runner = CanonicalExperimentRunner(use_sandbox=True, budget=ResourceBudget(per_experiment_timeout_s=0.01))
    policy = ExecutionPolicy(id="ptiny", schema_version="1", use_sandbox=True, per_experiment_timeout_s=0.01)

    def long_exec(s):
        time.sleep(0.05)
        return {"metrics": {"m": 0.1}}

    run, outcome, events = runner.run(spec, long_exec, execution_policy=policy)
    assert run.status.name in ("TIMEOUT", "FAILED")
    assert outcome is None


def test_sandbox_delegation_and_provenance_capture():
    spec = ExperimentSpec(id="s_prov", schema_version="1", metrics=["m"], provenance=Provenance(id="p2", schema_version="1", created_by="u", created_at="now"))
    runner = CanonicalExperimentRunner(use_sandbox=True)
    policy = ExecutionPolicy(id="pprov", schema_version="1", use_sandbox=True, per_experiment_timeout_s=2.0)

    def exec_ok(s):
        return {"metrics": {"m": 0.2}, "artifact_refs": ["a1"], "code_revision": "rev1", "environment_fingerprint": "env1", "dataset_fingerprint": "d1"}

    run, outcome, events = runner.run(spec, exec_ok, execution_policy=policy)
    assert run.experiment_spec_id == spec.id
    assert run.provenance is not None
    assert outcome is not None and outcome.artifact_refs
    # apply events to state
    s0 = ResearchState(id="s0", schema_version="1")
    s_fin = apply_events(s0, events)
    assert any(spec.id in (r or "") for r in (s_fin.recent_experiment_refs or []))
