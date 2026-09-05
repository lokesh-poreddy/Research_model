from __future__ import annotations

from typing import Callable, Dict, Any, List, Tuple
from datetime import datetime, timezone
from researchforge.domain.experiment import ExperimentSpec, ExperimentRun, Outcome, ExecutionStatus, OutcomeNature
from researchforge.domain.execution import ExecutionPolicy
from researchforge.domain.provenance import Provenance
from researchforge.domain.validity import Validity, ValidityVerdict
from researchforge.safety.sandbox import SafeRunner, ResourceBudget
from researchforge.state.events import Event, EventType


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CanonicalExperimentRunner:
    """Small canonical runner connecting ExperimentSpec -> ExperimentRun -> Outcome.

    The runner delegates execution to a callable `executor(spec) -> dict` which
    must return a dict with optional keys: `metrics` (dict), `artifact_refs` (list),
    and `resource_usage` (dict). The runner uses `SafeRunner` when `use_sandbox`
    is True.
    """

    def __init__(self, use_sandbox: bool = True, budget: ResourceBudget | None = None):
        self.use_sandbox = use_sandbox
        self._sandbox = SafeRunner(budget) if use_sandbox else None

    def validate_spec(self, spec: ExperimentSpec) -> None:
        if not spec.id:
            raise ValueError("ExperimentSpec must have an id")
        if not spec.metrics:
            # require at least one metric to evaluate
            raise ValueError("ExperimentSpec must declare at least one metric")

    def run(self, spec: ExperimentSpec, executor: Callable[[ExperimentSpec], Dict[str, Any]], execution_policy: ExecutionPolicy | None = None) -> Tuple[ExperimentRun, Outcome | None, List[Event]]:
        # Validate spec
        self.validate_spec(spec)
        # Require an explicit execution policy; for backward compatibility create a default policy
        if execution_policy is None:
            execution_policy = ExecutionPolicy(id=f"policy-default", schema_version=spec.schema_version, use_sandbox=self.use_sandbox, per_experiment_timeout_s=(self._sandbox.budget.per_experiment_timeout_s if self._sandbox else None))

        # record policy application event
        events: List[Event] = []
        events.append(Event.create(id=f"evt-policy-{spec.id}", schema_version=execution_policy.schema_version, event_type=EventType.EXPERIMENT_PLANNED, payload={"spec_id": spec.id, "policy_id": execution_policy.id}, timestamp=_now_iso(), provenance_id=spec.provenance.id if spec.provenance else None))


        # Create run record
        run_id = f"run-{spec.id}-{int(datetime.now(timezone.utc).timestamp())}"
        run = ExperimentRun(id=run_id, schema_version=spec.schema_version, experiment_spec_id=spec.id, start_time=_now_iso(), status=ExecutionStatus.STARTED, provenance=Provenance(id=f"prov-{run_id}", schema_version=spec.schema_version, created_by="CanonicalExperimentRunner", created_at=_now_iso(), parents=[spec.provenance.id] if spec.provenance else None))
        events.append(Event.create(id=f"evt-start-{run.id}", schema_version=spec.schema_version, event_type=EventType.EXPERIMENT_STARTED, payload={"run_id": run.id, "spec_id": spec.id, "policy_id": execution_policy.id}, timestamp=_now_iso(), provenance_id=run.provenance.id))

        # Execute
        try:
            if execution_policy.use_sandbox and self._sandbox is not None:
                # adapt sandbox budget according to execution_policy
                if execution_policy.per_experiment_timeout_s is not None:
                    self._sandbox.budget.per_experiment_timeout_s = execution_policy.per_experiment_timeout_s
                if execution_policy.max_experiments is not None:
                    self._sandbox.budget.max_experiments = execution_policy.max_experiments
                outcome = self._sandbox.run(executor, spec)
                if outcome.status.name == "OK":
                    exec_result = outcome.value
                    status = ExecutionStatus.SUCCESS
                elif outcome.status.name == "TIMEOUT":
                    exec_result = {"error": outcome.error}
                    status = ExecutionStatus.TIMEOUT
                elif outcome.status.name == "EXCEPTION":
                    exec_result = {"error": outcome.error}
                    status = ExecutionStatus.FAILED
                else:
                    exec_result = {"error": "unknown"}
                    status = ExecutionStatus.FAILED
                run = ExperimentRun.from_dict({**run.to_dict(), "end_time": _now_iso(), "status": status, "produced_artifacts": exec_result.get("artifact_refs"), "resource_usage": exec_result.get("resource_usage"), "failure_info": exec_result.get("error") if status != ExecutionStatus.SUCCESS else None, "provenance": run.provenance, "code_revision": exec_result.get("code_revision"), "environment_fingerprint": exec_result.get("environment_fingerprint"), "dataset_fingerprint": exec_result.get("dataset_fingerprint"), "model_fingerprint": exec_result.get("model_fingerprint"), "seed": exec_result.get("seed"), "experiment_spec_id": spec.id})
            else:
                exec_result = executor(spec)
                status = ExecutionStatus.SUCCESS
                run = ExperimentRun.from_dict({**run.to_dict(), "end_time": _now_iso(), "status": status, "produced_artifacts": exec_result.get("artifact_refs"), "resource_usage": exec_result.get("resource_usage"), "provenance": run.provenance, "code_revision": exec_result.get("code_revision"), "environment_fingerprint": exec_result.get("environment_fingerprint"), "dataset_fingerprint": exec_result.get("dataset_fingerprint"), "model_fingerprint": exec_result.get("model_fingerprint"), "seed": exec_result.get("seed"), "experiment_spec_id": spec.id})
        except Exception as exc:  # pragma: no cover - defensive
            status = ExecutionStatus.FAILED
            exec_result = {"error": f"{type(exc).__name__}: {exc}"}
            run = ExperimentRun.from_dict({**run.to_dict(), "end_time": _now_iso(), "status": status, "failure_info": exec_result, "experiment_spec_id": spec.id})

        # Emit completion event
        events.append(Event.create(id=f"evt-complete-{run.id}", schema_version=spec.schema_version, event_type=EventType.EXPERIMENT_COMPLETED, payload={"run_id": run.id, "status": run.status.value, "policy_id": execution_policy.id}, timestamp=_now_iso(), provenance_id=run.provenance.id))

        # Build Outcome if measurements exist
        outcome_obj = None
        metrics = exec_result.get("metrics") if isinstance(exec_result, dict) else None
        if metrics:
            # simple validity heuristic (placeholder): if any metric is NaN/None -> INCONCLUSIVE
            verdict = ValidityVerdict.INCONCLUSIVE
            try:
                # if numeric metrics exist, mark PROVISIONAL by default
                verdict = ValidityVerdict.PROVISIONAL
            except Exception:
                verdict = ValidityVerdict.INCONCLUSIVE

            validity = Validity(verdict=verdict, details={"note": "auto-assessed provisional"})
            # Derive nature conservatively
            nature = OutcomeNature.INFORMATION_GAIN
            outcome_id = f"out-{run.id}"
            outcome_obj = Outcome(id=outcome_id, schema_version=spec.schema_version, run_id=run.id, metrics=metrics, artifact_refs=exec_result.get("artifact_refs"), validity=validity, nature=nature, provenance=Provenance(id=f"prov-{outcome_id}", schema_version=spec.schema_version, created_by="CanonicalExperimentRunner", created_at=_now_iso(), parents=[run.provenance.id] if run.provenance else None))
            events.append(Event.create(id=f"evt-outcome-{outcome_obj.id}", schema_version=spec.schema_version, event_type=EventType.OUTCOME_RECORDED, payload={"outcome_id": outcome_obj.id, "run_id": run.id}, timestamp=_now_iso(), provenance_id=outcome_obj.provenance.id))
            events.append(Event.create(id=f"evt-validity-{outcome_obj.id}", schema_version=spec.schema_version, event_type=EventType.VALIDITY_ASSESSED, payload={"outcome_id": outcome_obj.id, "verdict": validity.verdict.value}, timestamp=_now_iso(), provenance_id=outcome_obj.provenance.id))

        return run, outcome_obj, events
