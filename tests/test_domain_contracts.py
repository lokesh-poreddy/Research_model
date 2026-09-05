import json
from researchforge.domain import (
    Hypothesis,
    ExperimentSpec,
    ExperimentRun,
    Outcome,
    Decision,
    Provenance,
    ResearchProblem,
)


def test_hypothesis_roundtrip_and_fingerprint():
    h = Hypothesis(id="h1", schema_version="1", research_question_id="q1", statement="X causes Y", prediction="improve", assumptions=("a1",))
    d = h.to_dict()
    h2 = Hypothesis.from_dict(d)
    assert h.fingerprint() == h2.fingerprint()


def test_experiment_spec_and_run_serialization():
    spec = ExperimentSpec(id="s1", schema_version="1", rsg_id="r1", tmg_id="t1", dataset_id="d1")
    run = ExperimentRun(id="run1", schema_version="1", spec_id=spec.id, start_time="2026-09-05T00:00:00Z")
    sdict = spec.to_dict()
    rdict = run.to_dict()
    assert isinstance(json.dumps(sdict), str)
    assert isinstance(json.dumps(rdict), str)


def test_outcome_validity_embedding():
    prov = Provenance(id="p1", schema_version="1", created_by="tester", created_at="2026-09-05T00:00:00Z")
    o = Outcome(id="o1", schema_version="1", run_id="run1", measured_metrics={"acc": 0.9}, validity=None)
    j = o.to_json()
    o2 = Outcome.from_dict(json.loads(j))
    assert o.fingerprint() == o2.fingerprint()


def test_decision_fields():
    dec = Decision(id="d1", schema_version="1", research_state_fingerprint="rsfp", rsg_id="rsg1", hypothesis_id="h1", selected_tmg_id="t1", selected_operator="mutate", decision_reason="test")
    d2 = Decision.from_dict(dec.to_dict())
    assert dec.fingerprint() == d2.fingerprint()
