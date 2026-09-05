import json
import os
import pytest
import jsonschema

from researchforge.domain import (
    Hypothesis,
    ExperimentSpec,
    ExperimentRun,
    Outcome,
    Decision,
    Provenance,
    ResearchProblem,
    ResearchQuestion,
    Evidence,
    Failure,
    ResearchState,
)

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "..", "researchforge", "schemas")


def load_schema(name: str):
    path = os.path.join(SCHEMA_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_hypothesis_schema_validation_roundtrip():
    schema = load_schema("hypothesis.schema.json")
    h = Hypothesis(id="h1", schema_version="1", research_question_id="q1", statement="s1", prediction="p", assumptions=("a1",), provenance_id="p1")
    d = h.to_dict()
    # validate
    jsonschema.validate(instance=d, schema=schema)


def test_experiment_spec_and_run_validate():
    schema = load_schema("experiment.schema.json")
    spec = ExperimentSpec(id="s1", schema_version="1", rsg_id="r1", tmg_id="t1", dataset_id="d1")
    run = ExperimentRun(id="run1", schema_version="1", spec_id=spec.id, start_time="2026-09-05T00:00:00Z")
    # validate against definitions
    spec_schema = schema["definitions"]["ExperimentSpec"]
    run_schema = schema["definitions"]["ExperimentRun"]
    jsonschema.validate(instance=spec.to_dict(), schema=spec_schema)
    jsonschema.validate(instance=run.to_dict(), schema=run_schema)


def test_outcome_schema_and_validity():
    schema = load_schema("outcome.schema.json")
    o = Outcome(id="o1", schema_version="1", run_id="run1", measured_metrics={"acc": 0.9})
    jsonschema.validate(instance=o.to_dict(), schema=schema)


def test_required_field_missing_fails():
    schema = load_schema("problem.schema.json")
    bad = {"id": "p1", "schema_version": "1"}  # missing title
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_invalid_enum_fails():
    schema = load_schema("question.schema.json")
    bad = {"id": "q1", "schema_version": "1", "problem_id": "p1", "question_text": "q", "status": "BAD"}
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_invalid_type_fails():
    schema = load_schema("hypothesis.schema.json")
    bad = {"id": "h1", "schema_version": "1", "research_question_id": "q1", "statement": 123}
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_python_object_validates_against_schema():
    schema = load_schema("evidence.schema.json")
    ev = Evidence(id="e1", schema_version="1", source="openalex", source_id="oa:1")
    jsonschema.validate(instance=ev.to_dict(), schema=schema)


def test_schema_contracts_do_not_reference_runtime_components():
    forbidden = ["controller", "evaluator", "runner", "memory", "api"]
    # ensure schema files do not contain these runtime component names
    import re
    def collect_property_names(obj, out=set()):
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.add(k)
                collect_property_names(v, out)
        elif isinstance(obj, list):
            for item in obj:
                collect_property_names(item, out)
        return out

    for fname in os.listdir(os.path.join(SCHEMA_DIR)):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(SCHEMA_DIR, fname)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
            try:
                j = json.loads(text)
            except Exception:
                j = None
            prop_names = set()
            if j is not None:
                prop_names = {p.lower() for p in collect_property_names(j, set())}
            for token in forbidden:
                # only fail if the forbidden token appears as a separate word
                if re.search(r"\b" + re.escape(token) + r"\b", text):
                    # allow if token appears only as a property name in the schema
                    if token not in prop_names:
                        # also allow if the occurrence is inside a $ref string (module path) check
                        if j is not None:
                            # detect $ref values containing the token
                            def ref_contains_token(o):
                                if isinstance(o, dict):
                                    for kk, vv in o.items():
                                        if kk == "$ref" and isinstance(vv, str) and token in vv.lower():
                                            return True
                                        if ref_contains_token(vv):
                                            return True
                                elif isinstance(o, list):
                                    for item in o:
                                        if ref_contains_token(item):
                                            return True
                                return False
                            if ref_contains_token(j):
                                pytest.fail(f"forbidden runtime token '{token}' found in $ref in {fname}")
                        pytest.fail(f"forbidden runtime token '{token}' found in {fname}")
