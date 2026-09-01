"""Model Genome: a structured, JSON-schema-validated description of a
candidate model (design doc Sec. 3.2 / technical report Sec. 3.2).

`build_estimator()` is what makes this a genuine implementation rather than a
paper exercise: it turns the genome's JSON into a real, trainable
scikit-learn Pipeline. Numeric fields are defensively clamped into each
estimator's valid range at build time (e.g. `min_samples_split >= 2`) so that
an evolution operator drifting a hyperparameter to an extreme value produces
a *weak* model to be diagnosed and selected against, rather than a spurious
constructor crash that would tell us nothing about the search process.
"""
from __future__ import annotations
import copy
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

GENOME_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Model Genome",
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "model_type": {"type": "string", "enum": [
            "MLPClassifier", "RandomForestClassifier", "SVC", "LogisticRegression"]},
        "architecture": {"type": "object"},
        "hyperparameters": {"type": "object"},
        "data_pipeline": {"type": "object"},
        "seed": {"type": "integer"},
        "parent_ids": {"type": "array"},
        "generation": {"type": "integer"},
    },
    "required": ["model_id", "model_type", "architecture", "hyperparameters"],
}

# Default architecture/hyperparameters per model family -- the "primitives"
# that `change_family` resets to and that seed a fresh population member.
DEFAULT_GENOMES = {
    "MLPClassifier": dict(
        architecture={"hidden_layer_sizes": [64], "activation": "relu"},
        hyperparameters={"alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 300},
    ),
    "RandomForestClassifier": dict(
        architecture={"n_estimators": 100, "max_depth": None},
        hyperparameters={"min_samples_leaf": 1, "min_samples_split": 2},
    ),
    "SVC": dict(
        architecture={"kernel": "rbf"},
        hyperparameters={"C": 1.0, "gamma": "scale"},
    ),
    "LogisticRegression": dict(
        architecture={},
        hyperparameters={"C": 1.0, "max_iter": 2000},
    ),
}


@dataclass
class ModelGenome:
    model_type: str
    architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    data_pipeline: Dict[str, Any] = field(
        default_factory=lambda: {"scale": True, "pca_components": None})
    seed: int = 0
    model_id: str = field(default_factory=lambda: f"g_{uuid.uuid4().hex[:10]}")
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0

    @classmethod
    def default(cls, model_type: str, seed: int = 0) -> "ModelGenome":
        base = DEFAULT_GENOMES[model_type]
        return cls(model_type=model_type,
                    architecture=copy.deepcopy(base["architecture"]),
                    hyperparameters=copy.deepcopy(base["hyperparameters"]),
                    seed=seed)

    def clone(self) -> "ModelGenome":
        g = copy.deepcopy(self)
        g.model_id = f"g_{uuid.uuid4().hex[:10]}"
        return g

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def validate(self) -> None:
        import jsonschema
        jsonschema.validate(self.to_dict(), GENOME_SCHEMA)

    def safety_check(self, max_units_per_layer: int = 1024, max_layers: int = 6,
                      max_n_estimators: int = 2000, max_depth_cap: int = 500,
                      max_c: float = 1e6) -> List[str]:
        """Pre-flight resource-bound check, distinct from validate(): validate()
        checks *shape* (does this look like a Model Genome at all); this checks
        *cost/safety* (would actually training this be a reasonable thing to
        attempt). Returns a list of violation strings -- empty means safe to
        run. This is the cheap, always-on complement to safety.sandbox.SafeRunner:
        catch the obviously-unreasonable genome before spending a process fork
        and a timeout window on it.
        """
        violations: List[str] = []
        if self.model_type == "MLPClassifier":
            sizes = self.architecture.get("hidden_layer_sizes", [])
            if len(sizes) > max_layers:
                violations.append(f"hidden_layer_sizes has {len(sizes)} layers (max {max_layers})")
            for s in sizes:
                if s is None or s <= 0:
                    violations.append(f"invalid layer width {s}")
                elif s > max_units_per_layer:
                    violations.append(f"layer width {s} exceeds cap {max_units_per_layer}")
            if self.hyperparameters.get("max_iter", 0) > 20_000:
                violations.append("max_iter exceeds 20000")
        elif self.model_type == "RandomForestClassifier":
            n_est = self.architecture.get("n_estimators", 0)
            if n_est <= 0 or n_est > max_n_estimators:
                violations.append(f"n_estimators {n_est} outside (0, {max_n_estimators}]")
            depth = self.architecture.get("max_depth")
            if depth is not None and depth > max_depth_cap:
                violations.append(f"max_depth {depth} exceeds cap {max_depth_cap}")
        elif self.model_type in ("SVC", "LogisticRegression"):
            c = self.hyperparameters.get("C", 1.0)
            if c <= 0 or c > max_c:
                violations.append(f"C {c} outside (0, {max_c}]")
        else:
            violations.append(f"unrecognised model_type {self.model_type}")
        return violations

    def build_estimator(self):
        """Materialise this genome into a real, trainable sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression

        steps = []
        if self.data_pipeline.get("scale"):
            steps.append(("scale", StandardScaler()))
        pca_n = self.data_pipeline.get("pca_components")
        if pca_n:
            steps.append(("pca", PCA(n_components=pca_n, random_state=self.seed)))

        if self.model_type == "MLPClassifier":
            sizes = self.architecture.get("hidden_layer_sizes", [64])
            sizes = tuple(max(1, int(round(s))) for s in sizes) or (8,)
            clf = MLPClassifier(
                hidden_layer_sizes=sizes,
                activation=self.architecture.get("activation", "relu"),
                alpha=max(1e-8, float(self.hyperparameters.get("alpha", 1e-4))),
                learning_rate_init=max(1e-6, float(self.hyperparameters.get("learning_rate_init", 1e-3))),
                max_iter=max(10, int(round(self.hyperparameters.get("max_iter", 300)))),
                random_state=self.seed)
        elif self.model_type == "RandomForestClassifier":
            depth = self.architecture.get("max_depth")
            clf = RandomForestClassifier(
                n_estimators=max(2, int(round(self.architecture.get("n_estimators", 100)))),
                max_depth=None if depth is None else max(1, int(round(depth))),
                min_samples_leaf=max(1, int(round(self.hyperparameters.get("min_samples_leaf", 1)))),
                min_samples_split=max(2, int(round(self.hyperparameters.get("min_samples_split", 2)))),
                random_state=self.seed)
        elif self.model_type == "SVC":
            clf = SVC(kernel=self.architecture.get("kernel", "rbf"),
                      C=max(1e-6, float(self.hyperparameters.get("C", 1.0))),
                      gamma=self.hyperparameters.get("gamma", "scale"),
                      probability=False, random_state=self.seed)
        elif self.model_type == "LogisticRegression":
            clf = LogisticRegression(
                C=max(1e-6, float(self.hyperparameters.get("C", 1.0))),
                max_iter=max(10, int(round(self.hyperparameters.get("max_iter", 2000)))),
                random_state=self.seed)
        else:
            raise ValueError(f"Unknown model_type {self.model_type}")

        steps.append(("clf", clf))
        return Pipeline(steps)
