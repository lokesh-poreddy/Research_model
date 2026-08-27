"""Network-free real training task for the RDE-Bench smoke benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


@dataclass
class DigitsTask:
    """Small-data digits classification with a held-out validation/test split."""
    seed: int = 0
    n_train: int = 200
    target_score: float = 0.90

    def __post_init__(self) -> None:
        data = load_digits()
        x_train, x_rest, y_train, y_rest = train_test_split(
            data.data, data.target, train_size=self.n_train, random_state=self.seed,
            stratify=data.target,
        )
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train, y_train, test_size=0.30, random_state=self.seed, stratify=y_train,
        )
        self.x_test, self.x_holdout, self.y_test, self.y_holdout = train_test_split(
            x_rest, y_rest, test_size=0.50, random_state=self.seed, stratify=y_rest,
        )
        self.baseline_score = 0.0

    def description(self) -> str:
        return "Improve 10-class handwritten digit accuracy under a small training-data budget."

    def evaluate(self, genome_dict: Dict[str, Any]) -> Dict[str, float]:
        hp = genome_dict.get("hyperparameters", {})
        strategy = genome_dict.get("strategy_description", "").lower()
        family = genome_dict.get("data_settings", {}).get("estimator", "logistic")
        if "svm" in strategy or "svc" in strategy:
            family = "svc"
        elif "forest" in strategy:
            family = "forest"
        elif "mlp" in strategy:
            family = "mlp"
        seed = int(genome_dict.get("seed", self.seed))
        if family == "svc":
            estimator = make_pipeline(StandardScaler(), SVC(C=2.0, gamma="scale"))
        elif family == "forest":
            estimator = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1)
        elif family == "mlp":
            estimator = make_pipeline(StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(64,), alpha=float(hp.get("weight_decay", 1e-4)),
                learning_rate_init=float(hp.get("learning_rate", 1e-3)), max_iter=250,
                random_state=seed,
            ))
        else:
            estimator = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0 / max(float(hp.get("weight_decay", 1e-4)), 1e-5),
                max_iter=1000, random_state=seed,
            ))
        estimator.fit(self.x_train, self.y_train)
        train_score = float(accuracy_score(self.y_train, estimator.predict(self.x_train)))
        score = float(accuracy_score(self.y_val, estimator.predict(self.x_val)))
        test_score = float(accuracy_score(self.y_test, estimator.predict(self.x_test)))
        return {"score": score, "train_score": train_score, "test_score": test_score,
                "success": score >= self.target_score, "baseline": self.baseline_score}
