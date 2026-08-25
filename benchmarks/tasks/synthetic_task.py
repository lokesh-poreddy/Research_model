"""
Synthetic time-series task for RDE-Bench.
Tests whether the agent can discover the underlying pattern of a
noisy dynamical system.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict


class SyntheticTimeSeriesTask:
    """
    RDE-Bench Synthetic Track.
    Generate toy sine/chaotic series; agent tries to fit it with minimum MSE.
    """

    name = "SyntheticTimeSeries"
    baseline_score = 0.5   # Baseline mean predictor MSE (inverted to score)
    target_score = 0.95

    def __init__(self, mock: bool = True, series_type: str = "sine"):
        self.mock = mock
        self.series_type = series_type  # "sine" | "lorenz" | "logistic"

    def evaluate(self, genome_dict: Dict[str, Any]) -> Dict[str, float]:
        if self.mock:
            return self._mock_eval(genome_dict)
        return self._real_eval(genome_dict)

    def _mock_eval(self, genome_dict: Dict[str, Any]) -> Dict[str, float]:
        gen = genome_dict.get("generation", 0)
        n_layers = len(genome_dict.get("architecture", {}).get("layers", []))
        score = self.baseline_score + gen * 0.015 + n_layers * 0.01 + random.gauss(0, 0.02)
        score = max(0.0, min(1.0, score))
        mse = (1.0 - score) * 0.5
        return {"score": score, "mse": mse}

    def _real_eval(self, genome_dict: Dict[str, Any]) -> Dict[str, float]:
        """Fit generated series using a simple linear model as baseline."""
        import numpy as np
        from sklearn.linear_model import Ridge

        n = 500
        t = np.linspace(0, 4 * math.pi, n)
        if self.series_type == "sine":
            y = np.sin(t) + np.random.normal(0, 0.1, n)
        else:
            y = np.cos(t) * np.sin(2 * t) + np.random.normal(0, 0.1, n)

        X = np.stack([t, t ** 2, np.sin(t)], axis=1)
        split = int(0.8 * n)
        model = Ridge(alpha=1.0)
        model.fit(X[:split], y[:split])
        preds = model.predict(X[split:])
        mse = float(np.mean((preds - y[split:]) ** 2))
        score = max(0.0, 1.0 - mse)
        return {"score": score, "mse": mse}

    def description(self) -> str:
        return (
            f"Discover the best model for a noisy {self.series_type} time-series. "
            "Score = 1 - normalized MSE. Baseline: 0.50. Target: >0.95."
        )
