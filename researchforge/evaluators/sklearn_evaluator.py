"""Executes a Model Genome as a real, trainable scikit-learn pipeline and
reports the outcome as an ExperimentResult -- the 'Experiment Engine' in the
design doc's Sec. 9 architecture diagrams. This is genuine model
training/evaluation, not a stub: every accuracy number this system reports
comes from an actual `fit()`/`predict()` call.
"""
from __future__ import annotations
import warnings

from ..diagnosis.failure_taxonomy import ExperimentResult
from ..genome.model_genome import ModelGenome


def evaluate_genome(genome: ModelGenome, X_train, y_train, X_val, y_val,
                     metric_fn, target: float = 0.0) -> ExperimentResult:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator = genome.build_estimator()
            estimator.fit(X_train, y_train)
            train_pred = estimator.predict(X_train)
            val_pred = estimator.predict(X_val)
            train_metric = float(metric_fn(y_train, train_pred))
            val_metric = float(metric_fn(y_val, val_pred))
        return ExperimentResult(metric=val_metric, train_metric=train_metric,
                                 success=True, target=target)
    except Exception as exc:  # a genuine failure path: e.g. an invalid genome
        return ExperimentResult(metric=0.0, success=False, exception=str(exc), target=target)
