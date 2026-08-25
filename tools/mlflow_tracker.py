"""
MLflow experiment tracker integration.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Thin wrapper around MLflow for experiment logging."""

    def __init__(self, tracking_uri: str, experiment_name: str):
        self._enabled = False
        try:
            import mlflow  # type: ignore
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            self._enabled = True
            logger.info("MLflow tracker initialised (uri=%s).", tracking_uri)
        except Exception as exc:
            logger.warning("MLflow unavailable: %s – tracking disabled.", exc)

    def start_run(self, run_name: Optional[str] = None) -> None:
        if self._enabled:
            self._mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        if self._enabled:
            self._mlflow.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._enabled:
            try:
                self._mlflow.log_params(params)
            except Exception as exc:
                logger.debug("MLflow log_params failed: %s", exc)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._enabled:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception as exc:
                logger.debug("MLflow log_metrics failed: %s", exc)

    def log_artifact(self, path: str) -> None:
        if self._enabled:
            try:
                self._mlflow.log_artifact(path)
            except Exception as exc:
                logger.debug("MLflow log_artifact failed: %s", exc)

    def set_tag(self, key: str, value: str) -> None:
        if self._enabled:
            try:
                self._mlflow.set_tag(key, value)
            except Exception as exc:
                logger.debug("MLflow set_tag failed: %s", exc)
