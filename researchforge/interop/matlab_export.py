"""MATLAB interoperability: exports RDE-Bench tasks and results to real
`.mat` files via `scipy.io.savemat`, which implements the actual MATLAB
MAT-file format (not a custom format that merely resembles one) -- files
written here load directly in MATLAB's `load()`, Octave's `load()`, and
`scipy.io.loadmat()` alike.

Why this exists: single-lead beat/ECG-style signal classification (the
domain `benchmarks.tasks.synthetic_ecg_task` stands in for) is commonly
analyzed with MATLAB's Signal Processing Toolbox, and the research-loop
metrics this project produces (best-so-far curves, RDE-Bench summary
statistics) are often easier to plot quickly in MATLAB than to wire up a new
Python plotting script for. Rather than re-implement any of the research
loop in MATLAB, this module exports its inputs and outputs as data, so
MATLAB is used for what it's good at (signal/matrix visualization) without
duplicating the actual research logic in a second language.

Every array is cast to `float64`/plain Python containers before export so
the resulting `.mat` file has no dependency on scikit-learn, dataclasses, or
any other Python-specific object graph -- it is pure numeric/struct data,
readable by MATLAB with no Python runtime involved.
"""
from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
from scipy.io import savemat, loadmat

from ..benchmarks.tasks import Task


def export_task_to_mat(task: Task, path: str) -> None:
    """Export one RDE-Bench task's train/val/test splits to a .mat file.

    In MATLAB:
        s = load('digits_task.mat');
        size(s.X_train)          % [n_train x n_features]
        s.target_metric
    """
    payload = {
        "task_name": task.name,
        "description": task.description,
        "X_train": np.asarray(task.X_train, dtype=np.float64),
        "y_train": np.asarray(task.y_train, dtype=np.float64).reshape(-1, 1),
        "X_val": np.asarray(task.X_val, dtype=np.float64),
        "y_val": np.asarray(task.y_val, dtype=np.float64).reshape(-1, 1),
        "X_test": np.asarray(task.X_test, dtype=np.float64),
        "y_test": np.asarray(task.y_test, dtype=np.float64).reshape(-1, 1),
        "target_metric": float(task.target_metric),
    }
    savemat(path, payload, do_compression=True)


def export_bench_results_to_mat(results: Dict[str, Dict[str, Any]], path: str) -> None:
    """Export an RDE-Bench report (as produced by benchmarks.rde_bench.run_rde_bench,
    already reduced to plain dict/JSON form as in run_demo.py) to a .mat file
    structured as one MATLAB struct per task, one field per condition.

    In MATLAB:
        r = load('rde_bench_results.mat');
        r.digits.full.best_metric_mean
        plot(mean(r.digits.full.curves, 1))   % mean best-so-far curve
    """
    mat_payload: Dict[str, Any] = {}
    for task_name, conditions in results.items():
        task_struct: Dict[str, Any] = {}
        for cond_name, summary in conditions.items():
            curves = np.asarray(summary["curves"], dtype=np.float64)  # (n_seeds, n_trials)
            task_struct[cond_name] = {
                "best_metric_mean": float(summary["best_metric_mean"]),
                "best_metric_std": float(summary["best_metric_std"]),
                "research_efficiency": float(summary["research_efficiency"]),
                "search_efficiency": float(summary["search_efficiency"]),
                "failure_repetition_rate": float(summary["failure_repetition_rate"]),
                "negative_transfer_rate": float(summary["negative_transfer_rate"]),
                "memory_utility": float(summary.get("memory_utility", 0.0)),
                "curves": curves,
            }
        mat_payload[_matlab_safe_name(task_name)] = task_struct
    savemat(path, mat_payload, do_compression=True)


def verify_roundtrip(path: str) -> Dict[str, Any]:
    """Load a .mat file back with scipy (the same reader MATLAB's own `load()`
    is format-compatible with) and return a small summary, so an export can
    be verified without needing MATLAB installed."""
    data = loadmat(path, simplify_cells=True)
    keys = [k for k in data.keys() if not k.startswith("__")]
    return {"path": path, "top_level_keys": keys}


def _matlab_safe_name(name: str) -> str:
    """MATLAB struct field names can't start with a digit or contain
    hyphens; sanitise task names accordingly."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if safe[:1].isdigit():
        safe = f"t_{safe}"
    return safe
