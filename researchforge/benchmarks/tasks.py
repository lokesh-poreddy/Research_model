"""RDE-Bench task definitions.

Real, runnable data with no network access required:
  - `digits_task`: scikit-learn's bundled `load_digits()` (ships with the
    package itself, no download). The training split is deliberately kept
    small (see `n_train`) so that model-family/hyperparameter choices
    actually matter -- with the full ~1080 training examples, even a default
    LogisticRegression clears 95%+ and there is nothing left for the search
    loop to demonstrate.
  - `synthetic_ecg_task`: a normal-vs-arrhythmic binary classification task
    over synthetic single-lead beats (parametric Gaussian bumps + noise),
    generated with numpy. This is a network-free surrogate for the
    PhysioNet/MIT-BIH task in the design doc's healthcare track (Sec. 6) --
    swap this loader for real PhysioNet data, or for the digitized output of
    Lokesh's own ECG-DigitizeNet pipeline, to move from the offline surrogate
    to the real benchmark the design doc specifies.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@dataclass
class Task:
    name: str
    description: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    metric_fn: Callable
    target_metric: float


def digits_task(seed: int = 0, n_train: int = 60) -> Task:
    from sklearn.datasets import load_digits
    data = load_digits()
    X, y = data.data, data.target

    # Small, fixed-size training split: enough signal to learn from, small
    # enough that architecture/hyperparameter choices move accuracy a lot.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=n_train, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)

    return Task(name="digits", description="10-class handwritten digit classification "
                                            f"(sklearn digits, {n_train}-sample training budget)",
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test, metric_fn=accuracy_score, target_metric=0.90)


def synthetic_ecg_task(seed: int = 0, n_samples: int = 700, noise: float = 0.30) -> Task:
    """Binary 'normal vs arrhythmic' classification over synthetic single-lead
    beats: parametric Gaussian-bump waveforms + noise. Deliberately noisy
    (real single-lead arrhythmia discrimination is a hard, overlapping-class
    problem) so a default model lands well below target and there's genuine,
    but not unlimited, room for search to close the gap -- consistent with
    the design doc's framing that RDE-Bench tasks have "no known optimum."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 96)

    def beat(centers, widths, heights, nz):
        sig = np.zeros_like(t)
        for c, w, h in zip(centers, widths, heights):
            sig = sig + h * np.exp(-((t - c) ** 2) / (2 * w ** 2))
        return sig + rng.normal(0, nz, size=t.shape)

    X, y = [], []
    n_pos = n_samples // 2
    for _ in range(n_pos):
        X.append(beat(centers=[0.3, 0.5, 0.7], widths=[0.05, 0.035, 0.06],
                       heights=[0.35, 0.95, 0.3], nz=noise))
        y.append(0)  # normal
    for _ in range(n_samples - n_pos):
        jitter = rng.uniform(-0.025, 0.025)
        X.append(beat(centers=[0.3 + jitter, 0.5 + jitter * 1.2, 0.7],
                       widths=[0.055, 0.03, 0.07],
                       heights=[0.45, rng.uniform(0.75, 1.15), 0.35], nz=noise))
        y.append(1)  # arrhythmic
    X, y = np.array(X), np.array(y)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=80, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)

    return Task(name="synthetic_ecg",
                description="Synthetic single-lead beat classification (normal vs "
                            "arrhythmic) -- offline surrogate for the PhysioNet/ECG track",
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test, metric_fn=accuracy_score, target_metric=0.78)
