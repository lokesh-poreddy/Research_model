"""Lightweight text embeddings for the ECRM memory store.

The design doc calls for 768/1024-dim vectors from an LLM encoder (Sec. 3.4:
"Vectors can be 768- or 1024-dim from an LLM encoder"). Running a real neural
encoder needs a downloaded pretrained model, which this offline sandbox has
no network access to fetch, so this module uses scikit-learn's
HashingVectorizer instead: a real, deterministic bag-of-words embedding with
no vocabulary-fitting step and no network dependency.

Every other module only depends on `embed(text) -> np.ndarray` and
`cosine_sim(a, b) -> float`, so swapping in a sentence-transformer or an
API-based embedding call is a one-function change, not a redesign.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

_VEC = HashingVectorizer(n_features=256, alternate_sign=False, norm="l2")


def embed(text: str) -> np.ndarray:
    return _VEC.transform([text]).toarray()[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
