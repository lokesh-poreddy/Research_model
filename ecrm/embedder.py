"""
Text → dense vector embedding using sentence-transformers.
Falls back to a simple TF-IDF-like hash when the library is absent.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Attempt to load sentence-transformers ───────────────────────────────────
_ST_MODEL = None


def _load_st(model_name: str = "all-MiniLM-L6-v2") -> Optional[object]:
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _ST_MODEL = SentenceTransformer(model_name)
        logger.info("SentenceTransformer loaded: %s", model_name)
        return _ST_MODEL
    except Exception as exc:
        logger.warning("sentence-transformers unavailable (%s); using fallback embedder.", exc)
        return None


def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2", dim: int = 384) -> np.ndarray:
    """
    Return a 1-D float32 numpy array of shape (dim,).
    Uses SentenceTransformer when available, else a deterministic hash-based fallback.
    """
    model = _load_st(model_name)
    if model is not None:
        vec = model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)
    # Fallback: deterministic random projection keyed on content
    return _hash_embed(text, dim)


def embed_batch(texts: List[str], model_name: str = "all-MiniLM-L6-v2", dim: int = 384) -> np.ndarray:
    """Embed a list of texts; returns (N, dim) float32 array."""
    model = _load_st(model_name)
    if model is not None:
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)
    return np.stack([_hash_embed(t, dim) for t in texts])


def _hash_embed(text: str, dim: int) -> np.ndarray:
    """Deterministic pseudo-embedding via MD5-seeded random projection."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)
