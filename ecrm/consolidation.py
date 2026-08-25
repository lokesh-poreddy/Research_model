"""
Memory consolidation and forgetting policies.

Implements:
  - Exponential decay retention score
  - Age-based pruning
  - NTR-based dampening
  - Memory half-life computation
"""
from __future__ import annotations

import math
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ecrm.memory_store import MemoryRecord

logger = logging.getLogger(__name__)


def retention_score(
    record,
    now: datetime,
    lam: float = 0.01,
) -> float:
    """
    P_retain = exp(-λ · age_in_days) · Reliability
    Higher retention = more useful, more recent.
    """
    age = (now - record.timestamp).total_seconds() / 86_400  # days
    decay = math.exp(-lam * age)
    return decay * max(record.reliability, 0.0)


def memory_half_life(lam: float) -> float:
    """Return the number of days until P_retain halves (Reliability=1 assumed)."""
    return math.log(2) / lam


def should_retain(record, now: datetime, threshold: float, lam: float) -> bool:
    return retention_score(record, now, lam) >= threshold


def prune_records(
    records: List,
    threshold: float,
    lam: float,
    max_records: int,
) -> List:
    """
    Remove records below retention threshold, then cap to max_records
    by evicting lowest-retention entries first.
    """
    now = datetime.now(timezone.utc)
    survivors = [r for r in records if should_retain(r, now, threshold, lam)]

    if len(survivors) > max_records:
        survivors.sort(key=lambda r: retention_score(r, now, lam), reverse=True)
        survivors = survivors[:max_records]
        logger.debug("Pruned memory to %d records.", max_records)

    removed = len(records) - len(survivors)
    if removed:
        logger.info("Memory consolidation: removed %d stale records.", removed)
    return survivors
