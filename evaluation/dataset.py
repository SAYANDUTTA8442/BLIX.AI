"""
evaluation/dataset.py
======================
Dataset utilities for Blix evaluation harness.

Provides temporal splitting to prevent data leakage in continual-learning
experiments where the system may store facts seen during adaptation and
later recall them during test — making benchmark scores overly optimistic.

B05: temporal_split() — sort by timestamp then slice into train/val/test
     so the system never sees test-phase facts during adaptation.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "temporal_split",
    "validate_split_ratios",
]


def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    tol: float = 1e-6,
) -> None:
    """
    Raise ``ValueError`` if the three ratios are invalid.

    Parameters
    ----------
    train_ratio, val_ratio, test_ratio : float
        Must each be in (0, 1) and sum to 1.0 within ``tol``.
    """
    for name, r in [("train_ratio", train_ratio),
                    ("val_ratio",   val_ratio),
                    ("test_ratio",  test_ratio)]:
        if not (0.0 < r < 1.0):
            raise ValueError(f"{name} must be in (0, 1), got {r!r}")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, "
            f"got {total} (train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )


def temporal_split(
    dataset: list[dict[str, Any]],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    date_key: str = "timestamp",
    *,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split a dataset chronologically to prevent leakage (B05).

    Samples are sorted by ``date_key`` (ascending) before splitting, so
    earlier data is always in train and later data is always in test.  This
    ensures that the system never sees future facts during adaptation, giving
    conservative and reproducible benchmark numbers.

    Parameters
    ----------
    dataset : list[dict]
        Records to split.  Each must contain ``date_key``.
    train_ratio : float
        Fraction of records for the training / adaptation phase.
    val_ratio : float
        Fraction of records for the validation / hyper-parameter phase.
    test_ratio : float
        Fraction of records for the held-out test phase.
    date_key : str
        Dictionary key containing the timestamp / date string.
        Values must be sortable (ISO-8601 strings, datetime objects, ints).
    strict : bool
        If ``True``, raise ``KeyError`` when a record is missing ``date_key``.
        If ``False`` (default), records without ``date_key`` sort last.

    Returns
    -------
    train, val, test : tuple of list[dict]
        Three non-overlapping subsets in chronological order.

    Raises
    ------
    ValueError
        If ratios are out of range or do not sum to 1.0.
    ValueError
        If ``dataset`` is empty.

    Examples
    --------
    >>> data = [{"id": i, "timestamp": f"2024-01-{10+i:02d}"} for i in range(10)]
    >>> train, val, test = temporal_split(data)
    >>> len(train), len(val), len(test)
    (6, 2, 2)
    >>> train[-1]["id"] < val[0]["id"] < test[0]["id"]
    True
    """
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    if not dataset:
        raise ValueError("temporal_split: dataset must not be empty")

    _SENTINEL = object()

    def _sort_key(record: dict[str, Any]) -> Any:
        v = record.get(date_key, _SENTINEL)
        if v is _SENTINEL:
            if strict:
                raise KeyError(
                    f"temporal_split: record missing date_key={date_key!r}: {record!r}"
                )
            # Put records without timestamps at the end (conservative)
            return "\xff" * 30
        return v

    sorted_data = sorted(dataset, key=_sort_key)
    n = len(sorted_data)

    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    # Ensure val and test are non-empty when n >= 3
    if n >= 3:
        train_end = max(1, min(train_end, n - 2))
        val_end   = max(train_end + 1, min(val_end, n - 1))

    train = sorted_data[:train_end]
    val   = sorted_data[train_end:val_end]
    test  = sorted_data[val_end:]

    log.debug(
        "temporal_split: n=%d → train=%d val=%d test=%d (key=%r)",
        n, len(train), len(val), len(test), date_key,
    )
    return train, val, test
