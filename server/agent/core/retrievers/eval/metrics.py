"""Tiny ranking-quality metrics used by the retrieval evaluation harness.

These are the standard IR metrics (Precision@K, Recall@K, MRR) over id lists, kept dependency-free
so the eval harness can run anywhere the test suite runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def precision_at_k(retrieved: Sequence[object], relevant: Iterable[object], k: int) -> float:
    """Fraction of the top-``k`` retrieved ids that are relevant.

    The denominator is ``min(k, len(retrieved))`` so a filter that narrows the result set is not
    penalised for returning fewer than ``k`` items.
    """
    relevant_set = set(relevant)
    top_k = list(retrieved)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[object], relevant: Iterable[object], k: int) -> float:
    """Fraction of relevant ids recovered in the top-``k`` (1.0 when all relevant are found)."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved: Sequence[object], relevant: Iterable[object]) -> float:
    """``1 / rank`` of the first relevant id (0.0 when none of the retrieved ids is relevant)."""
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0
