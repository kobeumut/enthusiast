"""Diversity via Maximal Marginal Relevance (MMR).

Without a diversity step, a vector retriever happily returns many overlapping chunks from the same
source document – the AC-2000 manual's "safety" section appears in four near-duplicate chunks and
crowds out the "cleaning" and "warranty" sections the user also needs. The context window fills with
redundancy and answer coverage drops.

Maximal Marginal Relevance (Carbonell & Goldstein, 1998) fixes this by selecting chunks that are
both *relevant* to the query and *diverse* relative to what is already selected::

    argmax_c  [ lambda * rel(c) - (1 - lambda) * max_{s in selected} sim(c, s) ]

``lambda = 1`` reproduces pure relevance ranking; ``lambda = 0`` maximises novelty; ``0.5`` balances
the two. The implementation is pure-Python over the candidate embeddings (already fetched for the
rerank stage), so it adds no extra DB round-trips.

The contract is duck-typed: candidates must expose ``.embedding`` and ``.distance`` (the annotated
cosine distance), matching the chunk querysets the retrievers build.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

#: Default relevance/diversity trade-off. ``0.5`` keeps the most relevant chunk while preventing
#: near-duplicates from dominating the returned context.
DEFAULT_MMR_LAMBDA = 0.5


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors, clamped to ``[-1, 1]``.

    Returns ``0.0`` for empty/zero-norm vectors instead of raising, so a candidate whose embedding
    is missing does not corrupt the MMR selection.
    """
    if a is None or b is None:
        return 0.0
    try:
        if len(a) != len(b) or len(a) == 0:
            return 0.0
    except TypeError:
        # ``a`` / ``b`` are not sized (unexpected type) – treat as no signal.
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    similarity = dot / math.sqrt(norm_a * norm_b)
    # Guard against float drift slightly outside the theoretical range.
    return max(-1.0, min(1.0, similarity))


def _relevance_from_distance(distance: float | None) -> float:
    """Convert an annotated cosine distance into a [0, 1] relevance (similarity) score."""
    if distance is None or math.isnan(distance):
        return 0.0
    return max(0.0, min(1.0, 1.0 - distance))


def maximal_marginal_relevance(
    candidates: Sequence[Any],
    lambda_: float = DEFAULT_MMR_LAMBDA,
    limit: int | None = None,
) -> list[Any]:
    """Select a relevant-yet-diverse subset of ``candidates`` via MMR.

    Args:
        candidates: The (already relevance-ordered or unordered) candidate pool. Each item must
            expose ``.embedding`` and ``.distance``. Order is only a tie-breaker.
        lambda_: Relevance/diversity trade-off in ``[0, 1]``. ``1`` = pure relevance (no diversity),
            ``0`` = pure novelty.
        limit: Maximum number of candidates to select. Defaults to the whole pool.

    Returns:
        The selected candidates, best-first, as a new list.
    """
    if not candidates:
        return []
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}.")
    if limit is None or limit < 0:
        limit = len(candidates)
    limit = min(limit, len(candidates))

    relevances = [_relevance_from_distance(getattr(c, "distance", None)) for c in candidates]
    embeddings = [getattr(c, "embedding", None) for c in candidates]

    selected_indices: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected_indices) < limit:
        best_index = None
        best_score = None
        for index in remaining:
            relevance = relevances[index]
            if selected_indices:
                max_sim = max(
                    cosine_similarity(embeddings[index], embeddings[s]) for s in selected_indices
                )
            else:
                max_sim = 0.0
            score = lambda_ * relevance - (1.0 - lambda_) * max_sim
            if best_score is None or score > best_score or (score == best_score and index < best_index):
                best_score = score
                best_index = index
        selected_indices.append(best_index)
        remaining.discard(best_index)

    return [candidates[index] for index in selected_indices]
