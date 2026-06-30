"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Pure-vector retrieval misses queries whose salient terms are rare/orthogonal to the embedding space
(exact SKUs, model numbers, branded terms). Pure keyword retrieval misses synonyms. *Hybrid*
retrieval combines both ranked lists so a hit in either survives.

The robust, parameter-light way to combine ranked lists is **Reciprocal Rank Fusion**: each item's
fused score is ``sum(1 / (k + rank_i))`` over every list it appears in, where ``k`` smooths the
contribution of very top ranks (the literature default is ``k=60``). RRF needs no score
calibration — only ranks — so it works across heterogeneous scorers (cosine distance vs. full-text
rank) without tuning weights.

This module is deliberately framework-free: it operates on ranked iterables of hashable ids and
returns a fused ordering, so both the product and document retrievers share one implementation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable, Sequence

#: Default RRF smoothing constant. The original Cormack et al. paper found ``k=60`` robust across
#: collections; it dampens the dominance of rank-1 hits so lower-ranked-but-consistent items can win.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Iterable[Hashable]],
    k: int = DEFAULT_RRF_K,
) -> list[tuple[Hashable, float]]:
    """Fuse several ranked id lists into one ordering using Reciprocal Rank Fusion.

    Args:
        ranked_lists: One or more iterables of ids, each already ordered best-first. An id may appear
            in several lists; duplicates within a single list are ignored (rank is by first sight).
        k: RRF smoothing constant (>= 1). Higher values flatten the contribution of top ranks.

    Returns:
        ``(id, fused_score)`` pairs sorted by descending fused score. Ties are broken by first-seen
        order across the input lists so the output is deterministic.
    """
    if k < 1:
        raise ValueError(f"RRF smoothing constant k must be >= 1, got {k}.")
    if not ranked_lists:
        return []

    scores: dict[Hashable, float] = defaultdict(float)
    first_seen: dict[Hashable, int] = {}
    order = 0
    for ranked in ranked_lists:
        # Each list is expected to be a proper ranking (unique items). If it is not, an item is
        # ranked by its first occurrence so duplicates do not artificially inflate its score.
        seen_in_list: set[Hashable] = set()
        rank = 0
        for item in ranked:
            if item in seen_in_list:
                continue
            seen_in_list.add(item)
            rank += 1
            if item not in first_seen:
                first_seen[item] = order
                order += 1
            scores[item] += 1.0 / (k + rank)

    # Descending score, then ascending first-seen order for a deterministic tie-break.
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
