"""Candidate reranking for the retrieval pipeline.

Vector nearest-neighbour gives a strong first cut, but the top chunk is not always the *best*
chunk for the query: embeddings blur exact terms (SKUs, model numbers, brand tokens), and short
chunks can rank high on surface similarity while missing the user's actual keywords. A cheap
second-stage reranker over a ~30–50 candidate pool is the single highest-leverage RAG quality
improvement, and unlike a cross-encoder it needs no model call and is fully deterministic/testable.

This module defines a small reranker contract (:class:`BaseReranker`) and a default
:class:`LexicalReranker` that blends the vector similarity (from the annotated cosine distance) with
a lexical coverage score (fraction of query terms present in the chunk). The blend is a convex
combination, so reranking never invents a candidate that was not retrieved – it only reorders the
pool. A cross-encoder / LLM reranker can be added later by subclassing ``BaseReranker``; the
retriever wiring is agnostic to the implementation.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

#: Tokens shorter than this are ignored as stop-ish noise ("a", "the", "for").
DEFAULT_MIN_TOKEN_LENGTH = 2

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str, min_length: int = DEFAULT_MIN_TOKEN_LENGTH) -> set[str]:
    """Lowercase ``text`` into a set of alphabetic tokens, dropping very short tokens.

    Args:
        text: Free-form text.
        min_length: Minimum token length to keep.

    Returns:
        A set of lowercased word tokens.
    """
    return {token.lower() for token in _TOKEN_RE.findall(text or "") if len(token) >= min_length}


class BaseReranker(ABC):
    """Reorders a pool of retrieved candidates given the original query."""

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[Any]) -> list[Any]:
        """Return ``candidates`` reordered best-first.

        Args:
            query: The original natural-language query.
            candidates: The candidate pool (objects exposing ``.content`` and ``.distance``).

        Returns:
            The same candidate objects, reordered, as a new list.
        """


class LexicalReranker(BaseReranker):
    """Blends vector similarity with lexical coverage to rerank a candidate pool.

    The rerank score for a candidate is::

        score = vector_weight * sim + lexical_weight * coverage

    where ``sim = clamp(1 - cosine_distance, 0, 1)`` (the annotated ``.distance``),
    ``coverage = |query_tokens ∩ content_tokens| / max(1, |query_tokens|)`` and the weights are
    normalised to sum to 1. Defaults weight vector and lexical signal equally (0.5 / 0.5).

    Candidates whose content shares no query term still survive on their vector score, so reranking
    is monotone with respect to the retrieved set – it only reorders, never drops.
    """

    def __init__(
        self,
        vector_weight: float = 0.5,
        lexical_weight: float = 0.5,
        min_token_length: int = DEFAULT_MIN_TOKEN_LENGTH,
    ):
        total = vector_weight + lexical_weight
        if total <= 0:
            raise ValueError("vector_weight + lexical_weight must be positive.")
        self.vector_weight = vector_weight / total
        self.lexical_weight = lexical_weight / total
        self.min_token_length = min_token_length

    def rerank(self, query: str, candidates: Sequence[Any]) -> list[Any]:
        query_tokens = tokenize(query, self.min_token_length)
        token_norm = max(1, len(query_tokens))

        scored: list[tuple[float, int, Any]] = []
        for tie_breaker, candidate in enumerate(candidates):
            content_tokens = tokenize(getattr(candidate, "content", ""), self.min_token_length)
            overlap = len(query_tokens & content_tokens) / token_norm if query_tokens else 0.0
            vector_similarity = self._vector_similarity(getattr(candidate, "distance", None))
            score = self.vector_weight * vector_similarity + self.lexical_weight * overlap
            scored.append((score, tie_breaker, candidate))

        # Descending score; ascending original order keeps reranking deterministic on ties.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [candidate for _, _, candidate in scored]

    @staticmethod
    def _vector_similarity(distance: float | None) -> float:
        """Convert an annotated cosine distance into a [0, 1] similarity.

        Cosine distance ranges over [0, 2]; for unit-normalised embeddings it is effectively [0, 1].
        Anything missing or outside the expected range is clamped rather than trusted blindly.
        """
        if distance is None or math.isnan(distance):
            return 0.0
        return max(0.0, min(1.0, 1.0 - distance))
