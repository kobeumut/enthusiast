"""Unit tests for Maximal Marginal Relevance (``agent.core.retrievers.diversity``)."""

import math
from types import SimpleNamespace

import pytest

from agent.core.retrievers.diversity import cosine_similarity, maximal_marginal_relevance


def _candidate(embedding, distance):
    return SimpleNamespace(embedding=embedding, distance=distance, id=tuple(embedding))


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_is_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_length_is_zero(self):
        assert cosine_similarity([1.0], [1.0, 1.0]) == 0.0


class TestMaximalMarginalRelevance:
    def test_lambda_one_reproduces_relevance_order(self):
        # With lambda=1 MMR ignores diversity and follows relevance (1 - distance).
        near = _candidate([1.0, 0.0, 0.0], 0.0)   # relevance 1.0
        mid = _candidate([0.0, 1.0, 0.0], 0.5)   # relevance 0.5
        far = _candidate([0.0, 0.0, 1.0], 1.0)   # relevance 0.0
        results = maximal_marginal_relevance([far, mid, near], lambda_=1.0)
        assert [r.id for r in results] == [near.id, mid.id, far.id]

    def test_lambda_zero_maximises_novelty(self):
        # With lambda=0 the second pick is the most dissimilar to the first, regardless of relevance.
        first = _candidate([1.0, 0.0], 0.0)       # most relevant -> picked first
        near_first = _candidate([0.95, 0.05], 0.01)  # very close to first, slightly less relevant
        orthogonal = _candidate([0.0, 1.0], 0.5)   # far from first
        results = maximal_marginal_relevance([first, near_first, orthogonal], lambda_=0.0, limit=2)
        assert results[0].id == first.id
        assert results[1].id == orthogonal.id

    def test_balanced_lambda_deprefers_near_duplicates(self):
        # Two near-duplicate "safety" chunks and one unrelated "cleaning" chunk.
        # relevance is equal; MMR should not return both duplicates back-to-back at limit=2.
        safety_a = _candidate([1.0, 0.0, 0.0], 0.0)
        safety_b = _candidate([0.99, 0.0, 0.0], 0.0)  # ~duplicate of safety_a
        cleaning = _candidate([0.0, 1.0, 0.0], 0.2)
        results = maximal_marginal_relevance([safety_a, safety_b, cleaning], lambda_=0.5, limit=2)
        ids = {r.id for r in results}
        assert ids == {safety_a.id, cleaning.id}

    def test_respects_limit(self):
        candidates = [_candidate([float(i == j) for j in range(5)], 0.1 * i) for i in range(5)]
        results = maximal_marginal_relevance(candidates, lambda_=0.5, limit=2)
        assert len(results) == 2

    def test_empty_returns_empty(self):
        assert maximal_marginal_relevance([], lambda_=0.5) == []

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            maximal_marginal_relevance([_candidate([1.0], 0.0)], lambda_=1.5)

    def test_missing_embedding_does_not_crash(self):
        candidate = SimpleNamespace(embedding=None, distance=0.0, id="x")
        results = maximal_marginal_relevance([candidate], lambda_=0.5)
        assert [r.id for r in results] == ["x"]

    def test_similarity_helper_is_clamped(self):
        # Slightly over-unit dot/norms should still clamp within [-1, 1].
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) <= 1.0
        assert math.isfinite(cosine_similarity([2.0, 0.0], [2.0, 0.0]))
