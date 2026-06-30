"""Unit tests for the lexical reranker (``agent.core.retrievers.reranking``)."""

from types import SimpleNamespace

import pytest

from agent.core.retrievers.reranking import LexicalReranker, tokenize


def _candidate(content: str, distance: float):
    """A duck-typed candidate exposing the ``.content`` / ``.distance`` the reranker reads."""
    return SimpleNamespace(content=content, distance=distance, id=content)


class TestTokenize:
    def test_lowercases_and_drops_short_tokens(self):
        assert tokenize("Red Running Shoes!") == {"red", "running", "shoes"}

    def test_empty_or_none(self):
        assert tokenize("") == set()
        assert tokenize(None) == set()

    def test_respects_min_length(self):
        assert tokenize("a bb ccc", min_length=2) == {"bb", "ccc"}


class TestLexicalReranker:
    def test_promotes_chunk_sharing_query_terms_over_closer_vector_match(self):
        # Chunk A is semantically closer (distance 0.0) but shares no query term.
        # Chunk B is farther (distance 0.5) but contains the exact query term "asic".
        # With balanced weights the lexical overlap should let B overtake A.
        reranker = LexicalReranker()
        a = _candidate("generic description text", distance=0.0)
        b = _candidate("asic cross-encoder model xyz", distance=0.5)
        results = reranker.rerank("asic model", [a, b])
        assert results[0].id == b.id

    def test_keeps_pool_intact_no_dropping(self):
        reranker = LexicalReranker()
        a = _candidate("alpha", distance=0.1)
        b = _candidate("beta", distance=0.2)
        results = reranker.rerank("query with no overlap", [a, b])
        assert {r.id for r in results} == {a.id, b.id}

    def test_pure_vector_when_no_query_tokens(self):
        # With an empty query the lexical signal is 0 for everyone, so ordering follows vector sim.
        reranker = LexicalReranker()
        near = _candidate("near", distance=0.0)
        far = _candidate("far", distance=0.9)
        results = reranker.rerank("", [far, near])
        assert [r.id for r in results] == [near.id, far.id]

    def test_tie_break_is_input_order(self):
        reranker = LexicalReranker()
        a = _candidate("same text", distance=0.3)
        b = _candidate("same text", distance=0.3)
        results = reranker.rerank("query", [a, b])
        assert [r.id for r in results] == [a.id, b.id]

    def test_weights_are_normalised(self):
        # Passing 4:1 should normalise to 0.8/0.2 and still accept the combination.
        reranker = LexicalReranker(vector_weight=4.0, lexical_weight=1.0)
        assert reranker.vector_weight == pytest.approx(0.8)
        assert reranker.lexical_weight == pytest.approx(0.2)

    def test_zero_total_weight_raises(self):
        with pytest.raises(ValueError):
            LexicalReranker(vector_weight=0.0, lexical_weight=0.0)

    def test_missing_distance_treated_as_zero_similarity(self):
        reranker = LexicalReranker()
        candidate = SimpleNamespace(content="asic model", distance=None, id="x")
        results = reranker.rerank("asic model", [candidate])
        assert results[0].id == "x"
