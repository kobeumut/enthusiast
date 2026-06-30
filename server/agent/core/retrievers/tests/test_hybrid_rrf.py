"""Unit tests for Reciprocal Rank Fusion (``agent.core.retrievers.hybrid``).

RRF is the shared hybrid-retrieval combiner, so it is tested in isolation here. The retrievers'
integration tests cover the wiring into the pgvector query path.
"""

from agent.core.retrievers.hybrid import DEFAULT_RRF_K, reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_fuses_two_lists_and_ranks_consistent_items_first(self):
        # Item 'b' is rank 1 in list A and rank 2 in list B -> highest fused score.
        # Item 'a' is rank 2 in A and absent in B. 'c' is rank 1 in B only.
        fused = reciprocal_rank_fusion([["b", "a"], ["c", "b"]])
        ids = [item_id for item_id, _ in fused]
        # 'b' appears in both lists so it must lead; the union has no other shared item.
        assert ids[0] == "b"
        assert set(ids) == {"a", "b", "c"}

    def test_scores_use_reciprocal_rank_formula(self):
        fused = dict(reciprocal_rank_fusion([["x"], ["x"]]))
        # 1/(k+1) from each of the two lists.
        assert fused["x"] == 2 * (1.0 / (DEFAULT_RRF_K + 1))

    def test_descending_score_order(self):
        # 'shared' is in both lists at rank 1; 'only_a' only in list A at rank 1.
        # shared: 2/(k+1); only_a: 1/(k+1) -> shared first.
        fused = reciprocal_rank_fusion([["shared", "only_a"], ["shared", "only_b"]])
        assert fused[0][0] == "shared"
        assert {pair[0] for pair in fused} == {"shared", "only_a", "only_b"}

    def test_ties_broken_by_first_seen_order(self):
        # Two items never co-occur and never share a rank, so determinism rests on first-seen.
        fused = reciprocal_rank_fusion([["a", "b", "c"], []])
        assert [item_id for item_id, _ in fused] == ["a", "b", "c"]

    def test_empty_input_returns_empty(self):
        assert reciprocal_rank_fusion([]) == []

    def test_duplicate_ids_within_one_list_counted_once(self):
        fused = dict(reciprocal_rank_fusion([["a", "a", "a"]]))
        # Rank is by first sight; duplicates do not stack the reciprocal contribution.
        assert fused["a"] == 1.0 / (DEFAULT_RRF_K + 1)

    def test_invalid_k_raises(self):
        import pytest

        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"]], k=0)
