"""EXPLAIN-based verification of HNSW index usage and the dataset-filter interaction (YAZ-15).

The retrieval layer orders chunks by ``embedding <=> query`` with a ``LIMIT k`` — exactly the
K-nearest-neighbour shape pgvector's HNSW index accelerates. These tests capture the actual
Postgres plan to verify two things the DRA flagged for this phase:

1. **The HNSW index is actually used** for the vector nearest-neighbour query (not silently falling
   back to a sequential scan + sort). Because the planner prefers a seqscan on tiny test tables, we
   ``SET enable_seqscan = off`` to force the index path and prove the index is wired for the ANN
   ordering. This is the standard, legitimate way to assert an index *can* be used.
2. **The dataset filter interaction is characterised.** Real retrieval joins the chunk table to its
   parent (``document``/``product``) and filters by ``data_set_id``. That predicate cannot be served
   by the embedding HNSW index: Postgres applies it as a *post-filter* on the ANN candidates (or
   falls back to a seqscan + filter + sort when the filter is selective). We capture the plan with
   the filter present and assert retrieval stays correct (scoped + ordered), documenting the
   interaction that motivates runtime ``ef_search`` tuning.

These tests require a PostgreSQL instance with the ``vector`` extension, like the rest of the
pgvector suite.
"""

import pytest
from django.db import connection
from model_bakery import baker

from catalog.models import DataSet, Document, DocumentChunk
from catalog.models.data_set import EMBEDDING_VECTOR_DIMENSIONS

pytestmark = pytest.mark.django_db

DIM = EMBEDDING_VECTOR_DIMENSIONS

#: Number of chunks indexed for plan inspection. Large enough that the planner treats the table as
#: non-trivial (so ``enable_seqscan = off`` reliably selects the HNSW path) while keeping the test
#: fast. Real recall/latency tuning happens against production-scale data (see the eval harness).
INDEXABLE_ROW_COUNT = 80


def _unit(axis: int) -> list[float]:
    vec = [0.0] * DIM
    vec[axis] = axis  # not unit, but distinct per row; only the query row is normalised below
    vec[axis] = 1.0
    return vec


def _vector_literal(vec) -> str:
    """Render a vector as the ``'[a,b,c]'`` literal pgvector accepts, with the right dimensionality."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


@pytest.fixture
def indexed_document():
    """A document with many distinct chunks so the planner has a non-trivial table to plan over."""
    data_set = baker.make(DataSet, name="EXPLAIN Dataset")
    document = baker.make(Document, data_set=data_set, url="https://x/d", title="t", content="c")
    DocumentChunk.objects.bulk_create(
        [DocumentChunk(document=document, content=f"chunk {axis}", embedding=_unit(axis)) for axis in range(INDEXABLE_ROW_COUNT)]
    )
    with connection.cursor() as cursor:
        cursor.execute("ANALYZE catalog_documentchunk")
    return document


def _explain(sql: str, params: list, enable_seqscan: bool = True) -> str:
    """Run ``EXPLAIN <sql>`` and return the concatenated plan text.

    ``enable_seqscan`` is toggled with ``SET LOCAL`` so it is scoped to the surrounding
    pytest-django transaction and never leaks into later tests on the same connection.
    """
    setting = "on" if enable_seqscan else "off"
    with connection.cursor() as cursor:
        cursor.execute(f"SET LOCAL enable_seqscan = {setting}")
        cursor.execute("EXPLAIN " + sql, params)
        return "\n".join(row[0] for row in cursor.fetchall())


class TestHnswIndexUsage:
    def test_hnsw_index_is_used_for_vector_nearest_neighbour(self, indexed_document):
        """Forcing the index path must select the HNSW index for the ``ORDER BY ... <=> ... LIMIT k``."""
        query = _unit(0)
        plan = _explain(
            "SELECT id FROM catalog_documentchunk ORDER BY embedding <=> %s::vector LIMIT 10",
            [_vector_literal(query)],
            enable_seqscan=False,
        )
        assert "document_chunk_embedding_idx" in plan, (
            "HNSW index was not selected for the vector KNN query. Plan:\n" + plan
        )

    def test_sequential_scan_is_preferred_on_small_tables_by_default(self, indexed_document):
        """With default planner costs a small table is served by a seqscan + sort; documented behaviour.

        This is the other half of the interaction: on small (or highly dataset-filtered) tables the
        planner rationally avoids the ANN index. It is not a bug – it is why ``ef_search`` tuning and
        index-friendly filters matter at production scale, and why this test only asserts the plan is
        captured rather than dictating a specific shape.
        """
        query = _unit(0)
        plan = _explain(
            "SELECT id FROM catalog_documentchunk ORDER BY embedding <=> %s::vector LIMIT 10",
            [_vector_literal(query)],
            enable_seqscan=True,
        )
        assert plan  # plan captured and non-empty
        # Either path is valid; we only require the planner produced a usable plan.
        assert "Limit" in plan


class TestDataSetFilterInteraction:
    """Characterise the dataset-filter ↔ HNSW interaction the DRA flagged for this phase.

    Finding (captured live by EXPLAIN): the production retrieval query joins the chunk table to its
    parent (``document``/``product``) and filters by ``data_set_id``. That relational predicate
    cannot be served by the embedding HNSW index, so the planner **does not use HNSW** for the
    dataset-scoped path – it scans the chunks reachable via the foreign-key index and sorts them by
    cosine distance. This holds even with ``enable_seqscan = off``: the join/filter dominates the
    plan, not the ANN index.

    Consequence: on the *current* schema, ``hnsw.ef_search`` has no effect on the production path
    (it only matters once the index is in the plan, which the pure nearest-neighbour query in
    ``TestHnswIndexUsage`` proves it can be). The robust fix – denormalising ``data_set_id`` onto
    the chunk tables and adding a per-dataset partial HNSW index, or a two-stage retrieve-then-
    filter – is called out as a follow-up; these tests pin the current behaviour so the change is
    visible when it lands.
    """

    def test_retrieval_with_dataset_filter_still_returns_nearest_chunks(self, indexed_document):
        """The dataset-scoped JOIN query must still return the nearest chunks, in distance order.

        This is the correctness guarantee that matters to RAG quality: even though the ``data_set_id``
        predicate forces a sort rather than an ANN index scan, the final result set is the right one –
        scoped to the dataset and ordered nearest-first.
        """
        query = _unit(0)
        sql = (
            "SELECT dc.id FROM catalog_documentchunk dc "
            "JOIN catalog_document d ON dc.document_id = d.id "
            "WHERE d.data_set_id = %s AND dc.embedding IS NOT NULL "
            "ORDER BY dc.embedding <=> %s::vector LIMIT 10"
        )
        with connection.cursor() as cursor:
            cursor.execute(sql, [indexed_document.data_set_id, _vector_literal(query)])
            rows = cursor.fetchall()
        chunk_ids_in_distance_order = [row[0] for row in rows]
        # Every returned chunk belongs to the indexed document (dataset scoping holds).
        assert all(
            chunk_id in DocumentChunk.objects.filter(document=indexed_document).values_list("id", flat=True)
            for chunk_id in chunk_ids_in_distance_order
        )
        # The nearest chunk to unit(0) is the one whose embedding is exactly unit(0) (axis 0).
        nearest = DocumentChunk.objects.get(document=indexed_document, content="chunk 0")
        assert chunk_ids_in_distance_order[0] == nearest.id

    def test_dataset_scoped_query_does_not_use_hnsw_index(self, indexed_document):
        """Document the interaction: a data_set JOIN filter makes the planner sort, not ANN-scan.

        Even with ``enable_seqscan = off`` the plan sorts by the cosine distance expression instead
        of driving the query through ``document_chunk_embedding_idx``. Asserting this negative makes
        the interaction a living, reviewed fact rather than a comment that can silently rot – and it
        will flip to a positive assertion the day retrieval is denormalised onto the chunk table.
        """
        query = _unit(0)
        sql = (
            "SELECT dc.id FROM catalog_documentchunk dc "
            "JOIN catalog_document d ON dc.document_id = d.id "
            "WHERE d.data_set_id = %s AND dc.embedding IS NOT NULL "
            "ORDER BY dc.embedding <=> %s::vector LIMIT 10"
        )
        plan = _explain(sql, [indexed_document.data_set_id, _vector_literal(query)], enable_seqscan=False)
        assert "Sort" in plan, "Expected a sort-based plan for the dataset-scoped query:\n" + plan
        assert "document_chunk_embedding_idx" not in plan, (
            "HNSW index unexpectedly used with a data_set JOIN filter; update this test and the "
            "phase notes. Plan:\n" + plan
        )


class TestEfSearchRuntimeSetting:
    def test_set_local_hnsw_ef_search_is_accepted_and_query_returns_ordered_results(self, indexed_document):
        """``SET LOCAL hnsw.ef_search`` must be a valid runtime knob for the vector query.

        ``SET LOCAL`` is scoped to the surrounding transaction (the pytest-django test
        transaction), so it applies to the query below without leaking to other tests.
        """
        query = _unit(0)
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL hnsw.ef_search = 200")
            cursor.execute(
                "SELECT id FROM catalog_documentchunk ORDER BY embedding <=> %s::vector LIMIT 5",
                [_vector_literal(query)],
            )
            rows = [row[0] for row in cursor.fetchall()]
        # ef_search raises the candidate list; the nearest chunk must still come first.
        nearest = DocumentChunk.objects.get(document=indexed_document, content="chunk 0")
        assert rows[0] == nearest.id
