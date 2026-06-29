"""Schema and retrieval verification for the pgvector ANN indexes (YAZ-3/YAZ-4).

These tests assert the deliberate decisions made for chunk-table embeddings:

* The embedding columns are fixed-dimension ``vector(N)`` so pgvector accepts an
  HNSW index (unbounded ``vector`` columns are rejected with
  ``ERROR: column does not have dimensions``).
* HNSW indexes with the ``vector_cosine_ops`` opclass exist for the cosine
  nearest-neighbor ordering used by the retrieval repositories.
* The data_set filter path used by retrieval is indexed.
* Retrieval still returns chunks ordered by ascending cosine distance.
* The ``catalog.W001`` system check flags data sets whose configured dimension
  drifts from the shared chunk-table column dimension.
"""

import pytest
from django.db import connection
from model_bakery import baker
from pgvector.django import CosineDistance

from catalog.checks import check_data_set_embedding_dimensions
from catalog.models import (
    EMBEDDING_VECTOR_DIMENSIONS,
    DataSet,
    Document,
    DocumentChunk,
    Product,
    ProductContentChunk,
)

DIM = EMBEDDING_VECTOR_DIMENSIONS


def _unit_vector(axis: int) -> list[float]:
    """A unit vector with a single 1.0 at ``axis`` and zeros elsewhere."""
    vec = [0.0] * DIM
    vec[axis] = 1.0
    return vec


def _format_type(table: str, column: str) -> str:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = %s AND a.attname = %s
            """,
            [table, column],
        )
        return cursor.fetchone()[0]


def _index_defs(table: str) -> dict[str, str]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s",
            [table],
        )
        return {name: definition for name, definition in cursor.fetchall()}


@pytest.mark.django_db
def test_embedding_columns_have_fixed_dimensions():
    """Both chunk embedding columns must be vector(N), not unbounded ``vector``."""
    assert _format_type("catalog_documentchunk", "embedding") == f"vector({DIM})"
    assert _format_type("catalog_productcontentchunk", "embedding") == f"vector({DIM})"


@pytest.mark.django_db
def test_hnsw_cosine_indexes_exist_on_embeddings():
    for table, expected_index in [
        ("catalog_documentchunk", "document_chunk_embedding_idx"),
        ("catalog_productcontentchunk", "product_chunk_embedding_idx"),
    ]:
        indexes = _index_defs(table)
        assert expected_index in indexes, f"missing index {expected_index} on {table}"
        definition = indexes[expected_index]
        assert "USING hnsw" in definition, f"{expected_index} is not an HNSW index: {definition}"
        assert "vector_cosine_ops" in definition, (
            f"{expected_index} does not use the cosine opclass: {definition}"
        )


@pytest.mark.django_db
def test_data_set_filter_indexes_exist():
    document_indexes = _index_defs("catalog_document")
    product_indexes = _index_defs("catalog_product")
    assert "catalog_document_data_set_idx" in document_indexes
    assert "catalog_product_data_set_idx" in product_indexes


@pytest.mark.django_db
def test_retrieval_returns_chunks_ordered_by_cosine_distance():
    """Retrieval must still order chunks by ascending cosine distance to the query."""
    data_set = baker.make(DataSet)
    document = baker.make(Document, data_set=data_set, url="https://example.com/doc", title="t", content="c")

    # chunk_a is identical to the query (distance 0), chunk_c is partially aligned,
    # chunk_b is orthogonal (distance 1). Expected ascending order: a, c, b.
    chunk_a = DocumentChunk.objects.create(document=document, content="a")
    chunk_a.set_embedding(_unit_vector(0))
    chunk_a.save()

    chunk_b = DocumentChunk.objects.create(document=document, content="b")
    chunk_b.set_embedding(_unit_vector(1))
    chunk_b.save()

    chunk_c = DocumentChunk.objects.create(document=document, content="c")
    diagonal = [0.0] * DIM
    diagonal[0] = 1.0
    diagonal[1] = 1.0
    chunk_c.set_embedding(diagonal)
    chunk_c.save()

    query = _unit_vector(0)
    results = list(
        DocumentChunk.objects.annotate(distance=CosineDistance("embedding", query))
        .order_by("distance")
        .values_list("content", flat=True)
    )

    assert results == ["a", "c", "b"]


@pytest.mark.django_db
def test_product_chunk_retrieval_orders_by_cosine_distance():
    data_set = baker.make(DataSet)
    product = baker.make(
        Product, data_set=data_set, entry_id="p1", name="n", slug="s", description="d", price=1.0
    )

    near = ProductContentChunk.objects.create(product=product, content="near")
    near.set_embedding(_unit_vector(0))
    near.save()

    far = ProductContentChunk.objects.create(product=product, content="far")
    far.set_embedding(_unit_vector(1))
    far.save()

    query = _unit_vector(0)
    results = list(
        ProductContentChunk.objects.annotate(distance=CosineDistance("embedding", query))
        .order_by("distance")
        .values_list("content", flat=True)
    )
    assert results == ["near", "far"]


@pytest.mark.django_db
def test_system_check_warns_on_dimension_mismatch():
    DataSet.objects.create(name="mismatched", embedding_vector_dimensions=DIM + 1)
    warnings = check_data_set_embedding_dimensions(None)
    assert any(w.id == "catalog.W001" for w in warnings)


@pytest.mark.django_db
def test_system_check_is_quiet_when_dimensions_match():
    DataSet.objects.create(name="ok", embedding_vector_dimensions=DIM)
    assert check_data_set_embedding_dimensions(None) == []
