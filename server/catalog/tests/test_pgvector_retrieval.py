"""Verifies that document and product retrieval are backed by pgvector embeddings stored in PostgreSQL.

These tests store real embeddings in the pgvector ``VectorField`` and exercise the same
``CosineDistance`` query path the agents use at runtime. They require a PostgreSQL instance with
the ``vector`` extension (the CI test service uses ``pgvector/pgvector``), which is the same store
the application uses in development and production.
"""

import pytest
from model_bakery import baker
from pgvector.django import CosineDistance

from agent.core.repositories import DjangoDocumentChunkRepository, DjangoProductChunkRepository
from catalog.models import DataSet, Document, DocumentChunk, Product, ProductContentChunk
from catalog.models.data_set import EMBEDDING_VECTOR_DIMENSIONS

pytestmark = pytest.mark.django_db


def _unit_vector(position: int, dimensions: int = EMBEDDING_VECTOR_DIMENSIONS) -> list[float]:
    """A deterministic unit vector with a single non-zero component at ``position``.

    The chunk embedding columns are ``vector(EMBEDDING_VECTOR_DIMENSIONS)`` (see migration
    ``0014_pgvector_ann_indexes``), so stored and query vectors must match that dimensionality.
    """
    vector = [0.0] * dimensions
    vector[position] = 1.0
    return vector


@pytest.fixture
def data_set():
    return baker.make(DataSet, name="pgvector Retrieval Dataset")


class TestProductRetrievalUsesPgvector:
    def test_returns_chunks_ordered_by_cosine_distance_and_scoped_to_data_set(self, data_set):
        product_a = baker.make(Product, data_set=data_set, entry_id="a", name="A", slug="a", price=1)
        product_b = baker.make(Product, data_set=data_set, entry_id="b", name="B", slug="b", price=1)
        # chunk_a is collinear with the query vector (distance 0), chunk_b is orthogonal (distance 1).
        chunk_a = ProductContentChunk.objects.create(
            product=product_a, content="red running shoes", embedding=_unit_vector(0)
        )
        ProductContentChunk.objects.create(product=product_b, content="blue cotton shirts", embedding=_unit_vector(1))

        repository = DjangoProductChunkRepository(ProductContentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id, distance=CosineDistance("embedding", _unit_vector(0))
            )
        )

        assert len(results) == 2
        assert results[0].id == chunk_a.id
        assert results[0].content == "red running shoes"
        # pgvector annotates the computed distance; the nearest chunk is closest to 0.
        assert results[0].distance <= results[1].distance

    def test_results_are_isolated_per_data_set(self, data_set):
        other_data_set = baker.make(DataSet, name="Other Dataset")
        other_product = baker.make(Product, data_set=other_data_set, entry_id="x", name="X", slug="x", price=1)
        ProductContentChunk.objects.create(
            product=other_product, content="should not match", embedding=_unit_vector(0)
        )

        repository = DjangoProductChunkRepository(ProductContentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id, distance=CosineDistance("embedding", _unit_vector(0))
            )
        )

        assert results == []


class TestDocumentRetrievalUsesPgvector:
    def test_returns_chunks_ordered_by_cosine_distance_and_scoped_to_data_set(self, data_set):
        document_a = baker.make(Document, data_set=data_set, url="https://example.com/a", title="A", content="A")
        document_b = baker.make(Document, data_set=data_set, url="https://example.com/b", title="B", content="B")
        chunk_a = DocumentChunk.objects.create(
            document=document_a, content="fiber optic broadband", embedding=_unit_vector(0)
        )
        DocumentChunk.objects.create(document=document_b, content="mobile roaming tariff", embedding=_unit_vector(1))

        repository = DjangoDocumentChunkRepository(DocumentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id, distance=CosineDistance("embedding", _unit_vector(0))
            )
        )

        assert len(results) == 2
        assert results[0].id == chunk_a.id
        assert results[0].content == "fiber optic broadband"
        assert results[0].distance <= results[1].distance


class TestNullEmbeddingsAreExcludedFromRetrieval:
    """Chunks with ``embedding IS NULL`` (not yet embedded) must never surface in retrieval.

    The embedding column is ``null=True`` (see migration ``0014_pgvector_ann_indexes``); if such rows
    reach the ``CosineDistance`` annotation they produce bogus / NULL distances and pollute results.
    All three retrieval methods filter them out.
    """

    def test_document_vector_retrieval_excludes_null_embeddings(self, data_set):
        document_embedded = baker.make(Document, data_set=data_set, url="https://example.com/a", title="A", content="A")
        document_pending = baker.make(Document, data_set=data_set, url="https://example.com/b", title="B", content="B")
        embedded_chunk = DocumentChunk.objects.create(
            document=document_embedded, content="fiber optic broadband", embedding=_unit_vector(0)
        )
        DocumentChunk.objects.create(document=document_pending, content="mobile roaming tariff", embedding=None)

        repository = DjangoDocumentChunkRepository(DocumentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id, distance=CosineDistance("embedding", _unit_vector(0))
            )
        )

        assert [chunk.id for chunk in results] == [embedded_chunk.id]

    def test_product_vector_retrieval_excludes_null_embeddings(self, data_set):
        product_embedded = baker.make(Product, data_set=data_set, entry_id="a", name="A", slug="a", price=1)
        product_pending = baker.make(Product, data_set=data_set, entry_id="b", name="B", slug="b", price=1)
        embedded_chunk = ProductContentChunk.objects.create(
            product=product_embedded, content="red running shoes", embedding=_unit_vector(0)
        )
        ProductContentChunk.objects.create(product=product_pending, content="blue cotton shirts", embedding=None)

        repository = DjangoProductChunkRepository(ProductContentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id, distance=CosineDistance("embedding", _unit_vector(0))
            )
        )

        assert [chunk.id for chunk in results] == [embedded_chunk.id]

    def test_product_hybrid_retrieval_excludes_null_embeddings(self, data_set):
        # Both chunks mention the keyword so their text ``rank`` passes the 0.05 threshold, but only
        # one has an embedding. The pending (NULL embedding) chunk must be filtered out even though
        # it matches the keyword.
        product_embedded = baker.make(Product, data_set=data_set, entry_id="a", name="A", slug="a", price=1)
        product_pending = baker.make(Product, data_set=data_set, entry_id="b", name="B", slug="b", price=1)
        embedded_chunk = ProductContentChunk.objects.create(
            product=product_embedded, content="running shoes for track", embedding=_unit_vector(0)
        )
        ProductContentChunk.objects.create(product=product_pending, content="running shoes on sale", embedding=None)

        repository = DjangoProductChunkRepository(ProductContentChunk)
        results = list(
            repository.get_chunks_by_keyword_for_data_set(
                data_set_id=data_set.id,
                distance=CosineDistance("embedding", _unit_vector(0)),
                keyword="shoes",
            )
        )

        assert [chunk.id for chunk in results] == [embedded_chunk.id]


class TestDistanceThresholdFiltersIrrelevantChunks:
    """The pure-vector path drops chunks whose cosine distance exceeds ``distance_threshold``.

    Without a similarity floor the ranking always returns ``top max_objects``, so irrelevant chunks
    leak into results (YAZ-9 / DRA). ``distance_threshold`` is an optional upper bound on cosine
    distance, pushed into the queryset as ``distance__lte`` so it composes with ``ef_search`` and the
    candidate-pool slice. ``None`` keeps the historical "always return top-K" behaviour.
    """

    def test_document_vector_retrieval_drops_far_chunks_when_threshold_set(self, data_set):
        document_near = baker.make(Document, data_set=data_set, url="https://example.com/near", title="N", content="N")
        document_far = baker.make(Document, data_set=data_set, url="https://example.com/far", title="F", content="F")
        near = DocumentChunk.objects.create(
            document=document_near, content="fiber optic broadband", embedding=_unit_vector(0)
        )
        # Orthogonal embedding -> cosine distance 1.0 from the query (clearly irrelevant).
        DocumentChunk.objects.create(document=document_far, content="unrelated filler", embedding=_unit_vector(1))

        repository = DjangoDocumentChunkRepository(DocumentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id,
                distance=CosineDistance("embedding", _unit_vector(0)),
                distance_threshold=0.6,
            )
        )

        assert [chunk.id for chunk in results] == [near.id]

    def test_product_vector_retrieval_drops_far_chunks_when_threshold_set(self, data_set):
        product_near = baker.make(Product, data_set=data_set, entry_id="near", name="Near", slug="near", price=1)
        product_far = baker.make(Product, data_set=data_set, entry_id="far", name="Far", slug="far", price=1)
        near = ProductContentChunk.objects.create(
            product=product_near, content="red running shoes", embedding=_unit_vector(0)
        )
        ProductContentChunk.objects.create(product=product_far, content="blue cotton shirts", embedding=_unit_vector(1))

        repository = DjangoProductChunkRepository(ProductContentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id,
                distance=CosineDistance("embedding", _unit_vector(0)),
                distance_threshold=0.6,
            )
        )

        assert [chunk.id for chunk in results] == [near.id]

    def test_threshold_none_keeps_historical_top_k_behaviour(self, data_set):
        product_near = baker.make(Product, data_set=data_set, entry_id="near", name="Near", slug="near", price=1)
        product_far = baker.make(Product, data_set=data_set, entry_id="far", name="Far", slug="far", price=1)
        near = ProductContentChunk.objects.create(
            product=product_near, content="red running shoes", embedding=_unit_vector(0)
        )
        far = ProductContentChunk.objects.create(product=product_far, content="blue cotton shirts", embedding=_unit_vector(1))

        repository = DjangoProductChunkRepository(ProductContentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id,
                distance=CosineDistance("embedding", _unit_vector(0)),
                distance_threshold=None,
            )
        )

        # No similarity floor -> the orthogonal far chunk is still returned (historical behaviour).
        assert [chunk.id for chunk in results] == [near.id, far.id]

    def test_high_threshold_keeps_everything_below_it(self, data_set):
        product_near = baker.make(Product, data_set=data_set, entry_id="near", name="Near", slug="near", price=1)
        product_far = baker.make(Product, data_set=data_set, entry_id="far", name="Far", slug="far", price=1)
        near = ProductContentChunk.objects.create(
            product=product_near, content="red running shoes", embedding=_unit_vector(0)
        )
        far = ProductContentChunk.objects.create(product=product_far, content="blue cotton shirts", embedding=_unit_vector(1))

        repository = DjangoProductChunkRepository(ProductContentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id,
                distance=CosineDistance("embedding", _unit_vector(0)),
                distance_threshold=1.5,
            )
        )

        assert [chunk.id for chunk in results] == [near.id, far.id]
