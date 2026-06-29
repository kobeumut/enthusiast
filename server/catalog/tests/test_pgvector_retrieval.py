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

pytestmark = pytest.mark.django_db


@pytest.fixture
def data_set():
    return baker.make(DataSet, name="pgvector Retrieval Dataset")


class TestProductRetrievalUsesPgvector:
    def test_returns_chunks_ordered_by_cosine_distance_and_scoped_to_data_set(self, data_set):
        product_a = baker.make(Product, data_set=data_set, entry_id="a", name="A", slug="a", price=1)
        product_b = baker.make(Product, data_set=data_set, entry_id="b", name="B", slug="b", price=1)
        # chunk_a is collinear with the query vector (distance 0), chunk_b is orthogonal (distance 1).
        chunk_a = ProductContentChunk.objects.create(product=product_a, content="red running shoes", embedding=[1.0, 0.0, 0.0])
        ProductContentChunk.objects.create(product=product_b, content="blue cotton shirts", embedding=[0.0, 1.0, 0.0])

        repository = DjangoProductChunkRepository(ProductContentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id, distance=CosineDistance("embedding", [1.0, 0.0, 0.0])
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
        ProductContentChunk.objects.create(product=other_product, content="should not match", embedding=[1.0, 0.0, 0.0])

        repository = DjangoProductChunkRepository(ProductContentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id, distance=CosineDistance("embedding", [1.0, 0.0, 0.0])
            )
        )

        assert results == []


class TestDocumentRetrievalUsesPgvector:
    def test_returns_chunks_ordered_by_cosine_distance_and_scoped_to_data_set(self, data_set):
        document_a = baker.make(Document, data_set=data_set, url="https://example.com/a", title="A", content="A")
        document_b = baker.make(Document, data_set=data_set, url="https://example.com/b", title="B", content="B")
        chunk_a = DocumentChunk.objects.create(document=document_a, content="fiber optic broadband", embedding=[1.0, 0.0, 0.0])
        DocumentChunk.objects.create(document=document_b, content="mobile roaming tariff", embedding=[0.0, 1.0, 0.0])

        repository = DjangoDocumentChunkRepository(DocumentChunk)
        results = list(
            repository.get_chunk_by_distance_for_data_set(
                data_set_id=data_set.id, distance=CosineDistance("embedding", [1.0, 0.0, 0.0])
            )
        )

        assert len(results) == 2
        assert results[0].id == chunk_a.id
        assert results[0].content == "fiber optic broadband"
        assert results[0].distance <= results[1].distance
