from io import StringIO
from unittest.mock import patch

import pytest
from django.core.management import CommandError, call_command
from model_bakery import baker

from catalog.models import DataSet, Document, DocumentChunk, Product, ProductContentChunk
from catalog.models.data_set import EMBEDDING_VECTOR_DIMENSIONS
from catalog.services import ProductEmbeddingGenerator

pytestmark = pytest.mark.django_db

# The chunk embedding columns are ``vector(EMBEDDING_VECTOR_DIMENSIONS)`` (see migration
# ``0014_pgvector_ann_indexes``), so the fake provider must return a vector of that dimensionality.
FAKE_EMBEDDING = [0.01] * EMBEDDING_VECTOR_DIMENSIONS
STALE_EMBEDDING = [0.9] * EMBEDDING_VECTOR_DIMENSIONS


class FakeEmbeddingProvider:
    """Stand-in for an embedding provider that returns a deterministic vector without calling an API."""

    NAME = "Fake"

    def __init__(self, model, dimensions):
        self._model = model
        self._dimensions = dimensions

    def generate_embeddings(self, content):  # noqa: ARG002
        return list(FAKE_EMBEDDING)

    def generate_embeddings_batch(self, contents):  # noqa: ARG002
        # Mirrors a real batched provider: one vector per input, in order. ``index_object``
        # now embeds every chunk through this method instead of one ``generate_embeddings``
        # call per chunk.
        return [list(FAKE_EMBEDDING) for _ in contents]


@pytest.fixture
def data_set_with_items():
    data_set = baker.make(DataSet, name="Reindex Dataset")
    baker.make(
        Product,
        data_set=data_set,
        entry_id="product-1",
        name="Red Running Shoes",
        slug="red-running-shoes",
        description="Lightweight red running shoes for everyday training.",
        price=10,
    )
    baker.make(
        Document,
        data_set=data_set,
        url="https://example.com/docs/shoes",
        title="Shoe care guide",
        content="How to clean and store your running shoes.",
    )
    return data_set


@patch("catalog.services.EmbeddingProviderRegistry")
class TestReindexCommand:
    def test_reindex_creates_product_and_document_chunks_with_embeddings(self, mock_registry_cls, data_set_with_items):
        mock_registry_cls.return_value.provider_for_dataset.return_value = FakeEmbeddingProvider

        call_command("reindex", data_set=data_set_with_items.id)

        product_chunks = ProductContentChunk.objects.filter(product__data_set=data_set_with_items)
        document_chunks = DocumentChunk.objects.filter(document__data_set=data_set_with_items)

        assert product_chunks.exists()
        assert document_chunks.exists()
        # Every regenerated chunk must carry an embedding vector stored in pgvector.
        assert all(chunk.embedding is not None for chunk in product_chunks)
        assert all(chunk.embedding is not None for chunk in document_chunks)
        for chunk in product_chunks:
            assert list(chunk.embedding) == pytest.approx(FAKE_EMBEDDING, rel=1e-3)

    def test_reindex_products_only_does_not_touch_documents(self, mock_registry_cls, data_set_with_items):
        mock_registry_cls.return_value.provider_for_dataset.return_value = FakeEmbeddingProvider

        call_command("reindex", data_set=data_set_with_items.id, products=True)

        assert ProductContentChunk.objects.filter(product__data_set=data_set_with_items).exists()
        assert not DocumentChunk.objects.filter(document__data_set=data_set_with_items).exists()

    def test_reindex_documents_only_does_not_touch_products(self, mock_registry_cls, data_set_with_items):
        mock_registry_cls.return_value.provider_for_dataset.return_value = FakeEmbeddingProvider

        call_command("reindex", data_set=data_set_with_items.id, documents=True)

        assert DocumentChunk.objects.filter(document__data_set=data_set_with_items).exists()
        assert not ProductContentChunk.objects.filter(product__data_set=data_set_with_items).exists()

    def test_reindex_replaces_existing_chunks(self, mock_registry_cls, data_set_with_items):
        mock_registry_cls.return_value.provider_for_dataset.return_value = FakeEmbeddingProvider
        product = data_set_with_items.products.first()

        # Seed a stale chunk that should be removed during re-split.
        baker.make(ProductContentChunk, product=product, content="stale", embedding=list(STALE_EMBEDDING))
        assert ProductContentChunk.objects.filter(product=product).count() == 1

        call_command("reindex", data_set=data_set_with_items.id, products=True)

        chunks = ProductContentChunk.objects.filter(product=product)
        assert chunks.count() == 1
        assert chunks.first().content != "stale"
        assert list(chunks.first().embedding) == pytest.approx(FAKE_EMBEDDING, rel=1e-3)

    def test_reindex_unknown_data_set_raises(self, mock_registry_cls):
        with pytest.raises(CommandError):
            call_command("reindex", data_set=999999)


@patch("catalog.services.EmbeddingProviderRegistry")
class TestReindexCommandFailureHandling:
    def test_continues_after_item_failure_and_raises_command_error(self, mock_registry_cls, data_set_with_items):
        mock_registry_cls.return_value.provider_for_dataset.return_value = FakeEmbeddingProvider
        out = StringIO()
        # Product indexing fails on every product, but the run must keep going and index documents.
        with patch.object(ProductEmbeddingGenerator, "index_object", side_effect=RuntimeError("boom")):
            with pytest.raises(CommandError):
                call_command("reindex", data_set=data_set_with_items.id, stdout=out)

        # The document was still indexed despite the product failure (the run did not abort early).
        assert DocumentChunk.objects.filter(document__data_set=data_set_with_items).exists()
        # No product chunks were committed for the failing product.
        assert not ProductContentChunk.objects.filter(product__data_set=data_set_with_items).exists()
        # The failure line includes the exception class name so it can be triaged from CLI output.
        assert "RuntimeError" in out.getvalue()
