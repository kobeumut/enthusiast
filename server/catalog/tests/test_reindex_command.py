from io import StringIO
from unittest.mock import Mock, patch

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


@pytest.fixture
def data_set_with_products():
    """A data set with three products, useful for error-isolation / resume tests."""
    data_set = baker.make(DataSet, name="Reindex Dataset")
    for n in range(1, 4):
        baker.make(
            Product,
            data_set=data_set,
            entry_id=f"product-{n}",
            name=f"Product {n}",
            slug=f"product-{n}",
            description=f"Description for product number {n}.",
            price=n * 10,
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

    # -- per-item error isolation -------------------------------------------------

    def test_reindex_continues_on_item_error_and_reports_summary(self, mock_registry_cls, data_set_with_products):
        mock_registry_cls.return_value.provider_for_dataset.return_value = FakeEmbeddingProvider
        # Capture the real indexer before patching so healthy items are actually indexed.
        real_index = ProductEmbeddingGenerator.index_object

        def flaky_index(obj):
            if obj.entry_id == "product-2":
                raise RuntimeError("embedding API down")
            real_index(obj)

        out = StringIO()
        with (
            patch.object(ProductEmbeddingGenerator, "index_object", side_effect=flaky_index),
            patch("catalog.management.commands.reindex.time.sleep"),
        ):
            call_command("reindex", data_set=data_set_with_products.id, products=True, stdout=out)

        # The two healthy items were indexed; the bad one was skipped.
        indexed = set(
            ProductContentChunk.objects.filter(product__data_set=data_set_with_products).values_list(
                "product__entry_id", flat=True
            )
        )
        assert indexed == {"product-1", "product-3"}
        # The summary reports exactly one failure and names the offending item.
        output = out.getvalue()
        assert "2 ok / 1 fail" in output
        assert "product-2" in output
        assert "Reindex complete with 1 failed item" in output

    def test_reindex_fail_fast_stops_on_first_error(self, mock_registry_cls, data_set_with_products):
        mock_registry_cls.return_value.provider_for_dataset.return_value = FakeEmbeddingProvider

        out = StringIO()
        with (
            patch.object(ProductEmbeddingGenerator, "index_object", side_effect=RuntimeError("down")),
            patch("catalog.management.commands.reindex.time.sleep"),
        ):
            with pytest.raises(CommandError):
                call_command(
                    "reindex",
                    data_set=data_set_with_products.id,
                    products=True,
                    fail_fast=True,
                    stdout=out,
                )

        # Nothing was indexed and the summary was still printed before the abort.
        assert not ProductContentChunk.objects.filter(product__data_set=data_set_with_products).exists()
        assert "1 fail" in out.getvalue()

    # -- retry / backoff ----------------------------------------------------------

    def test_reindex_retries_transient_error_then_succeeds(self, mock_registry_cls, data_set_with_items):
        mock_registry_cls.return_value.provider_for_dataset.return_value = FakeEmbeddingProvider
        # Default max_attempts is 3: raise on the first two attempts, succeed on the third.
        mock_index = Mock(side_effect=[RuntimeError("boom"), RuntimeError("boom"), None])

        out = StringIO()
        with (
            patch.object(ProductEmbeddingGenerator, "index_object", mock_index),
            patch("catalog.management.commands.reindex.time.sleep"),
        ):
            call_command("reindex", data_set=data_set_with_items.id, products=True, stdout=out)

        # The item was tried exactly three times (two retries) before succeeding.
        assert mock_index.call_count == 3
        assert "1 ok / 0 fail" in out.getvalue()
        assert "Reindex complete." in out.getvalue()

    def test_reindex_retries_exhausted_counts_as_failed(self, mock_registry_cls, data_set_with_items):
        mock_registry_cls.return_value.provider_for_dataset.return_value = FakeEmbeddingProvider
        # The provider always fails.
        mock_index = Mock(side_effect=RuntimeError("always down"))

        out = StringIO()
        with (
            patch.object(ProductEmbeddingGenerator, "index_object", mock_index),
            patch("catalog.management.commands.reindex.time.sleep"),
        ):
            call_command("reindex", data_set=data_set_with_items.id, products=True, max_attempts=2, stdout=out)

        # With max_attempts=2 the item is tried twice, then recorded as failed; the run continues.
        assert mock_index.call_count == 2
        assert "0 ok / 1 fail" in out.getvalue()
        assert "Reindex complete with 1 failed item" in out.getvalue()

    # -- resume ergonomics --------------------------------------------------------

    def test_reindex_resume_from_id_with_limit(self, mock_registry_cls, data_set_with_products):
        mock_registry_cls.return_value.provider_for_dataset.return_value = FakeEmbeddingProvider
        ordered_pks = list(
            Product.objects.filter(data_set=data_set_with_products).order_by("pk").values_list("pk", flat=True)
        )

        out = StringIO()
        call_command(
            "reindex",
            data_set=data_set_with_products.id,
            products=True,
            from_id=ordered_pks[1],
            limit=1,
            stdout=out,
        )

        # Only the single item at/after the resume id, within the limit, was reindexed.
        indexed = set(
            ProductContentChunk.objects.filter(product__data_set=data_set_with_products).values_list(
                "product__pk", flat=True
            )
        )
        assert indexed == {ordered_pks[1]}
        assert "1 ok / 0 fail" in out.getvalue()

    def test_reindex_default_verbosity_keeps_output_clean(self, mock_registry_cls, data_set_with_products):
        mock_registry_cls.return_value.provider_for_dataset.return_value = FakeEmbeddingProvider

        out = StringIO()
        call_command("reindex", data_set=data_set_with_products.id, products=True, stdout=out)

        # At the default verbosity there is no per-item progress line, only headings + summary.
        output = out.getvalue()
        assert "ok" in output  # summary present
        assert "[1/3]" not in output  # per-item progress suppressed
