"""Tests for the content-hash based re-index skipping in the sync managers (YAZ-10).

A no-op re-sync (identical content) must NOT re-enqueue ``index_*_task``, while a new item
or genuinely changed content must. The index tasks are mocked so we can assert on enqueue
behaviour without spinning up Celery / the embedding pipeline.
"""

from unittest.mock import patch

import pytest
from enthusiast_common import DocumentDetails, ProductDetails
from model_bakery import baker

from catalog.models import DataSet, Document, Product
from sync.document.manager import DocumentSyncManager
from sync.product.manager import ProductSyncManager

pytestmark = pytest.mark.django_db


# ---------------------------------------------------------------------------
# Product sync
# ---------------------------------------------------------------------------


def _product_details(**overrides):
    base = dict(
        entry_id="p-1",
        name="Red Running Shoes",
        slug="red-running-shoes",
        description="Lightweight red running shoes.",
        sku="RRS-1",
        properties='{"color":"red"}',
        categories='["shoes"]',
        price=129.0,
    )
    base.update(overrides)
    return ProductDetails(**base)


@pytest.fixture
def product_data_set():
    return baker.make(DataSet, name="Product Sync Dataset")


class TestProductSyncContentHash:
    def test_new_product_enqueues_index_task(self, product_data_set):
        manager = ProductSyncManager()
        with patch("sync.product.manager.index_product_task") as mock_task:
            manager._sync_item(data_set_id=product_data_set.id, item_data=_product_details())

        mock_task.apply_async.assert_called_once()
        assert Product.objects.filter(data_set=product_data_set).count() == 1
        # The canonical hash is stored on first sync.
        assert Product.objects.get(data_set=product_data_set, entry_id="p-1").content_hash is not None

    def test_unchanged_second_sync_skips_index_task(self, product_data_set):
        manager = ProductSyncManager()
        with patch("sync.product.manager.index_product_task") as mock_task:
            manager._sync_item(data_set_id=product_data_set.id, item_data=_product_details())
            manager._sync_item(data_set_id=product_data_set.id, item_data=_product_details())  # identical

        # Indexed exactly once (first sync); the no-op second sync must not re-enqueue.
        assert mock_task.apply_async.call_count == 1

    def test_changed_description_enqueues_index_task_again(self, product_data_set):
        manager = ProductSyncManager()
        with patch("sync.product.manager.index_product_task") as mock_task:
            manager._sync_item(data_set_id=product_data_set.id, item_data=_product_details())
            manager._sync_item(
                data_set_id=product_data_set.id,
                item_data=_product_details(description="Lightweight red running shoes for trails."),
            )

        assert mock_task.apply_async.call_count == 2

    def test_changed_price_enqueues_index_task_again(self, product_data_set):
        manager = ProductSyncManager()
        with patch("sync.product.manager.index_product_task") as mock_task:
            manager._sync_item(data_set_id=product_data_set.id, item_data=_product_details(price=129.0))
            manager._sync_item(data_set_id=product_data_set.id, item_data=_product_details(price=99.0))

        assert mock_task.apply_async.call_count == 2

    def test_legacy_null_hash_backfills_and_indexes_once(self, product_data_set):
        # An existing row from before this feature has content_hash=None. Syncing the same
        # content must index (backfill) and then set the hash, so the next identical sync is a no-op.
        manager = ProductSyncManager()
        details = _product_details()
        baker.make(
            Product,
            data_set=product_data_set,
            entry_id=details.entry_id,
            name=details.name,
            slug=details.slug,
            description=details.description,
            sku=details.sku,
            properties=details.properties,
            categories=details.categories,
            price=details.price,
            content_hash=None,
        )

        with patch("sync.product.manager.index_product_task") as mock_task:
            manager._sync_item(data_set_id=product_data_set.id, item_data=details)  # backfill
            manager._sync_item(data_set_id=product_data_set.id, item_data=details)  # now a no-op

        assert mock_task.apply_async.call_count == 1
        assert Product.objects.get(data_set=product_data_set, entry_id=details.entry_id).content_hash is not None


# ---------------------------------------------------------------------------
# Document sync
# ---------------------------------------------------------------------------


def _document_details(**overrides):
    base = dict(
        url="https://example.com/docs/shoes",
        title="Shoe care guide",
        content="How to clean and store your running shoes.",
    )
    base.update(overrides)
    return DocumentDetails(**base)


@pytest.fixture
def document_data_set():
    return baker.make(DataSet, name="Document Sync Dataset")


class TestDocumentSyncContentHash:
    def test_new_document_enqueues_index_task(self, document_data_set):
        manager = DocumentSyncManager()
        with patch("sync.document.manager.index_document_task") as mock_task:
            manager._sync_item(data_set_id=document_data_set.id, item_data=_document_details())

        mock_task.apply_async.assert_called_once()
        assert Document.objects.get(data_set=document_data_set, url=_document_details().url).content_hash is not None

    def test_unchanged_second_sync_skips_index_task(self, document_data_set):
        manager = DocumentSyncManager()
        with patch("sync.document.manager.index_document_task") as mock_task:
            manager._sync_item(data_set_id=document_data_set.id, item_data=_document_details())
            manager._sync_item(data_set_id=document_data_set.id, item_data=_document_details())  # identical

        assert mock_task.apply_async.call_count == 1

    def test_changed_content_enqueues_index_task_again(self, document_data_set):
        manager = DocumentSyncManager()
        with patch("sync.document.manager.index_document_task") as mock_task:
            manager._sync_item(data_set_id=document_data_set.id, item_data=_document_details())
            manager._sync_item(
                data_set_id=document_data_set.id,
                item_data=_document_details(content="How to clean, store and resole your running shoes."),
            )

        assert mock_task.apply_async.call_count == 2


# ---------------------------------------------------------------------------
# Hash determinism / scope
# ---------------------------------------------------------------------------


class TestContentHashScope:
    def test_product_hash_is_stable_and_scope_sensitive(self):
        same_a = Product.compute_content_hash(
            name="N", description="D", sku="S", properties="P", categories="C", price=10.0
        )
        same_b = Product.compute_content_hash(
            name="N", description="D", sku="S", properties="P", categories="C", price=10.0
        )
        assert same_a == same_b

        changed = Product.compute_content_hash(
            name="N", description="D-changed", sku="S", properties="P", categories="C", price=10.0
        )
        assert changed != same_a

    def test_document_hash_is_stable_and_scope_sensitive(self):
        same_a = Document.compute_content_hash(title="T", content="C")
        same_b = Document.compute_content_hash(title="T", content="C")
        assert same_a == same_b

        changed = Document.compute_content_hash(title="T", content="C-changed")
        assert changed != same_a
