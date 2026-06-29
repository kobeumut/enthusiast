"""Tests for the sync managers' reindex-on-change behavior.

Source sync must still index newly created/updated items, but it should not enqueue a redundant
re-split + embedding API call when the embedded content did not change. Product embeddings derive
from ``name`` + ``description`` (``Product.get_content``); document embeddings derive from ``content``
(``Document.split``). Only those fields (or a brand-new row) should trigger an indexing task.
"""

from unittest.mock import patch

import pytest
from enthusiast_common import DocumentDetails, ProductDetails
from model_bakery import baker

from catalog.models import DataSet, Document, Product
from sync.document.manager import DocumentSyncManager
from sync.product.manager import ProductSyncManager

pytestmark = pytest.mark.django_db


def _product_details(entry_id="a", name="Red Running Shoes", description="Fast shoes", **overrides):
    base = dict(
        entry_id=entry_id,
        name=name,
        slug=entry_id,
        description=description,
        sku="SKU-1",
        properties="",
        categories="",
        price=10.0,
    )
    base.update(overrides)
    return ProductDetails(**base)


@pytest.fixture
def data_set():
    return baker.make(DataSet, name="Sync Dataset")


@patch("sync.product.manager.index_product_task.apply_async")
class TestProductSyncReindexOnChange:
    def test_new_product_enqueues_reindex(self, mock_apply_async, data_set):
        manager = ProductSyncManager()

        manager._sync_item(data_set_id=data_set.id, item_data=_product_details(entry_id="new"))

        assert mock_apply_async.call_count == 1
        enqueued_product_id = mock_apply_async.call_args.args[0][0]
        assert Product.objects.filter(id=enqueued_product_id, data_set=data_set).exists()

    def test_unchanged_product_does_not_reindex(self, mock_apply_async, data_set):
        item = _product_details(entry_id="a", name="Red Running Shoes", description="Fast shoes")
        ProductSyncManager()._sync_item(data_set_id=data_set.id, item_data=item)

        mock_apply_async.reset_mock()
        ProductSyncManager()._sync_item(data_set_id=data_set.id, item_data=item)

        assert mock_apply_async.call_count == 0
        # And the row is still updated/created in the DB (sync still persists the source row).
        assert Product.objects.filter(data_set=data_set, entry_id="a").count() == 1

    def test_changed_name_enqueues_reindex(self, mock_apply_async, data_set):
        ProductSyncManager()._sync_item(data_set_id=data_set.id, item_data=_product_details(name="Old Name"))

        mock_apply_async.reset_mock()
        ProductSyncManager()._sync_item(
            data_set_id=data_set.id, item_data=_product_details(name="New Name")
        )

        assert mock_apply_async.call_count == 1

    def test_changed_description_enqueues_reindex(self, mock_apply_async, data_set):
        ProductSyncManager()._sync_item(
            data_set_id=data_set.id, item_data=_product_details(description="old description")
        )

        mock_apply_async.reset_mock()
        ProductSyncManager()._sync_item(
            data_set_id=data_set.id, item_data=_product_details(description="new description")
        )

        assert mock_apply_async.call_count == 1

    def test_changed_non_embedded_field_does_not_reindex(self, mock_apply_async, data_set):
        # price/sku/properties/categories are not part of Product.get_content(), so changing only
        # those must not enqueue a re-index.
        ProductSyncManager()._sync_item(
            data_set_id=data_set.id, item_data=_product_details(price=10.0, sku="OLD")
        )

        mock_apply_async.reset_mock()
        ProductSyncManager()._sync_item(
            data_set_id=data_set.id, item_data=_product_details(price=99.0, sku="NEW")
        )

        assert mock_apply_async.call_count == 0
        product = Product.objects.get(data_set=data_set, entry_id="a")
        assert product.price == 99.0
        assert product.sku == "NEW"


@patch("sync.document.manager.index_document_task.apply_async")
class TestDocumentSyncReindexOnChange:
    def test_new_document_enqueues_reindex(self, mock_apply_async, data_set):
        manager = DocumentSyncManager()

        manager._sync_item(
            data_set_id=data_set.id,
            item_data=DocumentDetails(url="https://example.com/a", title="A", content="A"),
        )

        assert mock_apply_async.call_count == 1
        enqueued_document_id = mock_apply_async.call_args.args[0][0]
        assert Document.objects.filter(id=enqueued_document_id, data_set=data_set).exists()

    def test_unchanged_content_does_not_reindex(self, mock_apply_async, data_set):
        details = DocumentDetails(url="https://example.com/a", title="A", content="A")
        DocumentSyncManager()._sync_item(data_set_id=data_set.id, item_data=details)

        mock_apply_async.reset_mock()
        # Title changes but content (what gets embedded) does not.
        DocumentSyncManager()._sync_item(
            data_set_id=data_set.id,
            item_data=DocumentDetails(url="https://example.com/a", title="New Title", content="A"),
        )

        assert mock_apply_async.call_count == 0
        document = Document.objects.get(data_set=data_set, url="https://example.com/a")
        assert document.title == "New Title"

    def test_changed_content_enqueues_reindex(self, mock_apply_async, data_set):
        DocumentSyncManager()._sync_item(
            data_set_id=data_set.id,
            item_data=DocumentDetails(url="https://example.com/a", title="A", content="old"),
        )

        mock_apply_async.reset_mock()
        DocumentSyncManager()._sync_item(
            data_set_id=data_set.id,
            item_data=DocumentDetails(url="https://example.com/a", title="A", content="new content"),
        )

        assert mock_apply_async.call_count == 1
