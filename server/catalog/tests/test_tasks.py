"""Tests for the catalog indexing tasks, in particular the dataset-scoped backfill tasks.

``index_all_products_task`` mirrors ``index_all_documents_task``: it queues a per-item
``index_product_task`` for every product in a dataset (and only that dataset). These tests pin the
enqueue fan-out and data-set scoping without depending on a Celery worker or an embedding API.
"""

from unittest.mock import patch

import pytest
from model_bakery import baker

from catalog.models import DataSet, Document, Product
from catalog.tasks import index_all_documents_task, index_all_products_task

pytestmark = pytest.mark.django_db


def _make_product(data_set, entry_id):
    return baker.make(
        Product,
        data_set=data_set,
        entry_id=entry_id,
        name=entry_id,
        slug=entry_id,
        price=1,
    )


def _make_document(data_set, url):
    return baker.make(Document, data_set=data_set, url=url, title=url, content=url)


@pytest.fixture
def data_set():
    return baker.make(DataSet, name="Backfill Dataset")


@patch("catalog.tasks.index_product_task.apply_async")
class TestIndexAllProductsTask:
    def test_enqueues_index_task_for_every_product_in_data_set(self, mock_apply_async, data_set):
        product_a = _make_product(data_set, "a")
        product_b = _make_product(data_set, "b")

        index_all_products_task(data_set.id)

        enqueued_ids = {call.args[0][0] for call in mock_apply_async.call_args_list}
        assert enqueued_ids == {product_a.id, product_b.id}
        assert mock_apply_async.call_count == 2

    def test_is_scoped_to_the_given_data_set(self, mock_apply_async, data_set):
        other_data_set = baker.make(DataSet, name="Other Dataset")
        # Products in another dataset must never be enqueued.
        _make_product(other_data_set, "x")
        _make_product(data_set, "a")

        index_all_products_task(data_set.id)

        enqueued_ids = {call.args[0][0] for call in mock_apply_async.call_args_list}
        assert enqueued_ids == {Product.objects.get(entry_id="a").id}

    def test_no_products_enqueues_nothing(self, mock_apply_async, data_set):
        index_all_products_task(data_set.id)
        assert mock_apply_async.call_count == 0


@patch("catalog.tasks.index_document_task.apply_async")
class TestIndexAllDocumentsTask:
    def test_enqueues_index_task_for_every_document_in_data_set(self, mock_apply_async, data_set):
        document_a = _make_document(data_set, "https://example.com/a")
        document_b = _make_document(data_set, "https://example.com/b")

        index_all_documents_task(data_set.id)

        enqueued_ids = {call.args[0][0] for call in mock_apply_async.call_args_list}
        assert enqueued_ids == {document_a.id, document_b.id}
        assert mock_apply_async.call_count == 2

    def test_is_scoped_to_the_given_data_set(self, mock_apply_async, data_set):
        other_data_set = baker.make(DataSet, name="Other Dataset")
        _make_document(other_data_set, "https://example.com/x")
        _make_document(data_set, "https://example.com/a")

        index_all_documents_task(data_set.id)

        enqueued_ids = {call.args[0][0] for call in mock_apply_async.call_args_list}
        assert enqueued_ids == {Document.objects.get(url="https://example.com/a").id}


@patch("catalog.tasks.ProductEmbeddingGenerator.index_object")
class TestIndexProductTaskFailureVisibility:
    def test_failure_is_reraised_after_logging(self, mock_index_object):
        from catalog.tasks import index_product_task

        data_set = baker.make(DataSet, name="Dataset")
        product = _make_product(data_set, "a")
        mock_index_object.side_effect = RuntimeError("embedding API down")

        with pytest.raises(RuntimeError):
            index_product_task(product.id)
