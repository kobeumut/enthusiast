"""Integration tests for pre-retrieval metadata filtering (YAZ-15, sub-step 1).

Filters are pushed into the chunk queryset *before* the vector ranking, so out-of-scope chunks never
enter the ranking. These tests build real pgvector data and assert the retriever's
``filters=`` argument actually narrows the result set by category / price (products) and url /
title (documents), while ``filters=None`` preserves the unfiltered behaviour.
"""

import math

import pytest
from model_bakery import baker

from agent.core.repositories import (
    DjangoDataSetRepository,
    DjangoDocumentChunkRepository,
    DjangoProductChunkRepository,
    DjangoProductRepository,
)
from agent.core.retrievers.document_retriever import DocumentRetriever
from agent.core.retrievers.filters import RetrievalFilters
from agent.core.retrievers.product_retriever import ProductRetriever
from catalog.models import DataSet, Document, DocumentChunk, Product, ProductContentChunk
from catalog.models.data_set import EMBEDDING_VECTOR_DIMENSIONS

pytestmark = pytest.mark.django_db

DIM = EMBEDDING_VECTOR_DIMENSIONS


def _unit(axis: int) -> list[float]:
    vec = [0.0] * DIM
    vec[axis] = 1.0
    return vec


def _slight_rotation(axis_a: int, axis_b: int, angle: float = 0.05) -> list[float]:
    """A unit vector mostly on ``axis_a`` with a small component on ``axis_b`` (near ``_unit``)."""
    vec = [0.0] * DIM
    vec[axis_a] = math.cos(angle)
    vec[axis_b] = math.sin(angle)
    return vec


class _StubEmbeddingsRegistry:
    """Returns a fixed query embedding regardless of input, mirroring the registry contract."""

    def __init__(self, query_vector: list[float]):
        self._query_vector = query_vector

    def provider_for_dataset(self, data_set_id):
        query_vector = self._query_vector

        class _Provider:
            def __init__(self, model, dimensions):
                pass

            def generate_embeddings(self, text):
                return query_vector

        return _Provider


@pytest.fixture
def data_set():
    return baker.make(DataSet, name="Filter Dataset")


# --------------------------------------------------------------------------- products


def _product_retriever(data_set, query_vector):
    return ProductRetriever(
        data_set_id=data_set.id,
        data_set_repo=DjangoDataSetRepository(DataSet),
        product_repo=DjangoProductRepository(Product),
        product_chunk_repo=DjangoProductChunkRepository(ProductContentChunk),
        embeddings_registry=_StubEmbeddingsRegistry(query_vector),
    )


class TestProductMetadataFilters:
    def test_category_filter_keeps_only_matching_products(self, data_set):
        shoes = baker.make(
            Product, data_set=data_set, entry_id="s", name="Shoes", slug="s", price=80, categories="Running,Footwear"
        )
        shirts = baker.make(
            Product, data_set=data_set, entry_id="t", name="Shirts", slug="t", price=30, categories="Shirts,Apparel"
        )
        # Both chunks sit close to the query, so without a filter both would be returned.
        ProductContentChunk.objects.create(product=shoes, content="running shoes", embedding=_unit(0))
        ProductContentChunk.objects.create(product=shirts, content="running apparel", embedding=_slight_rotation(0, 1))

        retriever = _product_retriever(data_set, _unit(0))

        unfiltered = [p.id for p in retriever.find_products_matching_query("running")]
        assert set(unfiltered) == {shoes.id, shirts.id}

        filtered = retriever.find_products_matching_query(
            "running", filters=RetrievalFilters(categories=["Footwear"])
        )
        assert [p.id for p in filtered] == [shoes.id]

    def test_price_range_filters(self, data_set):
        cheap = baker.make(Product, data_set=data_set, entry_id="c", name="Cheap", slug="c", price=30)
        pricey = baker.make(Product, data_set=data_set, entry_id="p", name="Pricey", slug="p", price=120)
        ProductContentChunk.objects.create(product=cheap, content="item", embedding=_unit(0))
        ProductContentChunk.objects.create(product=pricey, content="item", embedding=_slight_rotation(0, 1))

        retriever = _product_retriever(data_set, _unit(0))

        assert {p.id for p in retriever.find_products_matching_query("item")} == {cheap.id, pricey.id}
        assert [p.id for p in retriever.find_products_matching_query("item", filters=RetrievalFilters(price_max=50))] == [
            cheap.id
        ]
        assert [p.id for p in retriever.find_products_matching_query("item", filters=RetrievalFilters(price_min=100))] == [
            pricey.id
        ]

    def test_combined_category_and_price_filter(self, data_set):
        a = baker.make(Product, data_set=data_set, entry_id="a", name="A", slug="a", price=40, categories="Running")
        b = baker.make(Product, data_set=data_set, entry_id="b", name="B", slug="b", price=40, categories="Shirts")
        c = baker.make(Product, data_set=data_set, entry_id="c", name="C", slug="c", price=90, categories="Running")
        for product in (a, b, c):
            ProductContentChunk.objects.create(product=product, content="item", embedding=_unit(0))

        retriever = _product_retriever(data_set, _unit(0))
        results = retriever.find_products_matching_query(
            "item", filters=RetrievalFilters(categories=["Running"], price_max=50)
        )
        assert [p.id for p in results] == [a.id]


# --------------------------------------------------------------------------- documents


def _document_retriever(data_set, query_vector):
    return DocumentRetriever(
        data_set_id=data_set.id,
        data_set_repo=DjangoDataSetRepository(DataSet),
        model_chunk_repo=DjangoDocumentChunkRepository(DocumentChunk),
        embeddings_registry=_StubEmbeddingsRegistry(query_vector),
    )


class TestDocumentMetadataFilters:
    def test_url_and_title_filters(self, data_set):
        manual = baker.make(
            Document, data_set=data_set, url="https://shop.example/manuals/ac2000", title="AC-2000 Manual", content="c"
        )
        blog = baker.make(
            Document, data_set=data_set, url="https://shop.example/blog/news", title="Company News", content="c"
        )
        DocumentChunk.objects.create(document=manual, content="setup guide", embedding=_unit(0))
        DocumentChunk.objects.create(document=blog, content="setup guide", embedding=_slight_rotation(0, 1))

        retriever = _document_retriever(data_set, _unit(0))

        assert {c.document_id for c in retriever.find_content_matching_query("setup")} == {manual.id, blog.id}
        assert [c.document_id for c in retriever.find_content_matching_query("setup", filters=RetrievalFilters(url_contains="manuals"))] == [
            manual.id
        ]
        assert [c.document_id for c in retriever.find_content_matching_query("setup", filters=RetrievalFilters(title_contains="AC-2000"))] == [
            manual.id
        ]
