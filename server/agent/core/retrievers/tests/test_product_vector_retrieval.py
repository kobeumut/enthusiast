"""Tests for the pgvector-backed natural-language product retrieval path.

``ProductRetriever.find_products_matching_query`` embeds the query with the dataset's configured
embedding provider/model and ranks ``ProductContentChunk.embedding`` by cosine distance, the same
approach ``DocumentRetriever`` uses. These tests store real embeddings in the pgvector column and
exercise the full query -> embedding -> cosine ranking -> product JSON path.

They require a PostgreSQL instance with the ``vector`` extension (the CI test service uses
``pgvector/pgvector``), which is the same store the application uses in development and production.
"""

import pytest
from model_bakery import baker

from agent.core.repositories import (
    DjangoDataSetRepository,
    DjangoProductChunkRepository,
    DjangoProductRepository,
)
from agent.core.retrievers.product_retriever import ProductRetriever
from catalog.models import DataSet, Product, ProductContentChunk
from catalog.models.data_set import EMBEDDING_VECTOR_DIMENSIONS

pytestmark = pytest.mark.django_db


def _unit_vector(position: int, dimensions: int = EMBEDDING_VECTOR_DIMENSIONS) -> list[float]:
    """A deterministic unit vector with a single non-zero component at ``position``."""
    vector = [0.0] * dimensions
    vector[position] = 1.0
    return vector


class _StubEmbeddingsRegistry:
    """Stand-in for ``EmbeddingProviderRegistry`` that always returns a fixed query embedding.

    ``provider_for_dataset`` returns a provider class constructed with the dataset's embedding model
    and vector dimensions, mirroring the real registry contract used by ``DocumentRetriever``.
    """

    def __init__(self, query_vector: list[float]):
        self._query_vector = query_vector
        self.provider_for_dataset_calls: list[int] = []
        self.provider_init_calls: list[tuple[str, int]] = []

    def provider_for_dataset(self, data_set_id: int):
        self.provider_for_dataset_calls.append(data_set_id)
        registry = self

        class _EmbeddingProvider:
            def __init__(self, model: str, dimensions: int):
                registry.provider_init_calls.append((model, dimensions))

            def generate_embeddings(self, text: str) -> list[float]:
                return registry._query_vector

        return _EmbeddingProvider


def _build_retriever(data_set_id: int, query_vector: list[float], number_of_products: int = 12):
    return ProductRetriever(
        data_set_id=data_set_id,
        data_set_repo=DjangoDataSetRepository(DataSet),
        product_repo=DjangoProductRepository(Product),
        product_chunk_repo=DjangoProductChunkRepository(ProductContentChunk),
        embeddings_registry=_StubEmbeddingsRegistry(query_vector),
        number_of_products=number_of_products,
    )


@pytest.fixture
def data_set():
    return baker.make(DataSet, name="Vector Retrieval")


class TestProductVectorRetrieval:
    def test_returns_products_ranked_by_cosine_distance(self, data_set):
        product_a = baker.make(Product, data_set=data_set, entry_id="a", name="Running Shoes", slug="a", price=1)
        product_b = baker.make(Product, data_set=data_set, entry_id="b", name="Cotton Shirts", slug="b", price=1)
        # product_a's chunk is collinear with the query (distance 0), product_b's is orthogonal (distance 1).
        ProductContentChunk.objects.create(product=product_a, content="red running shoes", embedding=_unit_vector(0))
        ProductContentChunk.objects.create(product=product_b, content="blue cotton shirts", embedding=_unit_vector(1))

        retriever = _build_retriever(data_set.id, query_vector=_unit_vector(0))
        results = retriever.find_products_matching_query("shoes for running")

        assert [product.id for product in results] == [product_a.id, product_b.id]

    def test_no_chunks_returns_empty(self, data_set):
        # A product without any indexed chunks cannot match a vector query.
        baker.make(Product, data_set=data_set, entry_id="a", name="A", slug="a", price=1)

        retriever = _build_retriever(data_set.id, query_vector=_unit_vector(0))
        results = retriever.find_products_matching_query("anything")

        assert results == []

    def test_results_are_isolated_per_data_set(self, data_set):
        other_data_set = baker.make(DataSet, name="Other Dataset")
        other_product = baker.make(
            Product, data_set=other_data_set, entry_id="x", name="Exact Match", slug="x", price=1
        )
        # A perfect match living in another dataset must not leak into this dataset's results.
        ProductContentChunk.objects.create(product=other_product, content="perfect match", embedding=_unit_vector(0))

        retriever = _build_retriever(data_set.id, query_vector=_unit_vector(0))
        results = retriever.find_products_matching_query("perfect match")

        assert results == []

    def test_multiple_chunks_for_one_product_returned_once(self, data_set):
        product_a = baker.make(Product, data_set=data_set, entry_id="a", name="A", slug="a", price=1)
        product_b = baker.make(Product, data_set=data_set, entry_id="b", name="B", slug="b", price=1)
        # Two chunks for product_a, both closer to the query than product_b's single chunk.
        ProductContentChunk.objects.create(product=product_a, content="a first chunk", embedding=_unit_vector(0))
        ProductContentChunk.objects.create(product=product_a, content="a second chunk", embedding=_unit_vector(0))
        ProductContentChunk.objects.create(product=product_b, content="b chunk", embedding=_unit_vector(1))

        retriever = _build_retriever(data_set.id, query_vector=_unit_vector(0))
        results = retriever.find_products_matching_query("a")

        assert [product.id for product in results] == [product_a.id, product_b.id]

    def test_respects_number_of_products_limit(self, data_set):
        nearest = baker.make(Product, data_set=data_set, entry_id="near", name="Nearest", slug="near", price=1)
        other = baker.make(Product, data_set=data_set, entry_id="far", name="Far", slug="far", price=1)
        ProductContentChunk.objects.create(product=nearest, content="nearest", embedding=_unit_vector(0))
        ProductContentChunk.objects.create(product=other, content="other", embedding=_unit_vector(1))

        retriever = _build_retriever(data_set.id, query_vector=_unit_vector(0), number_of_products=1)
        results = retriever.find_products_matching_query("nearest")

        assert [product.id for product in results] == [nearest.id]

    def test_serializes_results_with_product_details_json(self, data_set):
        product = baker.make(
            Product,
            data_set=data_set,
            entry_id="a",
            name="Running Shoes",
            slug="running-shoes",
            description="Fast shoes",
            sku="SKU-1",
            price=42.0,
        )
        ProductContentChunk.objects.create(product=product, content="running shoes", embedding=_unit_vector(0))

        retriever = _build_retriever(data_set.id, query_vector=_unit_vector(0))
        results = retriever.find_products_matching_query("running")

        serialized = retriever.product_details_as_json(results)
        assert len(serialized) == 1
        assert serialized[0]["name"] == "Running Shoes"
        assert serialized[0]["entry_id"] == "a"
        assert serialized[0]["price"] == 42.0

    def test_reuses_dataset_embedding_config_via_provider_for_dataset(self, data_set):
        data_set.embedding_model = "text-embedding-test"
        data_set.embedding_vector_dimensions = EMBEDDING_VECTOR_DIMENSIONS
        data_set.save()

        product = baker.make(Product, data_set=data_set, entry_id="a", name="A", slug="a", price=1)
        ProductContentChunk.objects.create(product=product, content="a", embedding=_unit_vector(0))

        query_vector = _unit_vector(0)
        registry = _StubEmbeddingsRegistry(query_vector)
        retriever = ProductRetriever(
            data_set_id=data_set.id,
            data_set_repo=DjangoDataSetRepository(DataSet),
            product_repo=DjangoProductRepository(Product),
            product_chunk_repo=DjangoProductChunkRepository(ProductContentChunk),
            embeddings_registry=registry,
        )
        retriever.find_products_matching_query("a")

        assert registry.provider_for_dataset_calls == [data_set.id]
        assert registry.provider_init_calls == [(data_set.embedding_model, data_set.embedding_vector_dimensions)]
