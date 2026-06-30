"""Similarity-floor (cosine distance threshold) tests for the pure-vector retrieval path (YAZ-16).

The default pure-vector path always returns ``top max_objects`` / ``number_of_products`` even when
the nearest chunks are semantically unrelated to the query. ``distance_threshold`` is a configurable
upper bound on cosine distance: chunks farther than the threshold are dropped at the SQL level, so
irrelevant chunks no longer leak into results just to fill the limit.

These tests exercise the threshold end-to-end through the retrievers (which is the user-facing path
and where the platform default is wired in via ``extra_kwargs``), and confirm it composes correctly
with ``max_objects`` and the hybrid (keyword) ranklist.
"""

import math

import pytest
from model_bakery import baker

from agent.core.agents.default_config import DEFAULT_COSINE_DISTANCE_THRESHOLD
from agent.core.repositories import (
    DjangoDataSetRepository,
    DjangoDocumentChunkRepository,
    DjangoProductChunkRepository,
    DjangoProductRepository,
)
from agent.core.retrievers.document_retriever import DocumentRetriever
from agent.core.retrievers.product_retriever import ProductRetriever
from catalog.models import DataSet, Document, DocumentChunk, Product, ProductContentChunk
from catalog.models.data_set import EMBEDDING_VECTOR_DIMENSIONS

pytestmark = pytest.mark.django_db

DIM = EMBEDDING_VECTOR_DIMENSIONS


def _unit(axis: int) -> list[float]:
    vec = [0.0] * DIM
    vec[axis] = 1.0
    return vec


def _rotated(axis_a: int, axis_b: int, angle: float) -> list[float]:
    """A unit vector at ``angle`` radians between ``axis_a`` and ``axis_b`` (starts on ``axis_a``)."""
    vec = [0.0] * DIM
    vec[axis_a] = math.cos(angle)
    vec[axis_b] = math.sin(angle)
    return vec


class _StubEmbeddingsRegistry:
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
    return baker.make(DataSet, name="Similarity Threshold Dataset")


def _product_retriever(data_set, query_vector, **kwargs):
    return ProductRetriever(
        data_set_id=data_set.id,
        data_set_repo=DjangoDataSetRepository(DataSet),
        product_repo=DjangoProductRepository(Product),
        product_chunk_repo=DjangoProductChunkRepository(ProductContentChunk),
        embeddings_registry=_StubEmbeddingsRegistry(query_vector),
        **kwargs,
    )


def _document_retriever(data_set, query_vector, **kwargs):
    return DocumentRetriever(
        data_set_id=data_set.id,
        data_set_repo=DjangoDataSetRepository(DataSet),
        model_chunk_repo=DjangoDocumentChunkRepository(DocumentChunk),
        embeddings_registry=_StubEmbeddingsRegistry(query_vector),
        **kwargs,
    )


class TestDocumentSimilarityThreshold:
    def test_threshold_drops_irrelevant_far_chunk(self, data_set):
        # near is the exact query (distance 0); far is orthogonal (distance 1.0, clearly irrelevant).
        document = baker.make(Document, data_set=data_set, url="https://x", title="t", content="c")
        near = DocumentChunk.objects.create(document=document, content="near", embedding=_unit(0))
        DocumentChunk.objects.create(document=document, content="far", embedding=_unit(1))

        retriever = _document_retriever(data_set, _unit(0), max_objects=5, distance_threshold=0.6)
        assert [c.id for c in retriever.find_content_matching_query("near")] == [near.id]

    def test_threshold_none_keeps_historical_top_k(self, data_set):
        document = baker.make(Document, data_set=data_set, url="https://x", title="t", content="c")
        near = DocumentChunk.objects.create(document=document, content="near", embedding=_unit(0))
        far = DocumentChunk.objects.create(document=document, content="far", embedding=_unit(1))

        # No similarity floor -> the orthogonal far chunk is still returned (historical behaviour).
        retriever = _document_retriever(data_set, _unit(0), max_objects=5, distance_threshold=None)
        assert [c.id for c in retriever.find_content_matching_query("near")] == [near.id, far.id]

    def test_threshold_caps_results_below_max_objects(self, data_set):
        # Two near-ish chunks (distance 0 and 0.5) plus one far chunk (distance 1.0). With a 0.6
        # threshold and max_objects=5, the far chunk is dropped and the result is shorter than the
        # limit: ``max_objects`` stays an upper bound, the threshold reduces below it.
        document = baker.make(Document, data_set=data_set, url="https://x", title="t", content="c")
        near = DocumentChunk.objects.create(document=document, content="near", embedding=_unit(0))
        mid = DocumentChunk.objects.create(document=document, content="mid", embedding=_rotated(0, 1, math.pi / 3))
        DocumentChunk.objects.create(document=document, content="far", embedding=_unit(1))

        retriever = _document_retriever(data_set, _unit(0), max_objects=5, distance_threshold=0.6)
        results = [c.id for c in retriever.find_content_matching_query("query")]
        assert results == [near.id, mid.id]
        assert len(results) < retriever.max_objects


class TestProductSimilarityThreshold:
    def test_threshold_drops_irrelevant_far_product(self, data_set):
        near = baker.make(Product, data_set=data_set, entry_id="near", name="Near", slug="near", price=1)
        baker.make(Product, data_set=data_set, entry_id="far", name="Far", slug="far", price=1)
        ProductContentChunk.objects.create(product=near, content="red running shoes", embedding=_unit(0))
        ProductContentChunk.objects.create(product=near, content="blue cotton shirts", embedding=_unit(1))

        retriever = _product_retriever(data_set, _unit(0), distance_threshold=0.6)
        assert [p.id for p in retriever.find_products_matching_query("shoes")] == [near.id]

    def test_hybrid_keeps_keyword_match_even_when_vector_far(self, data_set):
        # The vector-far chunk matches the keyword exactly. The cosine threshold must not starve the
        # hybrid ranklist: keyword hits are relevant by text match regardless of embedding distance,
        # so the threshold is applied to the vector path only.
        product_a = baker.make(Product, data_set=data_set, entry_id="a", name="A", slug="a", price=1)
        product_b = baker.make(Product, data_set=data_set, entry_id="b", name="B", slug="b", price=1)
        ProductContentChunk.objects.create(product=product_a, content="generic description blurb", embedding=_unit(0))
        ProductContentChunk.objects.create(
            product=product_b, content="asic gtx cross encoder module", embedding=_unit(1)
        )

        retriever = _product_retriever(data_set, _unit(0), distance_threshold=0.6, hybrid_enabled=True)
        results = [p.id for p in retriever.find_products_matching_query("asic gtx")]
        # The keyword-only product survives hybrid fusion despite being vector-far.
        assert product_b.id in results


class TestPlatformDefault:
    """The platform default threshold is wired through ``extra_kwargs`` via the ``create`` factory."""

    def test_default_threshold_constant_is_within_recommended_range(self):
        # DRA / issue suggested 0.5–0.7 cosine distance as the starting value.
        assert 0.5 <= DEFAULT_COSINE_DISTANCE_THRESHOLD <= 0.7

    def test_create_passes_threshold_from_extra_kwargs(self, data_set):
        # Avoid a real embedding provider / LLM: only the kwarg plumbing matters here.
        from types import SimpleNamespace

        from enthusiast_common.config import RetrieverConfig

        repositories = SimpleNamespace(
            data_set=DjangoDataSetRepository(DataSet),
            document_chunk=DjangoDocumentChunkRepository(DocumentChunk),
        )
        config = SimpleNamespace(
            retrievers=SimpleNamespace(
                document=RetrieverConfig(
                    retriever_class=DocumentRetriever,
                    extra_kwargs={"distance_threshold": 0.42, "max_objects": 7},
                )
            )
        )

        retriever = DocumentRetriever.create(
            config=config,
            data_set_id=data_set.id,
            repositories=repositories,
            embeddings_registry=_StubEmbeddingsRegistry(_unit(0)),
            llm=None,
        )
        assert retriever.distance_threshold == 0.42
        assert retriever.max_objects == 7
