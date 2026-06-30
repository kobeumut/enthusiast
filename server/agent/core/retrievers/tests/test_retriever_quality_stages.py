"""Integration tests for the RAG-quality retrieval stages (YAZ-15, sub-steps 2–5).

Each stage is config-driven and disabled by default. These tests build real pgvector data and flip
one stage on at a time to show it changes the result ordering in the intended direction:

* hybrid (RRF) promotes an exact-keyword hit the vector ranker buried,
* lexical rerank promotes a query-term-rich chunk over a merely vector-close one,
* MMR replaces a near-duplicate chunk with a diverse one,
* ``ef_search`` runs the runtime HNSW setting end-to-end without error,
* with everything off the retriever keeps the historical pure-vector ordering.
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
    return baker.make(DataSet, name="Quality Stages Dataset")


# --------------------------------------------------------------------------- products


def _product_retriever(data_set, query_vector, **kwargs):
    return ProductRetriever(
        data_set_id=data_set.id,
        data_set_repo=DjangoDataSetRepository(DataSet),
        product_repo=DjangoProductRepository(Product),
        product_chunk_repo=DjangoProductChunkRepository(ProductContentChunk),
        embeddings_registry=_StubEmbeddingsRegistry(query_vector),
        **kwargs,
    )


class TestProductHybridAndRerank:
    def test_hybrid_rrf_promotes_exact_keyword_hit(self, data_set):
        # A is vector-close but its content lacks the keyword; B is vector-far but matches "asic".
        product_a = baker.make(Product, data_set=data_set, entry_id="a", name="A", slug="a", price=1)
        product_b = baker.make(Product, data_set=data_set, entry_id="b", name="B", slug="b", price=1)
        ProductContentChunk.objects.create(product=product_a, content="generic description blurb", embedding=_unit(0))
        ProductContentChunk.objects.create(
            product=product_b, content="asic gtx cross encoder module", embedding=_unit(1)
        )

        vector_only = _product_retriever(data_set, _unit(0))
        assert [p.id for p in vector_only.find_products_matching_query("asic gtx")] == [product_a.id, product_b.id]

        hybrid = _product_retriever(data_set, _unit(0), hybrid_enabled=True)
        assert [p.id for p in hybrid.find_products_matching_query("asic gtx")] == [product_b.id, product_a.id]

    def test_lexical_rerank_promotes_query_term_rich_chunk(self, data_set):
        # A is the closest vector but shares no query term; B is a 60° rotation (distance 0.5) but
        # contains every query term, so the lexical signal overtakes the vector gap.
        product_a = baker.make(Product, data_set=data_set, entry_id="a", name="A", slug="a", price=1)
        product_b = baker.make(Product, data_set=data_set, entry_id="b", name="B", slug="b", price=1)
        ProductContentChunk.objects.create(product=product_a, content="unrelated words here", embedding=_unit(0))
        ProductContentChunk.objects.create(product=product_b, content="asic cross encoder", embedding=_rotated(0, 1, math.pi / 3))

        vector_only = _product_retriever(data_set, _unit(0))
        assert [p.id for p in vector_only.find_products_matching_query("asic cross encoder")] == [
            product_a.id,
            product_b.id,
        ]

        reranking = _product_retriever(data_set, _unit(0), reranker_enabled=True)
        assert [p.id for p in reranking.find_products_matching_query("asic cross encoder")] == [
            product_b.id,
            product_a.id,
        ]

    def test_ef_search_runs_and_keeps_distance_order(self, data_set):
        near = baker.make(Product, data_set=data_set, entry_id="near", name="Near", slug="near", price=1)
        far = baker.make(Product, data_set=data_set, entry_id="far", name="Far", slug="far", price=1)
        ProductContentChunk.objects.create(product=near, content="near", embedding=_unit(0))
        ProductContentChunk.objects.create(product=far, content="far", embedding=_unit(1))

        retriever = _product_retriever(data_set, _unit(0), ef_search=100)
        results = retriever.find_products_matching_query("anything")
        # SET LOCAL hnsw.ef_search must not error and the vector ordering is preserved.
        assert [p.id for p in results] == [near.id, far.id]


# --------------------------------------------------------------------------- documents


def _document_retriever(data_set, query_vector, **kwargs):
    return DocumentRetriever(
        data_set_id=data_set.id,
        data_set_repo=DjangoDataSetRepository(DataSet),
        model_chunk_repo=DjangoDocumentChunkRepository(DocumentChunk),
        embeddings_registry=_StubEmbeddingsRegistry(query_vector),
        **kwargs,
    )


class TestDocumentMmrAndHybrid:
    def test_mmr_replaces_near_duplicate_with_diverse_chunk(self, data_set):
        document = baker.make(Document, data_set=data_set, url="https://x/d", title="t", content="c")
        # chunk_a is the exact query (distance 0); chunk_b is a near-duplicate (tiny rotation);
        # chunk_c is orthogonal (distance 1) but from the same document.
        chunk_a = DocumentChunk.objects.create(document=document, content="safety section", embedding=_unit(0))
        chunk_b = DocumentChunk.objects.create(
            document=document, content="safety section copy", embedding=_rotated(0, 1, 0.05)
        )
        chunk_c = DocumentChunk.objects.create(document=document, content="cleaning section", embedding=_unit(1))

        pure_vector = _document_retriever(data_set, _unit(0), max_objects=2)
        assert [c.id for c in pure_vector.find_content_matching_query("safety")] == [chunk_a.id, chunk_b.id]

        # lambda < 0.5 favours diversity: the near-duplicate (chunk_b) is penalised for being almost
        # identical to chunk_a, so the orthogonal-but-different chunk_c is preferred as the second pick.
        mmr = _document_retriever(data_set, _unit(0), max_objects=2, mmr_enabled=True, mmr_lambda=0.3)
        results = [c.id for c in mmr.find_content_matching_query("safety")]
        assert results[0] == chunk_a.id  # most relevant always picked first
        assert results == [chunk_a.id, chunk_c.id]

    def test_hybrid_rrf_promotes_exact_keyword_document(self, data_set):
        doc_a = baker.make(Document, data_set=data_set, url="https://x/a", title="a", content="c")
        doc_b = baker.make(Document, data_set=data_set, url="https://x/b", title="b", content="c")
        # chunk_a vector-close, no keyword; chunk_b vector-far, matches the keyword "ac2000".
        chunk_a = DocumentChunk.objects.create(document=doc_a, content="general overview text", embedding=_unit(0))
        doc_b_chunk_id = DocumentChunk.objects.create(
            document=doc_b, content="ac2000 error code fix", embedding=_unit(1)
        ).id

        vector_only = _document_retriever(data_set, _unit(0), max_objects=5)
        assert [c.id for c in vector_only.find_content_matching_query("ac2000")] == [chunk_a.id, doc_b_chunk_id]

        hybrid = _document_retriever(data_set, _unit(0), max_objects=5, hybrid_enabled=True)
        results = [c.id for c in hybrid.find_content_matching_query("ac2000")]
        # The keyword-only chunk must be promoted ahead of the vector-only chunk by RRF.
        assert results == [doc_b_chunk_id, chunk_a.id]


class TestDefaultsPreservePureVectorBehavior:
    """With no quality flags set, the retrievers reproduce the historical pure-vector ordering."""

    def test_product_default_is_pure_vector(self, data_set):
        near = baker.make(Product, data_set=data_set, entry_id="near", name="Near", slug="near", price=1)
        far = baker.make(Product, data_set=data_set, entry_id="far", name="Far", slug="far", price=1)
        ProductContentChunk.objects.create(product=near, content="near", embedding=_unit(0))
        ProductContentChunk.objects.create(product=far, content="far", embedding=_unit(1))

        retriever = _product_retriever(data_set, _unit(0))
        assert [p.id for p in retriever.find_products_matching_query("near")] == [near.id, far.id]
        # No optional knobs are on by default.
        assert retriever.hybrid_enabled is False
        assert retriever.reranker is None

    def test_document_default_is_pure_vector(self, data_set):
        document = baker.make(Document, data_set=data_set, url="https://x", title="t", content="c")
        near = DocumentChunk.objects.create(document=document, content="near", embedding=_unit(0))
        far = DocumentChunk.objects.create(document=document, content="far", embedding=_unit(1))

        retriever = _document_retriever(data_set, _unit(0), max_objects=5)
        assert [c.id for c in retriever.find_content_matching_query("near")] == [near.id, far.id]
        assert retriever.hybrid_enabled is False
        assert retriever.reranker is None
        assert retriever.mmr_enabled is False
