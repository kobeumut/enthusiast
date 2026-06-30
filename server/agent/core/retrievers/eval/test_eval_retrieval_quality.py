"""Labelled-corpus retrieval evaluation for the RAG quality phase (YAZ-15).

Builds a small, deterministic corpus on the real pgvector store and runs each quality stage in
isolation against a pure-vector baseline, measuring Precision@K / Recall@K / MRR before and after.
Each scenario is hand-crafted so the targeted stage moves the metric in the intended direction; the
test asserts the lift and prints a before/after table.

Scenarios:
* **A — metadata filter** (products): a wrong-category, vector-close item is removed → Precision@3.
* **B — lexical rerank** (products): a query-term-rich item overtakes a generic vector-close one → MRR.
* **C — hybrid RRF** (products): a vector-far, keyword-exact item is surfaced → Recall@4.
* **D — MMR diversity** (documents): near-duplicate sections stop crowding out diverse sections →
  distinct-section Recall@3.

This is a directional sanity check, not a statistically rigorous benchmark.
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
from agent.core.retrievers.eval.metrics import precision_at_k, recall_at_k, reciprocal_rank
from agent.core.retrievers.filters import RetrievalFilters
from agent.core.retrievers.product_retriever import ProductRetriever
from catalog.models import DataSet, Document, DocumentChunk, Product, ProductContentChunk
from catalog.models.data_set import EMBEDDING_VECTOR_DIMENSIONS

pytestmark = pytest.mark.django_db

DIM = EMBEDDING_VECTOR_DIMENSIONS


def vec(axes: set[int]) -> list[float]:
    """A unit vector with equal weight on each axis in ``axes`` (semantic-direction embedding)."""
    vector = [0.0] * DIM
    for axis in axes:
        vector[axis] = 1.0
    norm = math.sqrt(len(axes))
    return [value / norm for value in vector]


def rotated(axis_a: int, axis_b: int, angle: float) -> list[float]:
    """A unit vector at ``angle`` radians off ``axis_a`` toward ``axis_b`` (controls cosine distance)."""
    vector = [0.0] * DIM
    vector[axis_a] = math.cos(angle)
    vector[axis_b] = math.sin(angle)
    return vector


class _StubEmbeddingsRegistry:
    """Returns a fixed query embedding so the retriever's embedding call is deterministic."""

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


def _product_retriever(data_set, query_vector, number_of_products, **kwargs):
    return ProductRetriever(
        data_set_id=data_set.id,
        data_set_repo=DjangoDataSetRepository(DataSet),
        product_repo=DjangoProductRepository(Product),
        product_chunk_repo=DjangoProductChunkRepository(ProductContentChunk),
        embeddings_registry=_StubEmbeddingsRegistry(query_vector),
        number_of_products=number_of_products,
        **kwargs,
    )


def _document_retriever(data_set, query_vector, max_objects, **kwargs):
    return DocumentRetriever(
        data_set_id=data_set.id,
        data_set_repo=DjangoDataSetRepository(DataSet),
        model_chunk_repo=DjangoDocumentChunkRepository(DocumentChunk),
        embeddings_registry=_StubEmbeddingsRegistry(query_vector),
        max_objects=max_objects,
        **kwargs,
    )


# --------------------------------------------------------------------------- scenario rows


def _row(label, metric_name, before, after):
    """Format one before/after row for the summary table."""
    delta = after - before
    arrow = "↑" if delta > 0 else ("→" if delta == 0 else "↓")
    return f"  {label:<28} {metric_name:<14} {before:>5.3f}  ->  {after:>5.3f}   {arrow}"


class TestRetrievalQualityLift:
    def test_each_stage_improves_its_targeted_metric(self, capsys):
        rows = []

        # Each scenario gets its own dataset so products/documents never leak across scenarios.

        # --- Scenario A: metadata filter (Precision@3) ------------------------
        data_set = baker.make(DataSet, name="Eval A")
        relevant = baker.make(
            Product, data_set=data_set, entry_id="r", name="Runr Running Shoe", slug="r", price=40, categories="Running"
        )
        luxury_near_1 = baker.make(
            Product, data_set=data_set, entry_id="n1", name="Runr Premium", slug="n1", price=400, categories="Luxury"
        )
        luxury_near_2 = baker.make(
            Product, data_set=data_set, entry_id="n2", name="Runr Deluxe", slug="n2", price=500, categories="Luxury"
        )
        baker.make(Product, data_set=data_set, entry_id="d1", name="Cookbook", slug="d1", price=20, categories="Cooking")
        ProductContentChunk.objects.create(product=relevant, content="running shoe", embedding=vec({0}))
        ProductContentChunk.objects.create(product=luxury_near_1, content="running shoe premium", embedding=rotated(0, 6, 0.02))
        ProductContentChunk.objects.create(product=luxury_near_2, content="running shoe deluxe", embedding=rotated(0, 7, 0.04))

        query_vec = vec({0})
        relevant_ids = {relevant.id}
        baseline = _product_retriever(data_set, query_vec, number_of_products=3)
        before_ids = [p.id for p in baseline.find_products_matching_query("running shoe")]
        filtered = _product_retriever(data_set, query_vec, number_of_products=3)
        after_ids = [p.id for p in filtered.find_products_matching_query("running shoe", filters=RetrievalFilters(categories=["Running"]))]
        p_before = precision_at_k(before_ids, relevant_ids, k=3)
        p_after = precision_at_k(after_ids, relevant_ids, k=3)
        assert p_after > p_before
        rows.append(_row("metadata filter (P@3)", "precision@3", p_before, p_after))

        # --- Scenario B: lexical rerank (MRR) ---------------------------------
        data_set = baker.make(DataSet, name="Eval B")
        generic_close = baker.make(
            Product, data_set=data_set, entry_id="g", name="Generic", slug="g", price=50, categories="Running"
        )
        asic_mid = baker.make(
            Product, data_set=data_set, entry_id="asic", name="Asic", slug="asic", price=60, categories="Running"
        )
        ProductContentChunk.objects.create(product=generic_close, content="athletic footwear gear selection", embedding=vec({0}))
        ProductContentChunk.objects.create(product=asic_mid, content="asic gt 2000 running trainer", embedding=rotated(0, 1, math.pi / 3))

        query_vec_b = vec({0})
        relevant_b = {asic_mid.id}
        pure = _product_retriever(data_set, query_vec_b, number_of_products=4)
        before_b = [p.id for p in pure.find_products_matching_query("asic gt 2000")]
        reranking = _product_retriever(data_set, query_vec_b, number_of_products=4, reranker_enabled=True)
        after_b = [p.id for p in reranking.find_products_matching_query("asic gt 2000")]
        mrr_before = reciprocal_rank(before_b, relevant_b)
        mrr_after = reciprocal_rank(after_b, relevant_b)
        assert mrr_after > mrr_before
        rows.append(_row("lexical rerank (MRR)", "MRR", mrr_before, mrr_after))

        # --- Scenario C: hybrid RRF (Recall@4) --------------------------------
        data_set = baker.make(DataSet, name="Eval C")
        r1 = baker.make(Product, data_set=data_set, entry_id="c1", name="C1", slug="c1", price=70, categories="Running")
        r2 = baker.make(Product, data_set=data_set, entry_id="c2", name="C2", slug="c2", price=30, categories="Running")
        ProductContentChunk.objects.create(product=r1, content="asic trainer shoe", embedding=vec({0}))
        ProductContentChunk.objects.create(product=r2, content="asic gt 2000 spare cleat kit", embedding=vec({5}))
        # Three distractors that rank between R1 (distance 0) and R2 (distance 1) on the vector path.
        for i, angle in enumerate([0.7, 0.8, 0.9]):
            distractor = baker.make(Product, data_set=data_set, entry_id=f"dd{i}", name=f"DD{i}", slug=f"dd{i}", price=5)
            ProductContentChunk.objects.create(product=distractor, content=f"filler item {i}", embedding=rotated(0, 9 + i, angle))

        query_vec_c = vec({0})
        relevant_c = {r1.id, r2.id}
        pure_c = _product_retriever(data_set, query_vec_c, number_of_products=4)
        before_c = [p.id for p in pure_c.find_products_matching_query("asic gt 2000")]
        hybrid_c = _product_retriever(data_set, query_vec_c, number_of_products=4, hybrid_enabled=True)
        after_c = [p.id for p in hybrid_c.find_products_matching_query("asic gt 2000")]
        rec_before = recall_at_k(before_c, relevant_c, k=4)
        rec_after = recall_at_k(after_c, relevant_c, k=4)
        assert rec_after > rec_before
        rows.append(_row("hybrid RRF (Recall@4)", "recall@4", rec_before, rec_after))

        # --- Scenario D: MMR diversity (distinct-section Recall@3) ------------
        data_set = baker.make(DataSet, name="Eval D")
        document = baker.make(Document, data_set=data_set, url="https://x/manual", title="Manual", content="c")
        safety_a = DocumentChunk.objects.create(document=document, content="safety precautions section", embedding=vec({0}))
        DocumentChunk.objects.create(document=document, content="safety precautions duplicate", embedding=rotated(0, 2, 0.05))
        DocumentChunk.objects.create(document=document, content="safety warnings copy", embedding=rotated(0, 3, 0.06))
        cleaning = DocumentChunk.objects.create(document=document, content="cleaning maintenance section", embedding=vec({0, 10}))
        warranty = DocumentChunk.objects.create(document=document, content="warranty information section", embedding=vec({0, 11}))
        section_of = {safety_a.id: "safety", cleaning.id: "cleaning", warranty.id: "warranty"}
        relevant_sections = {"safety", "cleaning", "warranty"}

        def section_recall(chunk_ids, k):
            covered = {section_of[c_id] for c_id in chunk_ids[:k] if c_id in section_of}
            return len(covered & relevant_sections) / len(relevant_sections)

        query_vec_d = vec({0})
        pure_d = _document_retriever(data_set, query_vec_d, max_objects=3)
        before_d = [c.id for c in pure_d.find_content_matching_query("manual")]
        mmr_d = _document_retriever(data_set, query_vec_d, max_objects=3, mmr_enabled=True, mmr_lambda=0.3)
        after_d = [c.id for c in mmr_d.find_content_matching_query("manual")]
        srec_before = section_recall(before_d, k=3)
        srec_after = section_recall(after_d, k=3)
        assert srec_after > srec_before
        rows.append(_row("MMR diversity (SecRec@3)", "section-recall@3", srec_before, srec_after))

        # Print the before/after table for human review (visible with `pytest -s`).
        with capsys.disabled():
            print("\n" + "=" * 72)
            print("RAG quality phase — retrieval evaluation (before -> after)")
            print("=" * 72)
            for row in rows:
                print(row)
            print("=" * 72)
