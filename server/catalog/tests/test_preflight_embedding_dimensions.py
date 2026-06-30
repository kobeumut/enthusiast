"""Tests for the ``preflight_embedding_dimensions`` deploy-gate command (YAZ-17).

The command detects drift that would break ``catalog.0014``'s ``ALTER COLUMN ... TYPE
vector(512)``: a stored chunk embedding whose dimension is not the shared
``EMBEDDING_VECTOR_DIMENSIONS`` (512). Such rows make the ALTER fail with
``expected 512 dimensions, not N``.

Coverage:
* Clean database / matching vectors / matching data sets -> exit 0, OK.
* The pgvector ``vector_dims()`` detection path returns the real stored dimension.
* Stored non-512 vector -> BLOCKING, raises CommandError (non-zero exit).
* A data set configured for a non-512 dimension -> non-blocking warning (exit 0
  without ``--strict``, CommandError with ``--strict``).
"""

import io

import pytest
from django.core.management import CommandError, call_command
from model_bakery import baker

from catalog.management.commands.preflight_embedding_dimensions import Command
from catalog.models import (
    EMBEDDING_VECTOR_DIMENSIONS,
    DataSet,
    Document,
    DocumentChunk,
    Product,
    ProductContentChunk,
)

DIM = EMBEDDING_VECTOR_DIMENSIONS


def _run(*args):
    """Run the preflight command, returning the captured stdout.

    ``call_command`` lets CommandError propagate, so callers wrap it in
    ``pytest.raises(CommandError)`` for the failing cases.
    """
    out = io.StringIO()
    call_command("preflight_embedding_dimensions", *args, stdout=out, stderr=out)
    return out.getvalue()


@pytest.mark.django_db
def test_preflight_passes_on_clean_database():
    output = _run()
    assert "Preflight OK" in output
    assert f"target dimension: {DIM}" in output
    assert "DocumentChunk: 0 stored vector(s)" in output
    assert "ProductContentChunk: 0 stored vector(s)" in output
    assert "DataSet: all data sets configured" in output


@pytest.mark.django_db
def test_preflight_reports_counts_when_vectors_match_the_shared_dimension():
    data_set = baker.make(DataSet, embedding_vector_dimensions=DIM)
    document = baker.make(Document, data_set=data_set, url="u", title="t", content="c")
    product = baker.make(
        Product, data_set=data_set, entry_id="p", name="n", slug="s", description="d", price=1.0
    )

    doc_chunk = DocumentChunk.objects.create(document=document, content="doc")
    doc_chunk.set_embedding([0.0] * DIM)
    doc_chunk.save()

    prod_chunk = ProductContentChunk.objects.create(product=product, content="prod")
    prod_chunk.set_embedding([0.0] * DIM)
    prod_chunk.save()

    output = _run()
    assert "DocumentChunk: 1 stored vector(s), all 512d." in output
    assert "ProductContentChunk: 1 stored vector(s), all 512d." in output
    assert "Preflight OK" in output


@pytest.mark.django_db
def test_stored_dimension_distribution_reads_real_vector_dimensions():
    """The detection helper exercises pgvector's vector_dims() against real rows."""
    data_set = baker.make(DataSet, embedding_vector_dimensions=DIM)
    document = baker.make(Document, data_set=data_set, url="u", title="t", content="c")
    chunk = DocumentChunk.objects.create(document=document, content="c")
    chunk.set_embedding([0.0] * DIM)
    chunk.save()

    assert Command._stored_dimension_distribution("catalog_documentchunk") == {DIM: 1}
    # NULL embeddings are excluded by the WHERE embedding IS NOT NULL filter.
    DocumentChunk.objects.create(document=document, content="null-emb")
    assert Command._stored_dimension_distribution("catalog_documentchunk") == {DIM: 1}


@pytest.mark.django_db
def test_preflight_blocks_on_stored_vector_with_non_target_dimension(monkeypatch):
    """A stored vector whose dimension != the target is BLOCKING (non-zero exit).

    We simulate the pre-0014 state (an unbounded vector column holding a non-512 row)
    by feeding the detection helper a non-target dimension distribution. This keeps the
    test free of fragile in-test DDL (ALTER TYPE rebuilds the HNSW index, which pgvector
    rejects on unbounded columns); the real ``vector_dims()`` SQL is covered by the
    matching-vector and helper tests above.
    """

    def fake_distribution(table):
        return {"catalog_documentchunk": {DIM + 256: 2}, "catalog_productcontentchunk": {}}.get(table, {})

    monkeypatch.setattr(Command, "_stored_dimension_distribution", staticmethod(fake_distribution))

    out = io.StringIO()
    with pytest.raises(CommandError, match="BLOCKING"):
        call_command("preflight_embedding_dimensions", stdout=out, stderr=out)
    output = out.getvalue()
    assert f"{DIM + 256}d x2" in output
    assert "DocumentChunk" in output and "BLOCKING" in output


@pytest.mark.django_db
def test_preflight_warns_on_mismatched_dataset_config_without_strict():
    DataSet.objects.create(name="legacy", embedding_vector_dimensions=DIM + 1)

    output = _run()
    assert "non-blocking warnings" in output
    assert f"embedding_vector_dimensions != {DIM}" in output
    assert "Preflight OK" not in output


@pytest.mark.django_db
def test_preflight_strict_fails_on_mismatched_dataset_config():
    DataSet.objects.create(name="legacy", embedding_vector_dimensions=DIM + 1)

    with pytest.raises(CommandError, match="Strict mode"):
        _run("--strict")
