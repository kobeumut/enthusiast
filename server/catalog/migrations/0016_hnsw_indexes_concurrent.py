"""Build the HNSW ANN indexes with CREATE INDEX CONCURRENTLY (YAZ-17).

These two HNSW indexes originally lived in ``0014_pgvector_ann_indexes``. They are
created here, in a dedicated ``atomic = False`` migration, so PostgreSQL builds
them with ``CREATE INDEX CONCURRENTLY``: that does **not** take a table-level
``ACCESS EXCLUSIVE`` lock, so it is safe to run on the large ``catalog_documentchunk``
/ ``catalog_productcontentchunk`` tables in production (writes continue while the
index is built). ``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction,
hence ``atomic = False`` on the whole migration.

pgvector's HNSW index supports ``CONCURRENTLY`` (verified against
``pgvector/pgvector:pg17``), and ``AddIndexConcurrently`` renders the index through
the same ``HnswIndex`` class the models declare in their ``Meta.indexes``, so the
index definitions stay identical to the model state and ``makemigrations --check``
remains clean. Reverse builds ``DROP INDEX CONCURRENTLY``.
"""

import pgvector.django.indexes
from django.contrib.postgres.operations import AddIndexConcurrently
from django.db import migrations


class Migration(migrations.Migration):
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction.
    atomic = False

    dependencies = [
        ("catalog", "0015_product_document_content_hash"),
    ]

    operations = [
        AddIndexConcurrently(
            model_name="documentchunk",
            index=pgvector.django.indexes.HnswIndex(
                fields=["embedding"],
                name="document_chunk_embedding_idx",
                opclasses=["vector_cosine_ops"],
            ),
        ),
        AddIndexConcurrently(
            model_name="productcontentchunk",
            index=pgvector.django.indexes.HnswIndex(
                fields=["embedding"],
                name="product_chunk_embedding_idx",
                opclasses=["vector_cosine_ops"],
            ),
        ),
    ]
