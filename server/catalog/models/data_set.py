from django.conf import settings
from django.db import models

#: Fixed dimension of the chunk-table embedding columns.
#:
#: pgvector cannot build HNSW/IVFFlat ANN indexes on an unbounded ``vector``
#: column (``ERROR: column does not have dimensions``), so the
#: ``catalog_documentchunk.embedding`` and ``catalog_productcontentchunk.embedding``
#: columns are created as ``vector(EMBEDDING_VECTOR_DIMENSIONS)``. Because a column
#: can only carry one dimension, every ``DataSet`` MUST be configured with the
#: same ``embedding_vector_dimensions``; the ``DataSet`` default below tracks this
#: value so freshly created data sets are always compatible.
#:
#: Changing this constant requires a data migration that recreates both chunk
#: embedding columns at the new dimension and re-indexes them.
EMBEDDING_VECTOR_DIMENSIONS = 512


class DataSet(models.Model):
    name = models.CharField(max_length=30)
    language_model_provider = models.CharField(default="OpenAI")
    language_model = models.CharField(default="gpt-4o")
    embedding_provider = models.CharField(max_length=255, default="OpenAI")
    embedding_model = models.CharField(max_length=255, default="text-embedding-3-large")
    embedding_vector_dimensions = models.IntegerField(default=EMBEDDING_VECTOR_DIMENSIONS)
    embedding_chunk_size = models.IntegerField(default=3000)
    embedding_chunk_overlap = models.IntegerField(default=150)

    users = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name="data_sets")

    class Meta:
        db_table_comment = (
            "List of various data sets. One data set may be the whole company's content such as blog "
            "posts, or some part of it: a data set may be represent a brand or department."
        )
