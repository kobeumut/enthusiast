"""Project-wide system checks for the catalog app."""

from django.core.checks import Warning as DjangoWarning
from django.core.checks import register
from django.db import connection


@register()
def check_data_set_embedding_dimensions(app_configs, **kwargs):
    """Warn when a DataSet's configured embedding dimension drifts from the chunk-table column.

    pgvector can only build ANN indexes on a fixed-dimension ``vector(N)`` column, so every
    chunk table shares one dimension (``catalog.models.EMBEDDING_VECTOR_DIMENSIONS``). A data set
    configured with a different ``embedding_vector_dimensions`` cannot store its chunks in that
    column and must be recreated with the matching dimension. This check surfaces that
    misconfiguration.

    It is defensive about database availability: it is skipped entirely when the ``catalog_dataset``
    table is absent (e.g. during ``makemigrations`` on a fresh project) or when the database is
    unreachable, so it never breaks management commands.
    """
    warnings = []
    try:
        if "catalog_dataset" not in connection.introspection.table_names():
            return warnings
        # Imported here (not at module top) to avoid model access during app initialization.
        from catalog.models import EMBEDDING_VECTOR_DIMENSIONS, DataSet

        mismatched = DataSet.objects.exclude(embedding_vector_dimensions=EMBEDDING_VECTOR_DIMENSIONS).count()
    except Exception:  # noqa: BLE001 - intentionally broad: checks must never break a command.
        return warnings

    if mismatched:
        warnings.append(
            DjangoWarning(
                "%(n)d data set(s) have an embedding_vector_dimensions that differs from the "
                "chunk-table column dimension (%(dim)d). Their chunks cannot be stored in the "
                "fixed-dimension embedding column; recreate them with the matching dimension."
                % {"n": mismatched, "dim": EMBEDDING_VECTOR_DIMENSIONS},
                id="catalog.W001",
            )
        )
    return warnings
