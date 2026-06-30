"""Preflight management command for the pgvector dimension migration (YAZ-17).

Run ``python manage.py preflight_embedding_dimensions`` on an existing database
BEFORE deploying ``catalog.0014``. The migration ALTERs the two chunk embedding
columns to ``vector(EMBEDDING_VECTOR_DIMENSIONS)`` (512); that ALTER fails on live
data that already stores a different dimension (``ERROR: expected 512 dimensions,
not N``), so the offending rows must be re-embedded or removed first. This command
reports such drift and exits non-zero on blocking findings so it can gate a deploy.
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import connection

#: The chunk tables whose ``embedding`` column ``catalog.0014`` rewrites to the shared
#: fixed dimension. ``(db table, human label)``. The table names are module-level
#: constants (never user input), so interpolating them into the dimension-count query
#: below is safe.
CHUNK_TABLES = (
    ("catalog_documentchunk", "DocumentChunk"),
    ("catalog_productcontentchunk", "ProductContentChunk"),
)


class Command(BaseCommand):
    help = (
        "Preflight the pgvector dimension migration (catalog.0014). Reports embedding "
        "vectors whose stored dimension differs from the shared chunk-table dimension "
        "(catalog.EMBEDDING_VECTOR_DIMENSIONS) and data sets configured with a "
        "non-matching embedding_vector_dimensions. Stored-vector mismatches are "
        "BLOCKING: catalog.0014's ALTER to vector(N) fails when a row carries a "
        "different dimension. Run this before applying catalog.0014 and re-embed/clean "
        "any offenders. Exits non-zero on blocking findings (usable as a deploy/CI gate)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--strict",
            action="store_true",
            help="Also exit non-zero on non-blocking warnings (data sets whose configured "
            "embedding_vector_dimensions differs). Without this flag only blocking "
            "stored-vector mismatches cause a non-zero exit.",
        )

    def handle(self, *args, **options):
        strict = options["strict"]

        # Imported lazily so the command can be loaded (e.g. by --help or system checks)
        # even before the catalog tables exist, and so it always reflects the current
        # value of the shared dimension constant.
        from catalog.models import EMBEDDING_VECTOR_DIMENSIONS

        target = EMBEDDING_VECTOR_DIMENSIONS
        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"Preflight embedding-dimension check (target dimension: {target})"
            )
        )

        existing_tables = set(connection.introspection.table_names())

        blocking = self._check_stored_vector_dimensions(target, existing_tables)
        warnings = self._check_data_set_dimensions(target)

        if blocking:
            raise CommandError(
                f"BLOCKING findings: stored embedding vectors have a dimension other than "
                f"{target}. Migration catalog.0014's ALTER to vector({target}) will fail "
                "with 'expected N dimensions, not M'. Re-embed or remove the offending "
                "chunks first (see the pgvector migration runbook), then re-run this preflight."
            )

        if warnings:
            self.stdout.write(
                self.style.WARNING(
                    "Preflight completed with non-blocking warnings. Migration catalog.0014 "
                    "will not fail, but the flagged data sets cannot store chunks until realigned."
                )
            )
            if strict:
                raise CommandError("Strict mode: non-blocking warnings treated as failure.")
            return

        self.stdout.write(self.style.SUCCESS("Preflight OK: safe to apply migration catalog.0014."))

    def _check_stored_vector_dimensions(self, target, existing_tables):
        """Report chunk rows whose stored vector dimension != ``target``.

        Returns True when any blocking mismatch is found. Uses pgvector's
        ``vector_dims()``, which returns the actual stored dimension of each vector
        (NULL for NULL rows). This works whether the column is the legacy unbounded
        ``vector`` type or the post-0014 ``vector(N)`` type, so the check is meaningful
        both before and after the migration.
        """
        blocking = False
        for table, label in CHUNK_TABLES:
            if table not in existing_tables:
                self.stdout.write(f"  {label}: table {table} not present yet — skipped.")
                continue
            distribution = self._stored_dimension_distribution(table)
            mismatched = {dim: count for dim, count in distribution.items() if dim != target}
            if mismatched:
                blocking = True
                total = sum(mismatched.values())
                rendered = ", ".join(f"{dim}d x{count}" for dim, count in sorted(mismatched.items()))
                self.stdout.write(
                    self.style.ERROR(
                        f"  {label}: {total} stored vector(s) with dimension != {target} "
                        f"({rendered}) — BLOCKING"
                    )
                )
            else:
                total = sum(distribution.values())
                self.stdout.write(self.style.SUCCESS(f"  {label}: {total} stored vector(s), all {target}d."))
        return blocking

    def _check_data_set_dimensions(self, target):
        """Report data sets whose configured embedding_vector_dimensions != ``target``.

        Returns True when any such data set exists. This is a WARNING (not blocking):
        the catalog serializer already forces new data sets to the shared dimension and
        makes embedding config immutable, and ``catalog.W001`` flags drifted rows. It is
        surfaced here so a deploy operator sees the full picture in one place.
        """
        try:
            from catalog.models import DataSet

            mismatched = list(
                DataSet.objects.exclude(embedding_vector_dimensions=target)
                .order_by("id")
                .values("id", "name", "embedding_vector_dimensions")
            )
        except Exception:  # noqa: BLE001 - catalog_dataset absent / DB unreachable: non-fatal for preflight.
            return False

        if not mismatched:
            self.stdout.write(self.style.SUCCESS("  DataSet: all data sets configured for {0}d.".format(target)))
            return False

        self.stdout.write(
            self.style.WARNING(
                f"  DataSet: {len(mismatched)} data set(s) configured with "
                f"embedding_vector_dimensions != {target} (also reported by catalog.W001):"
            )
        )
        for ds in mismatched[:20]:
            self.stdout.write(
                self.style.WARNING(
                    f"    - id={ds['id']} name={ds['name']!r} dims={ds['embedding_vector_dimensions']}"
                )
            )
        if len(mismatched) > 20:
            self.stdout.write(self.style.WARNING(f"    ... and {len(mismatched) - 20} more"))
        return True

    @staticmethod
    def _stored_dimension_distribution(table):
        """Return ``{dimension: row_count}`` for non-NULL embeddings in ``table``.

        ``table`` is one of the hardcoded ``CHUNK_TABLES`` constants (never user input),
        so it is safe to interpolate into the query.
        """
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT vector_dims(embedding) AS dim, COUNT(*) "
                f"FROM {table} WHERE embedding IS NOT NULL GROUP BY dim"
            )
            return {dim: count for dim, count in cursor.fetchall()}
