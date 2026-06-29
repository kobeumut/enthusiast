from django.core.management.base import BaseCommand, CommandError

from catalog.models import DataSet
from catalog.services import DocumentEmbeddingGenerator, ProductEmbeddingGenerator


class Command(BaseCommand):
    help = (
        "Reindex (backfill) product and document content into the PostgreSQL pgvector store. "
        "Each item is re-split into chunks and its embeddings are regenerated using the data set's "
        "embedding configuration. The command runs synchronously in the foreground, so it does not "
        "depend on a running Celery worker — useful for an initial backfill or when recovering from a "
        "dimension/model change."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-set",
            type=int,
            default=None,
            help="ID of the data set to reindex. If omitted, every data set is reindexed.",
        )
        parser.add_argument(
            "--products",
            action="store_true",
            default=False,
            help="Reindex product content chunks only.",
        )
        parser.add_argument(
            "--documents",
            action="store_true",
            default=False,
            help="Reindex document chunks only.",
        )

    def handle(self, *args, **options):
        self.verbosity = options.get("verbosity", 1)
        data_set_id = options["data_set"]
        reindex_products = options["products"]
        reindex_documents = options["documents"]

        # When neither flag is provided, reindex both products and documents.
        if not reindex_products and not reindex_documents:
            reindex_products = True
            reindex_documents = True

        data_sets = DataSet.objects.all()
        if data_set_id is not None:
            data_sets = data_sets.filter(id=data_set_id)
            if not data_sets.exists():
                raise CommandError(f"Data set with id {data_set_id} does not exist.")

        total_failures = 0
        for data_set in data_sets:
            self.stdout.write(
                self.style.MIGRATE_HEADING(f"Reindexing data set '{data_set.name}' (id={data_set.id})")
            )
            if reindex_products:
                total_failures += self._reindex_queryset(
                    label="Products",
                    queryset=data_set.products.all(),
                    index_object=ProductEmbeddingGenerator.index_object,
                    identifier_attr="entry_id",
                )
            if reindex_documents:
                total_failures += self._reindex_queryset(
                    label="Documents",
                    queryset=data_set.documents.all(),
                    index_object=DocumentEmbeddingGenerator.index_object,
                    identifier_attr="url",
                )

        if total_failures:
            # Non-zero exit code + explicit summary so partial failures are never silent.
            raise CommandError(f"Reindex complete with {total_failures} failure(s); see log output above.")
        self.stdout.write(self.style.SUCCESS("Reindex complete."))

    def _reindex_queryset(self, label, queryset, index_object, identifier_attr) -> int:
        """Reindexes every object in ``queryset``, returning the number of items that failed.

        A failure on one item (e.g. an embedding API error) is logged with the item's identifier and
        the run continues, so a single bad item does not abort a large backfill. The caller raises a
        ``CommandError`` if any item failed, keeping failures visible instead of silent.
        """
        total = queryset.count()
        self.stdout.write(f"  {label}: {total} to reindex")
        failures = 0
        succeeded = 0
        for index, obj in enumerate(queryset.iterator(), start=1):
            identifier = getattr(obj, identifier_attr)
            try:
                index_object(obj)
            except Exception as exc:  # noqa: BLE001 — log every failure class and keep going
                failures += 1
                self.stdout.write(
                    self.style.ERROR(f"    [{index}/{total}] FAILED {label.lower()} {identifier}: {exc}")
                )
                continue
            succeeded += 1
            # Per-item success progress is noisy on large backfills; only print it at -v 2 and above.
            if self.verbosity >= 2:
                self.stdout.write(f"    [{index}/{total}] {label.lower()} {identifier}")
        summary = f"  {label} reindexed: {succeeded}/{total}"
        if failures:
            self.stdout.write(self.style.WARNING(f"{summary} ({failures} failed)"))
        else:
            self.stdout.write(self.style.SUCCESS(summary))
        return failures
