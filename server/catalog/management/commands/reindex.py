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

        for data_set in data_sets:
            self.stdout.write(
                self.style.MIGRATE_HEADING(f"Reindexing data set '{data_set.name}' (id={data_set.id})")
            )
            if reindex_products:
                self._reindex_queryset(
                    label="Products",
                    queryset=data_set.products.all(),
                    index_object=ProductEmbeddingGenerator.index_object,
                    identifier_attr="entry_id",
                )
            if reindex_documents:
                self._reindex_queryset(
                    label="Documents",
                    queryset=data_set.documents.all(),
                    index_object=DocumentEmbeddingGenerator.index_object,
                    identifier_attr="url",
                )

        self.stdout.write(self.style.SUCCESS("Reindex complete."))

    def _reindex_queryset(self, label, queryset, index_object, identifier_attr):
        total = queryset.count()
        self.stdout.write(f"  {label}: {total} to reindex")
        for index, obj in enumerate(queryset.iterator(), start=1):
            index_object(obj)
            # Per-item progress is noisy on large backfills; only print it at -v 2 and above.
            if self.verbosity >= 2:
                self.stdout.write(f"    [{index}/{total}] {label.lower()} {getattr(obj, identifier_attr)}")
        self.stdout.write(self.style.SUCCESS(f"  {label} reindexed: {total}"))
