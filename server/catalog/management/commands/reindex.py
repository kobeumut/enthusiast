import logging
import time

from django.core.management.base import BaseCommand, CommandError

from catalog.models import DataSet
from catalog.services import DocumentEmbeddingGenerator, ProductEmbeddingGenerator

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Reindex (backfill) product and document content into the PostgreSQL pgvector store. "
        "Each item is re-split into chunks and its embeddings are regenerated using the data set's "
        "embedding configuration. The command runs synchronously in the foreground, so it does not "
        "depend on a running Celery worker — useful for an initial backfill or when recovering from a "
        "dimension/model change.\n\n"
        "A single bad item (embedding API error, oversize chunk, transient network failure) never "
        "aborts the whole backfill: each item is retried with exponential backoff, and on terminal "
        "failure the error is recorded while the run continues. A summary of ok/failed items is "
        "printed at the end. Pass --fail-fast to instead stop on the first terminal failure "
        "(CI / debugging)."
    )

    #: Default maximum number of attempts (initial try + retries) per item before it counts as failed.
    DEFAULT_MAX_ATTEMPTS = 3
    #: Default base delay (seconds) for the exponential backoff applied between attempts.
    DEFAULT_RETRY_BACKOFF = 1.0

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
        parser.add_argument(
            "--fail-fast",
            action="store_true",
            default=False,
            help="Stop on the first item that terminally fails to index (after exhausting retries). "
            "Useful for CI / debugging: the command exits non-zero when it aborts.",
        )
        parser.add_argument(
            "--max-attempts",
            type=int,
            default=self.DEFAULT_MAX_ATTEMPTS,
            help="Maximum number of indexing attempts per item before it is counted as failed "
            "(default: %(default)s). The first attempt counts, so 3 means up to 2 retries.",
        )
        parser.add_argument(
            "--retry-backoff",
            type=float,
            default=self.DEFAULT_RETRY_BACKOFF,
            help="Base seconds for exponential backoff between attempts (default: %(default)s). "
            "Delays grow as backoff * 2 ** (attempt - 1).",
        )
        parser.add_argument(
            "--from-id",
            type=int,
            default=None,
            help="Resume a backfill: only reindex items whose primary key is greater than or equal "
            "to this id. Items are processed in primary-key order when set. "
            "Combine with --limit for batched runs.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Process at most this many items per queryset. Items are processed in primary-key "
            "order when set. Combine with --from-id to resume a large backfill in batches.",
        )

    def handle(self, *args, **options):
        self.verbosity = options.get("verbosity", 1)
        self.fail_fast = options["fail_fast"]

        data_set_id = options["data_set"]
        reindex_products = options["products"]
        reindex_documents = options["documents"]
        from_id = options["from_id"]
        limit = options["limit"]
        max_attempts = options["max_attempts"]
        retry_backoff = options["retry_backoff"]

        # Validate options up front with clear errors instead of silently clamping them or
        # crashing later (e.g. a negative --limit is not supported by queryset slicing).
        if max_attempts < 1:
            raise CommandError("--max-attempts must be at least 1.")
        if retry_backoff < 0.0:
            raise CommandError("--retry-backoff must not be negative.")
        if limit is not None and limit < 1:
            raise CommandError("--limit must be a positive integer.")
        self.max_attempts = max_attempts
        self.retry_backoff = retry_backoff

        # When neither flag is provided, reindex both products and documents.
        if not reindex_products and not reindex_documents:
            reindex_products = True
            reindex_documents = True

        data_sets = DataSet.objects.all()
        if data_set_id is not None:
            data_sets = data_sets.filter(id=data_set_id)
            if not data_sets.exists():
                raise CommandError(f"Data set with id {data_set_id} does not exist.")

        # Aggregate across every queryset so a single end-of-run summary lists all failures.
        self._succeeded = 0
        self._failures = []
        aborted = False

        try:
            for data_set in data_sets:
                self.stdout.write(
                    self.style.MIGRATE_HEADING(f"Reindexing data set '{data_set.name}' (id={data_set.id})")
                )
                if reindex_products:
                    aborted = self._reindex_queryset(
                        label="Products",
                        queryset=self._scope_queryset(data_set.products.all(), from_id, limit),
                        index_object=ProductEmbeddingGenerator.index_object,
                        identifier_attr="entry_id",
                    )
                    if aborted:
                        break
                if reindex_documents:
                    aborted = self._reindex_queryset(
                        label="Documents",
                        queryset=self._scope_queryset(data_set.documents.all(), from_id, limit),
                        index_object=DocumentEmbeddingGenerator.index_object,
                        identifier_attr="url",
                    )
                    if aborted:
                        break
        finally:
            # Always report what happened, even if an unexpected error aborted the run.
            self._print_summary()

        if aborted:
            raise CommandError(
                f"Reindex aborted (--fail-fast) after {len(self._failures)} failed item(s); see summary above."
            )
        if self._failures:
            self.stdout.write(
                self.style.WARNING(f"Reindex complete with {len(self._failures)} failed item(s); see summary above.")
            )
        else:
            self.stdout.write(self.style.SUCCESS("Reindex complete."))

    def _scope_queryset(self, queryset, from_id, limit):
        """Apply optional resume / pagination filters to a queryset.

        Items are iterated in deterministic primary-key order whenever a resume or limit option is
        used, so the same ``--from-id`` value continues exactly where a previous batch left off.
        When neither option is given the queryset is returned untouched (existing behavior).
        """
        if from_id is None and limit is None:
            return queryset
        queryset = queryset.order_by("pk")
        if from_id is not None:
            queryset = queryset.filter(pk__gte=from_id)
        if limit is not None:
            queryset = queryset[:limit]
        return queryset

    def _reindex_queryset(self, label, queryset, index_object, identifier_attr):
        """Reindex one queryset with per-item error isolation and retry.

        Args:
            label: Human-readable label used in log/summary lines (e.g. ``"Products"``).
            queryset: The objects to reindex.
            index_object: Callable that (re-)indexes a single object (e.g.
                ``ProductEmbeddingGenerator.index_object``).
            identifier_attr: Attribute used to identify an item in the output
                (``entry_id`` for products, ``url`` for documents).

        Returns:
            True if the run should abort (``--fail-fast`` triggered on a failure),
            False otherwise.
        """
        total = queryset.count()
        self.stdout.write(f"  {label}: {total} to reindex")
        ok = 0
        failed = 0

        for index, obj in enumerate(queryset.iterator(), start=1):
            identifier = getattr(obj, identifier_attr)
            error = self._index_with_retries(index_object, obj, identifier, label)
            if error is None:
                ok += 1
                self._succeeded += 1
                # Per-item progress is noisy on large backfills; only print it at -v 2 and above.
                if self.verbosity >= 2:
                    self.stdout.write(f"    [{index}/{total}] {label.lower()} {identifier} ok")
            else:
                failed += 1
                self._failures.append({"label": label, "identifier": identifier, "pk": obj.pk, "error": error})
                # Failures are the actionable signal — surface them even at the default verbosity.
                self.stdout.write(
                    self.style.WARNING(f"    [{index}/{total}] {label.lower()} {identifier} FAILED: {error}")
                )
                if self.fail_fast:
                    return True

        self.stdout.write(self.style.SUCCESS(f"  {label} reindexed: {ok} ok / {failed} fail"))
        return False

    def _index_with_retries(self, index_object, obj, identifier, label):
        """Index a single item, retrying transient failures with exponential backoff.

        Retrying is safe: ``index_object`` removes the item's existing chunks before re-splitting,
        so any partial state left by a failed attempt is replaced on the next try.

        Args:
            index_object: Callable that (re-)indexes ``obj``.
            obj: The object to index.
            identifier: Human-readable identifier of ``obj`` (for logging only).
            label: Human-readable label of the queryset ``obj`` belongs to.

        Returns:
            ``None`` on success, or the final error message after every attempt is exhausted.
        """
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                index_object(obj)
                return None
            except Exception as exc:
                # Isolate any per-item failure so one bad item can't abort the whole backfill.
                last_error = exc
                logger.warning(
                    "%s %s failed indexing (attempt %d/%d): %s",
                    label,
                    identifier,
                    attempt,
                    self.max_attempts,
                    exc,
                    exc_info=True,
                )
                if attempt < self.max_attempts:
                    delay = self.retry_backoff * (2 ** (attempt - 1))
                    if self.verbosity >= 2:
                        self.stdout.write(
                            f"    retry {attempt}/{self.max_attempts - 1} for {label.lower()} "
                            f"{identifier} in {delay:.1f}s: {exc}"
                        )
                    if delay > 0:
                        time.sleep(delay)
        # Include the exception class so the summary stays actionable even when the message is
        # blank or ambiguous (e.g. "ValueError" vs "KeyError: 'foo'").
        return f"{type(last_error).__name__}: {last_error}"

    def _print_summary(self):
        """Print the aggregated ok/failed summary, including the list of failed items."""
        total_fail = len(self._failures)
        summary_line = f"Reindex summary: {self._succeeded} ok / {total_fail} fail"
        # A run with failures is a warning, not a success — color the header accordingly.
        summary_style = self.style.WARNING if total_fail else self.style.SUCCESS
        self.stdout.write(summary_style(summary_line))
        if not total_fail:
            return
        self.stdout.write(self.style.WARNING(f"Failed items ({total_fail}):"))
        for failure in self._failures:
            self.stdout.write(
                self.style.WARNING(
                    f"  [{failure['label']}] {failure['identifier']} (pk={failure['pk']}): {failure['error']}"
                )
            )
