from enthusiast_common import DocumentDetails

from catalog.models import Document, DocumentSource
from catalog.tasks import index_document_task
from sync.base import DataSetSource, SyncManager
from sync.document.registry import DocumentSourcePluginRegistry


class DocumentSyncManager(SyncManager[DocumentDetails]):
    """Orchestrates synchronisation activities for document sync plugins."""

    def _build_registry(self):
        return DocumentSourcePluginRegistry()

    def _get_data_set_source(self, source_id: int) -> DataSetSource:
        source = DocumentSource.objects.get(id=source_id)
        return DataSetSource(plugin_name=source.plugin_name, data_set_id=source.data_set_id, config=source.config)

    def _sync_item(self, data_set_id: int, item_data: DocumentDetails):
        """Creates or updates a document and re-indexes it only when its content changed.

        A no-op re-sync (same ``title``/``content``) used to re-enqueue ``index_document_task``
        unconditionally, which made ``Document.split`` wipe and rebuild every chunk and re-run the
        embedding API. That is pure cost for unchanged content. We now store a canonical
        ``content_hash`` and skip the re-index when an existing document's content is
        byte-for-byte identical. New documents and genuinely changed documents are still re-indexed
        as before.

        Args:
            data_set_id (int): obligatory, a data set to which imported data belongs to.
            item_data (DocumentDetails): item details.
        """
        new_hash = Document.compute_content_hash(title=item_data.title, content=item_data.content)
        # Look up the previously stored hash *before* update_or_create overwrites it, so we can
        # decide whether the content actually changed. ``only('content_hash')`` keeps this cheap;
        # the (data_set, url) lookup is covered by the ``uq_document`` constraint.
        existing = (
            Document.objects.filter(data_set_id=data_set_id, url=item_data.url).only("content_hash").first()
        )

        item, created = Document.objects.update_or_create(
            data_set_id=data_set_id,
            url=item_data.url,
            defaults={
                "title": item_data.title,
                "content": item_data.content,
                "content_hash": new_hash,
            },
        )

        # created=True => brand new document (existing is None) => always index.
        # Otherwise index only when the content hash actually changed; an unchanged hash means
        # the embeddings are already up to date. A null old hash (legacy row) forces an index,
        # which backfills it.
        if created or existing.content_hash != new_hash:
            index_document_task.apply_async([item.id])
