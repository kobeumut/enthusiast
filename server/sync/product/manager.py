from enthusiast_common import ProductDetails

from catalog.models import Product, ProductSource
from catalog.tasks import index_product_task
from sync.base import DataSetSource, SyncManager
from sync.product.registry import ProductSourcePluginRegistry


class ProductSyncManager(SyncManager[ProductDetails]):
    """Orchestrates synchronisation activities of registered product plugins."""

    def _build_registry(self):
        return ProductSourcePluginRegistry()

    def _get_data_set_source(self, source_id: int) -> DataSetSource:
        source = ProductSource.objects.get(id=source_id)
        return DataSetSource(plugin_name=source.plugin_name, data_set_id=source.data_set_id, config=source.config)

    def _sync_item(self, data_set_id: int, item_data: ProductDetails):
        """Creates a product in the database and queues indexing only when needed.

        Product embeddings are derived from ``name`` + ``description`` (see ``Product.get_content()``),
        so a re-index is queued only for newly created products or when those fields actually changed.
        This avoids enqueuing a full re-split + embedding API call for every product on every sync when
        the content is unchanged.

        Args:
            data_set_id (int): obligatory, a data set to which imported data belongs to.
            item_data (ProductDetails): item details.
        """
        existing = Product.objects.filter(data_set_id=data_set_id, entry_id=item_data.entry_id).first()
        needs_reindex = existing is None or self._product_content_changed(existing, item_data)

        item, _created = Product.objects.update_or_create(
            data_set_id=data_set_id,
            entry_id=item_data.entry_id,
            defaults={
                "name": item_data.name,
                "slug": item_data.slug,
                "description": item_data.description,
                "sku": item_data.sku,
                "properties": item_data.properties,
                "categories": item_data.categories,
                "price": item_data.price,
            },
        )
        if needs_reindex:
            index_product_task.apply_async([item.id])

    @staticmethod
    def _product_content_changed(existing: Product, item_data: ProductDetails) -> bool:
        """Returns whether the product fields that feed the embedding changed during this sync.

        Only ``name`` and ``description`` are embedded (``Product.get_content``), so changes to the
        other catalog fields (price, sku, properties, categories) do not require a re-index.
        """
        return existing.name != item_data.name or existing.description != item_data.description
