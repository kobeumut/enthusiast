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
        """Creates or updates a product and re-indexes it only when its content changed.

        A no-op re-sync (same ``name``/``description``/``sku``/``properties``/``categories``/
        ``price``) used to re-enqueue ``index_product_task`` unconditionally, which made
        ``Product.split`` wipe and rebuild every chunk and re-run the embedding API. That is
        pure cost for unchanged content. We now store a canonical ``content_hash`` and skip the
        re-index when an existing product's content is byte-for-byte identical. New products and
        genuinely changed products are still re-indexed as before.

        Args:
            data_set_id (int): obligatory, a data set to which imported data belongs to.
            item_data (ProductDetails): item details.
        """
        new_hash = Product.compute_content_hash(
            name=item_data.name,
            description=item_data.description,
            sku=item_data.sku,
            properties=item_data.properties,
            categories=item_data.categories,
            price=item_data.price,
        )
        # Look up the previously stored hash *before* update_or_create overwrites it, so we can
        # decide whether the content actually changed. ``only('content_hash')`` keeps this cheap;
        # the (data_set, entry_id) lookup is covered by the ``uq_product`` constraint.
        existing = (
            Product.objects.filter(data_set_id=data_set_id, entry_id=item_data.entry_id)
            .only("content_hash")
            .first()
        )

        item, created = Product.objects.update_or_create(
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
                "content_hash": new_hash,
            },
        )

        # created=True => brand new product (existing is None) => always index.
        # Otherwise index only when the content hash actually changed; an unchanged hash means
        # the embeddings are already up to date. A null old hash (legacy row) forces an index,
        # which backfills it.
        if created or existing.content_hash != new_hash:
            index_product_task.apply_async([item.id])
