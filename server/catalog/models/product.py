import hashlib
import json

from django.db import models
from langchain_text_splitters import TokenTextSplitter

from .data_set import DataSet


class Product(models.Model):
    # ``db_index=False`` because the explicit ``catalog_product_data_set_idx`` index
    # (and the ``uq_product`` constraint on (data_set, entry_id)) already index the
    # data_set lookup path used by retrieval.
    data_set = models.ForeignKey(
        DataSet, on_delete=models.PROTECT, related_name="products", db_index=False
    )
    entry_id = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    slug = models.CharField(max_length=255)
    description = models.TextField()
    sku = models.CharField(max_length=255, blank=True)
    properties = models.CharField(max_length=65535, blank=True)
    categories = models.CharField(max_length=65535, blank=True)
    price = models.FloatField()
    # Canonical hash of the content fields that feed the chunker/embedder (see
    # ``compute_content_hash``). Stored so the sync layer can detect no-op re-syncs
    # and skip the expensive re-index. ``null=True`` keeps existing rows valid until
    # the first sync backfills it.
    content_hash = models.CharField(max_length=64, null=True, blank=True)

    class Meta:
        db_table_comment = "List of products from a given data set."
        constraints = [models.UniqueConstraint(fields=["data_set", "entry_id"], name="uq_product")]
        indexes = [models.Index(fields=["data_set"], name="catalog_product_data_set_idx")]

    #: Fields whose change should trigger a re-index (i.e. anything that affects the
    #: generated chunks/embeddings). ``entry_id``/``slug`` are identity/lookup fields
    #: and are intentionally excluded — changing them does not alter embeddings.
    CONTENT_HASH_FIELDS = ("name", "description", "sku", "properties", "categories", "price")

    @classmethod
    def compute_content_hash(cls, *, name, description, sku, properties, categories, price) -> str:
        """Return a stable sha256 hex digest over the content fields that drive embeddings.

        Hash scope (must stay in sync with ``CONTENT_HASH_FIELDS``):
        ``name``, ``description``, ``sku``, ``properties``, ``categories``, ``price``.
        Any of these changing means the chunked/embedded representation changed, so a
        re-index is required. ``json.dumps(..., sort_keys=True)`` gives a deterministic
        key order regardless of caller, and ``ensure_ascii=False`` keeps unicode stable.
        """
        payload = json.dumps(
            {
                "name": name,
                "description": description,
                "sku": sku,
                "properties": properties,
                "categories": categories,
                "price": price,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_content(self):
        return f"{self.name} {self.description}"

    def split(self, chunk_size, chunk_overlap):
        """
        Split a document into chunks that comply with the embedding model's token limits, removing old chunks if present.

        This function splits a document into one or more overlapping chunks to provide context for user queries.
        The main rule is that each chunk must stay within the token limit of the embedding model.
        For long documents that exceed this limit, the document is divided into multiple smaller chunks,
        while shorter documents are represented as a single chunk.
        If old chunks are present, they are removed before creating new ones.

        Args:
            chunk_size (int): The maximum number of tokens allowed in a single chunk.
            chunk_overlap (int): The number of overlapping tokens between adjacent chunks.
        """
        self.chunks.all().delete()

        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(self.get_content())

        for chunk in chunks:
            self.chunks.create(product=self, content=chunk)
