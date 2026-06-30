import hashlib
import json

from django.db import models
from langchain_text_splitters import TokenTextSplitter

from .data_set import DataSet


class Document(models.Model):
    # ``db_index=False`` because the explicit ``catalog_document_data_set_idx`` index
    # (and the ``uq_document`` constraint on (data_set, url)) already index the
    # data_set lookup path used by retrieval.
    data_set = models.ForeignKey(
        DataSet, related_name="documents", on_delete=models.PROTECT, db_index=False
    )
    url = models.CharField(max_length=255)
    title = models.CharField(max_length=1024)
    content = models.TextField()
    # Canonical hash of the content fields that feed the chunker/embedder (see
    # ``compute_content_hash``). Stored so the sync layer can detect no-op re-syncs
    # and skip the expensive re-index. ``null=True`` keeps existing rows valid until
    # the first sync backfills it.
    content_hash = models.CharField(max_length=64, null=True, blank=True)

    class Meta:
        db_table_comment = (
            "List of documents being part of a larger data set. A document may be for instance a blog "
            "post. This is the main entity being analysed by ECL engine when user asks questions "
            "regarding company's offer."
        )
        constraints = [models.UniqueConstraint(fields=["data_set", "url"], name="uq_document")]
        indexes = [models.Index(fields=["data_set"], name="catalog_document_data_set_idx")]

    #: Fields whose change should trigger a re-index (i.e. anything that affects the
    #: generated chunks/embeddings). ``url`` is the identity/lookup field and is
    #: intentionally excluded — it does not affect the generated embeddings.
    CONTENT_HASH_FIELDS = ("title", "content")

    @classmethod
    def compute_content_hash(cls, *, title, content) -> str:
        """Return a stable sha256 hex digest over the content fields that drive embeddings.

        Hash scope (must stay in sync with ``CONTENT_HASH_FIELDS``):
        ``title``, ``content``. Any of these changing means the chunked/embedded
        representation changed, so a re-index is required. ``json.dumps(...,
        sort_keys=True)`` gives a deterministic key order regardless of caller.
        """
        payload = json.dumps({"title": title, "content": content}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
        chunks = splitter.split_text(self.content)

        for chunk in chunks:
            self.chunks.create(document=self, content=chunk)
