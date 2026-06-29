from typing import Generic, TypeVar

from django.db import models

from agent.core.registries.embeddings import EmbeddingProviderRegistry
from catalog.models import Document, Product

T = TypeVar("T", bound=models.Model)


class DataSetObjectEmbeddingsGenerator(Generic[T]):
    @staticmethod
    def index_object(obj: T) -> None:
        """Splits the document into chunks and generates embeddings for them using data set's configuration.
        Removes the old chunks and embeddings if present.

        The embedding provider is resolved and instantiated once, outside the chunk
        loop, and all chunk contents are embedded through a single batched request
        (``generate_embeddings_batch``). This avoids one registry lookup, one provider
        instance and one HTTP client per chunk, which dominates indexing cost/latency
        on large catalogs.

        Args:
            obj (Document | Product): The object to (re-)index
        """
        data_set = obj.data_set
        obj.split(data_set.embedding_chunk_size, data_set.embedding_chunk_overlap)

        chunks = list(obj.chunks.all())
        if not chunks:
            return

        embedding_provider_class = EmbeddingProviderRegistry().provider_for_dataset(data_set.id)
        embedding_provider = embedding_provider_class(
            data_set.embedding_model, data_set.embedding_vector_dimensions
        )

        contents = [chunk.content for chunk in chunks]
        embeddings = embedding_provider.generate_embeddings_batch(contents)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.set_embedding(embedding)
            chunk.save()


class ProductEmbeddingGenerator(DataSetObjectEmbeddingsGenerator[Product]):
    pass


class DocumentEmbeddingGenerator(DataSetObjectEmbeddingsGenerator[Document]):
    pass
