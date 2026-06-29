from django.db import models
from pgvector.django import HnswIndex, VectorField

from .data_set import EMBEDDING_VECTOR_DIMENSIONS
from .product import Product


class ProductContentChunk(models.Model):
    product = models.ForeignKey(Product, related_name="chunks", on_delete=models.CASCADE)
    content = models.TextField()
    embedding = VectorField(dimensions=EMBEDDING_VECTOR_DIMENSIONS, null=True)

    def set_embedding(self, embedding_vector: list[float]):
        """Sets the embedding vector for this document chunk.

        Args:
            embedding_vector (list[float]): The embedding vector to associate with the document chunk.
        """
        self.embedding = embedding_vector

    class Meta:
        # HNSW is the default ANN index: it can be built incrementally, does not
        # require IVFFlat training data, and supports the cosine distance operator
        # class the retrieval layer orders by (CosineDistance). The column is
        # vector(N) so pgvector accepts the index (unbounded columns are rejected).
        indexes = [
            HnswIndex(
                name="product_chunk_embedding_idx",
                fields=["embedding"],
                opclasses=["vector_cosine_ops"],
            ),
        ]
