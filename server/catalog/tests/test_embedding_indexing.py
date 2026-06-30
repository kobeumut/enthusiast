from unittest.mock import MagicMock, patch

import pytest
from model_bakery import baker

from catalog.models import DataSet, Document, DocumentChunk, Product, ProductContentChunk
from catalog.models.data_set import EMBEDDING_VECTOR_DIMENSIONS
from catalog.services import DocumentEmbeddingGenerator, ProductEmbeddingGenerator

pytestmark = pytest.mark.django_db


class RecordingEmbeddingProvider:
    """Fake provider that records every call and returns a deterministic, content-dependent vector.

    The vector is a pure function of ``content`` so we can assert that each chunk
    received the embedding for its *own* content (i.e. the batch stayed aligned),
    without depending on chunk ordering.
    """

    NAME = "Recording"

    def __init__(self, model, dimensions):
        self._model = model
        self._dimensions = dimensions
        self.single_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def generate_embeddings(self, content: str) -> list[float]:
        self.single_calls.append(content)
        return self._vector_for(content)

    def generate_embeddings_batch(self, contents: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(contents))
        return [self._vector_for(content) for content in contents]

    @staticmethod
    def _vector_for(content: str) -> list[float]:
        value = (sum(ord(character) for character in content) % 97) / 97.0
        return [value] * EMBEDDING_VECTOR_DIMENSIONS


def patch_registry(provider: RecordingEmbeddingProvider):
    """Patches ``catalog.services.EmbeddingProviderRegistry`` so the data set resolves to ``provider``.

    Returns ``(registry_patch, mock_registry_cls, factory)``. ``factory`` is what
    ``provider_for_dataset`` returns; each time ``index_object`` "instantiates" the
    provider class it calls ``factory``, so ``factory.call_count`` is the number of
    provider instances (== HTTP clients) opened.
    """
    factory = MagicMock(side_effect=lambda model, dimensions: provider)
    registry_patch = patch("catalog.services.EmbeddingProviderRegistry")
    mock_registry_cls = registry_patch.start()
    mock_registry_cls.return_value.provider_for_dataset.return_value = factory
    return registry_patch, mock_registry_cls, factory


@pytest.fixture
def small_chunk_data_set():
    # A tiny chunk size guarantees several chunks per item, which makes the batch
    # path meaningful (1 item -> N chunks -> 1 batched call).
    return baker.make(
        DataSet,
        name="Indexing Test",
        embedding_chunk_size=10,
        embedding_chunk_overlap=0,
    )


class TestIndexObjectBatching:
    def test_product_chunks_are_embedded_in_a_single_batch_with_one_provider(self, small_chunk_data_set):
        provider = RecordingEmbeddingProvider(model="text-embedding-3-large", dimensions=EMBEDDING_VECTOR_DIMENSIONS)
        registry_patch, mock_registry_cls, factory = patch_registry(provider)
        try:
            long_description = " ".join(f"word{index}" for index in range(40))
            product = baker.make(
                Product,
                data_set=small_chunk_data_set,
                entry_id="product-batch",
                name="Batched Product",
                slug="batched-product",
                description=long_description,
                price=10,
            )

            ProductEmbeddingGenerator.index_object(product)
        finally:
            registry_patch.stop()

        chunks = list(ProductContentChunk.objects.filter(product=product))
        assert len(chunks) > 1, "expected multiple chunks so the batch call is exercised"

        # Provider/client is set up exactly once, OUTSIDE the chunk loop. The old
        # implementation did each of these once per chunk.
        assert mock_registry_cls.call_count == 1
        assert mock_registry_cls.return_value.provider_for_dataset.call_count == 1
        assert factory.call_count == 1
        # Exactly one batched request carrying every chunk's content.
        assert len(provider.batch_calls) == 1
        assert len(provider.batch_calls[0]) == len(chunks)
        assert set(provider.batch_calls[0]) == {chunk.content for chunk in chunks}
        # The single-string API must not be used on the indexing path.
        assert provider.single_calls == []

        # Each chunk received the embedding for its own content (batch stayed aligned).
        for chunk in chunks:
            assert list(chunk.embedding) == pytest.approx(
                RecordingEmbeddingProvider._vector_for(chunk.content), rel=1e-3
            )

    def test_document_chunks_are_embedded_in_a_single_batch(self, small_chunk_data_set):
        provider = RecordingEmbeddingProvider(model="text-embedding-3-large", dimensions=EMBEDDING_VECTOR_DIMENSIONS)
        registry_patch, _mock_registry_cls, factory = patch_registry(provider)
        try:
            long_content = " ".join(f"sentence{index}" for index in range(40))
            document = baker.make(
                Document,
                data_set=small_chunk_data_set,
                url="https://example.com/docs/batched",
                title="Batched Document",
                content=long_content,
            )

            DocumentEmbeddingGenerator.index_object(document)
        finally:
            registry_patch.stop()

        chunks = list(DocumentChunk.objects.filter(document=document))
        assert len(chunks) > 1
        assert factory.call_count == 1
        assert len(provider.batch_calls) == 1
        assert len(provider.batch_calls[0]) == len(chunks)
        assert provider.single_calls == []
        for chunk in chunks:
            assert list(chunk.embedding) == pytest.approx(
                RecordingEmbeddingProvider._vector_for(chunk.content), rel=1e-3
            )
