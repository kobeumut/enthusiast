from enthusiast_common.registry.embeddings import EmbeddingProvider
from enthusiast_common.utils import prioritize_items
from openai import OpenAI

PRIORITIZED_MODELS = ["text-embedding-3-large", "text-embedding-3-small"]

#: Maximum number of inputs sent in a single OpenAI embeddings request. The
#: OpenAI embeddings endpoint accepts up to 2048 inputs per call, so batching up
#: to this limit collapses N chunks into ~1 request, which is the main
#: latency/cost win when indexing large catalogs.
EMBEDDINGS_BATCH_SIZE = 2048


class OpenAIEmbeddingProvider(EmbeddingProvider):
    NAME = "OpenAI"

    def generate_embeddings(self, content: str) -> list[float]:
        """
        Generates and returns an embedding vector for the given content using OpenAI's embeddings API.

        Args:
            content (str): The input text for which the embedding vector is to be generated.
        """
        openai_embedding = OpenAI().embeddings.create(model=self._model, dimensions=self._dimensions, input=content)

        return openai_embedding.data[0].embedding

    def generate_embeddings_batch(self, contents: list[str]) -> list[list[float]]:
        """Generates embedding vectors for a batch of contents using a single OpenAI client.

        The OpenAI embeddings API accepts a list of inputs in one request (up to
        ``EMBEDDINGS_BATCH_SIZE`` per call) and returns one vector per input. We reuse
        a single ``OpenAI`` HTTP client across all requests, split ``contents`` into
        ``EMBEDDINGS_BATCH_SIZE``-sized requests, and realign the API output by each
        item's ``index`` field (which mirrors the input order) before returning.

        Args:
            contents (list[str]): The input texts to embed, in order.

        Returns:
            list[list[float]]: Embedding vectors aligned with ``contents`` order; the
            element at position ``i`` is the embedding of ``contents[i]``.

        Returns an empty list immediately when there is nothing to embed, so no
        ``OpenAI`` client is constructed (and thus validated/configured) for empty
        input.
        """
        if not contents:
            return []

        embeddings: list[list[float]] = [[] for _ in range(len(contents))]
        client = OpenAI()
        for start in range(0, len(contents), EMBEDDINGS_BATCH_SIZE):
            batch = contents[start : start + EMBEDDINGS_BATCH_SIZE]
            response = client.embeddings.create(
                model=self._model,
                dimensions=self._dimensions,
                input=batch,
            )
            for item in response.data:
                embeddings[start + item.index] = item.embedding
        return embeddings

    @staticmethod
    def available_models() -> list[str]:
        all_models = OpenAI().models.list().data
        embedding_models = [model.id for model in all_models if model.id.startswith("text-embedding")]
        return prioritize_items(embedding_models, PRIORITIZED_MODELS)
