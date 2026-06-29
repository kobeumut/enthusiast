from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from enthusiast_common.registry.embeddings import EmbeddingProvider
from enthusiast_model_openai.embedding import EMBEDDINGS_BATCH_SIZE, OpenAIEmbeddingProvider

# No ``django_db`` mark: these exercise pure provider/ABC logic and mock the OpenAI
# HTTP client, so they do not touch the database.


class TestEmbeddingProviderDefaultBatch:
    def test_default_batch_delegates_to_single_string_generate_embeddings(self):
        """Providers that only implement the single-string API get batching for free."""

        class SingleOnlyProvider(EmbeddingProvider):
            NAME = "SingleOnly"

            def generate_embeddings(self, content: str) -> list[float]:
                return [float(len(content))]

            @staticmethod
            def available_models() -> list[str]:
                return ["single-only"]

        provider = SingleOnlyProvider(model="single-only", dimensions=8)
        result = provider.generate_embeddings_batch(["a", "bb", "ccc"])

        assert result == [[1.0], [2.0], [3.0]]

    def test_single_string_generate_embeddings_is_unchanged(self):
        """The original single-string contract is preserved for existing call-sites (e.g. retrievers)."""

        class SingleOnlyProvider(EmbeddingProvider):
            NAME = "SingleOnly"

            def generate_embeddings(self, content: str) -> list[float]:
                return [42.0]

            @staticmethod
            def available_models() -> list[str]:
                return ["single-only"]

        provider = SingleOnlyProvider(model="single-only", dimensions=8)
        assert provider.generate_embeddings("hello") == [42.0]


class TestOpenAIEmbeddingProviderBatch:
    @patch("enthusiast_model_openai.embedding.OpenAI")
    def test_batch_uses_single_client_and_aligns_results_by_index(self, mock_openai_cls):
        # The API may return data out of order; alignment by ``index`` must fix that.
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=2, embedding=[0.2]),
                SimpleNamespace(index=0, embedding=[0.0]),
                SimpleNamespace(index=1, embedding=[0.1]),
            ]
        )
        mock_openai_cls.return_value = client

        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large", dimensions=512)
        result = provider.generate_embeddings_batch(["a", "b", "c"])

        # One HTTP client for the whole call, one request carrying the full input list.
        assert mock_openai_cls.call_count == 1
        assert client.embeddings.create.call_count == 1
        _, kwargs = client.embeddings.create.call_args
        assert kwargs["model"] == "text-embedding-3-large"
        assert kwargs["dimensions"] == 512
        assert kwargs["input"] == ["a", "b", "c"]
        # Reordered to match input order even though the response was shuffled.
        assert result == [[0.0], [0.1], [0.2]]

    @patch("enthusiast_model_openai.embedding.OpenAI")
    def test_batch_splits_inputs_larger_than_batch_size(self, mock_openai_cls):
        client = MagicMock()

        def fake_create(*, model, dimensions, input):  # `input` mirrors the OpenAI SDK kwarg name
            # Index is local to each response; the provider must offset by batch start.
            return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[i]) for i in range(len(input))])

        client.embeddings.create.side_effect = fake_create
        mock_openai_cls.return_value = client

        provider = OpenAIEmbeddingProvider(model="m", dimensions=8)
        contents = ["a", "b", "c", "d", "e"]
        with patch("enthusiast_model_openai.embedding.EMBEDDINGS_BATCH_SIZE", 2):
            result = provider.generate_embeddings_batch(contents)

        # 5 inputs with batch size 2 -> 3 requests, but still a single client.
        assert mock_openai_cls.call_count == 1
        assert client.embeddings.create.call_count == 3
        # No input sent to the API exceeds the configured batch size.
        for call in client.embeddings.create.call_args_list:
            assert len(call.kwargs["input"]) <= 2
        # Global reassembly across batches preserves input order.
        assert result == [[0], [1], [0], [1], [0]]

    @patch("enthusiast_model_openai.embedding.OpenAI")
    def test_batch_with_empty_input_makes_no_api_calls(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client

        provider = OpenAIEmbeddingProvider(model="m", dimensions=8)
        assert provider.generate_embeddings_batch([]) == []
        assert client.embeddings.create.call_count == 0

    def test_default_batch_size_matches_openai_request_limit(self):
        # OpenAI's embeddings endpoint accepts up to 2048 inputs per request.
        assert EMBEDDINGS_BATCH_SIZE == 2048
