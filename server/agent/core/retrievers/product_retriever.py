from typing import Self

import django
from django.core import serializers
from django.forms import model_to_dict
from enthusiast_common.builder import RepositoriesInstances
from enthusiast_common.config import AgentConfig
from enthusiast_common.registry import BaseEmbeddingProviderRegistry
from enthusiast_common.repositories import BaseDataSetRepository, BaseModelChunkRepository, BaseProductRepository
from enthusiast_common.retrievers import BaseProductRetriever
from langchain_core.language_models import BaseLanguageModel
from pgvector.django import CosineDistance

from agent.core.retrievers.retriever_sql_execution_error import RetrieverSQLExecutionError
from agent.core.retrievers.sql_validator import SQLValidator
from catalog.models import Product


class ProductRetriever(BaseProductRetriever):
    """Retrieves products for a dataset.

    Natural-language search (``find_products_matching_query``) is backed by pgvector: the query is
    embedded with the dataset's configured embedding provider/model and ranked against
    ``ProductContentChunk.embedding`` by cosine distance, the same path ``DocumentRetriever`` uses.
    The explicit raw-SQL path (``find_products_with_sql``) remains available for the dedicated SQL
    tool, but natural-language product search never depends on LLM-generated SQL.
    """

    def __init__(
        self,
        data_set_id: int,
        data_set_repo: BaseDataSetRepository,
        product_repo: BaseProductRepository,
        product_chunk_repo: BaseModelChunkRepository,
        embeddings_registry: BaseEmbeddingProviderRegistry,
        number_of_products: int = 12,
        max_sample_products: int = 12,
    ):
        self.data_set_id = data_set_id
        self.data_set_repo = data_set_repo
        self.product_repo = product_repo
        self.product_chunk_repo = product_chunk_repo
        self.embeddings_registry = embeddings_registry
        self.number_of_products = number_of_products
        self.max_sample_products = max_sample_products
        self._sql_validator = SQLValidator(allowed_table_name="catalog_product", data_set_id=self.data_set_id)

    def find_products_matching_query(self, user_query: str) -> list[Product]:
        """Find products matching a natural-language query using pgvector cosine ranking.

        The query is embedded with the dataset's embedding provider/model and the closest
        ``ProductContentChunk`` rows are ranked by cosine distance. Distinct products are returned,
        in distance order, up to ``number_of_products``.

        Args:
            user_query: A natural-language product search query.

        Returns:
            The best matching products for the query, limited to the current dataset.
        """
        embedding_vector = self._create_embedding_for_query(user_query)
        matching_chunks = self._find_chunks_matching_vector(embedding_vector)
        return self._distinct_products_from_chunks(matching_chunks, limit=self.number_of_products)

    def _create_embedding_for_query(self, query: str) -> list[float]:
        """Embeds the query using the dataset's configured embedding provider/model."""
        data_set = self.data_set_repo.get_by_id(self.data_set_id)
        embedding_provider = self.embeddings_registry.provider_for_dataset(self.data_set_id)
        return embedding_provider(data_set.embedding_model, data_set.embedding_vector_dimensions).generate_embeddings(
            query
        )

    def _find_chunks_matching_vector(self, embedding_vector: list[float]):
        """Returns the dataset's product content chunks ranked by cosine distance to the vector."""
        embedding_distance = CosineDistance("embedding", embedding_vector)
        return self.product_chunk_repo.get_chunk_by_distance_for_data_set(self.data_set_id, embedding_distance)

    def _distinct_products_from_chunks(self, chunks, limit: int) -> list[Product]:
        """Collects up to ``limit`` distinct products from distance-ordered chunks."""
        products: list[Product] = []
        seen_product_ids: set[int] = set()
        for chunk in chunks:
            product = chunk.product
            if product.id in seen_product_ids:
                continue
            seen_product_ids.add(product.id)
            products.append(product)
            if len(products) >= limit:
                break
        return products

    def find_products_with_sql(self, sql_query: str) -> list[Product]:
        sanitized_query = self._sql_validator.add_data_set_id_condition_and_raise_if_not_allowed(sql_query)
        cleaned_query = sanitized_query.replace("%", "%%")
        try:
            return list(Product.objects.raw(cleaned_query))
        except django.db.utils.ProgrammingError as e:
            raise RetrieverSQLExecutionError(e)

    def get_sample_products(self, num_sample_products: int = 12) -> list[Product]:
        sample_products = self.product_repo.filter(data_set_id=self.data_set_id)[: num_sample_products]
        return list(sample_products)

    def get_sample_products_json(self) -> str:
        return serializers.serialize("json", self.get_sample_products())

    def product_details_as_json(self, products: list[Product]) -> list[dict]:
        return [model_to_dict(product) for product in products]

    @classmethod
    def create(
        cls,
        config: AgentConfig,
        data_set_id: int,
        repositories: RepositoriesInstances,
        embeddings_registry: BaseEmbeddingProviderRegistry,
        llm: BaseLanguageModel,
    ) -> Self:
        return cls(
            data_set_id=data_set_id,
            data_set_repo=repositories.data_set,
            product_repo=repositories.product,
            product_chunk_repo=repositories.product_chunk,
            embeddings_registry=embeddings_registry,
            **config.retrievers.product.extra_kwargs,
        )
