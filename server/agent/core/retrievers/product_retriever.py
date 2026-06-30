from typing import Optional, Self

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

from agent.core.retrievers.filters import RetrievalFilters
from agent.core.retrievers.hybrid import reciprocal_rank_fusion
from agent.core.retrievers.reranking import BaseReranker, LexicalReranker
from agent.core.retrievers.retriever_sql_execution_error import RetrieverSQLExecutionError
from agent.core.retrievers.sql_validator import SQLValidator
from catalog.models import Product

#: Default candidate pool pulled from the index before rerank/hybrid fusion when those stages are
#: enabled but no explicit pool size was configured. Large enough to give the reranker room to
#: promote a good chunk that the vector ranker placed just outside the final top-K.
DEFAULT_CANDIDATE_POOL = 50


class ProductRetriever(BaseProductRetriever):
    """Retrieves products for a dataset.

    Natural-language search (``find_products_matching_query``) is backed by pgvector: the query is
    embedded with the dataset's configured embedding provider/model and ranked against
    ``ProductContentChunk.embedding`` by cosine distance. On top of that base path, the RAG quality
    phase adds four optional, config-driven stages (all disabled by default so the historical
    behaviour is preserved exactly until a dataset opts in):

    * **Metadata filtering** – ``filters`` pushes category/price predicates into the queryset before
      the vector ranking, so filter-violating chunks never enter the ranking.
    * **Hybrid retrieval** – ``hybrid_enabled`` fuses the vector ranklist with a PostgreSQL
      full-text keyword ranklist via Reciprocal Rank Fusion, recovering exact-term hits (SKUs,
      model numbers) the embedding space blurs.
    * **Reranking** – ``reranker_enabled`` (or an injected ``reranker``) applies a cheap lexical
      rerank over the candidate pool, promoting chunks that share the query's exact terms.
    * **HNSW tuning** – ``ef_search`` runs ``SET LOCAL hnsw.ef_search = N`` for the vector query,
      tuning recall at runtime without rebuilding the index.

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
        candidate_pool: Optional[int] = None,
        hybrid_enabled: bool = False,
        reranker_enabled: bool = False,
        reranker: Optional[BaseReranker] = None,
        ef_search: Optional[int] = None,
    ):
        self.data_set_id = data_set_id
        self.data_set_repo = data_set_repo
        self.product_repo = product_repo
        self.product_chunk_repo = product_chunk_repo
        self.embeddings_registry = embeddings_registry
        self.number_of_products = number_of_products
        self.max_sample_products = max_sample_products
        self.candidate_pool = candidate_pool
        self.hybrid_enabled = hybrid_enabled
        self.reranker_enabled = reranker_enabled or reranker is not None
        self.reranker = reranker or (LexicalReranker() if self.reranker_enabled else None)
        self.ef_search = ef_search
        self._sql_validator = SQLValidator(allowed_table_name="catalog_product", data_set_id=self.data_set_id)

    def find_products_matching_query(
        self, user_query: str, filters: Optional[RetrievalFilters] = None
    ) -> list[Product]:
        """Find products matching a natural-language query using pgvector cosine ranking.

        The query is embedded with the dataset's embedding provider/model and the closest
        ``ProductContentChunk`` rows are ranked by cosine distance. When the optional RAG-quality
        stages are enabled, a candidate pool is first expanded, fused with a keyword ranklist
        (hybrid), reranked (lexical), and finally collapsed to distinct products in rank order.

        Args:
            user_query: A natural-language product search query.
            filters: Optional pre-retrieval metadata predicates (category / price range).

        Returns:
            The best matching products for the query, limited to the current dataset.
        """
        embedding_vector = self._create_embedding_for_query(user_query)
        distance = CosineDistance("embedding", embedding_vector)
        candidate_chunks = self._candidate_chunks(user_query, distance, filters)
        return self._distinct_products_from_chunks(candidate_chunks, limit=self.number_of_products)

    def _candidate_chunks(self, user_query: str, distance, filters: Optional[RetrievalFilters]) -> list:
        """Build the ranked candidate chunk pool, applying hybrid fusion and lexical rerank."""
        pool = self._effective_pool()
        vector_chunks = self._vector_chunks(distance, filters, pool)

        if self.hybrid_enabled:
            keyword_chunks = self._keyword_chunks(user_query, distance, filters, pool)
            vector_chunks = self._fuse_hybrid(vector_chunks, keyword_chunks)

        if self.reranker is not None:
            vector_chunks = self.reranker.rerank(user_query, vector_chunks)
        return vector_chunks

    def _effective_pool(self) -> Optional[int]:
        """Candidate-pool size; defaults to a bounded pool only when fusion/rerank is enabled."""
        if self.candidate_pool is not None:
            return self.candidate_pool
        if self.hybrid_enabled or self.reranker is not None:
            return DEFAULT_CANDIDATE_POOL
        return None

    def _vector_chunks(self, distance, filters: Optional[RetrievalFilters], pool: Optional[int]) -> list:
        """The vector-ranked candidate chunks (nearest-first), bounded by ``pool`` when set."""
        chunks = self.product_chunk_repo.get_chunk_by_distance_for_data_set(
            self.data_set_id, distance, filters=filters, ef_search=self.ef_search
        )
        if pool is not None:
            return list(chunks[:pool])
        return list(chunks)

    def _keyword_chunks(
        self, user_query: str, distance, filters: Optional[RetrievalFilters], pool: Optional[int]
    ) -> list:
        """The keyword (full-text) ranked candidate chunks, bounded by ``pool`` when set."""
        chunks = self.product_chunk_repo.get_chunks_by_keyword_for_data_set(
            self.data_set_id, keyword=user_query, distance=distance, filters=filters
        )
        if pool is not None:
            return list(chunks[:pool])
        return list(chunks)

    @staticmethod
    def _fuse_hybrid(vector_chunks: list, keyword_chunks: list) -> list:
        """Reorder the union of chunks via Reciprocal Rank Fusion of the two ranklists."""
        if not keyword_chunks:
            return vector_chunks
        if not vector_chunks:
            return keyword_chunks
        fused = reciprocal_rank_fusion(
            [[chunk.id for chunk in vector_chunks], [chunk.id for chunk in keyword_chunks]]
        )
        chunks_by_id = {chunk.id: chunk for chunk in vector_chunks}
        chunks_by_id.update({chunk.id: chunk for chunk in keyword_chunks})
        return [chunks_by_id[chunk_id] for chunk_id, _ in fused if chunk_id in chunks_by_id]

    def _create_embedding_for_query(self, query: str) -> list[float]:
        """Embeds the query using the dataset's configured embedding provider/model."""
        data_set = self.data_set_repo.get_by_id(self.data_set_id)
        embedding_provider = self.embeddings_registry.provider_for_dataset(self.data_set_id)
        return embedding_provider(data_set.embedding_model, data_set.embedding_vector_dimensions).generate_embeddings(
            query
        )

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
