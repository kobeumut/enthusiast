from typing import Optional

from enthusiast_common.config import AgentConfig
from enthusiast_common.registry import BaseEmbeddingProviderRegistry
from enthusiast_common.retrievers import BaseVectorStoreRetriever
from enthusiast_common.structures import RepositoriesInstances
from langchain_core.language_models import BaseLanguageModel
from pgvector.django import CosineDistance

from agent.core.retrievers.diversity import DEFAULT_MMR_LAMBDA, maximal_marginal_relevance
from agent.core.retrievers.filters import RetrievalFilters
from agent.core.retrievers.hybrid import reciprocal_rank_fusion
from agent.core.retrievers.reranking import BaseReranker, LexicalReranker

#: Default candidate pool pulled from the index before fusion/rerank/MMR when those stages are
#: enabled but no explicit pool size was configured. Documents often split into many overlapping
#: chunks, so a slightly larger default pool gives MMR room to find diverse sections.
DEFAULT_CANDIDATE_POOL = 60


class DocumentRetriever(BaseVectorStoreRetriever):
    """Retrieves document chunks for a dataset via pgvector cosine ranking.

    On top of the base vector path, the RAG quality phase adds optional, config-driven stages (all
    disabled by default so the historical behaviour is preserved exactly):

    * **Metadata filtering** – ``filters`` pushes url/title predicates into the queryset before
      ranking.
    * **Hybrid retrieval** – ``hybrid_enabled`` fuses the vector ranklist with a full-text keyword
      ranklist via RRF, so exact-term matches (model numbers, error codes) survive.
    * **Reranking** – ``reranker_enabled`` applies a cheap lexical rerank over the candidate pool.
    * **MMR diversity** – ``mmr_enabled`` selects relevant-yet-diverse chunks so a single document's
      near-duplicate sections do not crowd out other sections the user needs.
    * **HNSW tuning** – ``ef_search`` runs ``SET LOCAL hnsw.ef_search = N`` for the vector query.
    """

    def __init__(
        self,
        data_set_id: int,
        data_set_repo,
        model_chunk_repo,
        embeddings_registry: BaseEmbeddingProviderRegistry,
        max_objects: int = 12,
        candidate_pool: Optional[int] = None,
        hybrid_enabled: bool = False,
        reranker_enabled: bool = False,
        reranker: Optional[BaseReranker] = None,
        mmr_enabled: bool = False,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        ef_search: Optional[int] = None,
    ):
        self.data_set_id = data_set_id
        self.data_set_repo = data_set_repo
        self.embeddings_registry = embeddings_registry
        self.max_objects = max_objects
        self.model_chunk_repo = model_chunk_repo
        self.candidate_pool = candidate_pool
        self.hybrid_enabled = hybrid_enabled
        self.reranker_enabled = reranker_enabled or reranker is not None
        self.reranker = reranker or (LexicalReranker() if self.reranker_enabled else None)
        self.mmr_enabled = mmr_enabled
        self.mmr_lambda = mmr_lambda
        self.ef_search = ef_search

    def find_content_matching_query(self, query: str, filters: Optional[RetrievalFilters] = None) -> list:
        """Find document chunks matching a natural-language query using pgvector cosine ranking.

        Args:
            query: A natural-language content search query.
            filters: Optional pre-retrieval metadata predicates (url / title scope).

        Returns:
            The best matching document chunks, limited to ``max_objects`` for the current dataset.
        """
        embedding_vector = self._create_embedding_for_query(query)
        distance = CosineDistance("embedding", embedding_vector)
        candidates = self._candidate_chunks(query, distance, filters)
        return self._finalize(candidates)

    def _candidate_chunks(self, query: str, distance, filters: Optional[RetrievalFilters]) -> list:
        """Build the ranked candidate pool, applying hybrid fusion and lexical rerank."""
        pool = self._effective_pool()
        vector_chunks = self._vector_chunks(distance, filters, pool)

        if self.hybrid_enabled:
            keyword_chunks = self._keyword_chunks(query, distance, filters, pool)
            vector_chunks = self._fuse_hybrid(vector_chunks, keyword_chunks)

        if self.reranker is not None:
            vector_chunks = self.reranker.rerank(query, vector_chunks)
        return vector_chunks

    def _effective_pool(self) -> Optional[int]:
        """Candidate-pool size; defaults to a bounded pool only when fusion/rerank/MMR is enabled."""
        if self.candidate_pool is not None:
            return self.candidate_pool
        if self.hybrid_enabled or self.reranker is not None or self.mmr_enabled:
            return DEFAULT_CANDIDATE_POOL
        return None

    def _vector_chunks(self, distance, filters: Optional[RetrievalFilters], pool: Optional[int]) -> list:
        chunks = self.model_chunk_repo.get_chunk_by_distance_for_data_set(
            self.data_set_id, distance, filters=filters, ef_search=self.ef_search
        )
        if pool is not None:
            return list(chunks[:pool])
        return list(chunks)

    def _keyword_chunks(self, query: str, distance, filters: Optional[RetrievalFilters], pool: Optional[int]) -> list:
        chunks = self.model_chunk_repo.get_chunks_by_keyword_for_data_set(
            self.data_set_id, keyword=query, distance=distance, filters=filters
        )
        if pool is not None:
            return list(chunks[:pool])
        return list(chunks)

    @staticmethod
    def _fuse_hybrid(vector_chunks: list, keyword_chunks: list) -> list:
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

    def _finalize(self, candidates: list) -> list:
        """Apply MMR diversity (or a plain top-K slice) to produce the final chunk list."""
        if self.mmr_enabled:
            return maximal_marginal_relevance(candidates, lambda_=self.mmr_lambda, limit=self.max_objects)
        return list(candidates[: self.max_objects])

    def _create_embedding_for_query(self, query: str) -> list[float]:
        data_set = self.data_set_repo.get_by_id(self.data_set_id)
        embedding_provider = self.embeddings_registry.provider_for_dataset(self.data_set_id)
        return embedding_provider(data_set.embedding_model, data_set.embedding_vector_dimensions).generate_embeddings(
            query
        )

    @classmethod
    def create(
        cls,
        config: AgentConfig,
        data_set_id: int,
        repositories: RepositoriesInstances,
        embeddings_registry: BaseEmbeddingProviderRegistry,
        llm: BaseLanguageModel,
    ) -> BaseVectorStoreRetriever:
        return cls(
            data_set_id=data_set_id,
            data_set_repo=repositories.data_set,
            model_chunk_repo=repositories.document_chunk,
            embeddings_registry=embeddings_registry,
            **config.retrievers.document.extra_kwargs,
        )
