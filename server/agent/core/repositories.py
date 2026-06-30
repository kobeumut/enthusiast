from typing import Any, Optional, Type, TypeVar

from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.db import connection, models, transaction
from django.db.models import QuerySet
from enthusiast_common.repositories import (
    BaseAgentRepository,
    BaseConversationRepository,
    BaseDataSetRepository,
    BaseMessageRepository,
    BaseModelChunkRepository,
    BaseProductRepository,
    BaseRepository,
    BaseUserRepository,
)
from enthusiast_common.structures import LLMFile
from pgvector.django import CosineDistance

from account.models import User
from agent.core.retrievers.filters import RetrievalFilters, document_filter_q, product_filter_q
from agent.models import Conversation, Message
from agent.models.agent import Agent
from agent.models.conversation import ConversationFile
from catalog.models import DataSet, DocumentChunk, Product, ProductContentChunk

T = TypeVar("T", bound=models.Model)


class BaseDjangoRepository(BaseRepository[T]):
    def __init__(self, model: Type[T]):
        super(BaseDjangoRepository, self).__init__(model)
        self.model = model

    def get_by_id(self, pk: int) -> Optional[T]:
        try:
            return self.model.objects.get(pk=pk)
        except self.model.DoesNotExist:
            return None

    def list(self) -> list[T]:
        return list(self.model.objects.all())

    def filter(self, **kwargs) -> QuerySet[T]:
        return self.model.objects.filter(**kwargs)

    def create(self, **kwargs) -> T:
        instance = self.model(**kwargs)
        instance.save()
        return instance

    def update(self, pk: int, **kwargs) -> Optional[T]:
        obj = self.get_by_id(pk)
        if obj:
            for key, value in kwargs.items():
                setattr(obj, key, value)
            obj.save()
        return obj

    def delete(self, pk: int) -> bool:
        deleted, _ = self.model.objects.filter(pk=pk).delete()
        return deleted > 0


class DjangoUserRepository(
    BaseDjangoRepository[User],
    BaseUserRepository,
):
    def get_user_dataset(self, user_id: int, data_set_id: int) -> DataSet:
        user = self.model.objects.get(pk=user_id)
        return user.datasets.get(pk=data_set_id)


class DjangoDocumentChunkRepository(BaseDjangoRepository[DocumentChunk], BaseModelChunkRepository[DocumentChunk]):
    def get_chunk_by_distance_for_data_set(
        self,
        data_set_id: int,
        distance: CosineDistance,
        filters: Optional[RetrievalFilters] = None,
        ef_search: Optional[int] = None,
    ) -> QuerySet[DocumentChunk] | list[DocumentChunk]:
        """Return this dataset's document chunks ranked by cosine distance to ``distance``.

        Args:
            data_set_id: Restrict chunks to documents in this dataset.
            distance: A ``CosineDistance`` annotation ordering the chunks nearest-first.
            filters: Optional pre-retrieval metadata predicates pushed into the queryset as ``Q``
                filters before the vector ranking (see ``agent.core.retrievers.filters``).
            ef_search: When set, runs ``SET LOCAL hnsw.ef_search = N`` inside a transaction before
                evaluating the query, tuning the HNSW candidate-list size at runtime. Because ``SET
                LOCAL`` is transaction-scoped and the queryset is lazy, the result is materialised to
                a list within that transaction. ``None`` keeps the server default and returns a lazy
                queryset (the historical behaviour).
        """
        queryset = (
            self.model.objects.annotate(distance=distance)
            .select_related("document")
            .filter(document__data_set_id__exact=data_set_id, embedding__isnull=False)
            .order_by("distance")
        )
        document_filter = document_filter_q(filters)
        if document_filter is not None:
            queryset = queryset.filter(document_filter)
        if ef_search is None:
            return queryset
        return _materialise_with_hnsw_ef_search(queryset, ef_search)

    def get_chunks_by_keyword_for_data_set(
        self,
        data_set_id: int,
        keyword: str,
        distance: CosineDistance,
        filters: Optional[RetrievalFilters] = None,
    ) -> QuerySet[DocumentChunk]:
        """Return this dataset's document chunks matching ``keyword`` via PostgreSQL full-text search.

        Chunks are annotated with a full-text ``rank`` (``SearchRank`` over a ``SearchVector`` of the
        chunk content) and ordered by rank descending, then by cosine distance as a deterministic
        tie-break. Only chunks whose content actually matches the query (``SearchVector @@
        SearchQuery``) are kept, so ``ts_rank``'s near-zero noise on non-matching rows does not leak
        into the ranklist. This is the keyword ranklist the document retriever fuses with the vector
        ranklist for hybrid (RRF) retrieval.
        """
        query = SearchQuery(keyword)
        vector = SearchVector("content")
        queryset = (
            self.model.objects.annotate(search=vector, rank=SearchRank(vector, query), distance=distance)
            .select_related("document")
            .filter(document__data_set_id__exact=data_set_id, embedding__isnull=False, search=query)
            .order_by("-rank", "distance")
        )
        document_filter = document_filter_q(filters)
        if document_filter is not None:
            queryset = queryset.filter(document_filter)
        return queryset


class DjangoProductChunkRepository(
    BaseDjangoRepository[ProductContentChunk], BaseModelChunkRepository[ProductContentChunk]
):
    def get_chunk_by_distance_for_data_set(
        self,
        data_set_id: int,
        distance: CosineDistance,
        filters: Optional[RetrievalFilters] = None,
        ef_search: Optional[int] = None,
    ) -> QuerySet[ProductContentChunk] | list[ProductContentChunk]:
        """Return this dataset's product chunks ranked by cosine distance to ``distance``.

        Args:
            data_set_id: Restrict chunks to products in this dataset.
            distance: A ``CosineDistance`` annotation ordering the chunks nearest-first.
            filters: Optional pre-retrieval metadata predicates (category / price range) pushed into
                the queryset as ``Q`` filters before the vector ranking.
            ef_search: When set, runs ``SET LOCAL hnsw.ef_search = N`` inside a transaction before
                evaluating the query (runtime HNSW tuning). See
                ``DjangoDocumentChunkRepository.get_chunk_by_distance_for_data_set``.
        """
        queryset = (
            self.model.objects.annotate(distance=distance)
            .select_related("product")
            .filter(product__data_set_id__exact=data_set_id, embedding__isnull=False)
            .order_by("distance")
        )
        product_filter = product_filter_q(filters)
        if product_filter is not None:
            queryset = queryset.filter(product_filter)
        if ef_search is None:
            return queryset
        return _materialise_with_hnsw_ef_search(queryset, ef_search)

    def get_chunks_by_keyword_for_data_set(
        self,
        data_set_id: int,
        keyword: str,
        distance: CosineDistance,
        filters: Optional[RetrievalFilters] = None,
    ) -> QuerySet[ProductContentChunk]:
        """Return this dataset's product chunks matching ``keyword`` via PostgreSQL full-text search.

        Chunks are annotated with a full-text ``rank`` and ordered by rank descending, then cosine
        distance. Only chunks whose content actually matches the query (``SearchVector @@
        SearchQuery``) are kept, so ``ts_rank``'s near-zero noise on non-matching rows does not leak
        into the ranklist. This is the keyword ranklist the product retriever fuses with the vector
        ranklist for hybrid (RRF) retrieval.
        """
        query = SearchQuery(keyword)
        vector = SearchVector("content")
        queryset = (
            self.model.objects.annotate(search=vector, rank=SearchRank(vector, query), distance=distance)
            .select_related("product")
            .filter(product__data_set_id__exact=data_set_id, embedding__isnull=False, search=query)
            .order_by("-rank", "distance")
        )
        product_filter = product_filter_q(filters)
        if product_filter is not None:
            queryset = queryset.filter(product_filter)
        return queryset


class DjangoProductRepository(BaseDjangoRepository[Product], BaseProductRepository[Product]):
    def extra(self, where_conditions: list[str]) -> QuerySet[Product]:
        return self.model.objects.extra(where=where_conditions)


class DjangoMessageRepository(BaseDjangoRepository[Message], BaseMessageRepository[Message]):
    pass


class DjangoConversationRepository(BaseDjangoRepository[Conversation], BaseConversationRepository[Conversation]):
    def get_data_set_id(self, conversation_id: int) -> int:
        return self.get_by_id(pk=conversation_id).data_set.id

    def get_agent_id(self, conversation_id: int) -> int:
        return self.get_by_id(pk=conversation_id).agent.id

    def list_files(self, conversation_id: int) -> list[LLMFile]:
        return [file.get_llm_file_object() for file in self.get_by_id(pk=conversation_id).files.filter(is_hidden=False)]

    def get_file_objects(self, conversation_id: Any, file_ids: list[Any]) -> list[LLMFile]:
        return [
            file.get_llm_file_object()
            for file in ConversationFile.objects.filter(
                conversation_id=conversation_id, id__in=file_ids, is_hidden=False
            ).order_by("created_at")
        ]


class DjangoDataSetRepository(BaseDjangoRepository[DataSet], BaseDataSetRepository[DataSet]):
    pass


class DjangoAgentRepository(BaseDjangoRepository[Agent], BaseAgentRepository[Agent]):
    def get_agent_configuration_by_id(self, agent_id: int) -> Any:
        return self.get_by_id(agent_id).config


def _materialise_with_hnsw_ef_search(queryset: QuerySet, ef_search: int) -> list:
    """Evaluate ``queryset`` with ``hnsw.ef_search`` set to ``ef_search`` for this transaction.

    ``hnsw.ef_search`` (default 40) is the size of the HNSW dynamic candidate list at query time: a
    larger value trades recall for latency. It is a *runtime* GUC, so unlike the build-time ``m`` /
    ``ef_construction`` parameters it can be tuned per-query without rebuilding the index.

    ``SET LOCAL`` scopes the change to the current transaction. Django evaluates ORM querysets
    lazily, so the queryset must be materialised *inside* the ``atomic`` block – otherwise the SET
    LOCAL would be reset (on commit) before the SQL runs. The function therefore returns a list, not
    a queryset. ``SET LOCAL`` takes effect immediately on the same connection within the same
    transaction.
    """
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL hnsw.ef_search = %s", [int(ef_search)])
        return list(queryset)
