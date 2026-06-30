from typing import Optional, Type

from enthusiast_common.config import (
    AgentCallbackHandlerConfig,
    AgentConfig,
    AgentConfigWithDefaults,
    EmbeddingsRegistryConfig,
    LLMConfig,
    LLMRegistryConfig,
    ModelsRegistryConfig,
    RegistryConfig,
    RepositoriesConfig,
    RetrieverConfig,
    RetrieversConfig,
)
from pydantic import BaseModel

from agent.core.callbacks import ConversationWebSocketCallbackHandler
from agent.core.injector import Injector
from agent.core.registries.embeddings import EmbeddingProviderRegistry
from agent.core.registries.language_models import LanguageModelRegistry
from agent.core.registries.models import BaseDjangoSettingsDBModelRegistry
from agent.core.repositories import (
    DjangoAgentRepository,
    DjangoConversationRepository,
    DjangoDataSetRepository,
    DjangoDocumentChunkRepository,
    DjangoMessageRepository,
    DjangoProductChunkRepository,
    DjangoProductRepository,
    DjangoUserRepository,
)
from agent.core.retrievers import DocumentRetriever, ProductRetriever


#: Platform-default cosine *distance* similarity floor for the pure-vector retrieval path. A chunk
#: whose distance to the query exceeds this bound is dropped at the SQL level, so the ranklist never
#: fills ``max_objects`` / ``number_of_products`` with irrelevant chunks. Cosine distance for
#: normalised embeddings sits in ``[0, 2]``; ``0.6`` (cosine similarity ``≈ 0.4``) keeps topically
#: related hits while cutting the orthogonal / unrelated tail. It is the *starting* value the DRA
#: recommended observing from — tune per agent/dataset via ``config.retrievers.*.extra_kwargs`` and
#: set to ``None`` to restore the historical "always return top-K" behaviour.
DEFAULT_COSINE_DISTANCE_THRESHOLD = 0.6


class DefaultAgentConfig(BaseModel):
    repositories: RepositoriesConfig
    retrievers: RetrieversConfig
    injector: Type[Injector]
    registry: RegistryConfig
    llm: LLMConfig
    agent_callback_handler: Optional[AgentCallbackHandlerConfig] = None


def get_default_config() -> DefaultAgentConfig:
    return DefaultAgentConfig(
        repositories=RepositoriesConfig(
            user=DjangoUserRepository,
            data_set=DjangoDataSetRepository,
            conversation=DjangoConversationRepository,
            message=DjangoMessageRepository,
            product=DjangoProductRepository,
            document_chunk=DjangoDocumentChunkRepository,
            product_chunk=DjangoProductChunkRepository,
            agent=DjangoAgentRepository,
        ),
        retrievers=RetrieversConfig(
            document=RetrieverConfig(
                retriever_class=DocumentRetriever,
                extra_kwargs={"distance_threshold": DEFAULT_COSINE_DISTANCE_THRESHOLD},
            ),
            product=RetrieverConfig(
                retriever_class=ProductRetriever,
                extra_kwargs={
                    "max_sample_products": 12,
                    "number_of_products": 12,
                    "distance_threshold": DEFAULT_COSINE_DISTANCE_THRESHOLD,
                },
            ),
        ),
        injector=Injector,
        registry=RegistryConfig(
            llm=LLMRegistryConfig(registry_class=LanguageModelRegistry),
            embeddings=EmbeddingsRegistryConfig(registry_class=EmbeddingProviderRegistry),
            model=ModelsRegistryConfig(registry_class=BaseDjangoSettingsDBModelRegistry),
        ),
        llm=LLMConfig(),
        agent_callback_handler=AgentCallbackHandlerConfig(handler_class=ConversationWebSocketCallbackHandler),
    )


def merge_config(
    partial: AgentConfigWithDefaults,
) -> AgentConfig:
    merged: dict[str, object] = {}
    defaults = get_default_config()
    for name in AgentConfig.model_fields:
        value = getattr(partial, name, None)

        if value is not None:
            merged[name] = value
        else:
            merged[name] = getattr(defaults, name, None)

    return AgentConfig(**merged)
