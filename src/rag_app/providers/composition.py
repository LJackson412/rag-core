from langchain.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.vectorstores import VectorStore
from typing_extensions import Protocol, runtime_checkable

from rag_app.providers.chat_model import get_chat_model
from rag_app.providers.embedding import get_embedding
from rag_app.providers.vstore import get_vstore

_DEFAULT_RATE_LIMITER = InMemoryRateLimiter()


def build_chat_model(
    model_name: str = "gpt-4.1",
    provider: str = "openai",
    temp: float = 0.0,
    max_retries: int = 5,
    rate_limiter: InMemoryRateLimiter | None = None,
) -> BaseChatModel:
    if rate_limiter is None:
        rate_limiter = _DEFAULT_RATE_LIMITER

    return get_chat_model(
        provider=provider,
        model_name=model_name,
        temp=temp,
        max_retries=max_retries,
        rate_limiter=rate_limiter,
    )

@runtime_checkable
class ProviderFactory(Protocol):
    def build_chat_model(
        self,
        model_name: str = "gpt-4.1",
        provider: str = "openai",
        temp: float = 0.0,
        max_retries: int = 5,
        rate_limiter: InMemoryRateLimiter | None = None,
    ) -> BaseChatModel: ...


class DefaultProviderFactory(ProviderFactory):
    def build_chat_model(
        self,
        model_name: str = "gpt-4.1",
        provider: str = "openai",
        temp: float = 0.0,
        max_retries: int = 5,
        rate_limiter: InMemoryRateLimiter | None = None,
    ) -> BaseChatModel:
        if rate_limiter is None:
            rate_limiter = _DEFAULT_RATE_LIMITER

        return get_chat_model(
            provider=provider,
            model_name=model_name,
            temp=temp,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        
_DEFAULT_PROVIDER_FACTORY = DefaultProviderFactory()


def get_provider_factory(provider_factory: ProviderFactory | None = None) -> ProviderFactory:
    """Return a model factory, optionally supplied via RunnableConfig."""

    if provider_factory is not None:
        if isinstance(provider_factory, ProviderFactory):
            return provider_factory

    return _DEFAULT_PROVIDER_FACTORY

def build_embeddings(
    model_name: str = "text-embedding-3-large",
    provider: str = "openai",
) -> Embeddings:
    return get_embedding(
        provider=provider,
        model_name=model_name,
    )


def build_vstore(
    embedding_model_name: str = "text-embedding-3-large",
    collection_name: str = "chroma_collection",
    persist_directory: str = ".chroma_directory",
    embedding_provider: str = "openai",
    vstore_provider: str = "chroma"
) -> VectorStore:
    embedding_model = build_embeddings(
        model_name=embedding_model_name,
        provider=embedding_provider,
    )
    return get_vstore(
        embedding_model=embedding_model,
        provider=vstore_provider,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
