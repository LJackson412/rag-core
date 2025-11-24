import asyncio

from langchain_chroma import Chroma
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_app.config.settings import settings

_DEFAULT_RATE_LIMITER = InMemoryRateLimiter()


def build_chat_model(
    model_name: str,
    temp: float = 0.0,
    max_retries: int = 5,
    rate_limiter: InMemoryRateLimiter | None = None,
) -> ChatOpenAI:
    if rate_limiter is None:
        rate_limiter = _DEFAULT_RATE_LIMITER

    return ChatOpenAI(
        api_key=settings.API_KEY,
        rate_limiter=rate_limiter,
        max_retries=max_retries,
        model=model_name,
        temperature=temp,
    )


def build_embeddings(model_name: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model_name, openai_api_key=settings.API_KEY)


async def abuild_vstore(
    embedding_model_name: str,
    collection_name: str = "chroma_collection",
    persist_directory: str = ".chroma_directory",
) -> Chroma:
    embedding_model = build_embeddings(embedding_model_name)
    return await asyncio.to_thread(
        Chroma,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )


def build_vstore(
    embedding_model_name: str,
    collection_name: str = "chroma_collection",
    persist_directory: str = ".chroma_directory",
) -> Chroma:
    embedding_model = build_embeddings(embedding_model_name)
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )
