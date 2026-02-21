import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any, Dict, cast

from langchain_core.documents import Document
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from rag_core.retrieval.config import RetrievalConfig
from rag_core.retrieval.schema import LLMDecision, LLMQuestions
from rag_core.retrieval.state import (
    InputRetrievalState,
    OutputRetrievalState,
    OverallRetrievalState,
)
from rag_core.utils.utils import extract_provider_and_model

logger = logging.getLogger(__name__)

PERSIST_DIRECTORY = ".chroma"


def _log_ctx(
    state: OverallRetrievalState, retrieval_config: RetrievalConfig
) -> dict[str, Any]:
    return {
        "collection_id": state.collection_id,
        "doc_id": state.doc_id,
        "generate_questions_model": retrieval_config.generate_questions_model,
        "embedding_model": retrieval_config.embedding_model,
        "compress_docs_model": retrieval_config.compress_docs_model,
        "generate_answer_model": retrieval_config.generate_answer_model,
    }


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


async def generate_questions(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, list[str]]:
    retrieval_config = RetrievalConfig.from_runnable_config(config)
    ctx = _log_ctx(state, retrieval_config)

    provider_factory = retrieval_config.provider_factory
    generate_questions_model = retrieval_config.generate_questions_model
    number_of_llm_generated_questions = (
        retrieval_config.number_of_llm_generated_questions
    )
    generate_questions_prompt = retrieval_config.generate_questions_prompt

    generate_questions_provider, model_name = extract_provider_and_model(
        generate_questions_model
    )

    number = number_of_llm_generated_questions
    user_question = state.messages[-1].content

    structured_llm = provider_factory.build_chat_model(
        provider=generate_questions_provider,
        model_name=model_name,
        temp=0.3,  # for semantic relevant questions
    ).with_structured_output(LLMQuestions)

    text_prompt = generate_questions_prompt.format(
        question=user_question, number=number
    )
    llm_input = [HumanMessage(content=text_prompt)]

    t0 = time.perf_counter()
    logger.info("generate questions start", extra={**ctx, "n_requested": number})

    try:
        llm_output = cast(
            LLMQuestions,
            await structured_llm.ainvoke(
                llm_input,
            ),
        )
    except Exception as e:
        logger.warning(
            "Error during generating questions",
            extra={
                **ctx,
                "error": str(e),
            },
        )
        raise e

    logger.info(
        "generate questions done",
        extra={
            **ctx,
            "n_requested": number,
            "n_questions": len(llm_output.questions),
            "ms": _ms(t0),
        },
    )

    return {"llm_questions": llm_output.questions}


async def retrieve_docs(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, list[Document]]:
    retrieval_config = RetrievalConfig.from_runnable_config(config)
    ctx = _log_ctx(state, retrieval_config)

    provider_factory = retrieval_config.provider_factory
    embedding_name = retrieval_config.embedding_model
    number_of_docs_to_retrieve = retrieval_config.number_of_docs_to_retrieve
    include_original_question = retrieval_config.include_original_question
    vstore_provider = retrieval_config.vstore

    embedding_provider, model_name = extract_provider_and_model(embedding_name)
    embedding_model = provider_factory.build_embeddings(
        provider=embedding_provider,
        model_name=model_name,
    )

    vstore = await asyncio.to_thread(
        provider_factory.build_vstore,
        embedding_model,
        provider=vstore_provider,
        collection_name=state.collection_id,
        persist_directory=PERSIST_DIRECTORY,
    )

    search_kwargs: Dict[str, Any] = {"k": number_of_docs_to_retrieve}

    if state.doc_id is not None:
        search_kwargs["filter"] = {"doc_id": state.doc_id}

    retriever = vstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    user_question = cast(str, state.messages[-1].content)
    if include_original_question:
        queries = [user_question] + state.llm_questions
    else:
        queries = state.llm_questions

    t0 = time.perf_counter()
    logger.info("retrieve docs start", extra={**ctx, "n_queries": len(queries)})

    docs_per_query = await retriever.abatch(queries)
    all_docs = [doc for docs in docs_per_query for doc in docs]

    def _unique_documents(documents: Sequence[Document]) -> list[Document]:
        seen_contents = set()
        unique_docs = []

        for doc in documents:
            content = doc.metadata.get("id")
            if content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)

        return unique_docs

    unique_docs = _unique_documents(all_docs)

    logger.info(
        "retrieve docs done",
        extra={
            **ctx,
            "n_queries": len(queries),
            "n_docs": len(all_docs),
            "n_unique_docs": len(unique_docs),
            "ms": _ms(t0),
        },
    )

    return {"retrieved_docs": unique_docs}


async def compress_docs(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, list[Document]]:
    retrieval_config = RetrievalConfig.from_runnable_config(config)
    ctx = _log_ctx(state, retrieval_config)

    provider_factory = retrieval_config.provider_factory
    compress_docs_model = retrieval_config.compress_docs_model

    compress_docs_provider, model_name = extract_provider_and_model(compress_docs_model)

    compress_docs_prompt: str = retrieval_config.compress_docs_prompt
    user_question = state.messages[-1].content

    base_llm = provider_factory.build_chat_model(
        provider=compress_docs_provider, model_name=model_name
    )
    structured_llm = base_llm.with_structured_output(LLMDecision)

    t0 = time.perf_counter()
    logger.info(
        "compress docs start",
        extra={**ctx, "n_retrieved_docs": len(state.retrieved_docs)},
    )
    
    llm_inputs: list[LanguageModelInput] = []
    input_counts: list[int] = []
    for doc in state.retrieved_docs:
        if doc.metadata.get("category") == "Image":
            content = doc.metadata.get("img_url", "")
        elif doc.metadata.get("category") == "Table":
            content = doc.metadata.get("text_as_html") or doc.metadata.get("text", "")
        elif doc.metadata.get("category") == "Text":
            content = doc.metadata.get("text", "")

        if doc.metadata.get("category") == "Image":
            text_prompt = compress_docs_prompt.format(
                question=user_question,
                doc_content="Image",
            )

            llm_inputs.append(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": content},
                            },
                        ]
                    )
                ]
            )
            input_counts.append(len(text_prompt.split()))
        else:
            text_prompt = compress_docs_prompt.format(
                question=user_question,
                doc_content=content,
            )

            llm_inputs.append([HumanMessage(content=text_prompt)])
            input_counts.append(len(text_prompt.split()))


    logger.info(
        "compress docs start",
        extra={
            **ctx,
            "n_retrieved_docs": len(state.retrieved_docs),

            "ms": _ms(t0),
        },
    )


    filtered_docs = []
    for llm_input, doc, input_count in zip(
        llm_inputs, state.retrieved_docs, input_counts, strict=True
    ):
        try:
            logger.debug(
                "debug compress doc",
                extra={
                    **ctx,
                    "chunk_id": doc.metadata.get("id"),
                    "input_count": input_count,
                },
            )

            llm_res = cast(LLMDecision, await structured_llm.ainvoke(llm_input))

            if llm_res.is_relevant:
                filtered_docs.append(doc)

        except Exception as e:
            logger.error(
                "Error during compressing doc",
                extra={
                    **ctx,
                    "error": str(e),
                    "chunk_id": doc.metadata.get("id"),
                    "n_retrieved_docs": len(state.retrieved_docs),
                    "input_count": input_count,
                },
                exc_info=True,  # Stacktrace ins Log
            )
            
    logger.info(
        "compress docs done",
        extra={
            **ctx,
            "n_retrieved_docs": len(state.retrieved_docs),
            "n_filtered_docs": len(filtered_docs),
            "ms": _ms(t0),
        },
    )

    return {"filtered_docs": filtered_docs}


async def generate_answer(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, BaseModel | list[Document]]:

    retrieval_config = RetrievalConfig.from_runnable_config(config)
    ctx = _log_ctx(state, retrieval_config)

    provider_factory = retrieval_config.provider_factory
    generate_answer_model = retrieval_config.generate_answer_model

    generate_answer_provider, model_name = extract_provider_and_model(
        generate_answer_model
    )
    generate_answer_prompt = retrieval_config.generate_answer_prompt
    user_question = cast(str, state.messages[-1].content)

    answer_schema = retrieval_config.generate_answer_schema
    structured_llm = provider_factory.build_chat_model(
        provider=generate_answer_provider, model_name=model_name
    ).with_structured_output(answer_schema)

    def _doc_text_for_prompt(doc: Document) -> str:
        cat = doc.metadata.get("category")
        if cat == "Image":
            return "Image"
        if cat == "Table":
            return (
                doc.metadata.get("text_as_html") or doc.metadata.get("text", "") or ""
            )
        if cat == "Text":
            return doc.metadata.get("text", "") or ""
        return doc.page_content or doc.metadata.get("text", "") or ""

    def _prepare_docs_for_prompt(docs: list[Document]) -> str:
        if not docs:
            return "No Documents found"

        chunk_tpl = (
            "Document-Metadata:\n"
            "Chunk-ID: {chunk_id}\n"
            "Document Segment:\n"
            "{doc_content}\n"
            "------------------------------------------------------------\n"
        )

        parts: list[str] = []
        for doc in docs:
            parts.append(
                chunk_tpl.format(
                    chunk_id=doc.metadata.get("id", "N/A"),
                    doc_content=_doc_text_for_prompt(doc),
                )
            )
        return "\n".join(parts)

    def _build_user_message(prompt: str, docs: list[Document]) -> HumanMessage:
        image_urls = []
        for doc in docs:
            if doc.metadata.get("category") == "Image":
                url = doc.metadata.get("img_url")
                if url:
                    image_urls.append(url)

        # no images -> plain text prompt only
        if not image_urls:
            return HumanMessage(content=prompt)

        # images exist -> multimodal
        content_parts: list[str | dict[str, Any]] = [{"type": "text", "text": prompt}]
        for url in image_urls:
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

        return HumanMessage(content=content_parts)

    filtered_docs = state.filtered_docs

    t0 = time.perf_counter()
    logger.info(
        "generate answer start",
        extra={**ctx, "n_filtered_docs": len(filtered_docs)},
    )

    prompt = generate_answer_prompt.format(
        question=user_question,
        docs=_prepare_docs_for_prompt(filtered_docs),
    )

    llm_input = _build_user_message(prompt, filtered_docs)

    # TODO: reduce llm input to match model context size
    try:
        llm_answer = cast(
            BaseModel,
            await structured_llm.ainvoke(
                [llm_input],
            ),
        )
    except Exception as e:
        logger.warning(
            "Error during generating answer",
            extra={
                **ctx,
                "error": str(e),
            },
        )
        raise e

    chunk_ids = set(getattr(llm_answer, "chunk_ids", None) or [])
    llm_evidence_docs = [
        doc for doc in filtered_docs if doc.metadata.get("id") in chunk_ids
    ]

    logger.info(
        "generate answer done",
        extra={
            **ctx,
            "n_filtered_docs": len(filtered_docs),
            "n_evidence_docs": len(llm_evidence_docs),
            "ms": _ms(t0),
        },
    )

    return {"llm_answer": llm_answer, "llm_evidence_docs": llm_evidence_docs}


builder = StateGraph(
    state_schema=OverallRetrievalState,
    input_schema=InputRetrievalState,
    output_schema=OutputRetrievalState,
    context_schema=RetrievalConfig,
)

builder.add_node("generate_questions", generate_questions)
builder.add_node("retrieve", retrieve_docs)
builder.add_node("compress_docs", compress_docs)
builder.add_node("generate_answer", generate_answer)


builder.add_edge(START, "generate_questions")
builder.add_edge("generate_questions", "retrieve")
builder.add_edge("retrieve", "compress_docs")
builder.add_edge("compress_docs", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()
graph.name = "Retriever"
