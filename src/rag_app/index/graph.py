from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from rag_app.factory.factory import abuild_vstore, build_chat_model
from rag_app.index.config import IndexConfig
from rag_app.index.extractor import extract_from_pdf
from rag_app.index.schema import ExtractedData, LLMExtractedData, map_to_docs
from rag_app.index.state import InputIndexState, OutputIndexState, OverallIndexState
from rag_app.utils.utils import make_chunk_id


async def extract(
    state: OverallIndexState, config: RunnableConfig
) -> dict[str, list[ExtractedData]]:
    index_config = IndexConfig.from_runnable_config(config)
    extract_model = index_config.extract_model
    extract_data_prompt = index_config.extract_data_prompt

    extracted_data: list[ExtractedData] = await extract_from_pdf(
        pdf_path=state.path,
        llm=build_chat_model(extract_model),
        extract_data_prompt=extract_data_prompt,
        extraction_data=LLMExtractedData,
    )

    return {"extracted_data": extracted_data}


async def save(
    state: OverallIndexState, config: RunnableConfig
) -> dict[str, list[Document]]:
    index_config = IndexConfig.from_runnable_config(config)
    doc_id = index_config.doc_id
    collection_id = index_config.collection_id
    embedding_model = index_config.embedding_model
    extract_model = index_config.extract_model

    vstore = await abuild_vstore(embedding_model, collection_id)

    docs = map_to_docs(state.extracted_data)
    filterd_docs = filter_complex_metadata(docs)

    for i, doc in enumerate(filterd_docs, start=0):
        chunk_id = make_chunk_id(
            collection_id=collection_id, doc_id=doc_id, chunk_index=i
        )
        doc.metadata = {
            **getattr(doc, "metadata", {}),
            "doc_id": doc_id,
            "collection_id": collection_id,
            "chunk_index": i,
            "chunk_id": chunk_id,
            "embedding_model": embedding_model,
            "extract_model": extract_model,
        }

    if filterd_docs:
        await vstore.aadd_documents(filterd_docs)

    return {"index_docs": filterd_docs}


builder = StateGraph(
    state_schema=OverallIndexState,
    input_schema=InputIndexState,
    output_schema=OutputIndexState,
    context_schema=IndexConfig,
)

builder.add_node("extract", extract)
builder.add_node("save", save)

builder.add_edge(START, "extract")
builder.add_edge("extract", "save")
builder.add_edge("save", END)

graph = builder.compile()
graph.name = "Indexer"
