from typing import Generator, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from rag_app.llm_index.config import IndexConfig
from rag_app.llm_index.graph import graph as index_graph
from rag_app.llm_index.state import InputIndexState

from rag_app.factory.factory import build_vstore
from rag_app.retrieval.graph import graph as retrieval_graph
from rag_app.retrieval.state import InputRetrievalState


class RagAppData(TypedDict):
    index_config: RunnableConfig
    retrieval_config: RunnableConfig
    index_state: InputIndexState
    retrieval_state: InputRetrievalState


@pytest.fixture()
def create_collection_db_zb_s55_default() -> Generator[RagAppData, None, None]:
    index_config = RunnableConfig(
        configurable={
            "doc_id": "DB_ZB_25_Test",
            "collection_id": "DB_ZB_Test",
        }
    )
    retrieval_config = RunnableConfig(
        configurable={
            "doc_id": "DB_ZB_25_Test",
            "collection_id": "DB_ZB_Test",
        }
    )
    index_state = InputIndexState(path="./data/DB_ZB25_S55.pdf")
    retrieval_state = InputRetrievalState(
        question="Wie viele Mitarbeiter hatte die DB InfraGo 2024?"
    )

    yield {
        "index_config": index_config,
        "retrieval_config": retrieval_config,
        "index_state": index_state,
        "retrieval_state": retrieval_state,
    }
    config = IndexConfig.from_runnable_config(index_config)
    vstore = build_vstore(config.embedding_model, config.collection_id)
    vstore.delete_collection()


@pytest.mark.asyncio
async def test_rag_app(create_collection_db_zb_s55_default: RagAppData) -> None:
    data = create_collection_db_zb_s55_default

    input_index_state = data["index_state"]
    index_config = data["index_config"]

    index_res = await index_graph.ainvoke(
        input=input_index_state,
        config=index_config,
    )

    index_docs = index_res["index_docs"]
    assert len(index_docs) > 0

    retrieval_state = data["retrieval_state"]
    retrieval_config = data["retrieval_config"]

    retrieval_res = await retrieval_graph.ainvoke(
        input=retrieval_state, config=retrieval_config
    )

    llm_answer = retrieval_res["llm_answer"]
    assert llm_answer is not None
