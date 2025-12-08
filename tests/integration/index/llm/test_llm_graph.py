from typing import Generator, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig

from rag_app.factory.factory import build_vstore
from rag_app.index.llm.config import IndexConfig
from rag_app.index.llm.graph import graph
from rag_app.index.llm.state import InputIndexState


class IndexGraphData(TypedDict):
    index_config: RunnableConfig
    index_state: InputIndexState


@pytest.fixture(
    params=[
        {
            "doc_id": "Test_Cancom_240514",
            "path": "./data/Cancom/240514_CANCOM_Zwischenmitteilung.pdf",
        },
        {
            "doc_id": "Test_Cancom_20241112",
            "path": "./data/Cancom/20241112_CANCOM_Zwischenmitteilung.pdf",
        },
    ],
    ids=["Test_Cancom_240514", "Test_Cancom_20241112"],
)
def create_config_and_input(
    request: pytest.FixtureRequest,
) -> Generator[IndexGraphData, None, None]:
    doc_data = request.param
    index_config = RunnableConfig(
        configurable={
            "doc_id": doc_data["doc_id"],
            "collection_id": "Test_Cancom_LLM",
        }
    )
    index_state = InputIndexState(path=doc_data["path"])


    yield {
        "index_config": index_config,
        "index_state": index_state,
    }
    config = IndexConfig.from_runnable_config(index_config)
    vstore = build_vstore(config.embedding_model, config.collection_id)
    vstore.delete_collection()



@pytest.mark.asyncio
async def test_index_graph(create_config_and_input: IndexGraphData) -> None:
    data = create_config_and_input

    input_index_state = data["index_state"]
    index_config = data["index_config"]

    index_res = await graph.ainvoke(
        input=input_index_state,
        config=index_config,
    )

    index_docs = index_res["index_docs"]
    assert len(index_docs) > 0
