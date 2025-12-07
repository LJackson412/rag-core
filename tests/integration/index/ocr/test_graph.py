from typing import Generator, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig

from rag_app.factory.factory import build_vstore
from rag_app.index.ocr.config import IndexConfig
from rag_app.index.ocr.graph import graph
from rag_app.index.ocr.state import InputIndexState


class IndexGraphData(TypedDict):
    index_config: RunnableConfig
    index_state: InputIndexState


@pytest.fixture()
def create_config_and_input() -> Generator[IndexGraphData, None, None]:
    index_config = RunnableConfig(
        configurable={
            "doc_id": "Test_M_1",
            "collection_id": "Test_M_1",
        }
    )
    index_state = InputIndexState(path="./data/Test_Maßnahme_1.pdf")


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
