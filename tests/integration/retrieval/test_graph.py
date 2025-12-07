from typing import Generator, TypedDict

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from rag_app.retrieval.graph import graph
from rag_app.retrieval.state import InputRetrievalState


class RetrievalGraphData(TypedDict):
    retrieval_config: RunnableConfig
    retrieval_state: InputRetrievalState


@pytest.fixture()
def create_config_and_input() -> Generator[RetrievalGraphData, None, None]:
    retrieval_config = RunnableConfig(
        configurable={
            "doc_id": "Test_M_1",
            "collection_id": "Test_M_1",
        }
    )
    
    human_message = HumanMessage(
        content="Wie viele Mitarbeiter hatte die DB InfraGo 2024?"
    )
    retrieval_state = InputRetrievalState(
        messages=[human_message]
    )

    yield {
        "retrieval_config": retrieval_config,
        "retrieval_state": retrieval_state,
    }
 


@pytest.mark.asyncio
async def test_rag_app(create_config_and_input: RetrievalGraphData) -> None:
    data = create_config_and_input

    retrieval_config = data["retrieval_config"]
    retrieval_state = data["retrieval_state"]

    index_res = await graph.ainvoke(
        input=retrieval_state,
        config=retrieval_config,
    )

    index_docs = index_res["index_docs"]
    assert len(index_docs) > 0

