from typing import cast

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from rag_app.llm_index.schema import BaseAttributes
from rag_app.ocr_index.prompts import EXTRACT_DATA_FROM_DOCS_PROMPT

from rag_app.factory.factory import build_chat_model
from rag_app.index.llm.extractor import extract_from_docs


@pytest.fixture()
def create_docs_and_llm() -> dict[str, BaseChatModel | list[Document]]:

    llm = build_chat_model(model_name="gpt-4o")

    docs = [
        Document(page_content="Erste Seite"),
        Document(page_content="Zweite Seite"),
        Document(page_content="Dritte Seite"),
    ]

    return {"docs": docs, "llm": llm}


@pytest.mark.asyncio
async def test_extract_from_docs(
    create_docs_and_llm: dict[str, BaseChatModel | list[Document]],
) -> None:

    llm = cast(BaseChatModel, create_docs_and_llm["llm"])
    docs = cast(list[Document], create_docs_and_llm["docs"])

    base_attributes = await extract_from_docs(
        docs, llm, EXTRACT_DATA_FROM_DOCS_PROMPT, BaseAttributes
    )

    assert len(base_attributes) == len(docs)
