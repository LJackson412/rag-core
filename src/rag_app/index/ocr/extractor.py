import logging
from typing import cast

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import HumanMessage

from rag_app.index.ocr.schema import ExtractedData, Text

logger = logging.getLogger(__name__)

# Todo: Mapping auf ExtractedData in "schema.py"?


async def extract_from_docs(
    docs: list[Document],
    llm: BaseChatModel,
    extract_data_prompt: str,
    extraction_data: type[Text],
) -> list[ExtractedData]:
    """Extrahiere strukturierte Daten aus einer Liste von LangChain-Documents."""

    if not docs:
        return []

    structured_llm = llm.with_structured_output(extraction_data)

    inputs: list[LanguageModelInput] = []
    for doc in docs:
        prompt = extract_data_prompt.format(page_content=doc.page_content)
        messages = [HumanMessage(content=prompt)]
        inputs.append(messages)

    texts = cast(list[Text], await structured_llm.abatch(inputs))

    extracted_data = []
    for text, doc in zip(texts, docs):
        metadata = dict(doc.metadata) if doc.metadata is not None else {}
        metadata["extracted_content"] = doc.page_content
        metadata["page"] = metadata["page"] + 1

        extracted_data.append(
            ExtractedData(text=text, figure=None, table=None, metadata=metadata)
        )

    return extracted_data
