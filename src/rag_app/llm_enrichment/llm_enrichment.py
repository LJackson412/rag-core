
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


async def gen_llm_metadata(
    chunks: list[str],
    llm: BaseChatModel,
    gen_prompt: str,
    gen_data: type[BaseModel]
) -> list[BaseModel]:
    """Extrahiere strukturierte Daten aus einer Liste von LangChain-Documents."""

    if not chunks:
        return []

    structured_llm = llm.with_structured_output(gen_data)

    inputs: list[LanguageModelInput] = []
    for chunk in chunks:
        prompt = gen_prompt.format(content=chunk)
        messages = [HumanMessage(content=prompt)]
        inputs.append(messages)

    res = cast(list[BaseModel], await structured_llm.abatch(inputs))
    return res
