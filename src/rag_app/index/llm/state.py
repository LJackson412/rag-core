from operator import add
from typing import Annotated, Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag_app.index.llm.schema import (
    CodeOrFormulaSegment,
    ImageSegment,
    OtherSegment,
    TableOrListSegment,
    TextSegment,
)
from rag_app.index.schema import LLMException


class InputIndexState(BaseModel):
    path: Annotated[
        str,
        Field(
            ...,
            description=(
                "Path to the PDF that should be indexed"
            ),
        ),
    ]


class OutputIndexState(BaseModel):
    text_segments: Annotated[
        list[TextSegment],
        Field(
            default_factory=list,
            description=(
                "Segments of text produced by the structured extraction model, including the "
                "LLM-assigned categories and positional data."
            ),
        ),
    ]
    image_segments: Annotated[
        list[ImageSegment],
        Field(
            default_factory=list,
            description=(
                "Image or figure regions extracted by the LLM with associated labels and "
                "layout coordinates."
            ),
        ),
    ]
    table_segments: Annotated[
        list[TableOrListSegment],
        Field(
            default_factory=list,
            description=(
                "LLM-generated representations of tables and lists, including HTML content "
                "and bounding boxes for downstream processing."
            ),
        ),
    ]
    code_or_formula_segments: Annotated[
        list[CodeOrFormulaSegment],
        Field(
            default_factory=list,
            description=(
                "Code blocks or mathematical expressions identified by the LLM with their "
                "locations in the source document."
            ),
        ),
    ]
    other_segments: Annotated[
        list[OtherSegment],
        Field(
            default_factory=list,
            description=(
                "Miscellaneous content segments (e.g., separators or unclassified snippets) "
                "captured to avoid data loss during indexing."
            ),
        ),
    ]
    index_docs: Annotated[
        list[Document],
        Field(
            default_factory=list,
            description=(
                "LangChain Document instances built from the extracted segments that are "
                "saved to the vector store for retrieval."
            ),
        ),
    ]
    llm_exceptions: Annotated[
        list[LLMException],
        add,
        Field(
            default_factory=list,
            description=(
                "Errors raised by the LLM structured extraction nodes. Clients can inspect these "
                "to selectively retry failed chunks without rerunning the entire indexing job."
            ),
        ),
    ]


class OverallIndexState(InputIndexState, OutputIndexState):
    """Combined input/output schema used as the shared state across the graph."""

    document_metadata: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description=(
                "Metadata about the indexed document (e.g., title, source path, page count, "
                "and ingestion timestamps) propagated through the graph."
            ),
        ),
    ]
    
