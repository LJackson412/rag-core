from operator import add
from typing import Annotated, Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag_app.index.ocr.schema import ImageSegment, TableSegment, TextSegment
from rag_app.index.schema import LLMException


class InputIndexState(BaseModel):
    path: Annotated[
        str,
        Field(
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
                "Text segments extracted from the document, including bounding boxes, page "
                "numbers, and the LLM-provided text classification metadata."
            ),
        ),
    ]
    table_segments: Annotated[
        list[TableSegment],
        Field(
            default_factory=list,
            description=(
                "Structured table or list segments, each with HTML content and positional "
                "information for later chunking and storage."
            ),
        ),
    ]
    image_segments: Annotated[
        list[ImageSegment],
        Field(
            default_factory=list,
            description=(
                "Image-based segments (figures and graphics) with spatial metadata that can "
                "be re-rendered or summarized downstream."
            ),
        ),
    ]
    index_docs: Annotated[
        list[Document],
        Field(
            default_factory=list,
            description=(
                "LangChain Document objects persisted in Chroma during the save node. Includes chunk metadata such as "
                "doc_id, collection_id, and chunk_id that the retrieval graph relies on."
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
 
