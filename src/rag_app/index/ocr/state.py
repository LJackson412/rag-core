from typing import Annotated

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag_app.index.ocr.schema import Segment


class InputIndexState(BaseModel):
    path: Annotated[
        str,
        Field(
            description=("Path to the PDF that should be indexed"),
        ),
    ]


class OutputIndexState(BaseModel):
    texts: Annotated[
        list[Segment],
        Field(
            default_factory=list,
            description=(""),
        ),
    ]
    tables: Annotated[
        list[Segment],
        Field(
            default_factory=list,
            description=(""),
        ),
    ]
    imgs: Annotated[
        list[Segment],
        Field(
            default_factory=list,
            description=(""),
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


class OverallIndexState(InputIndexState, OutputIndexState):
    """Combined input/output schema used as the shared state across the graph."""
