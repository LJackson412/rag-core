from typing import Any

from langchain_core.documents import Document

from rag_app.index.ocr.schema import ExtractedData, Text


def map_to_docs(data: list[ExtractedData]) -> list[Document]:
    docs: list[Document] = []

    def add_chunk(
        chunk: Text,
        chunk_type: str,
        chunk_metadata: dict[str, Any]
    ) -> None:
        metadata: dict[str, Any] = {
            **chunk_metadata,
            "chunk_type": chunk_type,
            "language": chunk.language,
            "title": chunk.title,
            "labels": chunk.labels,
            "category": chunk.category
        }

        docs.append(
            Document(
                page_content=chunk.retrieval_summary,
                metadata=metadata,
            )
        )

    for doc_data in data:
        metadata = doc_data.metadata

        if doc_data.text:
            add_chunk(doc_data.text, "text", metadata)

    return docs
