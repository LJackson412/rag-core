from typing import Any

from langchain_core.documents import Document

from rag_app.index.llm.schema import BaseAttributes, ExtractedData


def map_to_docs(data: list[ExtractedData]) -> list[Document]:
    docs: list[Document] = []

    def add_chunk(
        chunk: BaseAttributes,
        chunk_type: str,
        page_metadata: dict[str, Any],
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata: dict[str, Any] = {
            **page_metadata,
            "chunk_type": chunk_type,
            "language": chunk.language,
            "title": chunk.title,
            "extracted_content": getattr(chunk, "extracted_content", None),
            "labels": chunk.labels,
            "category": getattr(chunk, "category", None),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        docs.append(
            Document(
                page_content=chunk.retrieval_summary,
                metadata=metadata,
            )
        )

    for page_data in data:
        page_metadata = page_data.metadata

        for text_chunk in page_data.texts:
            add_chunk(text_chunk, "text", page_metadata)

        for fig in page_data.figures:
            add_chunk(fig, "figure", page_metadata)

        for table in page_data.tables:
            add_chunk(table, "table_or_list", page_metadata)

        for code_block in page_data.code_or_formulas:
            add_chunk(code_block, "code_or_formula", page_metadata)

        for other in page_data.others:
            add_chunk(other, "other", page_metadata)

    return docs
