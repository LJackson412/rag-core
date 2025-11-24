from __future__ import annotations

from typing import Annotated, Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class BaseAttributes(BaseModel):
    language: Annotated[
        str,
        Field(
            description="Detected document segment language; all outputs must use this language.",
        ),
    ]
    title: Annotated[
        str,
        Field(
            description=(
                "Original title of this document section, if present. "
                "If the section has no explicit title, create a short, precise, and informative title "
                "that clearly describes the content of this section."
            ),
        ),
    ]
    retrieval_summary: Annotated[
        str,
        Field(
            description=(
                "Write a short, retrieval-oriented summary of this document section in the same language as the original text."
                "Capture the main topic, important entities and concepts, their relationships, and any crucial numbers, dates, or identifiers. "
                "Use key domain terms and phrases that users might search for. 1-3 sentences, no bullet points."
            ),
        ),
    ]
    labels: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "List of short tags to enrich this section with metadata, "
            ),
        ),
    ]


class Text(BaseAttributes):
    extracted_content: Annotated[
        str,
        Field(
            description=(
                "Complete, verbatim, and unmodified text content of this section. "
                "No summarization, no interpretation, and no omitted sentences. "
                "Preserve the original wording, punctuation, and line breaks where they matter "
            ),
        ),
    ]
    category: Annotated[
        str,
        Field(
            description=(
                "Short textual category describing the text type, e.g. "
                "'Heading', 'Subheading', 'Body text', 'Bullet list', "
                "'Footnote', 'Quote', 'Caption'."
            ),
        ),
    ]


class Figure(BaseAttributes):
    extracted_content: Annotated[
        str,
        Field(
            description=(
                "Describe the content of the figure and extract any visible text. "
                "Answer format:\n\n"
                "Figure description:\n"
                "<Put the description of the figure here>\n\n"
                "Extracted text:\n"
                "<Put the original text from the figure here (e.g. axis labels, legends, labels)>"
                "\n\nImportant: Never use this element for tables and lists. Even if a table appears as an image,"
                " extract it as structured HTML and return it in the `tables` list instead."
            ),
        ),
    ]

    category: Annotated[
        str,
        Field(
            description=(
                "Short label for the visual type of the figure, e.g. "
                "'Bar chart', 'Line chart', 'Pie chart', 'Photo', 'Logo', "
                "'Screenshot', 'Technical drawing'. "
                "Do not use a whole PDF page as a category, and never classify tables here."
            ),
        ),
    ]


class TableOrList(BaseAttributes):
    extracted_content: Annotated[
        str,
        Field(
            description=(
                "The extracted structure as a complete HTML fragment.\n\n"
                "- For tables: use <table>, <thead>, <tbody>, <tr>, <th>, <td>.\n"
                "- For lists: use <ul> or <ol> with <li> (nested if required).\n\n"
                "The HTML structure should reflect the original layout as closely as possible "
                "(rows, columns, header rows, list hierarchy). "
                "Cell values and list items must be taken verbatim from the document. "
                "Include every tabular structure here, even if the page labels it as a figure or "
                "the table appears as an embedded image."
            ),
        ),
    ]

    category: Annotated[
        str,
        Field(
            description=(
                "Short label for the structural type, e.g. "
                "'Table', 'Financial table', 'Matrix', 'Data overview', "
                "'Bullet list', 'Numbered list', 'Checklist'."
            ),
        ),
    ]


class CodeOrFormula(BaseAttributes):
    extracted_content: Annotated[
        str,
        Field(
            description=(
                "Verbatim, unmodified content of the code or formula block.\n\n"
                "- For source code: return it as a Markdown code block with the appropriate language, e.g.:\n"
                "  ```python\n"
                "  <original code>\n"
                "  ```\n"
                "- For mathematical formulas: preferably use LaTeX, e.g.:\n"
                "  $$<LaTeX formula>$$\n"
                "- For other notations (e.g. chemical formulas): keep the original notation "
                "without any changes."
            ),
        ),
    ]
    category: Annotated[
        str,
        Field(
            description=(
                "Short category of the block, e.g. "
                "'Code example', 'Configuration snippet', 'Algorithm', "
                "'Mathematical formula', 'Statistical formula', 'Chemical formula'."
            ),
        ),
    ]


class Other(BaseAttributes):
    extracted_content: Annotated[
        str,
        Field(
            ...,
            description=(
                "Original content of this element as text. "
                "Use this for elements that clearly belong together but do not fit any other category."
            ),
        ),
    ]
    category: Annotated[
        str,
        Field(
            ...,
            description=("Short category for this element, e.g. "),
        ),
    ]


class LLMExtractedData(BaseModel):
    texts: Annotated[
        list[Text],
        Field(
            default_factory=list,
            description=(
                "List of semantically coherent document sections "
                "(e.g. heading plus associated paragraph, individual paragraphs, captions, lists). "
                "Split the content so that each entry is a meaningful RAG chunk "
                "for question-answering systems."
            ),
        ),
    ]
    figures: Annotated[
        list[Figure],
        Field(
            default_factory=list,
            description=("List of detected figures and graphical elements"),
        ),
    ]
    tables: Annotated[
        list[TableOrList],
        Field(
            default_factory=list,
            description=(
                "List of detected tables and lists with a structured HTML representation."
            ),
        ),
    ]
    code_or_formulas: Annotated[
        list[CodeOrFormula],
        Field(
            default_factory=list,
            description=(
                "List of all detected code blocks and formulas "
                "(programming code, pseudocode, mathematical and other formulas)."
            ),
        ),
    ]
    others: Annotated[
        list[Other],
        Field(
            default_factory=list,
            description=(
                "List of all content that cannot be clearly assigned to text, table/list, "
                "figure, or code/formula, but is still relevant for understanding the document."
            ),
        ),
    ]
class ExtractedData(LLMExtractedData):
    metadata: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Additional metadata such as source, page number or PDF information.",
        ),
    ]


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
            "title": chunk.title,  # (you had "titel" here)
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
