from typing import Annotated

from pydantic import BaseModel, Field


class LLMQuestions(BaseModel):
    questions: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "List of alternative query formulations for semantic search in a vector database. "
                "Each item must be a single, well-formed query that is semantically related to the "
                "original user question but phrased differently (e.g., using synonyms, different "
                "levels of specificity, or focusing on sub-aspects of the question). "
                "Return only the raw queries without numbering, explanations, or additional text."
            ),
        ),
    ]


class LLMDecision(BaseModel):
    is_relevant: Annotated[
        bool,
        Field(
            description=(
                "Set to true if the given document segment is relevant to the users question, i.e., "
                "it directly answers the question or provides useful supporting context. "
                "Set to false if the passage is not helpful for answering the question or is off-topic."
            ),
        ),
    ]


class LLMAnswer(BaseModel):
    answer: Annotated[
        str,
        Field(
            description=(
                "A natural-language answer to the user's question based only on the "
                "provided document segments. Write the answer in the same language "
                "as the question. If the necessary information is not present in the "
                "documents, explicitly state that the answer cannot be derived from "
                "the provided context and avoid guessing."
            )
        ),
    ]
    chunk_ids: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "A list of the IDs of the document chunks that directly support your "
                "answer. Include only chunks that are relevant as evidence. Use the "
                "chunk IDs exactly as they appear in the input. If no document "
                "segment is relevant, leave the answer blank"
            ),
        ),
    ]
