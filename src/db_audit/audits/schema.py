from typing import Annotated, Literal

from pydantic import BaseModel, Field

Judgment = Literal[
    "nicht_erfuellt",
    "teilweise_erfuellt",
    "vollstaendig_erfuellt",
    "nicht_beurteilbar",
]

class SubRequirementAudit(BaseModel):
    requirement: Annotated[
        str,
        Field(description="Atomare, prüfbare Teilanforderung (Kurztext)."),
    ]
    judgment: Annotated[
        Judgment,
        Field(description="Urteil zur Teilanforderung."),
    ]
    statement: Annotated[
        str,
        Field(description="Kurze Begründung. Bei nicht_beurteilbar MUSS enthalten: 'Keine Evidenz im Kontext gefunden.'"),
    ]
    evidence_chunk_ids: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "IDs der Chunks, die das Urteil direkt belegen. "
                "Nur direkt relevante Chunk-IDs. "
                "Bei nicht_beurteilbar: []."
            ),
        ),
    ]

class AuditResult(BaseModel):
    sub_requirements: Annotated[
        list[SubRequirementAudit],
        Field(description="Liste der Audits pro Teilanforderung."),
    ]
    overall_judgment: Annotated[
        Judgment,
        Field(description="Gesamturteil gemäß Aggregationslogik im Prompt."),
    ]
    overall_statement: Annotated[
        str,
        Field(description="Kurze Gesamtbegründung (Referenz auf Muster/Mehrheit/Wesentliches aus den Teilurteilen)."),
    ]
