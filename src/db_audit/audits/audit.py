"""
- Verwendet rag/index.py und rag/retrieval.py für audit use-case 
- Input sind "reqs" zu auditierende Anforderungen und spezifische Audit-Prompts

"""

import asyncio
from typing import Literal

from db_audit.audits.prompts import AUDIT_PROMPT
from db_audit.audits.schema import AuditResult
from db_audit.rag.index import index_files
from db_audit.rag.retrieval import retrieve_docs


async def audit(
    workitem_element_id: str,
    paths: list[str], 
    doc_ids: list[str],
    reqs: list[str],
    index_mode: Literal["none", "all"] = "none",
    skip_indexing: bool = False,
) -> None:
    
    if not skip_indexing:
        await index_files(
            collection_id=workitem_element_id,
            paths=paths,
            doc_ids=doc_ids,
            mode=index_mode
        )
    
    await retrieve_docs(
        collection_id=workitem_element_id,
        queries=reqs,
        number_of_docs_to_retrieve = 38, # 4 x 38,
        generate_answer_prompt=AUDIT_PROMPT,
        generate_answer_schema=AuditResult
    )
    
async def main() -> None:
    WORKITEM_ELEMENT_ID = "Test"
    PATHS = [
        "./data/Test/Test.pdf",
        "./data/Test/Test.docx",
        "./data/Test/Test.csv",
        "./data/Test/Test.png",
        "./data/Test/Test.xlsx"
        
    ]
    DOC_IDS = [
        "./data/Test/Test.pdf",
        "./data/Test/Test.docx",
        "./data/Test/Test.csv",
        "./data/Test/Test.png",
        "./data/Test/Test.xlsx"
    ]
    REQUIREMENTS = [
        "Zugriffsberechtigungen von Benutzern unter Verantwortung des Unternehmens (interne und externe Mitarbeiter) werden mindestens jährlich überprüft, um diese zeitnah auf Änderungen im Beschäftigungsverhältnis (Kündigung, Versetzung, längerer Abwesenheit/Sabbatical/Elternzeit) anzupassen.",
        "Die Überprüfung erfolgt durch hierzu autorisierte Personen aus den Unternehmensbereichen des Unternehmens, die aufgrund ihres Wissens über die Zuständigkeiten die Angemessenheit der vergebenen Berechtigungen überprüfen können.",
        "Die Überprüfung sowie die sich daraus ergebenden Berechtigungsanpassungen werden nachvollziehbar dokumentiert.",
        "Administrative Berechtigungen werden regelmäßig (mind. jährlich) überprüft.",
    ]

    await audit(
        WORKITEM_ELEMENT_ID,
        PATHS,
        DOC_IDS,
        REQUIREMENTS,
        skip_indexing=True
    )


if __name__ == "__main__":
    asyncio.run(main())