from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from db_audit.audits.schema import AuditResult, SubRequirementAudit

BADGE_COLORS = {
    "vollstaendig_erfuellt": ("#0E9F6E", "#ECFDF3"),
    "teilweise_erfuellt": ("#F59E0B", "#FFFBEB"),
    "nicht_erfuellt": ("#DC2626", "#FEF2F2"),
    "nicht_beurteilbar": ("#6B7280", "#F3F4F6"),
}


def _judgment_badge(judgment: str) -> str:
    color, background = BADGE_COLORS.get(judgment, ("#2563EB", "#EFF6FF"))
    return (
        f"<span class='badge' style='color:{color}; background:{background}; border-color:{color};'>"
        f"{escape(judgment.replace('_', ' '))}</span>"
    )


def _document_summary(doc: Document) -> str:
    meta = doc.metadata or {}

    # Immer anzeigen:
    title = "Evidenz"
    llm_title = meta.get("llm_title") or "N/A"
    filename = meta.get("filename") or "N/A"
    page_number = meta.get("page_number") or "N/A"
    category = meta.get("category") or "N/A"

    # Ausklappbarer Teil -> expanded_content:
    category_norm = str(category).strip().lower()

    expanded_content = ""
    if "table" in category_norm:
        table_html = meta.get("text_as_html") or ""
        if table_html.strip():
            # HTML-Tabelle NICHT escapen
            expanded_content = f"<div class='table-wrap'>{table_html}</div>"
        else:
            expanded_content = (
                "<pre class='text-block'>"
                f"{escape(doc.page_content or '')}"
                "</pre>"
            )
    elif "image" in category_norm:
        img_url = meta.get("img_url") or meta.get("image_url") or ""
        if img_url.strip():
            expanded_content = (
                "<figure class='image-wrap'>"
                f"<img class='evidence-image' src='{escape(img_url)}' alt='Evidenz Bild' loading='lazy' />"
                "</figure>"
            )
        else:
            expanded_content = "<div class='muted small'>Kein Bild-Link (img_url) vorhanden.</div>"
    else:
        expanded_content = (
            "<pre class='text-block'>"
            f"{escape(doc.page_content or '')}"
            "</pre>"
        )

    # Summary-Text: (Test.pdf, S. 1, Text)
    summary_meta = f"({filename}, S. {page_number}, {category})"

    return (
        "<div class='evidence-card'>"
        "<div class='evidence-top'>"
        f"<div class='evidence-title'>{escape(title)}</div>"
        # Nur die Titelbeschreibung (ohne Label "Titel") + Kürzung per CSS
        f"<div class='evidence-doc-title' title='{escape(str(llm_title))}'>{escape(str(llm_title))}</div>"
        "</div>"
        "<details class='evidence-details'>"
        f"<summary class='evidence-summary evidence-summary-meta'>{escape(summary_meta)}</summary>"
        f"<div class='evidence-expanded'>{expanded_content}</div>"
        "</details>"
        "</div>"
    )


def _render_sub_requirement(sub_requirement: SubRequirementAudit) -> str:
    evidence_blocks = "".join(_document_summary(doc) for doc in sub_requirement.evidence_docs)
    no_evidence_html = '<div class="no-evidence">Keine Evidenz dokumentiert.</div>'

    return (
        "<div class='sub-card'>"
        f"<div class='sub-header'>"
        f"<div class='sub-title'>{escape(sub_requirement.requirement)}</div>"
        f"{_judgment_badge(sub_requirement.judgment)}"
        "</div>"
        f"<p class='sub-statement'>{escape(sub_requirement.statement)}</p>"
        f"<div class='evidence-grid'>{evidence_blocks or no_evidence_html}</div>"
        "</div>"
    )


def _render_requirement_card(requirement: str, audit_result: AuditResult, index: int) -> str:
    sub_sections = "".join(_render_sub_requirement(sr) for sr in audit_result.sub_requirements)
    return (
        "<section class='card'>"
        f"<div class='card-header'>"
        f"<div><div class='eyebrow'>Anforderung {index}</div>"
        f"<h2>{escape(requirement)}</h2></div>"
        f"{_judgment_badge(audit_result.overall_judgment)}"
        "</div>"
        f"<p class='overall-statement'>{escape(audit_result.overall_statement)}</p>"
        f"<div class='sub-grid'>{sub_sections}</div>"
        "</section>"
    )


def generate_audit_report(
    workitem_id: str,
    requirements_with_results: Iterable[tuple[str, AuditResult]],
    base_dir: str | Path = "data/Audits",
) -> str:
    """Render audit results into a reusable HTML string.

    Args:
        workitem_id: Identifier of the audited work item used for folder and file names.
        requirements_with_results: Iterable of ``(requirement, AuditResult)`` tuples.
            The iterable should already be filtered to contain only successfully
            generated audit results and must be ordered as they should appear in
            the report (typically matching the original requirements input).
        base_dir: Root folder in which the report directory will be created.

    The report is stored under ``Audits/<workitem_element_id>/<workitem_element_id>.html``
    and the rendered HTML string is returned for downstream use.
    """

    requirement_cards: list[str] = []
    for idx, (requirement_text, audit_result) in enumerate(requirements_with_results, start=1):
        requirement_cards.append(_render_requirement_card(requirement_text, audit_result, idx))

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    workitem_id = workitem_id or "unbekannt"

    html = f"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Audit Report – {escape(workitem_id)}</title>
  <style>
    :root {{
      --bg: #0b1221;
      --panel: #0f172a;
      --card: #111827;
      --muted: #9ca3af;
      --text: #e5e7eb;
      --accent: #2563eb;
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      color: var(--text);
      background: radial-gradient(circle at 20% 20%, rgba(37,99,235,0.15), transparent 30%),
                  radial-gradient(circle at 80% 0%, rgba(16,185,129,0.12), transparent 25%),
                  var(--bg);
    }}
    .page {{
      max-width: 1100px;
      margin: 0 auto;
      background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(17,24,39,0.85));
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 28px;
      box-shadow: 0 30px 80px rgba(0,0,0,0.35);
      backdrop-filter: blur(8px);
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 28px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      padding-bottom: 16px;
    }}
    .title-block h1 {{
      margin: 0;
      font-size: 26px;
      letter-spacing: -0.01em;
    }}
    .muted {{ color: var(--muted); margin-top: 4px; }}
    .muted.small {{ font-size: 13px; margin-top: 0; }}
    .badge {{
      border: 1px solid;
      padding: 6px 10px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 12px;
      text-transform: capitalize;
    }}
    .card {{
      background: var(--card);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 14px;
      padding: 18px 20px;
      margin-bottom: 18px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.25);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }}
    .eyebrow {{
      text-transform: uppercase;
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.1em;
    }}
    h2 {{ margin: 4px 0 10px; font-size: 20px; }}
    .overall-statement {{ color: #cbd5e1; margin: 0 0 12px; line-height: 1.5; overflow-wrap: anywhere; }}
    .sub-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .sub-card {{
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px;
      padding: 14px;
      background: rgba(255,255,255,0.02);
      min-width: 0;
    }}
    .sub-header {{ display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; }}
    .sub-title {{ font-weight: 700; overflow-wrap: anywhere; }}
    .sub-statement {{ color: #d1d5db; margin: 8px 0 12px; line-height: 1.5; overflow-wrap: anywhere; }}
    .evidence-grid {{ display: grid; gap: 10px; grid-template-columns: 1fr; }}

    .evidence-card {{
      border: 1px dashed rgba(255,255,255,0.1);
      border-radius: 10px;
      padding: 10px 12px;
      background: rgba(37, 99, 235, 0.05);
      min-width: 0;
      overflow: hidden;
    }}
    .evidence-top {{
      display: grid;
      gap: 4px;
      margin-bottom: 8px;
      min-width: 0;
    }}
    .evidence-title {{ font-weight: 700; }}

    /* Titel: nur bis zu einer bestimmten Länge anzeigen (optisch gekürzt) */
    .evidence-doc-title {{
      color: #e5e7eb;
      font-size: 13px;
      line-height: 1.35;
      overflow: hidden;
      display: -webkit-box;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2; /* max. 2 Zeilen */
      overflow-wrap: anywhere;
      word-break: break-word;
      min-width: 0;
    }}

    details.evidence-details {{
      border-top: 1px solid rgba(255,255,255,0.08);
      padding-top: 8px;
    }}
    .evidence-summary {{
      cursor: pointer;
      user-select: none;
      color: #cbd5e1;
      font-weight: 600;
      font-size: 13px;
      outline: none;
      list-style: none;
    }}
    .evidence-summary::-webkit-details-marker {{ display: none; }}
    .evidence-summary::before {{
      content: "▸";
      display: inline-block;
      margin-right: 8px;
      transform: translateY(-1px);
      color: var(--muted);
    }}
    details[open] .evidence-summary::before {{
      content: "▾";
    }}

    .evidence-summary-meta {{
      color: var(--muted);
      font-weight: 600;
    }}

    .evidence-expanded {{
      margin-top: 10px;
      min-width: 0;
    }}

    .table-wrap {{
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      background: rgba(0,0,0,0.15);
      padding: 8px;
    }}
    .table-wrap table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 560px;
    }}
    .table-wrap th,
    .table-wrap td {{
      border: 1px solid rgba(255,255,255,0.08);
      padding: 8px;
      vertical-align: top;
      font-size: 13px;
      color: #e5e7eb;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .table-wrap th {{
      background: rgba(255,255,255,0.04);
      font-weight: 700;
    }}

    .image-wrap {{
      margin: 0;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      overflow: hidden;
      background: rgba(0,0,0,0.15);
    }}
    .evidence-image {{
      display: block;
      width: 100%;
      height: auto;
    }}

    .text-block {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      padding: 10px;
      max-height: 320px;
      overflow: auto;
      font-size: 13px;
      line-height: 1.45;
      color: #e5e7eb;
    }}

    .no-evidence {{ color: var(--muted); font-style: italic; }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <div class="title-block">
        <h1>Audit Report</h1>
        <div class="muted">Workitem: {escape(workitem_id)}</div>
        <div class="muted">Generiert am {escape(generated_at)}</div>
      </div>
      <div class="badge" style="border-color: var(--accent); color: var(--accent); background: rgba(37,99,235,0.1);">{len(requirement_cards)} Anforderungen</div>
    </header>
    {''.join(requirement_cards) or '<p class="muted">Keine Audit-Ergebnisse vorhanden.</p>'}
  </div>
</body>
</html>
    """

    output_dir = Path(base_dir) / workitem_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{workitem_id}.html"
    output_path.write_text(html, encoding="utf-8")

    return html



def example_html_report(base_dir: str | Path = "data/Audits") -> str:
    """Erzeugt einen Beispiel-HTML-Report mit Demo-Daten.

    - Demonstriert Text-, Tabellen- und Bild-Evidenz
    - Verknüpft chunk_ids -> Document über AuditResult.attach_documents()
    - Schreibt die HTML-Datei nach Audits/<workitem_id>/<workitem_id>.html
    """
    workitem_id = "demo_workitem_001"

    # --- Beispieldokumente (Evidenzen) -------------------------------------
    # Wichtig: attach_documents() matcht chunk_ids gegen metadata["id"] (als str).
    docs: list[Document] = [
        Document(
            page_content=(
                "Im Dokument wird beschrieben, dass ein Vier-Augen-Prinzip "
                "für produktive Änderungen angewendet wird."
            ),
            metadata={
                "id": "c1",
                "llm_title": "Vier-Augen-Prinzip – Prozessbeschreibung",
                "filename": "Test.pdf",
                "page_number": 1,
                "category": "text",
            },
        ),
        Document(
            page_content="Fallback-Text, falls keine HTML-Tabelle vorhanden ist.",
            metadata={
                "id": "c2",
                "llm_title": "Rollen- und Verantwortlichkeitsmatrix",
                "filename": "Test.pdf",
                "page_number": 2,
                "category": "table",
                # Diese Tabelle wird in _document_summary NICHT escaped gerendert:
                "text_as_html": """
<table>
  <thead>
    <tr><th>Rolle</th><th>Verantwortung</th><th>Freigabe</th></tr>
  </thead>
  <tbody>
    <tr><td>Entwicklung</td><td>Implementierung</td><td>Nein</td></tr>
    <tr><td>Review</td><td>Code-Review</td><td>Ja</td></tr>
    <tr><td>Security</td><td>Risikoanalyse</td><td>Ja</td></tr>
  </tbody>
</table>
""".strip(),
            },
        ),
        Document(
            page_content="(Kein Text nötig für Bild-Evidenz, aber erlaubt.)",
            metadata={
                "id": "c3",
                "llm_title": "Screenshot: Freigabe-Workflow im Tool",
                "filename": "Screenshot.png",
                "page_number": "N/A",
                "category": "image",
                # Beispiel-URL (wenn offline, lädt das Bild halt nicht – Report bleibt ok):
                "img_url": "https://via.placeholder.com/1000x520.png?text=Evidenzbild",
            },
        ),
        Document(
            page_content=(
                "Die Protokollierung erfolgt zentral. Aufbewahrungsfristen sind "
                "in der Policy dokumentiert, jedoch fehlt ein Hinweis auf Zugriffskontrollen."
            ),
            metadata={
                "id": "c4",
                "llm_title": "Logging- & Retention-Policy (Auszug)",
                "filename": "Policy.pdf",
                "page_number": 7,
                "category": "text",
            },
        ),
    ]

    # --- Anforderung 1 ------------------------------------------------------
    ar1 = AuditResult(
        sub_requirements=[
            SubRequirementAudit(
                requirement="Änderungen werden vor Deployment überprüft (Code-Review).",
                judgment="vollstaendig_erfuellt",
                statement="Ein formalisierter Review-Schritt ist beschrieben und wird angewendet.",
                chunk_ids=["c1", "c2"],
            ),
            SubRequirementAudit(
                requirement="Deployment erfordert eine Freigabe durch eine zweite Person.",
                judgment="teilweise_erfuellt",
                statement="Freigaben sind vorgesehen, aber nicht für alle Deployment-Typen eindeutig dokumentiert.",
                chunk_ids=["c2", "c3"],
            ),
        ],
        overall_judgment="teilweise_erfuellt",
        overall_statement=(
            "Die wesentlichen Kontrollmechanismen sind vorhanden. "
            "Es gibt jedoch Lücken in der konsistenten Dokumentation von Freigaben."
        ),
    )
    ar1.attach_documents(docs, id_key="id")

    # --- Anforderung 2 ------------------------------------------------------
    ar2 = AuditResult(
        sub_requirements=[
            SubRequirementAudit(
                requirement="Sicherheitsrelevante Ereignisse werden protokolliert.",
                judgment="vollstaendig_erfuellt",
                statement="Zentrale Protokollierung und Retention sind dokumentiert.",
                chunk_ids=["c4"],
            ),
            SubRequirementAudit(
                requirement="Zugriffe auf Logs sind rollenbasiert eingeschränkt.",
                judgment="nicht_erfuellt",
                statement="Es fehlt eine eindeutige Beschreibung von Zugriffskontrollen auf Logdaten.",
                chunk_ids=["c4"],
            ),
        ],
        overall_judgment="nicht_erfuellt",
        overall_statement=(
            "Logging ist zwar vorhanden, aber Zugriffskontrollen sind nicht ausreichend nachgewiesen."
        ),
    )
    ar2.attach_documents(docs, id_key="id")

    requirements_with_results: list[tuple[str, AuditResult]] = [
        ("Change Management & Freigaben", ar1),
        ("Logging & Monitoring", ar2),
    ]

    html = generate_audit_report(
        workitem_id=workitem_id,
        requirements_with_results=requirements_with_results,
        base_dir=base_dir,
    )
    return html
  
  
if __name__ == "__main__":
  
  example_html_report()