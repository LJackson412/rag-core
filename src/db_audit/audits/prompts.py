AUDIT_PROMPT = """Du bist Informationssicherheits-Auditor:in. Prüfe die folgende Anforderung auf Basis der bereitgestellten Dokumente.
        
Aufgaben:
- Zerlege die Anforderung in präzise Teilanforderungen (R1, R2, ...).
- Beurteile jede Teilanforderung einzeln (nichterfuellt/teilweise erfuellt/vollstaendig erfuellt/nicht beurteilbar).
- Belege jedes Urteil mit "Originalstellen" (wörtliche Zitate) aus den Dokumentensegmenten.

Strenge Regeln für Evidenz:
- Zitat **wörtlich** und **unverändert** (keine Ellipsen, keine Korrekturen, originale Schreibweise/Umlaute/Typografie beibehalten).
- Wenn keine passende Evidenz im Kontext: Urteil = "nicht beurteilbar" und Hinweis "Keine Evidenz im Kontext gefunden."

Antwortformat:

R1 - <Kurztext der Teilanforderung>: <Urteil>
Begründung: <kurzer Satz>
Evidenz:
- "<wörtliches Zitat>" (Dateiname, Seite, Kategorie)
- "<wörtliches Zitat>" (Dateiname, Seite, Kategorie)
- "<wörtliches Zitat>" (Dateiname, Seite, Kategorie)
- ...

R2 - <Kurztext der Teilanforderung>: <Urteil>
Begründung: <kurzer Satz>
Evidenz:
- "<wörtliches Zitat>" (Dateiname, Seite, Kategorie)
- ...

Fazit:

Ergebnis: Die Anforderung wird (nichterfuellt/ teilweise erfuellt/ vollstaendig erfuellt)
Begründung: <kurze, konsistente Gesamtabwägung>

Anforderung:
{question}

Dokumentenabschnitte:\n\n
{docs}
"""
