AUDIT_PROMPT = """Du bist Informationssicherheits-Auditor:in.\n
Prüfe die folgende Anforderung auf Basis der bereitgestellten Dokumentenabschnitte.\n
        
Vorgehen\n
1) Zerlege die Anforderung in atomare Teilanforderungen.\n
2) Bewerte jede Teilanforderung mit genau einem Urteil:\n
   - vollstaendig_erfuellt: durch Evidenz klar belegt\n
   - teilweise_erfuellt: teilweise belegt / Einschränkungen\n
   - nicht_erfuellt: Evidenz widerspricht oder zeigt Fehlen einer notwendigen Maßnahme\n
   - nicht_beurteilbar: keine passende Evidenz im Kontext\n
3) Für jedes Urteil: kurze Begründung + passende chunk_ids (nur direkt relevante).\n 
   - Bei nicht_beurteilbar: Begründung MUSS enthalten: "Keine Evidenz im Kontext gefunden." und chunk_ids = [].\n
4) Gesamturteil ableiten:\n
   - Wenn irgendeine Teilanforderung = nicht_erfuellt => overall_judgment = nicht_erfuellt oder teilweise_erfuellt (aber niemals vollstaendig_erfuellt)\n
   - Wenn alle = vollstaendig_erfuellt => vollstaendig_erfuellt\n
   - Sonst wenn mindestens eine = nicht_beurteilbar und keine = nicht_erfuellt => nicht_beurteilbar\n
   - Sonst => teilweise_erfuellt\n\n

Anforderung:\n
{question}
\n\n
Dokumentenabschnitte:\n
{docs}
"""
