"""
Parses free-text user input into structured ForecastAssumption rows via Claude.
"""
import json
from ai.claude_client import call_claude_json


_SYSTEM_PROMPT = """
Ești un AI specializat în planificare financiară pentru o bancă/companie.
Rolul tău este să transformi textul liber al utilizatorului în ipoteze structurate de forecast.

Returnezi EXCLUSIV un array JSON valid (fără explicații, fără markdown extra).

Fiecare ipoteză din array are structura:
{
  "description": "titlu scurt pentru ipoteză",
  "assumption_type": unul din: growth_pct | inflation_pct | fixed_value | delta_value | fx_adjustment_pct | headcount_growth_pct | ytd_target | yoy_pct,
  "assumption_value": număr (procente ca 0.03 pentru 3%, valori fixe în unitatea monetară),
  "account_category": "OpEx" | "Revenue" | "Salary" | "Marketing" | "CapEx" | null (null dacă nu e specificat),
  "account_code": codul sau numele contului dacă e specificat, altfel null,
  "entity_scope": "all" | cod entitate (null = all),
  "department_scope": "all" | cod departament (null = all),
  "fiscal_year": an fiscal (întreg), null dacă nu e specificat,
  "period_from": 1-12 (luna de start), null dacă se aplică tot anul,
  "period_to": 1-12 (luna de final), null dacă se aplică tot anul,
  "input_source": "AI",
  "needs_review": true dacă există ambiguitate dimensională (entitate/departament nespecificat sau neclar)
}

Reguli de interpretare:
- "H1" = period_from: 1, period_to: 6
- "H2" = period_from: 7, period_to: 12
- "Q1"=1-3, "Q2"=4-6, "Q3"=7-9, "Q4"=10-12
- "Ianuarie"=1, "Februarie"=2, "Martie"=3, "Aprilie"=4, "Mai"=5, "Iunie"=6
  "Iulie"=7, "August"=8, "Septembrie"=9, "Octombrie"=10, "Noiembrie"=11, "Decembrie"=12
- "Jan"=1, "Feb"=2, "Mar"=3, "Apr"=4, "May/Mai"=5, "Jun"=6, "Jul"=7, "Aug"=8, "Sep"=9, "Oct"=10, "Nov"=11, "Dec"=12
- Dacă utilizatorul spune "toate conturile OpEx" → account_category: "OpEx", account_code: null
- Dacă utilizatorul specifică un cost fix (ex: "200k") → assumption_type: fixed_value, assumption_value: 200000
- Procente se normalizează: "3%" → 0.03, "2.5%" → 0.025
- Dacă utilizatorul nu specifică entitate sau departament → entity_scope: "all", department_scope: "all", needs_review: true
- Dacă textul este ambiguu → needs_review: true

Returnează DOAR array JSON, fără alt text.
"""


def parse_assumptions_from_text(
    text: str,
    scenario_code: str,
    active_year: int,
) -> list[dict]:
    """
    Parses free-text into structured assumption dicts.
    Returns a list ready for display and confirmation before saving.
    """
    user_message = f"""
Scenariul activ: {scenario_code}
Anul fiscal: {active_year}

Textul utilizatorului:
{text}

Parsează ipotezele din text și returnează array JSON.
"""
    try:
        parsed = call_claude_json(_SYSTEM_PROMPT, user_message, temperature=0.1)
        if not isinstance(parsed, list):
            parsed = [parsed]

        # Normalize and add scenario_code
        for item in parsed:
            item["scenario_code"] = scenario_code
            if "fiscal_year" not in item or item.get("fiscal_year") is None:
                item["fiscal_year"] = active_year
            if "input_source" not in item:
                item["input_source"] = "AI"
            if "needs_review" not in item:
                item["needs_review"] = False

        return parsed

    except (json.JSONDecodeError, Exception) as e:
        return [{
            "description": "Eroare de parsare",
            "error": str(e),
            "raw_text": text,
            "needs_review": True,
        }]


def estimate_assumption_impact(
    assumption: dict,
    current_value: float,
) -> float | None:
    """
    Simple client-side impact estimate (no Claude call needed for this).
    Returns estimated delta in absolute monetary terms.
    """
    atype = assumption.get("assumption_type")
    value = float(assumption.get("assumption_value", 0))

    if atype in ("growth_pct", "inflation_pct", "headcount_growth_pct"):
        return current_value * value
    if atype in ("fixed_value",):
        return value - current_value
    if atype in ("delta_value",):
        return value
    if atype == "fx_adjustment_pct":
        return current_value * value
    if atype == "ytd_target":
        return value - current_value
    if atype == "yoy_pct":
        return current_value * value
    return None
