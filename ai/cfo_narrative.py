"""
Auto-generates CFO-level narrative commentary from financial data.
Used in the CFO Reports page.
"""
import pandas as pd
from ai.claude_client import call_claude, stream_claude


_SYSTEM_PROMPT = """
Ești un asistent AI care generează rapoarte executive pentru CFO.
Stilul tău:
- Profesionist și concis (executive summary)
- Bazat pe date concrete (citezi cifre exacte)
- Structurat: Performanță Generală → KPI-uri cheie → Varianțe principale → Outlook
- Ton: obiectiv, fără subiectivism, fără predicții nefondate
- Lungime: 300-500 cuvinte pentru executive summary, mai mult pentru secțiuni detaliate
- Răspunde în română
"""


def generate_executive_summary(
    financial_context: str,
    scenario_code: str,
    period_label: str,
    currency_symbol: str = "€",
) -> str:
    """
    Generates a full executive summary for the CFO report.
    """
    user_message = f"""
Generează un Executive Summary pentru raportul CFO.

Scenariul: {scenario_code}
Perioada: {period_label}
Moneda raportare: {currency_symbol}

{financial_context}

Structura dorită:
## 1. Performanță Generală
## 2. KPI-uri Cheie (Revenue, EBITDA, Cost/Income Ratio)
## 3. Varianțe Principale față de Buget
## 4. Riscuri și Atenționări
## 5. Outlook și Pași Următori
"""
    return call_claude(_SYSTEM_PROMPT, user_message, temperature=0.3)


def generate_variance_commentary(
    variance_df: pd.DataFrame,
    scenario_code: str,
    vs_scenario: str = "BDG",
    top_n: int = 5,
    currency_symbol: str = "€",
) -> str:
    """
    Generates commentary specifically for the top N variances.
    """
    if variance_df.empty:
        return "Nu există date de varianță disponibile."

    top_variances = variance_df.nlargest(top_n, "variance_abs")
    rows = []
    for _, row in top_variances.iterrows():
        account = row.get("account_name", row.get("account_code", "N/A"))
        dept = row.get("department_name", "")
        var = float(row.get("variance", 0))
        var_pct = float(row.get("variance_pct", 0))
        rows.append(f"- {account} ({dept}): {currency_symbol}{var/1000:.1f}k ({var_pct:+.1f}%)")

    variances_text = "\n".join(rows)

    user_message = f"""
Cele mai mari {top_n} varianțe față de {vs_scenario} în scenariul {scenario_code}:

{variances_text}

Generează un comentariu de 100-150 cuvinte care explică varianțele principale
și sugerează acțiuni de urmărit. Fii specific la conturi și departamente menționate.
"""
    return call_claude(_SYSTEM_PROMPT, user_message, temperature=0.3)


def generate_stream_narrative(
    financial_context: str,
    user_instruction: str,
    history: list[dict] | None = None,
):
    """
    Streaming narrative generation for interactive CFO report building.
    """
    user_message = f"""
{financial_context}

Instrucțiunea utilizatorului: {user_instruction}
"""
    yield from stream_claude(_SYSTEM_PROMPT, user_message, history=history)


def generate_forecast_approval_memo(
    forecast_summary: str,
    scenario_code: str,
    submitted_by: str = "Finance Team",
    currency_symbol: str = "€",
) -> str:
    """
    Generates a formal memo for forecast approval workflow.
    """
    user_message = f"""
Generează un memo formal de aprobare pentru forecast.

Scenariu: {scenario_code}
Submis de: {submitted_by}

{forecast_summary}

Memo-ul trebuie să includă:
1. Rezumatul forecast-ului (cifre cheie)
2. Ipoteze principale utilizate
3. Varianțe semnificative față de versiunea anterioară
4. Recomandare de aprobare (sau puncte de clarificat)
5. Semnătură placeholder pentru CFO
"""
    return call_claude(_SYSTEM_PROMPT, user_message, temperature=0.2)
