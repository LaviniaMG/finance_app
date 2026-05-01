"""
EBITDA optimizer — prescriptive AI recommendations for closing the EBITDA gap.
Given current P&L data and a target, Claude identifies specific actionable levers.
"""
from typing import Generator
import pandas as pd

from ai.claude_client import call_claude, stream_claude


_SYSTEM_PROMPT = """
Ești un consilier financiar CFO AI pentru o bancă/companie de servicii financiare.
Ai acces la datele financiare actuale și rolul tău este să oferi recomandări SPECIFICE și ACȚIONABILE
pentru atingerea unui target de EBITDA.

Regulile tale:
1. Fii specific: menționează conturi, departamente, entități reale din datele primite
2. Cuantifică impactul estimat al fiecărui lever în unitate monetară
3. Prioritizează levere după impact și ușurință de implementare
4. Separă levere pe venituri (creștere) vs costuri (reducere)
5. Identifică și riscurile fiecărui lever
6. Formatul răspunsului: structurat cu secțiuni clare
7. Dacă datele sunt insuficiente, spune-o explicit

Răspunde în română dacă nu ți se cere altfel.
"""


def _format_pl_summary(pl_df: pd.DataFrame, currency_symbol: str = "€") -> str:
    """Formats a P&L dataframe into readable text for Claude's context."""
    if pl_df.empty:
        return "Nu există date P&L disponibile."

    lines = ["SITUAȚIA P&L ACTUALĂ (YTD, în mii):"]
    lines.append(f"{'Cont':<40} {'Actual':>12} {'Forecast':>12} {'Budget':>12} {'Var vs BDG':>12}")
    lines.append("-" * 90)

    for _, row in pl_df.iterrows():
        account = str(row.get("account_name", row.get("account_code", "N/A")))[:38]
        actual = row.get("actual_ytd", 0) or 0
        forecast = row.get("forecast_ytd", 0) or 0
        budget = row.get("budget_ytd", 0) or 0
        var = forecast - budget
        var_pct = (var / abs(budget) * 100) if budget != 0 else 0

        lines.append(
            f"{account:<40} {actual/1000:>11.1f} {forecast/1000:>11.1f} "
            f"{budget/1000:>11.1f} {var_pct:>+10.1f}%"
        )

    return "\n".join(lines)


def analyze_ebitda_gap(
    pl_summary_df: pd.DataFrame,
    current_ebitda: float,
    target_ebitda: float,
    scenario_code: str,
    currency_symbol: str = "€",
    additional_context: str = "",
) -> str:
    """
    Analyzes the EBITDA gap and returns prescriptive recommendations.
    """
    gap = target_ebitda - current_ebitda
    gap_pct = (gap / abs(current_ebitda) * 100) if current_ebitda != 0 else 0

    pl_text = _format_pl_summary(pl_summary_df, currency_symbol)

    user_message = f"""
Scenariul: {scenario_code}
EBITDA actual (forecast curent): {currency_symbol}{current_ebitda/1000:.1f}k
EBITDA target: {currency_symbol}{target_ebitda/1000:.1f}k
GAP de închis: {currency_symbol}{gap/1000:.1f}k ({gap_pct:+.1f}%)

{pl_text}

{f'Context suplimentar: {additional_context}' if additional_context else ''}

Analizează situația și oferă recomandări specifice pentru închiderea gap-ului de {currency_symbol}{gap/1000:.1f}k.
Structurează răspunsul astfel:
1. ANALIZA SITUAȚIEI (2-3 rânduri)
2. LEVERE PE VENITURI (cu impact estimat)
3. LEVERE PE COSTURI (cu impact estimat)
4. PLAN DE ACȚIUNE RECOMANDAT (prioritizat)
5. RISCURI ȘI CONSIDERENTE
"""
    return call_claude(_SYSTEM_PROMPT, user_message, temperature=0.4)


def stream_ebitda_analysis(
    pl_summary_df: pd.DataFrame,
    current_ebitda: float,
    target_ebitda: float,
    scenario_code: str,
    history: list[dict] | None = None,
    currency_symbol: str = "€",
) -> Generator[str, None, None]:
    """Streaming version for real-time display in Streamlit."""
    gap = target_ebitda - current_ebitda
    pl_text = _format_pl_summary(pl_summary_df, currency_symbol)

    user_message = f"""
EBITDA actual: {currency_symbol}{current_ebitda/1000:.1f}k | Target: {currency_symbol}{target_ebitda/1000:.1f}k | GAP: {currency_symbol}{gap/1000:.1f}k

{pl_text}

Oferă recomandări specifice pentru atingerea targetului.
"""
    yield from stream_claude(_SYSTEM_PROMPT, user_message, history=history)


def answer_financial_question(
    question: str,
    financial_context: str,
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """
    Answers any financial question using the provided data context.
    Streaming — use with st.write_stream().

    question: the user's question (e.g., "Ce conduce creșterea costurilor în Marketing?")
    financial_context: formatted string with relevant financial data
    history: conversation history for multi-turn
    """
    system = """
Ești un analist financiar AI cu acces la datele companiei.
Răspunzi la întrebări despre performanța financiară, varianțe, forecast și buget.

Regulile tale:
- Fii concis și direct
- Citezi cifre concrete din datele primite
- Dacă datele nu conțin răspunsul, spune-o explicit
- Poți sugera ce date suplimentare ar fi utile
- Răspunde în română

Dacă utilizatorul întreabă "dacă vreau EBITDA X, ce trebuie să fac?" →
  analizează gap-ul față de curent și sugerează levere concrete.
"""
    user_message = f"""
DATE FINANCIARE DISPONIBILE:
{financial_context}

ÎNTREBAREA UTILIZATORULUI:
{question}
"""
    yield from stream_claude(system, user_message, history=history)


def compute_ebitda_from_pl(pl_df: pd.DataFrame, ebitda_account_codes: list[str] | None = None) -> dict:
    """
    Computes current EBITDA from P&L data.
    If ebitda_account_codes is provided, sums those accounts.
    Otherwise estimates as Revenue - OpEx.
    Returns {"current": float, "budget": float, "variance": float}
    """
    if pl_df.empty:
        return {"current": 0.0, "budget": 0.0, "variance": 0.0}

    if ebitda_account_codes:
        mask = pl_df["account_code"].isin(ebitda_account_codes)
        subset = pl_df[mask]
        current = float(subset.get("forecast_ytd", pd.Series([0])).fillna(0).sum())
        budget = float(subset.get("budget_ytd", pd.Series([0])).fillna(0).sum())
    else:
        revenue_mask = pl_df.get("account_category", pd.Series(dtype=str)).str.lower().isin(["revenue", "venituri"])
        cost_mask = pl_df.get("account_category", pd.Series(dtype=str)).str.lower().isin(["cost", "costuri", "opex"])

        revenue_fc = float(pl_df.loc[revenue_mask, "forecast_ytd"].fillna(0).sum()) if revenue_mask.any() else 0
        cost_fc = float(pl_df.loc[cost_mask, "forecast_ytd"].fillna(0).sum()) if cost_mask.any() else 0
        revenue_bdg = float(pl_df.loc[revenue_mask, "budget_ytd"].fillna(0).sum()) if revenue_mask.any() else 0
        cost_bdg = float(pl_df.loc[cost_mask, "budget_ytd"].fillna(0).sum()) if cost_mask.any() else 0

        current = revenue_fc - cost_fc
        budget = revenue_bdg - cost_bdg

    return {
        "current": current,
        "budget": budget,
        "variance": current - budget,
        "variance_pct": (current - budget) / abs(budget) * 100 if budget != 0 else 0,
    }
