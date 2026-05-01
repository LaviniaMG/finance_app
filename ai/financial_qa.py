"""
Conversational financial Q&A with data context.
Supports multi-turn dialogue with the full financial dataset as context.
"""
from typing import Generator
import pandas as pd

from ai.claude_client import stream_claude, call_claude


_SYSTEM_PROMPT = """
Ești un analist financiar AI pentru FinPlan — o platformă de planificare financiară.
Ai acces la datele financiare ale companiei: actual, forecast și buget.

Capacitățile tale:
- Răspunzi la întrebări despre varianțe (Actual vs Budget vs RF)
- Explici ce conduce creșterea/scăderea unui cont sau departament
- Calculezi și interpretezi KPI-uri (EBITDA, Cost/Income Ratio, etc.)
- Faci drill-down: de la total → entitate → departament → cont
- Dacă utilizatorul întreabă "dacă vreau EBITDA X, ce trebuie să fac?" → analizezi gap-ul și propui levere concrete
- Poți face comparații: "cum stăm față de aceeași perioadă din anul trecut?"
- Poți identifica outlieri: "care e cel mai mare deviant față de buget?"

Stil de comunicare:
- Direct și concis (răspunsuri de max 200-300 cuvinte pentru întrebări simple)
- Citezi cifre exacte din date
- Dacă datele sunt insuficiente, spune ce lipsește
- Răspunde în română
- Dacă utilizatorul schimbă filtrul ("arată-mi pe țara X"), confirmă și adaptează
"""


def build_financial_context(
    fact_df: pd.DataFrame,
    scenario_code: str,
    actuals_scenario: str = "ACT_2025",
    budget_scenario: str = "BDG_2025",
    max_rows: int = 100,
) -> str:
    """
    Builds a text summary of financial data to pass to Claude as context.
    Summarizes at account level to keep tokens manageable.
    """
    if fact_df.empty:
        return "Nu există date financiare disponibile."

    fc = fact_df[fact_df["scenario_code"] == scenario_code].copy()
    act = fact_df[fact_df["scenario_code"].str.startswith("ACT")].copy()
    bdg = fact_df[fact_df["scenario_code"] == budget_scenario].copy()

    group_keys = ["account_id", "account_name", "account_code", "statement_type"]
    group_keys_present = [k for k in group_keys if k in fc.columns]

    def agg(df):
        if df.empty:
            return pd.DataFrame()
        k = [g for g in group_keys_present if g in df.columns]
        if not k:
            return pd.DataFrame()
        return df.groupby(k, dropna=False).agg(
            ytd_rpt=("ytd_rpt_amount", "sum"),
            mtd_rpt=("mtd_rpt_amount", "sum"),
        ).reset_index()

    fc_agg = agg(fc)
    act_agg = agg(act)
    bdg_agg = agg(bdg)

    merge_keys = [k for k in ["account_id", "account_name", "account_code"] if k in fc_agg.columns]
    if fc_agg.empty:
        return "Nu există date forecast disponibile."

    summary = fc_agg.rename(columns={"ytd_rpt": "fc_ytd", "mtd_rpt": "fc_mtd"})
    if not act_agg.empty and merge_keys:
        summary = summary.merge(
            act_agg[merge_keys + ["ytd_rpt", "mtd_rpt"]].rename(
                columns={"ytd_rpt": "act_ytd", "mtd_rpt": "act_mtd"}
            ),
            on=merge_keys, how="left"
        )
    if not bdg_agg.empty and merge_keys:
        summary = summary.merge(
            bdg_agg[merge_keys + ["ytd_rpt"]].rename(columns={"ytd_rpt": "bdg_ytd"}),
            on=merge_keys, how="left"
        )

    for col in ["act_ytd", "act_mtd", "bdg_ytd"]:
        if col not in summary.columns:
            summary[col] = 0

    summary = summary.head(max_rows)

    lines = [
        f"CONTEXT FINANCIAR — Scenariul: {scenario_code}",
        f"Rânduri afișate: {len(summary)} conturi (sumarizat)",
        "",
        f"{'Cont':<35} {'FC YTD':>12} {'ACT YTD':>12} {'BDG YTD':>12} {'Var vs BDG':>12}",
        "-" * 80,
    ]

    for _, row in summary.iterrows():
        name = str(row.get("account_name", row.get("account_code", "?")))[:33]
        fc_ytd = float(row.get("fc_ytd", 0) or 0)
        act_ytd = float(row.get("act_ytd", 0) or 0)
        bdg_ytd = float(row.get("bdg_ytd", 0) or 0)
        var = fc_ytd - bdg_ytd
        var_pct = (var / abs(bdg_ytd) * 100) if bdg_ytd != 0 else 0
        lines.append(
            f"{name:<35} {fc_ytd/1000:>11.1f} {act_ytd/1000:>11.1f} "
            f"{bdg_ytd/1000:>11.1f} {var_pct:>+10.1f}%"
        )

    return "\n".join(lines)


def ask_financial_question(
    question: str,
    financial_context: str,
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """
    Streaming financial Q&A. Use with st.write_stream().
    """
    user_message = f"""
DATE DISPONIBILE:
{financial_context}

ÎNTREBARE:
{question}
"""
    yield from stream_claude(_SYSTEM_PROMPT, user_message, history=history)


def get_quick_insights(
    financial_context: str,
    scenario_code: str,
) -> str:
    """
    Returns 3-5 bullet point insights about the current financial situation.
    Non-streaming, used for auto-generating dashboard insights.
    """
    user_message = f"""
{financial_context}

Generează 3-5 observații cheie despre situația financiară curentă ({scenario_code}).
Fiecare observație pe un rând nou, precedată de un bullet point (•).
Fii concis: max 20 cuvinte per observație.
Focus pe: cele mai mari varianțe, trenduri îngrijorătoare, performanțe bune.
"""
    return call_claude(_SYSTEM_PROMPT, user_message, temperature=0.3)
