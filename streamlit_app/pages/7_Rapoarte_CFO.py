"""Rapoarte CFO — Auto-generated CFO reports with AI narrative and export."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
from io import BytesIO

from streamlit_app.utils.state import init_session_state, get_chat_history, append_chat
from streamlit_app.utils.cache import (
    get_fact_multi, get_actuals, get_dim_accounts, get_dim_entities,
    get_dim_departments, build_dim_lookup,
)
from streamlit_app.utils.formatters import fmt_currency, fmt_number, fmt_pct, MONTH_NAMES_RO
from streamlit_app.components.charts import line_chart_actuals_vs_forecast, waterfall_chart
from streamlit_app.components.filters import scenario_selector
from ai.cfo_narrative import generate_executive_summary, generate_variance_commentary, generate_stream_narrative
from ai.financial_qa import build_financial_context

st.set_page_config(page_title="Rapoarte CFO — FinPlan", layout="wide")
init_session_state()

with st.sidebar:
    st.markdown("### FinPlan")
    st.caption("Platformă de planificare financiară")
    st.divider()
    active_scenario = scenario_selector(key="cfo_scenario")
    st.divider()
    period_label_sel = st.selectbox(
        "Perioadă raport",
        ["YTD", "Q1", "Q2", "Q3", "Q4", "Luna curentă"],
        key="cfo_period_label",
    )

st.title("Rapoarte CFO")
st.caption("Rapoarte executive cu narativă AI, grafice și export PDF/Excel")

if not active_scenario:
    st.warning("Selectează un scenariu activ din sidebar.")
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────
budget_year = active_scenario.split("_")[-1] if "_" in active_scenario else "2025"
budget_code = f"BDG_{budget_year}"
fact_df = get_fact_multi(tuple({active_scenario, budget_code}))
actuals_df = get_actuals()

account_lookup = build_dim_lookup(get_dim_accounts(), "account_id", "account_name")
entity_lookup = build_dim_lookup(get_dim_entities(), "entity_id", "entity_name")

financial_context = build_financial_context(fact_df, active_scenario, budget_scenario=budget_code)

fc_df = fact_df[fact_df["scenario_code"] == active_scenario] if not fact_df.empty else pd.DataFrame()
bdg_df = fact_df[fact_df["scenario_code"] == budget_code] if not fact_df.empty else pd.DataFrame()

fc_total = float(fc_df["ytd_rpt_amount"].fillna(0).sum()) if not fc_df.empty else 0
bdg_total = float(bdg_df["ytd_rpt_amount"].fillna(0).sum()) if not bdg_df.empty else 0

# ── Report tabs ───────────────────────────────────────────────────────────────
tab_exec, tab_variance, tab_custom, tab_export = st.tabs([
    "📋 Executive Summary",
    "📊 Analiza Varianțelor",
    "✍️ Narativă Personalizată",
    "📤 Export",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Summary
# ══════════════════════════════════════════════════════════════════════════════
with tab_exec:
    col_gen, col_preview = st.columns([1, 2])

    with col_gen:
        st.markdown("#### Parametri raport")
        report_author = st.text_input("Autor", value="Finance Team")
        include_charts = st.checkbox("Include grafice", value=True)
        include_assumptions = st.checkbox("Include ipoteze aplicate", value=True)

        if st.button("🤖 Generează Executive Summary", type="primary", use_container_width=True):
            with st.spinner("AI generează raportul..."):
                try:
                    summary = generate_executive_summary(
                        financial_context=financial_context,
                        scenario_code=active_scenario,
                        period_label=f"{period_label_sel} {budget_year}",
                    )
                    st.session_state["cfo_exec_summary"] = summary
                except Exception as e:
                    st.error(f"Eroare AI: {e}")

    with col_preview:
        st.markdown("#### Preview")
        exec_summary = st.session_state.get("cfo_exec_summary", "")
        if exec_summary:
            st.markdown(exec_summary)
        else:
            st.info("Apasă 'Generează' pentru a crea narativa AI.")

    # KPI summary always shown
    st.markdown("---")
    st.markdown("#### KPI-uri cheie")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Revenue Forecast YTD", fmt_currency(fc_total))
    with k2:
        st.metric("Budget YTD", fmt_currency(bdg_total))
    with k3:
        var = fc_total - bdg_total
        var_pct = (var / abs(bdg_total) * 100) if bdg_total != 0 else 0
        st.metric("Varianță", fmt_currency(var), delta=f"{var_pct:+.1f}%")
    with k4:
        st.metric("Acoperire BDG", f"{(fc_total / bdg_total * 100):.1f}%" if bdg_total != 0 else "N/A")

    if include_charts and not fc_df.empty:
        monthly = fc_df.groupby("period_id").agg(forecast_ytd=("ytd_rpt_amount", "sum")).reset_index()
        if not actuals_df.empty:
            act_m = actuals_df.groupby("period_id").agg(actual_ytd=("ytd_rpt_amount", "sum")).reset_index()
            monthly = monthly.merge(act_m, on="period_id", how="left")
        if not bdg_df.empty:
            bdg_m = bdg_df.groupby("period_id").agg(budget_ytd=("ytd_rpt_amount", "sum")).reset_index()
            monthly = monthly.merge(bdg_m, on="period_id", how="left")
        monthly["period_label"] = monthly["period_id"].map(MONTH_NAMES_RO)
        fig = line_chart_actuals_vs_forecast(monthly, title=f"YTD Trend — {active_scenario}")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Variance Commentary
# ══════════════════════════════════════════════════════════════════════════════
with tab_variance:
    st.markdown("#### Top varianțe cu comentariu AI")

    if not fc_df.empty and not bdg_df.empty and "account_id" in fc_df.columns:
        fc_acc = fc_df.groupby("account_id")["ytd_rpt_amount"].sum().reset_index().rename(columns={"ytd_rpt_amount": "fc_ytd"})
        bdg_acc = bdg_df.groupby("account_id")["ytd_rpt_amount"].sum().reset_index().rename(columns={"ytd_rpt_amount": "bdg_ytd"})
        var_df = fc_acc.merge(bdg_acc, on="account_id", how="inner")
        var_df["variance"] = var_df["fc_ytd"] - var_df["bdg_ytd"]
        var_df["variance_abs"] = var_df["variance"].abs()
        var_df["variance_pct"] = (var_df["variance"] / var_df["bdg_ytd"].abs() * 100).where(var_df["bdg_ytd"] != 0)
        var_df["account_name"] = var_df["account_id"].map(account_lookup).fillna("N/A")
        var_df["department_name"] = ""

        top_n = st.slider("Număr varianțe", 3, 10, 5, key="cfo_var_n")

        st.dataframe(
            var_df.nlargest(top_n, "variance_abs")[
                ["account_name", "fc_ytd", "bdg_ytd", "variance", "variance_pct"]
            ].rename(columns={
                "account_name": "Cont",
                "fc_ytd": "Forecast YTD",
                "bdg_ytd": "Budget YTD",
                "variance": "Varianță",
                "variance_pct": "Var %",
            }),
            use_container_width=True,
        )

        if st.button("📝 Generează comentariu varianțe", type="primary"):
            with st.spinner("AI analizează varianțele..."):
                try:
                    commentary = generate_variance_commentary(var_df, active_scenario, top_n=top_n)
                    st.session_state["cfo_var_commentary"] = commentary
                except Exception as e:
                    st.error(f"Eroare AI: {e}")

        var_commentary = st.session_state.get("cfo_var_commentary", "")
        if var_commentary:
            st.markdown("---")
            st.markdown(var_commentary)
    else:
        st.info("Date insuficiente pentru analiza varianțelor.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Custom narrative
# ══════════════════════════════════════════════════════════════════════════════
with tab_custom:
    st.markdown("#### Narativă personalizată")
    st.caption("Dă instrucțiuni specifice AI-ului pentru secțiunea dorită din raport.")

    custom_instruction = st.text_area(
        "Instrucțiune pentru AI",
        height=100,
        placeholder=(
            "Ex: Scrie un paragraf despre riscurile valutare pentru H2 2025\n"
            "Ex: Compară performanța față de aceeași perioadă din 2024\n"
            "Ex: Formulează o recomandare pentru management bazată pe EBITDA actual"
        ),
    )

    if st.button("✍️ Generează", type="primary"):
        if custom_instruction.strip():
            custom_area = st.empty()
            full_text = ""
            try:
                for chunk in generate_stream_narrative(financial_context, custom_instruction):
                    full_text += chunk
                    custom_area.markdown(full_text + "▌")
                custom_area.markdown(full_text)
                st.session_state["cfo_custom_text"] = (
                    st.session_state.get("cfo_custom_text", "") + "\n\n" + full_text
                )
            except Exception as e:
                st.error(f"Eroare AI: {e}")
        else:
            st.warning("Introdu o instrucțiune.")

    custom_text = st.session_state.get("cfo_custom_text", "")
    if custom_text:
        st.markdown("---")
        st.markdown("**Textul generat:**")
        edited = st.text_area("Editează narativa", value=custom_text, height=200, key="cfo_custom_edit")
        st.session_state["cfo_custom_text"] = edited


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Export
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown("#### Export raport")

    exec_summary = st.session_state.get("cfo_exec_summary", "")
    var_commentary = st.session_state.get("cfo_var_commentary", "")
    custom_text = st.session_state.get("cfo_custom_text", "")

    full_report = f"""# Raport CFO — {active_scenario}
Perioada: {period_label_sel} {budget_year}

---

## Executive Summary
{exec_summary or "Ne-generat."}

---

## Analiza Varianțelor
{var_commentary or "Ne-generat."}

---

## Narativă Personalizată
{custom_text or "Ne-generat."}
"""

    st.text_area("Preview raport complet", value=full_report, height=300, disabled=True)

    exp_cols = st.columns(3)
    with exp_cols[0]:
        st.download_button(
            "📄 Export TXT",
            full_report.encode("utf-8"),
            f"raport_cfo_{active_scenario}.txt",
            "text/plain",
            use_container_width=True,
        )
    with exp_cols[1]:
        if not fc_df.empty:
            excel_buf = BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                fc_df.to_excel(writer, sheet_name="Forecast", index=False)
                if not bdg_df.empty:
                    bdg_df.to_excel(writer, sheet_name="Budget", index=False)
                pd.DataFrame({"Raport": [full_report]}).to_excel(writer, sheet_name="Narativa", index=False)
            st.download_button(
                "📊 Export Excel",
                excel_buf.getvalue(),
                f"raport_cfo_{active_scenario}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    with exp_cols[2]:
        st.button("📧 Trimite pe email", disabled=True, use_container_width=True, help="În curând")
