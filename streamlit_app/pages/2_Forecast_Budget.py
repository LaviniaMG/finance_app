"""
Forecast & Budget — view forecast output with drill-down, year-level adjustment inputs,
and detailed report view with scenario comparison.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd

from streamlit_app.utils.state import init_session_state
from streamlit_app.utils.cache import (
    get_fact_multi, get_actuals, get_dim_accounts, get_dim_entities,
    get_dim_departments, get_dim_coverages, build_dim_lookup, build_select_options,
    get_dim_scenarios,
)
from streamlit_app.utils.formatters import fmt_currency, fmt_number, fmt_pct, MONTH_NAMES_RO
from streamlit_app.components.charts import line_chart_actuals_vs_forecast, waterfall_chart
from streamlit_app.components.filters import scenario_selector, dimension_filters, fiscal_year_selector

st.set_page_config(page_title="Forecast & Budget — FinPlan", layout="wide")
init_session_state()

with st.sidebar:
    st.markdown("### FinPlan")
    st.caption("Platformă de planificare financiară")
    st.divider()
    active_scenario = scenario_selector(key="fb_scenario")
    st.divider()
    st.markdown("**Comparare cu:**")
    scenarios_df = get_dim_scenarios()
    compare_opts = build_select_options(scenarios_df, "scenario_code", "scenario_code", add_all=True)
    compare_label = st.selectbox("Scenariu referință", [o[0] for o in compare_opts], key="fb_compare")
    compare_scenario = dict(compare_opts)[compare_label]

st.title("Forecast & Budget")

if not active_scenario:
    st.warning("Selectează un scenariu activ din sidebar.")
    st.stop()

# ── Data loading ─────────────────────────────────────────────────────────────
budget_year = active_scenario.split("_")[-1] if "_" in active_scenario else "2025"
codes = list({active_scenario})
if compare_scenario:
    codes.append(compare_scenario)

fact_df = get_fact_multi(tuple(codes))
actuals_df = get_actuals()

account_lookup = build_dim_lookup(get_dim_accounts(), "account_id", "account_name")
entity_lookup = build_dim_lookup(get_dim_entities(), "entity_id", "entity_name")
dept_lookup = build_dim_lookup(get_dim_departments(), "department_id", "department_name")

# ── Filters ──────────────────────────────────────────────────────────────────
st.markdown("### Filtre")
filter_cols = st.columns(5)

entities_df = get_dim_entities()
departments_df = get_dim_departments()
coverages_df = get_dim_coverages()
accounts_df = get_dim_accounts()

with filter_cols[0]:
    ent_opts = build_select_options(entities_df, "entity_id", "entity_name")
    ent_label = st.selectbox("Entitate", [o[0] for o in ent_opts], key="fb_entity")
    selected_entity = dict(ent_opts)[ent_label]

with filter_cols[1]:
    dept_opts = build_select_options(departments_df, "department_id", "department_name")
    dept_label = st.selectbox("Departament", [o[0] for o in dept_opts], key="fb_dept")
    selected_dept = dict(dept_opts)[dept_label]

with filter_cols[2]:
    cov_opts = build_select_options(get_dim_coverages(), "coverage_id", "coverage_name")
    cov_label = st.selectbox("Coverage", [o[0] for o in cov_opts], key="fb_cov")
    selected_cov = dict(cov_opts)[cov_label]

with filter_cols[3]:
    acc_opts = build_select_options(accounts_df, "account_id", "account_name")
    acc_label = st.selectbox("Cont", [o[0] for o in acc_opts], key="fb_acc")
    selected_acc = dict(acc_opts)[acc_label]

with filter_cols[4]:
    fiscal_year = st.selectbox("An fiscal", [2024, 2025, 2026], index=1, key="fb_year")

# ── Apply filters ─────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["fiscal_year"] == fiscal_year
    if selected_entity:
        mask &= df["entity_id"] == selected_entity
    if selected_dept:
        mask &= df["department_id"] == selected_dept
    if selected_cov:
        mask &= df["coverage_id"] == selected_cov
    if selected_acc:
        mask &= df["account_id"] == selected_acc
    return df[mask].copy()

fc_df = apply_filters(fact_df[fact_df["scenario_code"] == active_scenario]) if not fact_df.empty else pd.DataFrame()
cmp_df = apply_filters(fact_df[fact_df["scenario_code"] == compare_scenario]) if compare_scenario and not fact_df.empty else pd.DataFrame()
act_df = apply_filters(actuals_df) if not actuals_df.empty else pd.DataFrame()

# ── View selector ─────────────────────────────────────────────────────────────
view = st.radio(
    "Vedere",
    ["📊 Sumar lunar", "📋 Tabel detaliat", "🌉 Bridge varianță"],
    horizontal=True,
    key="fb_view",
)

st.markdown("---")

# ── View 1: Monthly summary ──────────────────────────────────────────────────
if view == "📊 Sumar lunar":
    if not fc_df.empty:
        monthly_fc = fc_df.groupby("period_id").agg(
            forecast_ytd=("ytd_rpt_amount", "sum"),
            forecast_mtd=("mtd_rpt_amount", "sum"),
        ).reset_index()

        if not act_df.empty:
            act_monthly = act_df.groupby("period_id").agg(actual_ytd=("ytd_rpt_amount", "sum")).reset_index()
            monthly_fc = monthly_fc.merge(act_monthly, on="period_id", how="left")

        if not cmp_df.empty:
            cmp_monthly = cmp_df.groupby("period_id").agg(budget_ytd=("ytd_rpt_amount", "sum")).reset_index()
            monthly_fc = monthly_fc.merge(cmp_monthly, on="period_id", how="left")

        monthly_fc["period_label"] = monthly_fc["period_id"].map(MONTH_NAMES_RO)
        fig = line_chart_actuals_vs_forecast(monthly_fc, title=f"YTD — {active_scenario}")
        st.plotly_chart(fig, use_container_width=True)

        # MTD table
        st.markdown("#### Valori MTD pe luni")
        display = monthly_fc[["period_label", "forecast_mtd"] +
                              (["actual_ytd"] if "actual_ytd" in monthly_fc.columns else []) +
                              (["budget_ytd"] if "budget_ytd" in monthly_fc.columns else [])].copy()

        for col in display.columns:
            if col != "period_label":
                display[col] = display[col].apply(lambda x: fmt_number(x))

        st.dataframe(display.rename(columns={
            "period_label": "Lună",
            "forecast_mtd": f"{active_scenario} MTD",
            "actual_ytd": "Actual YTD",
            "budget_ytd": f"{compare_scenario} YTD",
        }), use_container_width=True)
    else:
        st.info("Nu există date pentru filtrele selectate.")

# ── View 2: Detailed table ────────────────────────────────────────────────────
elif view == "📋 Tabel detaliat":
    if not fc_df.empty:
        detail = fc_df.copy()
        detail["Entitate"] = detail["entity_id"].map(entity_lookup).fillna("N/A")
        detail["Departament"] = detail["department_id"].map(dept_lookup).fillna("N/A")
        detail["Cont"] = detail["account_id"].map(account_lookup).fillna("N/A")
        detail["Lună"] = detail["period_id"].map(MONTH_NAMES_RO)

        if not cmp_df.empty:
            cmp_acc = cmp_df.groupby(["account_id", "period_id"])["ytd_rpt_amount"].sum().reset_index().rename(columns={"ytd_rpt_amount": "ref_ytd"})
            detail = detail.merge(cmp_acc, on=["account_id", "period_id"], how="left")
            detail["Varianță"] = detail["ytd_rpt_amount"] - detail["ref_ytd"].fillna(0)
            detail["Var %"] = (detail["Varianță"] / detail["ref_ytd"].abs() * 100).where(detail["ref_ytd"] != 0).round(1)

        show_cols = ["Entitate", "Departament", "Cont", "Lună",
                     "mtd_rpt_amount", "ytd_rpt_amount"]
        if "ref_ytd" in detail.columns:
            show_cols += ["ref_ytd", "Varianță", "Var %"]

        display = detail[show_cols].rename(columns={
            "mtd_rpt_amount": "MTD RPT",
            "ytd_rpt_amount": "YTD RPT",
            "ref_ytd": f"{compare_scenario} YTD",
        })

        st.dataframe(display, use_container_width=True, height=500)

        csv = display.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export CSV", csv, "forecast_detail.csv", "text/csv")
    else:
        st.info("Nu există date pentru filtrele selectate.")

# ── View 3: Waterfall bridge ──────────────────────────────────────────────────
elif view == "🌉 Bridge varianță":
    if not fc_df.empty and not cmp_df.empty:
        fc_total = float(fc_df["ytd_rpt_amount"].fillna(0).sum())
        cmp_total = float(cmp_df["ytd_rpt_amount"].fillna(0).sum())
        gap = fc_total - cmp_total

        categories = [compare_scenario, "Volume", "Price/Mix", "FX", "Other", active_scenario]
        values = [cmp_total, gap * 0.4, gap * 0.3, gap * 0.15, gap * 0.15, fc_total]

        fig = waterfall_chart(
            categories=categories,
            values=values,
            title=f"Bridge: {compare_scenario} → {active_scenario}",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Notă: Descompunerea (Volume/Price/FX/Other) este estimată. Configurează CalculationRule pentru descompunere exactă.")
    else:
        st.info("Selectează un scenariu de referință pentru bridge.")
