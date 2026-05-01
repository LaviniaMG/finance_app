import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd

from streamlit_app.utils.state import init_session_state
from streamlit_app.utils.cache import (
    get_fact_multi, get_dim_scenarios, get_forecast_runs, get_business_warnings,
    get_assumptions, build_dim_lookup, get_dim_accounts, get_dim_entities,
)
from streamlit_app.utils.formatters import fmt_currency, fmt_pct, fmt_variance, status_badge, MONTH_NAMES_RO
from streamlit_app.components.charts import monthly_bar_chart, line_chart_actuals_vs_forecast
from streamlit_app.components.filters import scenario_selector, fiscal_year_selector

st.set_page_config(page_title="Dashboard — FinPlan", layout="wide")
init_session_state()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### FinPlan")
    st.caption("Platformă de planificare financiară")
    st.divider()
    active_scenario = scenario_selector(key="dash_scenario")
    st.divider()
    col_run = st.columns(2)
    with col_run[0]:
        if st.button("▶ Rulează forecast", type="primary", use_container_width=True):
            st.toast("Forecast pornit...", icon="🔄")
    with col_run[1]:
        if st.button("📤 Exportă", use_container_width=True):
            st.toast("Export în lucru...", icon="📤")

st.title("Dashboard")

if not active_scenario:
    st.warning("Selectează un scenariu activ din sidebar pentru a vedea datele.")
    st.stop()

# ── Data loading ─────────────────────────────────────────────────────────────
scenarios_df = get_dim_scenarios()
runs_df = get_forecast_runs()
account_lookup = build_dim_lookup(get_dim_accounts(), "account_id", "account_name")
entity_lookup = build_dim_lookup(get_dim_entities(), "entity_id", "entity_name")

# Detect budget and actuals scenario codes
budget_year = active_scenario.split("_")[-1] if "_" in active_scenario else "2025"
budget_code = f"BDG_{budget_year}"
actuals_prefix = f"ACT_{budget_year}"

all_codes = tuple({active_scenario, budget_code, actuals_prefix})
fact_df = get_fact_multi(all_codes)

fc_df = fact_df[fact_df["scenario_code"] == active_scenario] if not fact_df.empty else pd.DataFrame()
bdg_df = fact_df[fact_df["scenario_code"] == budget_code] if not fact_df.empty else pd.DataFrame()
act_df = fact_df[fact_df["scenario_code"].str.startswith("ACT")] if not fact_df.empty else pd.DataFrame()

warnings_df = get_business_warnings(active_scenario)

def _ytd_sum(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["ytd_rpt_amount"].fillna(0).sum())

def _ytd_pct_change(current: float, reference: float) -> float:
    if reference == 0:
        return 0.0
    return (current - reference) / abs(reference) * 100

revenue_fc = _ytd_sum(fc_df[fc_df.get("account_name", pd.Series(dtype=str)).str.contains("Revenue|Venit", case=False, na=False)] if "account_name" in fc_df.columns else pd.DataFrame())
revenue_bdg = _ytd_sum(bdg_df[bdg_df.get("account_name", pd.Series(dtype=str)).str.contains("Revenue|Venit", case=False, na=False)] if "account_name" in bdg_df.columns else pd.DataFrame())
revenue_act = _ytd_sum(act_df)

total_fc_ytd = _ytd_sum(fc_df)
total_bdg_ytd = _ytd_sum(bdg_df)
total_act_ytd = _ytd_sum(act_df)

opex_fc = total_fc_ytd * 0.45  # placeholder — replace with real OPEX account filter
ebitda_fc = total_fc_ytd - opex_fc

# ── KPI Cards ────────────────────────────────────────────────────────────────
st.markdown("---")
k1, k2, k3, k4 = st.columns(4)

with k1:
    delta_act = _ytd_pct_change(total_act_ytd, total_bdg_ytd)
    st.metric(
        label="REVENUE YTD ACTUAL",
        value=fmt_currency(total_act_ytd, unit=1_000_000, decimals=1).replace("k", "M"),
        delta=f"{delta_act:+.1f}% vs BDG",
        delta_color="normal",
    )

with k2:
    delta_fc = _ytd_pct_change(total_fc_ytd, total_bdg_ytd)
    st.metric(
        label="REVENUE FORECAST FY",
        value=fmt_currency(total_fc_ytd, unit=1_000_000, decimals=1).replace("k", "M"),
        delta=f"{delta_fc:+.1f}% vs BDG",
    )

with k3:
    st.metric(
        label="OPEX FORECAST FY",
        value=fmt_currency(opex_fc, unit=1_000_000, decimals=1).replace("k", "M"),
        delta="-1.4% vs BDG",
        delta_color="inverse",
    )

with k4:
    ebitda_bdg = total_bdg_ytd - total_bdg_ytd * 0.45
    ebitda_delta = _ytd_pct_change(ebitda_fc, ebitda_bdg)
    st.metric(
        label="EBITDA FORECAST FY",
        value=fmt_currency(ebitda_fc, unit=1_000_000, decimals=1).replace("k", "M"),
        delta=f"{ebitda_delta:+.1f}% vs RF prev.",
    )

st.markdown("---")

# ── Main chart + right panels ─────────────────────────────────────────────
chart_col, right_col = st.columns([2, 1])

with chart_col:
    # Build monthly comparison data
    if not fc_df.empty:
        monthly = fc_df.groupby("period_id").agg(
            forecast_mtd=("mtd_rpt_amount", "sum")
        ).reset_index()
        if not act_df.empty:
            act_monthly = act_df.groupby("period_id").agg(actual_mtd=("mtd_rpt_amount", "sum")).reset_index()
            monthly = monthly.merge(act_monthly, on="period_id", how="left")
        if not bdg_df.empty:
            bdg_monthly = bdg_df.groupby("period_id").agg(budget_mtd=("mtd_rpt_amount", "sum")).reset_index()
            monthly = monthly.merge(bdg_monthly, on="period_id", how="left")
        monthly["period_label"] = monthly["period_id"].map(MONTH_NAMES_RO)
        fig = monthly_bar_chart(
            monthly,
            title="Venit lunar — Actual vs Forecast vs Budget",
            actual_col="actual_mtd" if "actual_mtd" in monthly.columns else None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nu există date forecast pentru scenariul selectat.")

with right_col:
    st.markdown("#### Scenarii active")
    if not scenarios_df.empty:
        for _, row in scenarios_df.iterrows():
            code = row.get("scenario_code", "")
            stype = row.get("scenario_type", "")
            run_row = runs_df[runs_df["scenario_code"] == code].head(1) if not runs_df.empty else pd.DataFrame()
            status = run_row.iloc[0].get("status", "Draft") if not run_row.empty else "Draft"
            badge_color = {"Aprobat": "green", "Draft": "blue", "In review": "orange"}.get(status, "gray")
            st.markdown(
                f"**{code}** &nbsp; :{badge_color}[{status}]",
                unsafe_allow_html=False,
            )
    else:
        st.info("Nu există scenarii.")

    st.markdown("---")
    st.markdown("#### Alerte active")
    if not warnings_df.empty:
        for _, w in warnings_df.head(5).iterrows():
            account = account_lookup.get(w.get("account_id"), "N/A")
            warn_text = w.get("business_warning_text", "")
            level = w.get("confidence_level", "Medium")
            css_class = "alert-red" if level == "Low" else "alert-yellow"
            st.markdown(
                f'<div class="{css_class}">⚠️ <b>{account}</b><br><small>{warn_text[:80]}</small></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="alert-green">✅ Fără alerte active</div>', unsafe_allow_html=True)

# ── Bottom row: Top variances + Recent adjustments + Forecast status ─────────
st.markdown("---")
bot1, bot2, bot3 = st.columns(3)

with bot1:
    st.markdown("#### Top varianțe față de BDG")
    if not fc_df.empty and not bdg_df.empty and "account_id" in fc_df.columns:
        fc_acc = fc_df.groupby("account_id")["ytd_rpt_amount"].sum().reset_index().rename(columns={"ytd_rpt_amount": "fc_ytd"})
        bdg_acc = bdg_df.groupby("account_id")["ytd_rpt_amount"].sum().reset_index().rename(columns={"ytd_rpt_amount": "bdg_ytd"})
        var_df = fc_acc.merge(bdg_acc, on="account_id", how="inner")
        var_df["variance"] = var_df["fc_ytd"] - var_df["bdg_ytd"]
        var_df["var_pct"] = (var_df["variance"] / var_df["bdg_ytd"].abs() * 100).where(var_df["bdg_ytd"] != 0)
        var_df["account_name"] = var_df["account_id"].map(account_lookup).fillna("N/A")
        var_df = var_df.nlargest(5, "variance", keep="all").reset_index(drop=True)

        for _, row in var_df.iterrows():
            color = "🟢" if float(row["var_pct"] or 0) > 0 else "🔴"
            st.markdown(
                f"{color} **{row['account_name']}** &nbsp;|&nbsp; "
                f"FC: {fmt_currency(row['fc_ytd'])} &nbsp; BDG: {fmt_currency(row['bdg_ytd'])} &nbsp; "
                f"Var: {fmt_pct(row['var_pct'])}"
            )
    else:
        st.info("Date insuficiente pentru varianțe.")

with bot2:
    st.markdown("#### Ajustări recente")
    assumptions_df = get_assumptions(active_scenario)
    if not assumptions_df.empty:
        for _, a in assumptions_df.sort_values("created_at", ascending=False).head(5).iterrows():
            atype = a.get("assumption_type", "")
            aval = a.get("assumption_value", 0)
            asrc = a.get("input_source", "MANUAL")
            atext = a.get("assumption_text") or f"{atype}: {aval}"
            badge = "✅" if a.get("is_active") else "📝"
            st.markdown(f"{badge} {atext[:50]}")
    else:
        st.info("Nu există ajustări recente.")

with bot3:
    st.markdown(f"#### Status forecast — {active_scenario}")
    if not runs_df.empty:
        run = runs_df[runs_df["scenario_code"] == active_scenario].head(1)
        if not run.empty:
            r = run.iloc[0]
            st.markdown(f"**Model:** {r.get('model_name', 'N/A')}")
            st.markdown(f"**Creat:** {r.get('created_at', 'N/A')}")
            status = r.get("status", "Draft")
            st.markdown(f"**Status:** {status_badge(status)}")
        else:
            st.info("Forecast nerulatat pentru acest scenariu.")
    else:
        st.info("Nu există rulări de forecast.")
