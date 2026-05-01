"""Analiza Abaterilor — Variance analysis: Actual vs Budget vs RF with drill-down."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from streamlit_app.utils.state import init_session_state
from streamlit_app.utils.cache import (
    get_fact_multi, get_actuals, get_dim_accounts, get_dim_entities,
    get_dim_departments, get_dim_coverages, build_dim_lookup, build_select_options,
    get_dim_scenarios,
)
from streamlit_app.utils.formatters import fmt_currency, fmt_number, fmt_pct, MONTH_NAMES_RO
from streamlit_app.components.charts import waterfall_chart, variance_heatmap
from streamlit_app.components.filters import scenario_selector

st.set_page_config(page_title="Analiza Abaterilor — FinPlan", layout="wide")
init_session_state()

with st.sidebar:
    st.markdown("### FinPlan")
    st.caption("Platformă de planificare financiară")
    st.divider()
    active_scenario = scenario_selector(key="var_scenario")
    st.divider()
    scenarios_df = get_dim_scenarios()
    ref_opts = build_select_options(scenarios_df, "scenario_code", "scenario_code", add_all=False)
    ref_label = st.selectbox("Compară cu", [o[0] for o in ref_opts], key="var_ref")
    ref_scenario = dict(ref_opts).get(ref_label)

st.title("Analiza Abaterilor")
st.caption("Varianțe Actual vs Forecast vs Budget cu drill-down pe dimensiuni")

if not active_scenario:
    st.warning("Selectează un scenariu activ din sidebar.")
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────
budget_year = active_scenario.split("_")[-1] if "_" in active_scenario else "2025"
budget_code = f"BDG_{budget_year}"
codes = list({active_scenario, budget_code})
if ref_scenario and ref_scenario not in codes:
    codes.append(ref_scenario)

fact_df = get_fact_multi(tuple(codes))
actuals_df = get_actuals()

account_lookup = build_dim_lookup(get_dim_accounts(), "account_id", "account_name")
entity_lookup = build_dim_lookup(get_dim_entities(), "entity_id", "entity_name")
dept_lookup = build_dim_lookup(get_dim_departments(), "department_id", "department_name")

# ── Filters ───────────────────────────────────────────────────────────────────
f_cols = st.columns(3)
with f_cols[0]:
    fiscal_year = st.selectbox("An fiscal", [2024, 2025, 2026], index=1, key="var_year")
with f_cols[1]:
    period_labels = ["Toate"] + [f"{n} (P{i})" for i, n in MONTH_NAMES_RO.items()]
    period_sel = st.selectbox("Perioadă", period_labels, key="var_period")
    period_id = None if period_sel == "Toate" else int(period_sel.split("(P")[1].rstrip(")"))
with f_cols[2]:
    drill_dim = st.selectbox("Drill-down by", ["Entitate", "Departament", "Cont", "Coverage"], key="var_drill")

st.markdown("---")

# ── Compute variances ─────────────────────────────────────────────────────────
if fact_df.empty:
    st.info("Nu există date pentru scenariile selectate.")
    st.stop()

def _filter_by_year_period(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    m = df["fiscal_year"] == fiscal_year
    if period_id:
        m &= df["period_id"] == period_id
    return df[m].copy()

fc_df = _filter_by_year_period(fact_df[fact_df["scenario_code"] == active_scenario])
bdg_df = _filter_by_year_period(fact_df[fact_df["scenario_code"] == budget_code])
act_df = _filter_by_year_period(actuals_df) if not actuals_df.empty else pd.DataFrame()
ref_df = _filter_by_year_period(fact_df[fact_df["scenario_code"] == ref_scenario]) if ref_scenario else pd.DataFrame()

dim_col_map = {
    "Entitate": ("entity_id", entity_lookup),
    "Departament": ("department_id", dept_lookup),
    "Cont": ("account_id", account_lookup),
    "Coverage": ("coverage_id", {}),
}
group_col, lookup = dim_col_map[drill_dim]

def agg_by_dim(df: pd.DataFrame, val_col: str = "ytd_rpt_amount") -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    return df.groupby(group_col)[val_col].sum().reset_index()

fc_grp = agg_by_dim(fc_df).rename(columns={"ytd_rpt_amount": "fc_ytd"})
bdg_grp = agg_by_dim(bdg_df).rename(columns={"ytd_rpt_amount": "bdg_ytd"})
act_grp = agg_by_dim(act_df).rename(columns={"ytd_rpt_amount": "act_ytd"}) if not act_df.empty else pd.DataFrame()
ref_grp = agg_by_dim(ref_df).rename(columns={"ytd_rpt_amount": "ref_ytd"}) if not ref_df.empty else pd.DataFrame()

combined = fc_grp
if not bdg_grp.empty:
    combined = combined.merge(bdg_grp, on=group_col, how="outer")
if not act_grp.empty:
    combined = combined.merge(act_grp, on=group_col, how="outer")
if not ref_grp.empty:
    combined = combined.merge(ref_grp, on=group_col, how="outer")

for col in ["fc_ytd", "bdg_ytd", "act_ytd", "ref_ytd"]:
    if col not in combined.columns:
        combined[col] = 0
    combined[col] = combined[col].fillna(0)

combined["var_vs_bdg"] = combined["fc_ytd"] - combined["bdg_ytd"]
combined["var_vs_bdg_pct"] = (combined["var_vs_bdg"] / combined["bdg_ytd"].abs() * 100).where(combined["bdg_ytd"] != 0)
combined["dim_label"] = combined[group_col].map(lookup).fillna(combined[group_col].astype(str))

combined = combined.sort_values("var_vs_bdg", ascending=False).reset_index(drop=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k_cols = st.columns(4)
with k_cols[0]:
    st.metric("Forecast YTD total", fmt_currency(combined["fc_ytd"].sum()))
with k_cols[1]:
    st.metric("Budget YTD total", fmt_currency(combined["bdg_ytd"].sum()))
with k_cols[2]:
    total_var = combined["var_vs_bdg"].sum()
    total_var_pct = (total_var / abs(combined["bdg_ytd"].sum()) * 100) if combined["bdg_ytd"].sum() != 0 else 0
    st.metric("Varianță vs BDG", fmt_currency(total_var), delta=f"{total_var_pct:+.1f}%")
with k_cols[3]:
    st.metric("Conturi cu varianță negativă", int((combined["var_vs_bdg"] < 0).sum()))

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
chart_col, table_col = st.columns([1, 1])

with chart_col:
    fig = px.bar(
        combined.head(15),
        x="var_vs_bdg",
        y="dim_label",
        orientation="h",
        color="var_vs_bdg",
        color_continuous_scale="RdYlGn",
        title=f"Varianță vs BDG — by {drill_dim}",
        labels={"var_vs_bdg": "Varianță", "dim_label": drill_dim},
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with table_col:
    st.markdown("#### Detaliu varianțe")
    display_cols = ["dim_label", "fc_ytd", "bdg_ytd", "var_vs_bdg", "var_vs_bdg_pct"]
    if "act_ytd" in combined.columns and combined["act_ytd"].any():
        display_cols.insert(2, "act_ytd")

    display = combined[display_cols].rename(columns={
        "dim_label": drill_dim,
        "fc_ytd": "Forecast YTD",
        "act_ytd": "Actual YTD",
        "bdg_ytd": "Budget YTD",
        "var_vs_bdg": "Varianță",
        "var_vs_bdg_pct": "Var %",
    })

    for col in ["Forecast YTD", "Actual YTD", "Budget YTD", "Varianță"]:
        if col in display.columns:
            display[col] = display[col].apply(fmt_number)
    if "Var %" in display.columns:
        display["Var %"] = display["Var %"].apply(lambda x: fmt_pct(x) if x is not None else "—")

    st.dataframe(display, use_container_width=True, height=380)

    csv = combined.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export CSV", csv, "variance_analysis.csv", "text/csv")

# ── Top positive / negative variances ────────────────────────────────────────
st.markdown("---")
pos_col, neg_col = st.columns(2)

with pos_col:
    st.markdown("#### 🟢 Top performeri (vs BDG)")
    top_pos = combined[combined["var_vs_bdg"] > 0].nlargest(5, "var_vs_bdg")
    for _, row in top_pos.iterrows():
        st.markdown(
            f"**{row['dim_label']}** — {fmt_currency(row['var_vs_bdg'])} "
            f"({fmt_pct(row.get('var_vs_bdg_pct'))})"
        )

with neg_col:
    st.markdown("#### 🔴 Sub-performeri (vs BDG)")
    top_neg = combined[combined["var_vs_bdg"] < 0].nsmallest(5, "var_vs_bdg")
    for _, row in top_neg.iterrows():
        st.markdown(
            f"**{row['dim_label']}** — {fmt_currency(row['var_vs_bdg'])} "
            f"({fmt_pct(row.get('var_vs_bdg_pct'))})"
        )
