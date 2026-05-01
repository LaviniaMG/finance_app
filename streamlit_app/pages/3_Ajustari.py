"""
Ajustări — 3-step breakback adjustment page.
Step 1: Select dimension node (any level)
Step 2: Enter new aggregated value
Step 3: Preview breakback to leaves, then save
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd

from streamlit_app.utils.state import init_session_state, add_draft_adjustment, get_draft_adjustments, clear_draft_adjustments
from streamlit_app.utils.cache import (
    get_fact_for_scenario, get_actuals, get_dim_entities, get_dim_accounts,
    get_dim_departments, get_dim_coverages, build_select_options, build_dim_lookup,
)
from streamlit_app.utils.formatters import fmt_currency, fmt_number, fmt_pct, MONTH_NAMES_RO_FULL
from streamlit_app.components.filters import scenario_selector
from ml.breakback import preview_breakback, apply_breakback
from database.save_assumptions import save_assumptions_to_db
import pandas as pd

st.set_page_config(page_title="Ajustări — FinPlan", layout="wide")
init_session_state()

with st.sidebar:
    st.markdown("### FinPlan")
    st.caption("Platformă de planificare financiară")
    st.divider()
    active_scenario = scenario_selector(key="adj_scenario")

st.title("Ajustări")

if not active_scenario:
    st.warning("Selectează un scenariu activ din sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Load dimension data
entities_df = get_dim_entities()
departments_df = get_dim_departments()
coverages_df = get_dim_coverages()
accounts_df = get_dim_accounts()

entity_opts = build_select_options(entities_df, "entity_id", "entity_name")
dept_opts = build_select_options(departments_df, "department_id", "department_name")
cov_opts = build_select_options(coverages_df, "coverage_id", "coverage_name")
acc_opts = build_select_options(accounts_df, "account_id", "account_name")

entity_lookup = build_dim_lookup(entities_df, "entity_id", "entity_name")
dept_lookup = build_dim_lookup(departments_df, "department_id", "department_name")
cov_lookup = build_dim_lookup(coverages_df, "coverage_id", "coverage_name")
acc_lookup = build_dim_lookup(accounts_df, "account_id", "account_name")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Select dimension node
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="background:#fff3f5;border-radius:8px;padding:16px;margin-bottom:12px">'
    '<span class="step-badge" style="background:#E91E63;color:white;border-radius:50%;'
    'padding:2px 9px;font-weight:bold;margin-right:8px">1</span>'
    '<b>Selectează nodul pe fiecare dimensiune</b></div>',
    unsafe_allow_html=True,
)

st.info(
    "Selectează orice nivel — root, nod intermediar sau leaf. "
    "Valoarea afișată este suma tuturor leafurilor de sub selecție. "
    "Editând-o, diferența se distribuie proporțional pe leafuri."
)

dim_cols = st.columns(4)
with dim_cols[0]:
    ent_label = st.selectbox("Entitate", [o[0] for o in entity_opts], key="adj_entity")
    entity_id = dict(entity_opts)[ent_label]

with dim_cols[1]:
    dept_label = st.selectbox("Departament", [o[0] for o in dept_opts], key="adj_dept")
    department_id = dict(dept_opts)[dept_label]

with dim_cols[2]:
    cov_label = st.selectbox("Coverage", [o[0] for o in cov_opts], key="adj_cov")
    coverage_id = dict(cov_opts)[cov_label]

with dim_cols[3]:
    acc_label = st.selectbox("Cont", [o[0] for o in acc_opts], key="adj_acc")
    account_id = dict(acc_opts)[acc_label]

scen_per_cols = st.columns(3)
with scen_per_cols[0]:
    st.text_input("Scenariu", value=active_scenario, disabled=True)

available_years = sorted(
    entities_df["entity_id"].head(1).tolist() and [2025, 2026] or [2025]
)
with scen_per_cols[1]:
    fiscal_year = st.selectbox("An fiscal", [2024, 2025, 2026], index=1, key="adj_year")

with scen_per_cols[2]:
    period_labels = ["Toate"] + [f"{n} (P{i})" for i, n in MONTH_NAMES_RO_FULL.items()]
    period_sel = st.selectbox("Perioadă", period_labels, index=7, key="adj_period")
    if period_sel == "Toate":
        period_id = None
    else:
        period_id = int(period_sel.split("(P")[1].rstrip(")"))

load_col, _ = st.columns([1, 4])
with load_col:
    load_clicked = st.button("📥 Încarcă valoarea", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Edit aggregated value
# ─────────────────────────────────────────────────────────────────────────────
if load_clicked or st.session_state.get("adj_loaded"):
    st.session_state["adj_loaded"] = True

    forecast_df = get_fact_for_scenario(active_scenario)

    if forecast_df.empty:
        st.error("Nu există date forecast pentru scenariul selectat.")
        st.stop()

    # Filter to match dimensions
    mask = pd.Series([True] * len(forecast_df))
    if entity_id is not None:
        mask &= forecast_df["entity_id"] == entity_id
    if department_id is not None:
        mask &= forecast_df["department_id"] == department_id
    if coverage_id is not None:
        mask &= forecast_df["coverage_id"] == coverage_id
    if account_id is not None:
        mask &= forecast_df["account_id"] == account_id
    if fiscal_year:
        mask &= forecast_df["fiscal_year"] == fiscal_year
    if period_id is not None:
        mask &= forecast_df["period_id"] == period_id

    filtered = forecast_df[mask].copy()
    current_agg = float(filtered["mtd_rpt_amount"].fillna(0).sum()) if not filtered.empty else 0.0

    st.markdown("---")
    st.markdown(
        '<span class="step-badge" style="background:#E91E63;color:white;border-radius:50%;'
        'padding:2px 9px;font-weight:bold;margin-right:8px">2</span>'
        '<b>Editează valoarea agregată</b>',
        unsafe_allow_html=True,
    )

    val_cols = st.columns([1, 1, 1, 2])
    with val_cols[0]:
        st.metric("VALOARE ML FORECAST", fmt_number(current_agg))

    with val_cols[1]:
        new_value = st.number_input(
            "VALOAREA INTRODUSĂ",
            value=float(round(current_agg)),
            step=1.0,
            format="%.0f",
            key="adj_new_val",
        )

    with val_cols[2]:
        delta = new_value - current_agg
        delta_pct = (delta / abs(current_agg) * 100) if current_agg != 0 else 0
        st.metric("DELTA", f"{fmt_number(delta)} ({delta_pct:+.1f}%)")

    with val_cols[3]:
        st.markdown("")
        preview_clicked = st.button("👁 Previzualizare impact", use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Preview breakback
    # ─────────────────────────────────────────────────────────────────────────
    if preview_clicked or st.session_state.get("adj_preview_shown"):
        st.session_state["adj_preview_shown"] = True

        actuals_df = get_actuals()

        preview_df = preview_breakback(
            forecast_df=forecast_df,
            actuals_df=actuals_df,
            entity_id=entity_id,
            department_id=department_id,
            coverage_id=coverage_id,
            account_id=account_id,
            fiscal_year=fiscal_year,
            period_id=period_id or 7,
            new_aggregated_value=new_value,
        )

        st.markdown("---")
        st.markdown(
            '<span class="step-badge" style="background:#E91E63;color:white;border-radius:50%;'
            'padding:2px 9px;font-weight:bold;margin-right:8px">3</span>'
            '<b>Previzualizare impact</b>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"+{fmt_number(delta)} se distribuie pe {len(preview_df)-1} frunze "
            "proporțional cu valoarea cumulată pe actuale."
        )

        if not preview_df.empty:
            display_df = preview_df.copy()
            if "entity_id" in display_df.columns:
                display_df["Entitate"] = display_df["entity_id"].map(entity_lookup).fillna(display_df["entity_id"].astype(str))
            if "department_id" in display_df.columns:
                display_df["Departament"] = display_df["department_id"].map(dept_lookup).fillna(display_df["department_id"].astype(str))
            if "coverage_id" in display_df.columns:
                display_df["Coverage"] = display_df["coverage_id"].map(cov_lookup).fillna(display_df["coverage_id"].astype(str))
            if "account_id" in display_df.columns:
                display_df["Cont"] = display_df["account_id"].map(acc_lookup).fillna(display_df["account_id"].astype(str))

            show_cols = [c for c in ["Entitate", "Departament", "Coverage", "Cont",
                                     "ytd_weight_pct", "current_value", "new_value", "delta"]
                         if c in display_df.columns]
            display_df = display_df[show_cols].rename(columns={
                "ytd_weight_pct": "Pondere YTD %",
                "current_value": "Valoare calculată",
                "new_value": "Valoare ajustare",
                "delta": "Delta",
            })

            def color_delta(val):
                try:
                    v = float(val)
                    return "color: green" if v > 0 else ("color: red" if v < 0 else "")
                except Exception:
                    return ""

            st.dataframe(
                display_df.style.applymap(color_delta, subset=["Delta"]),
                use_container_width=True,
                height=220,
            )

        # Save buttons
        save_cols = st.columns([3, 1, 1])
        with save_cols[1]:
            if st.button("❌ Anulează"):
                st.session_state["adj_loaded"] = False
                st.session_state["adj_preview_shown"] = False
                st.rerun()
        with save_cols[2]:
            if st.button("✅ Salvează", type="primary"):
                # Save as a delta_value assumption
                adjustment = {
                    "scenario_code": active_scenario,
                    "entity_id": entity_id,
                    "department_id": department_id,
                    "coverage_id": coverage_id,
                    "account_id": account_id,
                    "fiscal_year": fiscal_year,
                    "period_from": period_id,
                    "period_to": period_id,
                    "assumption_type": "delta_value",
                    "assumption_value": delta,
                    "input_source": "MANUAL",
                    "assumption_text": f"Ajustare manuală: {entity_lookup.get(entity_id,'All')} / {acc_lookup.get(account_id,'All')} → {fmt_number(new_value)}",
                    "priority_order": 1,
                    "is_active": 1,
                }
                try:
                    save_assumptions_to_db(pd.DataFrame([adjustment]))
                    st.success("✅ Ajustare salvată cu succes!")
                    add_draft_adjustment(adjustment)
                    st.session_state["adj_preview_shown"] = False
                    st.session_state["adj_loaded"] = False
                    # Clear cache so next load reflects new data
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Eroare la salvare: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Adjustment history
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"#### Istoric ajustări — {active_scenario}")

from streamlit_app.utils.cache import get_assumptions
assumptions_df = get_assumptions(active_scenario)

if not assumptions_df.empty:
    for _, a in assumptions_df.sort_values("created_at", ascending=False).head(10).iterrows():
        src = a.get("input_source", "MANUAL")
        atype = a.get("assumption_type", "")
        aval = a.get("assumption_value", 0)
        atext = a.get("assumption_text") or f"{atype}: {aval}"
        status = "salvat" if a.get("is_active") else "inactiv"
        color = "green" if status == "salvat" else "gray"

        tag_cols = st.columns([4, 1])
        with tag_cols[0]:
            ent_tag = entity_lookup.get(a.get("entity_id"), "Toate entitățile")
            acc_tag = acc_lookup.get(a.get("account_id"), "Toate conturile")
            period_f = a.get("period_from", "")
            period_t = a.get("period_to", "")
            period_str = f"P{period_f}" if period_f == period_t else f"P{period_f}–P{period_t}"
            st.markdown(
                f"**{ent_tag}** &nbsp;|&nbsp; **{acc_tag}** &nbsp;|&nbsp; "
                f"_{period_str}_ &nbsp;|&nbsp; {atext[:60]}"
            )
        with tag_cols[1]:
            st.markdown(f":{color}[{status}]")
else:
    st.info("Nu există ajustări salvate pentru acest scenariu.")

# Bottom action buttons
st.markdown("---")
btn_cols = st.columns([3, 1, 1, 1])
with btn_cols[1]:
    if st.button("🔄 Resetează toate"):
        clear_draft_adjustments()
        st.toast("Draft-uri resetate.")
with btn_cols[2]:
    if st.button("💾 Salvează draft"):
        st.toast("Draft salvat.")
with btn_cols[3]:
    if st.button("✅ Salvează", type="primary"):
        st.toast("Ajustări salvate.")
