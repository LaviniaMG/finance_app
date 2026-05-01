"""
Ipoteze AI — Full AI assistant page.
Features:
  - Parse assumptions from natural language text
  - EBITDA optimizer: "What do I need to do to reach EBITDA X?"
  - Financial Q&A: any question about the data (with streaming)
  - Voice-style chat (text-based with conversation history)
  - History of applied assumptions
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd

from streamlit_app.utils.state import (
    init_session_state, get_chat_history, append_chat, clear_chat,
    set_draft_assumptions, get_draft_assumptions,
)
from streamlit_app.utils.cache import (
    get_fact_multi, get_actuals, get_dim_accounts, get_dim_entities,
    get_dim_departments, build_dim_lookup, get_assumptions,
)
from streamlit_app.utils.formatters import fmt_currency, fmt_number
from streamlit_app.components.filters import scenario_selector
from ai.assumption_parser import parse_assumptions_from_text, estimate_assumption_impact
from ai.ebitda_optimizer import stream_ebitda_analysis, answer_financial_question, compute_ebitda_from_pl
from ai.financial_qa import ask_financial_question, build_financial_context, get_quick_insights
from database.save_assumptions import save_assumptions_to_db

st.set_page_config(page_title="Ipoteze AI — FinPlan", layout="wide")
init_session_state()

with st.sidebar:
    st.markdown("### FinPlan")
    st.caption("Platformă de planificare financiară")
    st.divider()
    active_scenario = scenario_selector(key="ai_scenario")
    st.divider()
    if st.button("🗑 Șterge conversație"):
        clear_chat()
        st.rerun()

st.title("Ipoteze AI")

if not active_scenario:
    st.warning("Selectează un scenariu activ din sidebar.")
    st.stop()

# Load data for context
budget_year = active_scenario.split("_")[-1] if "_" in active_scenario else "2025"
budget_code = f"BDG_{budget_year}"
all_codes = tuple({active_scenario, budget_code})
fact_df = get_fact_multi(all_codes)
actuals_df = get_actuals()

account_lookup = build_dim_lookup(get_dim_accounts(), "account_id", "account_name")
entity_lookup = build_dim_lookup(get_dim_entities(), "entity_id", "entity_name")

financial_context = build_financial_context(
    fact_df, active_scenario, budget_scenario=budget_code
) if not fact_df.empty else "Nu există date financiare disponibile."

ebitda_data = compute_ebitda_from_pl(
    fact_df[fact_df["scenario_code"] == active_scenario] if not fact_df.empty else pd.DataFrame()
)
current_ebitda = ebitda_data.get("current", 0)

# ─────────────────────────────────────────────────────────────────────────────
# TABS: Ipoteze | Chat AI | EBITDA Optimizer
# ─────────────────────────────────────────────────────────────────────────────
tab_ipoteze, tab_chat, tab_ebitda = st.tabs([
    "📝 Scriere Ipoteze",
    "💬 Chat Financial AI",
    "🎯 EBITDA Optimizer",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Assumption Parser
# ═══════════════════════════════════════════════════════════════════════════
with tab_ipoteze:
    st.markdown(
        '<span style="background:#E91E63;color:white;border-radius:50%;'
        'padding:2px 9px;font-weight:bold;margin-right:8px">1</span>'
        '<b>Scriere ipoteze</b>',
        unsafe_allow_html=True,
    )

    assumption_text = st.text_area(
        "Descrie ipotezele în limbaj natural",
        height=120,
        placeholder=(
            "Exemplu:\n"
            "Aplică inflație de 3% pentru toate conturile din OPEX din Ian 2026\n"
            "Pentru costurile salariale aplică o creștere de 2.5% în H2 2025\n"
            "Setează costul cu marketing la 200k pentru Octombrie"
        ),
        key="assumption_text_input",
    )

    col_scen, col_parse = st.columns([2, 1])
    with col_scen:
        st.text_input("Scenariu activ", value=active_scenario, disabled=True)
    with col_parse:
        parse_clicked = st.button("🔍 Parsează", type="primary", use_container_width=True)

    if parse_clicked and assumption_text.strip():
        with st.spinner("AI parsează ipotezele..."):
            parsed = parse_assumptions_from_text(
                text=assumption_text,
                scenario_code=active_scenario,
                active_year=int(budget_year),
            )
            set_draft_assumptions(parsed)

    # ── Review section ──────────────────────────────────────────────────────
    draft_assumptions = get_draft_assumptions()
    if draft_assumptions:
        st.markdown("---")
        st.markdown(
            '<span style="background:#E91E63;color:white;border-radius:50%;'
            'padding:2px 9px;font-weight:bold;margin-right:8px">2</span>'
            '<b>Revizuire</b>',
            unsafe_allow_html=True,
        )

        any_needs_review = any(a.get("needs_review") for a in draft_assumptions)
        if any_needs_review:
            st.warning(
                "⚠️ Verifică fiecare ipoteză înainte de aplicare. "
                "AI-ul poate interpreta greșit contextul dimensional dacă textul e ambiguu — "
                "câmpurile marcate necesită confirmare."
            )

        for i, assumption in enumerate(draft_assumptions):
            if "error" in assumption:
                st.error(f"Eroare la parsare: {assumption['error']}")
                continue

            with st.expander(
                f"**{assumption.get('description', f'Ipoteză {i+1}')}** "
                f"{'⚠️' if assumption.get('needs_review') else '✅'}",
                expanded=True,
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Tip:** {assumption.get('assumption_type', 'N/A')}")
                    st.markdown(f"**Valoare:** {assumption.get('assumption_value', 0)}")
                    st.markdown(f"**Entitate:** {assumption.get('entity_scope', 'Toate')} {'— AI a presupus' if assumption.get('entity_scope') == 'all' else ''}")
                    st.markdown(f"**Departament:** {assumption.get('department_scope', 'Toate')} {'— AI a presupus' if assumption.get('department_scope') == 'all' else ''}")
                with col_b:
                    st.markdown(f"**Cont:** {assumption.get('account_code') or assumption.get('account_category') or 'Toate'}")
                    st.markdown(f"**An fiscal:** {assumption.get('fiscal_year', budget_year)}")
                    pf = assumption.get('period_from', '')
                    pt = assumption.get('period_to', '')
                    period_str = f"P{pf}–P{pt}" if pf and pt and pf != pt else (f"P{pf}" if pf else "Tot anul")
                    st.markdown(f"**Perioadă:** {period_str}")

                impact = estimate_assumption_impact(assumption, current_ebitda)
                if impact is not None:
                    impact_color = "green" if impact > 0 else "red"
                    st.markdown(
                        f"**Impact estimat:** :{impact_color}[{fmt_currency(impact)}]"
                        f" &nbsp; `{active_scenario}`"
                    )

                btn_cols = st.columns(3)
                with btn_cols[0]:
                    if st.button("✅ Aplică", key=f"apply_{i}", type="primary"):
                        try:
                            row = {
                                "scenario_code": active_scenario,
                                "assumption_type": assumption.get("assumption_type", "growth_pct"),
                                "assumption_value": float(assumption.get("assumption_value", 0)),
                                "input_source": "AI",
                                "assumption_text": assumption.get("description", ""),
                                "fiscal_year": assumption.get("fiscal_year"),
                                "period_from": assumption.get("period_from"),
                                "period_to": assumption.get("period_to"),
                            }
                            save_assumptions_to_db(pd.DataFrame([row]))
                            st.success("✅ Ipoteză aplicată!")
                            st.cache_data.clear()
                        except Exception as e:
                            st.error(f"Eroare: {e}")
                with btn_cols[1]:
                    if st.button("✏️ Editează", key=f"edit_{i}"):
                        st.info("Editare directă în tabel — în curând.")
                with btn_cols[2]:
                    if st.button("❌ Respinge", key=f"reject_{i}"):
                        draft_assumptions[i]["rejected"] = True
                        set_draft_assumptions(draft_assumptions)
                        st.rerun()

    # ── Applied history ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### Istoric ipoteze aplicate — {active_scenario}")
    applied_df = get_assumptions(active_scenario)
    ai_applied = applied_df[applied_df["input_source"] == "AI"] if not applied_df.empty else pd.DataFrame()

    if not ai_applied.empty:
        for _, a in ai_applied.sort_values("created_at", ascending=False).head(10).iterrows():
            atext = a.get("assumption_text") or a.get("assumption_type", "")
            aval = a.get("assumption_value", 0)
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"📌 **{atext}** — {fmt_currency(aval)}")
            with cols[1]:
                st.markdown(":green[aplicat]")
    else:
        st.info("Nu există ipoteze AI aplicate pentru acest scenariu.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Financial Chat AI
# ═══════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("#### 💬 Conversație financiară AI")
    st.caption(
        "Întreabă orice despre datele financiare: varianțe, trenduri, KPI-uri, "
        "sau ce trebuie să faci pentru a atinge un anumit target."
    )

    # Quick insight button
    if st.button("✨ Generează insight-uri rapide"):
        with st.spinner("AI analizează datele..."):
            try:
                insights = get_quick_insights(financial_context, active_scenario)
                append_chat("assistant", insights)
            except Exception as e:
                st.error(f"Eroare AI: {e}")

    # Conversation display
    chat_history = get_chat_history()
    for msg in chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Întreabă ceva... (ex: 'Dacă vreau EBITDA de 10M, ce trebuie să fac?')"
    )

    if user_input:
        append_chat("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                stream = ask_financial_question(
                    question=user_input,
                    financial_context=financial_context,
                    history=chat_history[:-1],  # exclude last user message (already included)
                )
                for chunk in stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                append_chat("assistant", full_response)
            except Exception as e:
                err_msg = f"Eroare la conectarea cu AI: {str(e)}"
                response_placeholder.error(err_msg)
                append_chat("assistant", err_msg)

    # Suggested questions
    st.markdown("---")
    st.caption("💡 Întrebări sugerate:")
    sugg_cols = st.columns(3)
    suggested = [
        "Care sunt cele mai mari varianțe față de buget?",
        "Ce conduce creșterea costurilor în Marketing?",
        "Cum stăm față de aceeași perioadă din anul trecut?",
        "Care e cel mai profitabil segment/entitate?",
        "Dacă vreau EBITDA cu 10% mai mare, ce trebuie să fac?",
        "Care conturi au cele mai mari riscuri (low confidence)?",
    ]
    for i, q in enumerate(suggested):
        with sugg_cols[i % 3]:
            if st.button(q, key=f"sugg_{i}", use_container_width=True):
                append_chat("user", q)
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    try:
                        stream = ask_financial_question(
                            question=q,
                            financial_context=financial_context,
                            history=get_chat_history()[:-1],
                        )
                        for chunk in stream:
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                        append_chat("assistant", full_response)
                        st.rerun()
                    except Exception as e:
                        response_placeholder.error(str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — EBITDA Optimizer
# ═══════════════════════════════════════════════════════════════════════════
with tab_ebitda:
    st.markdown("#### 🎯 EBITDA Optimizer")
    st.caption(
        "Spune-i AI-ului ce target de EBITDA vrei să atingi și îți va oferi "
        "recomandări specifice și cuantificate pentru a închide gap-ul."
    )

    ebitda_cols = st.columns(3)
    with ebitda_cols[0]:
        st.metric("EBITDA curent (forecast)", fmt_currency(current_ebitda))
    with ebitda_cols[1]:
        ebitda_target = st.number_input(
            "Target EBITDA",
            value=float(round(current_ebitda * 1.1)),
            step=100_000.0,
            format="%.0f",
            key="ebitda_target",
        )
    with ebitda_cols[2]:
        gap = ebitda_target - current_ebitda
        gap_pct = (gap / abs(current_ebitda) * 100) if current_ebitda != 0 else 0
        st.metric("GAP de închis", fmt_currency(gap), delta=f"{gap_pct:+.1f}%")

    additional_context = st.text_area(
        "Context suplimentar (opțional)",
        height=80,
        placeholder="Ex: suntem în Q3, avem o campanie nouă în H2, bugetul HR este fix...",
        key="ebitda_context",
    )

    if st.button("🚀 Analizează și recomandă", type="primary", use_container_width=False):
        fc_pl = fact_df[fact_df["scenario_code"] == active_scenario] if not fact_df.empty else pd.DataFrame()
        bdg_pl = fact_df[fact_df["scenario_code"] == budget_code] if not fact_df.empty else pd.DataFrame()

        pl_summary = pd.DataFrame()
        if not fc_pl.empty and "account_id" in fc_pl.columns:
            fc_acc = fc_pl.groupby("account_id")["ytd_rpt_amount"].sum().reset_index().rename(columns={"ytd_rpt_amount": "forecast_ytd"})
            if not bdg_pl.empty:
                bdg_acc = bdg_pl.groupby("account_id")["ytd_rpt_amount"].sum().reset_index().rename(columns={"ytd_rpt_amount": "budget_ytd"})
                fc_acc = fc_acc.merge(bdg_acc, on="account_id", how="left")
            fc_acc["account_name"] = fc_acc["account_id"].map(account_lookup).fillna("N/A")
            pl_summary = fc_acc

        with st.spinner("AI analizează situația P&L și generează recomandări..."):
            response_area = st.empty()
            full_response = ""
            try:
                for chunk in stream_ebitda_analysis(
                    pl_summary_df=pl_summary,
                    current_ebitda=current_ebitda,
                    target_ebitda=ebitda_target,
                    scenario_code=active_scenario,
                    history=get_chat_history(),
                ):
                    full_response += chunk
                    response_area.markdown(full_response + "▌")
                response_area.markdown(full_response)
                append_chat("user", f"Vreau EBITDA de {fmt_currency(ebitda_target)}. Ce trebuie să fac?")
                append_chat("assistant", full_response)
            except Exception as e:
                response_area.error(f"Eroare AI: {e}")

    st.markdown("---")
    st.markdown("**Sfaturi rapide pentru EBITDA:**")
    tips = [
        "💡 Optimizare costuri salariale prin eficiență operațională",
        "💡 Creșterea veniturilor din fee-uri și comisioane",
        "💡 Reducerea costurilor IT prin renegociere contracte",
        "💡 Îmbunătățirea marjei prin mix de produse mai profitabile",
    ]
    for t in tips:
        st.markdown(t)
