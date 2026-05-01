"""Session state helpers."""
import streamlit as st


def init_session_state():
    defaults = {
        "active_scenario": None,
        "active_fiscal_year": None,
        "active_entity_id": None,
        "active_department_id": None,
        "active_coverage_id": None,
        "active_account_id": None,
        "chat_history": [],          # AI Q&A conversation
        "draft_adjustments": [],     # Pending breakback adjustments
        "draft_assumptions": [],     # Parsed AI assumptions awaiting confirmation
        "forecast_df": None,         # In-memory forecast (after adjustments)
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_active_scenario() -> str | None:
    return st.session_state.get("active_scenario")


def set_active_scenario(scenario_code: str):
    st.session_state["active_scenario"] = scenario_code


def get_chat_history() -> list[dict]:
    return st.session_state.get("chat_history", [])


def append_chat(role: str, content: str):
    history = st.session_state.get("chat_history", [])
    history.append({"role": role, "content": content})
    st.session_state["chat_history"] = history


def clear_chat():
    st.session_state["chat_history"] = []


def add_draft_adjustment(adjustment: dict):
    drafts = st.session_state.get("draft_adjustments", [])
    drafts.append(adjustment)
    st.session_state["draft_adjustments"] = drafts


def clear_draft_adjustments():
    st.session_state["draft_adjustments"] = []


def get_draft_adjustments() -> list[dict]:
    return st.session_state.get("draft_adjustments", [])


def set_draft_assumptions(assumptions: list[dict]):
    st.session_state["draft_assumptions"] = assumptions


def get_draft_assumptions() -> list[dict]:
    return st.session_state.get("draft_assumptions", [])
