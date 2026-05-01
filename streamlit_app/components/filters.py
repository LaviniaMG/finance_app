"""Reusable dimension filter widgets."""
import streamlit as st
import pandas as pd

from streamlit_app.utils.cache import (
    get_dim_entities, get_dim_accounts, get_dim_departments,
    get_dim_coverages, get_dim_scenarios, build_select_options,
)


def scenario_selector(key: str = "scenario_sel", label: str = "Scenariu activ") -> str | None:
    scenarios_df = get_dim_scenarios()
    options = build_select_options(scenarios_df, "scenario_code", "scenario_code", add_all=False)

    if not options:
        st.warning("Nu există scenarii disponibile.")
        return None

    labels = [o[0] for o in options]
    values = [o[1] for o in options]

    current = st.session_state.get("active_scenario")
    idx = values.index(current) if current in values else 0

    selected_label = st.selectbox(label, labels, index=idx, key=key)
    selected_val = values[labels.index(selected_label)]
    st.session_state["active_scenario"] = selected_val
    return selected_val


def dimension_filters(
    col_layout: list | None = None,
    show_entity: bool = True,
    show_department: bool = True,
    show_coverage: bool = True,
    show_account: bool = True,
    key_prefix: str = "",
) -> dict:
    """
    Renders dimension filter selectboxes. Returns dict with selected IDs.
    """
    entities_df = get_dim_entities()
    departments_df = get_dim_departments()
    coverages_df = get_dim_coverages()
    accounts_df = get_dim_accounts()

    entity_opts = build_select_options(entities_df, "entity_id", "entity_name")
    dept_opts = build_select_options(departments_df, "department_id", "department_name")
    cov_opts = build_select_options(coverages_df, "coverage_id", "coverage_name")
    acc_opts = build_select_options(accounts_df, "account_id", "account_name")

    cols = col_layout or st.columns(4)
    result = {}
    col_idx = 0

    if show_entity and col_idx < len(cols):
        with cols[col_idx]:
            sel = st.selectbox(
                "Entitate",
                [o[0] for o in entity_opts],
                key=f"{key_prefix}entity_sel",
            )
            result["entity_id"] = dict(entity_opts)[sel]
        col_idx += 1

    if show_department and col_idx < len(cols):
        with cols[col_idx]:
            sel = st.selectbox(
                "Departament",
                [o[0] for o in dept_opts],
                key=f"{key_prefix}dept_sel",
            )
            result["department_id"] = dict(dept_opts)[sel]
        col_idx += 1

    if show_coverage and col_idx < len(cols):
        with cols[col_idx]:
            sel = st.selectbox(
                "Coverage",
                [o[0] for o in cov_opts],
                key=f"{key_prefix}cov_sel",
            )
            result["coverage_id"] = dict(cov_opts)[sel]
        col_idx += 1

    if show_account and col_idx < len(cols):
        with cols[col_idx]:
            sel = st.selectbox(
                "Cont",
                [o[0] for o in acc_opts],
                key=f"{key_prefix}acc_sel",
            )
            result["account_id"] = dict(acc_opts)[sel]
        col_idx += 1

    return result


def fiscal_year_selector(
    available_years: list[int],
    key: str = "fiscal_year_sel",
    label: str = "An fiscal",
) -> int | None:
    if not available_years:
        return None
    selected = st.selectbox(label, available_years, key=key)
    st.session_state["active_fiscal_year"] = selected
    return selected


def period_selector(
    key: str = "period_sel",
    label: str = "Perioadă",
    include_all: bool = True,
) -> int | None:
    from streamlit_app.utils.formatters import MONTH_NAMES_RO_FULL
    options = [(name, i) for i, name in MONTH_NAMES_RO_FULL.items()]
    if include_all:
        options = [("Toate perioadele", None)] + options
    labels = [o[0] for o in options]
    values = [o[1] for o in options]
    sel = st.selectbox(label, labels, key=key)
    return values[labels.index(sel)]
