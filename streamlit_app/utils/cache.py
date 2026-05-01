"""Cached data loaders for Streamlit."""
import streamlit as st
import pandas as pd
import sys
import os

# Ensure project root is on path when running from streamlit_app/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from database.load_data import (
    load_dim_entities, load_dim_accounts, load_dim_departments,
    load_dim_coverages, load_dim_scenarios, load_dim_periods,
    load_fact_by_scenario_code, load_fact_multi_scenario,
    load_actuals_for_ml, load_active_forecast_runs,
    load_business_warnings, load_import_batches,
    load_calculation_rules,
)
from database.load_assumptions import load_assumptions_by_scenario


@st.cache_data(ttl=600, show_spinner=False)
def get_dim_entities() -> pd.DataFrame:
    return load_dim_entities()


@st.cache_data(ttl=600, show_spinner=False)
def get_dim_accounts() -> pd.DataFrame:
    return load_dim_accounts()


@st.cache_data(ttl=600, show_spinner=False)
def get_dim_departments() -> pd.DataFrame:
    return load_dim_departments()


@st.cache_data(ttl=600, show_spinner=False)
def get_dim_coverages() -> pd.DataFrame:
    return load_dim_coverages()


@st.cache_data(ttl=600, show_spinner=False)
def get_dim_scenarios() -> pd.DataFrame:
    return load_dim_scenarios()


@st.cache_data(ttl=600, show_spinner=False)
def get_dim_periods() -> pd.DataFrame:
    return load_dim_periods()


@st.cache_data(ttl=60, show_spinner=False)
def get_fact_for_scenario(scenario_code: str) -> pd.DataFrame:
    return load_fact_by_scenario_code(scenario_code)


@st.cache_data(ttl=60, show_spinner=False)
def get_fact_multi(scenario_codes: tuple[str, ...]) -> pd.DataFrame:
    return load_fact_multi_scenario(list(scenario_codes))


@st.cache_data(ttl=60, show_spinner=False)
def get_actuals() -> pd.DataFrame:
    return load_actuals_for_ml()


@st.cache_data(ttl=30, show_spinner=False)
def get_forecast_runs() -> pd.DataFrame:
    return load_active_forecast_runs()


@st.cache_data(ttl=30, show_spinner=False)
def get_business_warnings(scenario_code: str) -> pd.DataFrame:
    return load_business_warnings(scenario_code)


@st.cache_data(ttl=60, show_spinner=False)
def get_assumptions(scenario_code: str) -> pd.DataFrame:
    return load_assumptions_by_scenario(scenario_code)


@st.cache_data(ttl=120, show_spinner=False)
def get_import_batches() -> pd.DataFrame:
    return load_import_batches()


@st.cache_data(ttl=600, show_spinner=False)
def get_calculation_rules() -> pd.DataFrame:
    return load_calculation_rules()


def build_dim_lookup(df: pd.DataFrame, id_col: str, name_col: str) -> dict:
    """Returns {id: name} dict from a dimension dataframe."""
    if df.empty or id_col not in df.columns or name_col not in df.columns:
        return {}
    return dict(zip(df[id_col], df[name_col]))


def build_select_options(df: pd.DataFrame, id_col: str, name_col: str, add_all: bool = True) -> list[tuple]:
    """Returns [(label, id), ...] for selectbox use. Optionally prepends 'Toate'."""
    if df.empty:
        return [("Toate", None)] if add_all else []
    options = list(zip(df[name_col].fillna("?"), df[id_col]))
    if add_all:
        options = [("Toate", None)] + options
    return options
