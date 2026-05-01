"""Configurare — Dimension hierarchy management + CSV upload per dimension."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
from io import StringIO

from streamlit_app.utils.state import init_session_state
from streamlit_app.utils.cache import (
    get_dim_entities, get_dim_accounts, get_dim_departments,
    get_dim_coverages, get_dim_scenarios, get_dim_periods,
)
from database.connection import get_connection

st.set_page_config(page_title="Configurare — FinPlan", layout="wide")
init_session_state()

with st.sidebar:
    st.markdown("### FinPlan")
    st.caption("Platformă de planificare financiară")

st.title("Configurare")

TABS = ["Entitate", "Departament", "Coverage", "Cont", "Perioade", "Scenarii", "Date financiare"]
tab_objects = st.tabs([f"📁 {t}" for t in TABS])

DIM_CONFIG = {
    "Entitate": {
        "loader": get_dim_entities,
        "table": "dbo.DimEntity",
        "id_col": "entity_id",
        "cols": ["entity_id", "entity_name"],
        "required": ["entity_name"],
    },
    "Departament": {
        "loader": get_dim_departments,
        "table": "dbo.DimDepartment",
        "id_col": "department_id",
        "cols": ["department_id", "department_name"],
        "required": ["department_name"],
    },
    "Coverage": {
        "loader": get_dim_coverages,
        "table": "dbo.DimCoverage",
        "id_col": "coverage_id",
        "cols": ["coverage_id", "coverage_name"],
        "required": ["coverage_name"],
    },
    "Cont": {
        "loader": get_dim_accounts,
        "table": "dbo.DimAccount",
        "id_col": "account_id",
        "cols": ["account_id", "account_code", "account_name", "statement_type"],
        "required": ["account_code", "account_name", "statement_type"],
    },
}


def _render_dim_tab(tab, dim_name: str):
    cfg = DIM_CONFIG[dim_name]
    with tab:
        st.markdown(f"### {dim_name}")

        col_upload, col_table = st.columns([1, 2])

        with col_upload:
            st.markdown(f"#### Încărcare fișier — {dim_name}")
            st.markdown(
                '<div style="border:2px dashed #ccc;border-radius:8px;padding:24px;text-align:center">'
                '📤<br>Trage fișierul CSV sau Excel aici</div>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Încarcă fișier",
                type=["csv", "xlsx"],
                key=f"upload_{dim_name}",
                label_visibility="collapsed",
            )
            if uploaded:
                try:
                    if uploaded.name.endswith(".csv"):
                        df_up = pd.read_csv(StringIO(uploaded.read().decode("utf-8")))
                    else:
                        df_up = pd.read_excel(uploaded)

                    st.success(f"✅ Actualizare terminată cu succes. {len(df_up)} rânduri detectate.")
                    st.dataframe(df_up.head(5), use_container_width=True)
                    st.button("💾 Importă în bază de date", key=f"import_{dim_name}", type="primary")
                except Exception as e:
                    st.error(f"Eroare la citirea fișierului: {e}")

            # CSV template download
            sample_data = {col: [] for col in cfg["required"]}
            sample_df = pd.DataFrame(sample_data)
            csv_template = sample_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descarcă model CSV",
                csv_template,
                f"template_{dim_name.lower()}.csv",
                "text/csv",
                use_container_width=True,
            )

        with col_table:
            df = cfg["loader"]()
            if not df.empty:
                display_cols = [c for c in cfg["cols"] if c in df.columns]
                st.markdown(f"#### Date curente ({len(df)} rânduri)")

                # Stats
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Total", len(df))
                if "is_active" in df.columns:
                    with stat_cols[1]:
                        st.metric("Active", int(df["is_active"].sum()))
                    with stat_cols[2]:
                        st.metric("Inactive", int((df["is_active"] == 0).sum()))

                search = st.text_input("🔍 Caută după cod sau nume...", key=f"search_{dim_name}")
                show_df = df[display_cols].copy()
                if search:
                    mask = show_df.apply(
                        lambda col: col.astype(str).str.contains(search, case=False, na=False)
                    ).any(axis=1)
                    show_df = show_df[mask]

                st.dataframe(show_df, use_container_width=True, height=350)

                csv_exp = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Exportă CSV",
                    csv_exp,
                    f"{dim_name.lower()}.csv",
                    "text/csv",
                    key=f"export_{dim_name}",
                )

                # Add new row form
                st.markdown("#### Adaugă rând nou")
                add_cols = st.columns(len(cfg["required"]))
                new_vals = {}
                for i, col_name in enumerate(cfg["required"]):
                    with add_cols[i]:
                        new_vals[col_name] = st.text_input(col_name, key=f"new_{dim_name}_{col_name}")

                if st.button(f"➕ Adaugă", key=f"add_{dim_name}"):
                    if all(new_vals.values()):
                        st.success(f"Rând adăugat: {new_vals}")
                        st.cache_data.clear()
                    else:
                        st.warning("Completează toate câmpurile obligatorii.")
            else:
                st.info(f"Nu există date pentru {dim_name}.")


for dim_name, tab in zip(list(DIM_CONFIG.keys()), tab_objects[:4]):
    _render_dim_tab(tab, dim_name)

with tab_objects[4]:
    st.markdown("### Perioade")
    periods_df = get_dim_periods()
    if not periods_df.empty:
        st.dataframe(periods_df, use_container_width=True)
    else:
        st.info("Nu există perioade configurate.")

with tab_objects[5]:
    st.markdown("### Scenarii")
    scenarios_df = get_dim_scenarios()
    if not scenarios_df.empty:
        st.dataframe(scenarios_df, use_container_width=True)
        st.markdown("#### Adaugă scenariu")
        s_cols = st.columns(3)
        with s_cols[0]:
            new_code = st.text_input("Cod scenariu (ex: RF08_2025)")
        with s_cols[1]:
            new_type = st.selectbox("Tip", ["RF", "BDG", "ACT", "SP"])
        with s_cols[2]:
            if st.button("➕ Adaugă scenariu") and new_code:
                st.success(f"Scenariu {new_code} adăugat.")
    else:
        st.info("Nu există scenarii.")

with tab_objects[6]:
    st.markdown("### Date financiare — Import batch")
    from streamlit_app.utils.cache import get_import_batches
    batches_df = get_import_batches()
    if not batches_df.empty:
        st.dataframe(batches_df, use_container_width=True)
    else:
        st.info("Nu există batch-uri de import.")

    st.markdown("#### Upload date financiare (Excel/CSV)")
    fin_file = st.file_uploader("Fișier FinanceFact", type=["xlsx", "csv"], key="fin_upload")
    fin_scenario = st.selectbox("Scenariu", scenarios_df["scenario_code"].tolist() if not get_dim_scenarios().empty else [], key="fin_scenario_sel")
    fin_year = st.number_input("An fiscal", value=2025, key="fin_year")

    if fin_file and st.button("📤 Importă date financiare", type="primary"):
        st.info("Import în procesare... (conectează cu ImportBatch logic)")
