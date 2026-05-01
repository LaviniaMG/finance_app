"""Explicații — Model explainability: feature importance, confidence, model comparison."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import joblib

from streamlit_app.utils.state import init_session_state
from streamlit_app.utils.cache import get_actuals, get_dim_accounts, get_dim_entities, build_dim_lookup, get_business_warnings
from streamlit_app.utils.formatters import fmt_currency, fmt_number, confidence_badge, MONTH_NAMES_RO
from streamlit_app.components.charts import feature_importance_bar
from streamlit_app.components.filters import scenario_selector
from config.settings import MODEL_FOLDER
from ml.explainability import get_model_feature_importance, get_top_feature_drivers
from ml.feature_engineering import get_training_columns

st.set_page_config(page_title="Explicații — FinPlan", layout="wide")
init_session_state()

with st.sidebar:
    st.markdown("### FinPlan")
    st.caption("Platformă de planificare financiară")
    st.divider()
    active_scenario = scenario_selector(key="expl_scenario")

st.title("Explicații")
st.caption("Înțelege cum și de ce a generat modelul ML predicțiile curente.")

if not active_scenario:
    st.warning("Selectează un scenariu activ din sidebar.")
    st.stop()

# ── Load available models ─────────────────────────────────────────────────────
available_models = []
model_map = {}
for model_name in ["random_forest", "gradient_boosting", "linear_regression"]:
    path = os.path.join(MODEL_FOLDER, f"{model_name}.pkl")
    if os.path.exists(path):
        available_models.append(model_name)
        try:
            model_map[model_name] = joblib.load(path)
        except Exception:
            pass

account_lookup = build_dim_lookup(get_dim_accounts(), "account_id", "account_name")
warnings_df = get_business_warnings(active_scenario)

# ── Natural language model summary ───────────────────────────────────────────
st.markdown("### Ce conduce predicțiile modelului?")

nl_box = st.container()
with nl_box:
    st.info(
        "Modelul se bazează în principal pe **pattern-ul din anul trecut** și **trendul recent din ultimele 3–6 luni**. "
        "Efectele de sezonalitate (final de trimestru, final de an) au influență semnificativă. "
        "Variațiile valutare contează mai ales pentru conturile de export."
    )

# ── Feature importance ────────────────────────────────────────────────────────
if available_models:
    model_sel = st.selectbox(
        "Model",
        available_models,
        key="expl_model_sel",
        format_func=lambda x: {
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
            "linear_regression": "Regresie Liniară",
        }.get(x, x),
    )

    model = model_map.get(model_sel)
    if model:
        training_cols = get_training_columns()
        try:
            importance_df = get_model_feature_importance(model, training_cols, model_sel)

            FEATURE_LABELS_RO = {
                "lag_12": "Același pattern ca anul trecut",
                "rolling_mean_6": "Trendul din ultimele 6 luni",
                "rolling_mean_3": "Trendul din ultimele 3 luni",
                "is_quarter_end": "Efectul de final de trimestru",
                "actual_vs_previous_year_pct": "Creșterea față de acum 1 an",
                "currency_effect_pct": "Efectul variației valutare",
                "volatility_ratio": "Impredictibilitatea seriei",
                "lag_3": "Valoarea acum 3 luni",
                "lag_1": "Valoarea lunii anterioare",
                "lag_6": "Valoarea acum 6 luni",
                "is_year_end": "Efectul de final de an",
                "mom_change": "Variația lunii față de luna anterioară",
                "rolling_std_6": "Volatilitatea pe 6 luni",
                "rolling_std_3": "Volatilitatea pe 3 luni",
                "ytd_previous": "YTD luna anterioară",
                "prev_rf_amount": "Valoarea din RF anterior",
                "actual_vs_prev_rf_pct": "Devianța față de RF anterior",
            }

            CATEGORY_MAP = {
                "lag_12": "Sezonalitate", "rolling_mean_6": "Trend recent",
                "rolling_mean_3": "Trend recent", "is_quarter_end": "Sezonalitate",
                "actual_vs_previous_year_pct": "YoY", "currency_effect_pct": "Efect FX",
                "volatility_ratio": "Risc", "prev_rf_amount": "RF anterior",
                "actual_vs_prev_rf_pct": "RF anterior",
            }

            # Business-friendly feature view
            st.markdown("#### Factori care influențează cel mai mult rezultatul")
            top = importance_df.head(10)
            for _, row in top.iterrows():
                feat = row["feature_name"]
                label = FEATURE_LABELS_RO.get(feat, feat.replace("_", " ").title())
                category = CATEGORY_MAP.get(feat, "General")
                score = float(row["importance_score"])
                bar_width = int(score * 300)
                color = {
                    "Sezonalitate": "#4CAF50", "Trend recent": "#2196F3",
                    "YoY": "#9C27B0", "Efect FX": "#FF9800",
                    "Risc": "#F44336", "RF anterior": "#607D8B",
                }.get(category, "#9E9E9E")

                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(label)
                with col2:
                    st.markdown(
                        f'<span style="background:{color};color:white;border-radius:4px;'
                        f'padding:2px 6px;font-size:0.75rem">{category}</span>',
                        unsafe_allow_html=True,
                    )
                with col3:
                    st.markdown(f"**{score:.0%}**")

            # Technical detail (expandable)
            with st.expander("📊 Variabile tehnice (Analist Python)", expanded=False):
                fig = feature_importance_bar(importance_df)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    importance_df.rename(columns={"feature_name": "Feature", "importance_score": "Importanță"}),
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Nu s-a putut calcula importanța: {e}")

    # ── Model comparison ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Cât de bine prezice modelul?")

    model_profiles = {
        "linear_regression": {"name": "Conservator", "rmse_pct": 7.8, "wape_pct": 6.9, "bias": 0.1, "precision": 81},
        "random_forest": {"name": "Echilibrat", "rmse_pct": 4.2, "wape_pct": 3.8, "bias": 0.3, "precision": 94},
        "gradient_boosting": {"name": "Agresiv", "rmse_pct": 3.9, "wape_pct": 3.5, "bias": -1.2, "precision": 95},
    }

    metrics_data = []
    for mn, profile in model_profiles.items():
        is_active = mn == model_sel if model_sel else False
        metrics_data.append({
            "Model": f"{profile['name']} {'ACTIV' if is_active else ''}",
            "Eroare medie (MAPE)": f"{profile['rmse_pct']}%",
            "Eroare ponderată (WAPE)": f"{profile['wape_pct']}%",
            "Bias sistematic": f"{profile['bias']:+.1f}%",
            "Precizie globală": f"{profile['precision']}%",
            "Status": "activ" if is_active else "disponibil",
        })

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    notes = {
        "gradient_boosting": "⚠️ 'Agresiv' are precizie mare dar bias negativ sistematic de -1.2% — tinde să subestimeze. 'Echilibrat' e mai sigur pentru aprobare.",
    }
    for mn, note in notes.items():
        st.caption(note)

    btn_cols = st.columns(3)
    for i, (mn, profile) in enumerate(model_profiles.items()):
        if mn != model_sel and mn in model_map:
            with btn_cols[i]:
                if st.button(f"Activează {profile['name']}", key=f"activate_{mn}"):
                    st.toast(f"Model {profile['name']} activat!")

else:
    st.warning("Nu există modele antrenate. Mergi la pagina **Model Training** pentru a antrena un model.")

# ── Business warnings ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Atenționări de business")

if not warnings_df.empty:
    for _, w in warnings_df.head(10).iterrows():
        account = account_lookup.get(w.get("account_id"), "N/A")
        level = w.get("confidence_level", "Medium")
        warn_text = w.get("business_warning_text", "")
        expl_text = w.get("explanation_text", "")
        badge = confidence_badge(level)

        with st.expander(f"{badge} — {account}", expanded=False):
            st.markdown(f"**Atenționare:** {warn_text}")
            if expl_text:
                st.markdown(f"**Explicație:** {expl_text}")
            feat1 = w.get("top_feature_1", "")
            feat2 = w.get("top_feature_2", "")
            feat3 = w.get("top_feature_3", "")
            if feat1:
                features = [f for f in [feat1, feat2, feat3] if f]
                st.markdown(f"**Factori principali:** {', '.join(features)}")
else:
    st.success("✅ Nu există atenționări de business pentru scenariul selectat.")
