"""Model Training — Train, compare and manage ML models."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import joblib

from streamlit_app.utils.state import init_session_state
from streamlit_app.utils.cache import get_actuals
from streamlit_app.components.charts import feature_importance_bar
from ml.feature_engineering import build_feature_dataset, get_training_columns
from ml.train_model import standard_mode_training, advanced_mode_training, save_trained_model
from ml.explainability import get_model_feature_importance
from config.settings import MODEL_FOLDER

st.set_page_config(page_title="Model Training — FinPlan", layout="wide")
init_session_state()

with st.sidebar:
    st.markdown("### FinPlan")
    st.caption("Platformă de planificare financiară")

st.title("Model Training")
st.caption("Antrenează și compară modele ML pentru forecast financiar")

# ── Current models ────────────────────────────────────────────────────────────
st.markdown("### Modele curente")
model_names = ["random_forest", "gradient_boosting", "linear_regression"]
model_status = {}
for mn in model_names:
    path = os.path.join(MODEL_FOLDER, f"{mn}.pkl")
    model_status[mn] = {
        "exists": os.path.exists(path),
        "modified": pd.Timestamp(os.path.getmtime(path), unit="s").strftime("%Y-%m-%d %H:%M") if os.path.exists(path) else "—",
    }

stat_cols = st.columns(len(model_names))
display_names = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "linear_regression": "Regresie Liniară",
}
for i, (mn, stat) in enumerate(model_status.items()):
    with stat_cols[i]:
        icon = "✅" if stat["exists"] else "❌"
        st.metric(
            f"{icon} {display_names[mn]}",
            "Disponibil" if stat["exists"] else "Ne-antrenat",
            stat["modified"],
        )

st.markdown("---")

# ── Training config ────────────────────────────────────────────────────────────
st.markdown("### Configurare antrenament")

mode_col, data_col = st.columns(2)
with mode_col:
    training_mode = st.radio(
        "Mod antrenament",
        ["Standard (compară toate modelele)", "Avansat (un singur model)"],
        key="training_mode",
    )

with data_col:
    amount_basis = st.selectbox("Baza de calcul", ["rpt", "lcl", "ccy"], key="train_basis")
    min_history = st.slider("Istoric minim (luni)", 3, 24, 6, key="train_min_hist")
    include_prev_rf = st.checkbox("Include features RF anterior (model operațional)", value=False, key="train_prev_rf")

if "Avansat" in training_mode:
    adv_cols = st.columns(3)
    with adv_cols[0]:
        adv_model = st.selectbox("Model", ["random_forest", "gradient_boosting", "linear_regression"])
    with adv_cols[1]:
        n_estimators = st.number_input("n_estimators", value=200, min_value=10, max_value=1000)
    with adv_cols[2]:
        max_depth = st.number_input("max_depth", value=8, min_value=1, max_value=30)

st.markdown("---")

# ── Train button ──────────────────────────────────────────────────────────────
if st.button("🚀 Pornește antrenamentul", type="primary"):
    with st.spinner("Se încarcă datele..."):
        actuals_df = get_actuals()
        if actuals_df.empty:
            st.error("Nu există date actuale. Încarcă date înainte de antrenament.")
            st.stop()

    progress = st.progress(0, text="Feature engineering...")
    try:
        feature_df = build_feature_dataset(
            actuals_df,
            amount_basis=amount_basis,
            min_history=min_history,
            include_prev_rf=include_prev_rf,
        )

        progress.progress(25, text="Antrenament modele...")

        if "Standard" in training_mode:
            results = standard_mode_training(feature_df)
            progress.progress(80, text="Salvare modele...")

            recommended = results["recommended_model_name"]
            results_df = results["results_table"]

            for mn, model in results.items():
                if mn == "results_table" or not hasattr(model, "predict"):
                    continue

            # Save all trained models
            from ml.train_model import get_available_models
            from sklearn.base import clone
            from ml.train_model import prepare_training_data
            df_train, X, y, _ = prepare_training_data(feature_df)
            for mn, m in get_available_models().items():
                m_clone = clone(m)
                m_clone.fit(X, y)
                save_trained_model(m_clone, mn)

            progress.progress(100, text="Antrenament complet!")
            st.success(f"✅ Antrenament complet! Model recomandat: **{recommended}**")

            st.markdown("### Rezultate comparație modele")
            display_results = results_df.copy()
            for col in display_results.columns:
                if col not in ["model_name"]:
                    try:
                        display_results[col] = display_results[col].apply(
                            lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                        )
                    except Exception:
                        pass
            st.dataframe(display_results, use_container_width=True)

        else:
            params = {"n_estimators": n_estimators, "max_depth": max_depth}
            results = advanced_mode_training(feature_df, adv_model, params)
            progress.progress(80, text="Salvare model...")
            save_trained_model(results["model"], adv_model)
            progress.progress(100, text="Antrenament complet!")
            st.success(f"✅ Model {adv_model} antrenat!")

            metrics = results["metrics"]
            m_cols = st.columns(4)
            metric_display = [
                ("RMSE_mean", "RMSE mediu"),
                ("MAPE_mean", "MAPE mediu"),
                ("WAPE_mean", "WAPE mediu"),
                ("Bias_mean", "Bias"),
            ]
            for i, (key, label) in enumerate(metric_display):
                val = metrics.get(key)
                with m_cols[i]:
                    st.metric(label, f"{val:.4f}" if val is not None and pd.notna(val) else "—")

        # Feature importance for recommended model
        st.markdown("---")
        st.markdown("### Importanța features")

        try:
            model_path = os.path.join(MODEL_FOLDER, f"{recommended if 'Standard' in training_mode else adv_model}.pkl")
            if os.path.exists(model_path):
                loaded_model = joblib.load(model_path)
                importance_df = get_model_feature_importance(
                    loaded_model,
                    get_training_columns(include_prev_rf=include_prev_rf),
                    recommended if "Standard" in training_mode else adv_model,
                )
                fig = feature_importance_bar(importance_df)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Nu s-a putut afișa importanța features: {e}")

        st.cache_data.clear()

    except Exception as e:
        progress.progress(0)
        st.error(f"Eroare la antrenament: {str(e)}")
        st.exception(e)

# ── Delete models ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("⚠️ Gestionare modele (avansat)"):
    del_model = st.selectbox("Șterge model", model_names, key="del_model_sel")
    if st.button("🗑 Șterge model selectat", type="secondary"):
        path = os.path.join(MODEL_FOLDER, f"{del_model}.pkl")
        if os.path.exists(path):
            os.remove(path)
            st.success(f"Model {del_model} șters.")
            st.rerun()
        else:
            st.warning("Modelul nu există.")
