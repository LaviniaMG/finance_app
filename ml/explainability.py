import numpy as np
import pandas as pd


def _normalize_importance_series(importance_series: pd.Series) -> pd.Series:
    s = importance_series.copy().astype(float)

    if s.abs().sum() == 0:
        return s

    return s / s.abs().sum()


def get_model_feature_importance(model, training_columns: list, model_name: str) -> pd.DataFrame:
    """
    Returns feature importance depending on model type.
    - tree models -> feature_importances_
    - linear regression -> coefficients
    """
    model_name = model_name.lower().strip()

    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=training_columns)

    elif hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        importance = pd.Series(coef, index=training_columns).abs()

    else:
        raise ValueError(f"Model '{model_name}' does not expose importances/coefs in current implementation.")

    importance = _normalize_importance_series(importance)

    df = pd.DataFrame({
        "feature_name": importance.index,
        "importance_score": importance.values
    }).sort_values("importance_score", ascending=False).reset_index(drop=True)

    return df


def get_top_feature_drivers(importance_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    return importance_df.head(top_n).copy()


def get_prediction_context_from_row(prediction_row: pd.Series) -> dict:
    """
    Extracts useful business context from one forecast row.
    """
    return {
        "period_id": prediction_row.get("period_id"),
        "statement_type": prediction_row.get("statement_type"),
        "mtd_rpt_amount": prediction_row.get("mtd_rpt_amount"),
        "ytd_rpt_amount": prediction_row.get("ytd_rpt_amount"),
        "run_rate_recent": prediction_row.get("run_rate_recent"),
        "volatility_ratio": prediction_row.get("volatility_ratio"),
        "month_number": prediction_row.get("period_id", 0) % 100 if pd.notna(prediction_row.get("period_id")) else None,
    }


def explain_growth_vs_run_rate(prediction_row: pd.Series) -> str:
    run_rate = prediction_row.get("run_rate_recent", np.nan)
    current = prediction_row.get("mtd_rpt_amount", np.nan)

    if pd.isna(run_rate) or pd.isna(current) or run_rate == 0:
        return "Nu există suficient istoric pentru a compara forecastul cu run-rate-ul recent."

    pct = (current - run_rate) / abs(run_rate)

    if pct > 0.20:
        return f"Forecastul este peste run-rate-ul recent cu aproximativ {pct:.1%}, ceea ce indică o creștere semnificativă."
    if pct < -0.20:
        return f"Forecastul este sub run-rate-ul recent cu aproximativ {abs(pct):.1%}, ceea ce indică o scădere semnificativă."
    return f"Forecastul este apropiat de run-rate-ul recent, diferența fiind de aproximativ {pct:.1%}."


def explain_volatility(prediction_row: pd.Series) -> str:
    vol = prediction_row.get("volatility_ratio", np.nan)

    if pd.isna(vol):
        return "Nu există suficiente date pentru evaluarea stabilității seriei."

    if vol < 0.10:
        return "Seria istorică este stabilă, ceea ce crește încrederea în predicție."
    if vol < 0.25:
        return "Seria are volatilitate moderată, iar predicția trebuie interpretată cu atenție."
    return "Seria este volatilă, ceea ce reduce stabilitatea și încrederea în forecast."


def explain_time_effect(prediction_row: pd.Series) -> str:
    month = prediction_row.get("period_id", np.nan)
    if pd.isna(month):
        return "Nu a putut fi determinată componenta calendaristică."

    month_number = int(month) % 100

    if month_number in [3, 6, 9, 12]:
        return "Perioada forecastată este la final de trimestru, ceea ce poate influența comportamentul valorilor financiare."
    if month_number == 12:
        return "Perioada forecastată este la final de an, unde pot apărea ajustări și efecte sezoniere specifice."
    return "Nu există un indicator calendaristic puternic de final de trimestru sau final de an pentru această perioadă."


def build_natural_language_explanation(
    prediction_row: pd.Series,
    top_features_df: pd.DataFrame,
    model_name: str
) -> str:
    """
    Business-friendly explanation for one forecasted row.
    """
    top_features = top_features_df["feature_name"].head(3).tolist()

    if not top_features:
        top_features_text = "factori generali ai seriei istorice"
    elif len(top_features) == 1:
        top_features_text = top_features[0]
    elif len(top_features) == 2:
        top_features_text = f"{top_features[0]} și {top_features[1]}"
    else:
        top_features_text = f"{top_features[0]}, {top_features[1]} și {top_features[2]}"

    statement_type = str(prediction_row.get("statement_type", "UNKNOWN")).upper()

    if statement_type == "BS":
        statement_text = "Pentru acest cont de bilanț, modelul a previzionat evoluția pe baza dinamicii valorilor YTD."
    else:
        statement_text = "Pentru acest cont de tip P&L, modelul a previzionat valoarea lunară pe baza comportamentului MTD."

    growth_text = explain_growth_vs_run_rate(prediction_row)
    volatility_text = explain_volatility(prediction_row)
    time_text = explain_time_effect(prediction_row)

    return (
        f"Modelul selectat este {model_name}. "
        f"Principalii factori care influențează predicția sunt {top_features_text}. "
        f"{statement_text} {growth_text} {volatility_text} {time_text}"
    )


def explain_model_recommendation(results_table: pd.DataFrame, recommended_model_name: str) -> str:
    """
    Explains why a model was recommended.
    """
    if results_table.empty:
        return "Nu există rezultate suficiente pentru recomandarea unui model."

    row = results_table[results_table["model_name"] == recommended_model_name]
    if row.empty:
        return "Modelul recomandat nu a fost găsit în tabelul de rezultate."

    row = row.iloc[0]

    rmse = row.get("RMSE_mean", np.nan)
    mae = row.get("MAE_mean", np.nan)

    return (
        f"Modelul recomandat este {recommended_model_name} deoarece a obținut una dintre cele mai bune performanțe "
        f"în backtesting, cu RMSE mediu de {rmse:.2f} și MAE mediu de {mae:.2f}, "
        f"menținând în același timp un nivel bun de interpretabilitate."
    )


def explain_forecast_output(
    forecast_df: pd.DataFrame,
    model,
    model_name: str,
    training_columns: list,
    results_table: pd.DataFrame | None = None,
    top_n_features: int = 5
) -> dict:
    """
    Main explainability function.

    Returns:
    - feature_importance_df
    - top_features_df
    - model_recommendation_text
    - row_level_explanations_df
    """
    feature_importance_df = get_model_feature_importance(
        model=model,
        training_columns=training_columns,
        model_name=model_name
    )

    top_features_df = get_top_feature_drivers(feature_importance_df, top_n=top_n_features)

    if results_table is not None:
        model_recommendation_text = explain_model_recommendation(results_table, model_name)
    else:
        model_recommendation_text = f"Modelul utilizat este {model_name}."

    explanations = []
    for _, row in forecast_df.iterrows():
        explanation_text = build_natural_language_explanation(
            prediction_row=row,
            top_features_df=top_features_df,
            model_name=model_name
        )

        explanations.append({
            "fiscal_year": row.get("fiscal_year"),
            "period_id": row.get("period_id"),
            "scenario_code": row.get("scenario_code"),
            "mtd_rpt_amount": row.get("mtd_rpt_amount"),
            "ytd_rpt_amount": row.get("ytd_rpt_amount"),
            "confidence_level": row.get("confidence_level"),
            "business_warning_text": row.get("business_warning_text"),
            "explanation_text": explanation_text
        })

    row_level_explanations_df = pd.DataFrame(explanations)

    return {
        "feature_importance_df": feature_importance_df,
        "top_features_df": top_features_df,
        "model_recommendation_text": model_recommendation_text,
        "row_level_explanations_df": row_level_explanations_df,
    }