import pandas as pd
from database.connection import get_connection


def validate_explanations_dataframe(explanations_df: pd.DataFrame):
    required_columns = [
        "fiscal_year",
        "period_id",
        "scenario_code",
        "explanation_text",
    ]

    missing = [c for c in required_columns if c not in explanations_df.columns]
    if missing:
        raise ValueError(f"explanations_df is missing required columns: {missing}")


def save_explanations_to_db(
    explanations_df: pd.DataFrame,
    top_features_df: pd.DataFrame,
    forecast_run_id: int
):
    if explanations_df.empty:
        raise ValueError("explanations_df is empty. Nothing to save.")

    validate_explanations_dataframe(explanations_df)

    top_features = top_features_df["feature_name"].head(3).tolist()
    top_feature_1 = top_features[0] if len(top_features) > 0 else None
    top_feature_2 = top_features[1] if len(top_features) > 1 else None
    top_feature_3 = top_features[2] if len(top_features) > 2 else None

    conn = get_connection()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO dbo.ForecastExplanation
        (
            forecast_run_id,
            fiscal_year,
            period_id,
            scenario_code,
            top_feature_1,
            top_feature_2,
            top_feature_3,
            confidence_level,
            business_warning_text,
            explanation_text,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
    """

    for _, row in explanations_df.iterrows():
        cursor.execute(
            insert_sql,
            int(forecast_run_id),
            int(row["fiscal_year"]),
            int(row["period_id"]),
            str(row["scenario_code"]),
            top_feature_1,
            top_feature_2,
            top_feature_3,
            row["confidence_level"] if "confidence_level" in row and pd.notna(row["confidence_level"]) else None,
            row["business_warning_text"] if "business_warning_text" in row and pd.notna(row["business_warning_text"]) else None,
            str(row["explanation_text"]),
        )

    conn.commit()
    conn.close()

    return len(explanations_df)