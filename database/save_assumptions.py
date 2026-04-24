import pandas as pd
from database.connection import get_connection


ALLOWED_ASSUMPTION_TYPES = {
    "growth_pct",
    "inflation_pct",
    "fixed_value",
    "delta_value",
    "fx_adjustment_pct",
    "headcount_growth_pct",
}


ALLOWED_INPUT_SOURCES = {
    "MANUAL",
    "AI",
    "SUGGESTED",
}


def validate_assumptions_dataframe(assumptions_df: pd.DataFrame):
    required_columns = [
        "scenario_code",
        "assumption_type",
        "assumption_value",
        "input_source",
    ]

    missing = [c for c in required_columns if c not in assumptions_df.columns]
    if missing:
        raise ValueError(f"assumptions_df is missing required columns: {missing}")

    invalid_types = set(assumptions_df["assumption_type"].dropna().unique()) - ALLOWED_ASSUMPTION_TYPES
    if invalid_types:
        raise ValueError(f"Invalid assumption_type values found: {sorted(invalid_types)}")

    invalid_sources = {str(x).upper() for x in assumptions_df["input_source"].dropna().unique()} - ALLOWED_INPUT_SOURCES
    if invalid_sources:
        raise ValueError(f"Invalid input_source values found: {sorted(invalid_sources)}")


def normalize_assumptions_dataframe(assumptions_df: pd.DataFrame) -> pd.DataFrame:
    df = assumptions_df.copy()

    optional_cols_with_defaults = {
        "entity_id": None,
        "coverage_id": None,
        "department_id": None,
        "segment_id": None,
        "account_id": None,
        "fiscal_year": None,
        "period_from": None,
        "period_to": None,
        "assumption_text": None,
        "priority_order": 1,
        "is_active": 1,
    }

    for col, default_value in optional_cols_with_defaults.items():
        if col not in df.columns:
            df[col] = default_value

    df["scenario_code"] = df["scenario_code"].astype(str).str.upper().str.strip()
    df["assumption_type"] = df["assumption_type"].astype(str).str.strip()
    df["input_source"] = df["input_source"].astype(str).str.upper().str.strip()

    return df


def save_assumptions_to_db(assumptions_df: pd.DataFrame) -> int:
    if assumptions_df.empty:
        raise ValueError("assumptions_df is empty. Nothing to save.")

    df = normalize_assumptions_dataframe(assumptions_df)
    validate_assumptions_dataframe(df)

    conn = get_connection()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO dbo.ForecastAssumption
        (
            scenario_code,
            entity_id,
            coverage_id,
            department_id,
            segment_id,
            account_id,
            fiscal_year,
            period_from,
            period_to,
            assumption_type,
            assumption_value,
            input_source,
            assumption_text,
            priority_order,
            is_active,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
    """

    rows_saved = 0

    for _, row in df.iterrows():
        cursor.execute(
            insert_sql,
            row["scenario_code"],
            int(row["entity_id"]) if pd.notna(row["entity_id"]) else None,
            int(row["coverage_id"]) if pd.notna(row["coverage_id"]) else None,
            int(row["department_id"]) if pd.notna(row["department_id"]) else None,
            int(row["segment_id"]) if pd.notna(row["segment_id"]) else None,
            int(row["account_id"]) if pd.notna(row["account_id"]) else None,
            int(row["fiscal_year"]) if pd.notna(row["fiscal_year"]) else None,
            int(row["period_from"]) if pd.notna(row["period_from"]) else None,
            int(row["period_to"]) if pd.notna(row["period_to"]) else None,
            row["assumption_type"],
            float(row["assumption_value"]),
            row["input_source"],
            row["assumption_text"] if pd.notna(row["assumption_text"]) else None,
            int(row["priority_order"]) if pd.notna(row["priority_order"]) else 1,
            int(row["is_active"]) if pd.notna(row["is_active"]) else 1,
        )
        rows_saved += 1

    conn.commit()
    conn.close()

    return rows_saved