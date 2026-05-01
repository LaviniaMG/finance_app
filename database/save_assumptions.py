import pandas as pd
import numpy as np
from database.connection import get_connection


ALLOWED_ASSUMPTION_TYPES = {
    "growth_pct",
    "inflation_pct",
    "fixed_value",
    "delta_value",
    "fx_adjustment_pct",
    "headcount_growth_pct",
    "ytd_target",   # year-level input: user gives total YTD for the year
    "yoy_pct",      # year-over-year %: user gives growth rate for future years
}

ALLOWED_INPUT_SOURCES = {"MANUAL", "AI", "SUGGESTED"}


# ---------------------------------------------------------------------------
# Monthly spread computation (year-level input logic)
# ---------------------------------------------------------------------------

def compute_monthly_from_ytd_target(
    ytd_target: float,
    actuals_ytd: float,
    open_rf_months: list[int],
    seasonal_weights: dict | None = None,
) -> dict[int, float]:
    """
    Year 1 logic: user gives YTD target for the full year.
    Remaining = YTD target - actuals already posted.
    Spread remaining evenly (or seasonally) over open RF months.
    Returns {period_id: monthly_amount}.
    """
    remaining = ytd_target - actuals_ytd
    if not open_rf_months:
        return {}

    if seasonal_weights:
        relevant_weights = {m: seasonal_weights.get(m, 1.0) for m in open_rf_months}
        total_w = sum(relevant_weights.values())
        if total_w == 0:
            total_w = len(open_rf_months)
        return {m: remaining * (relevant_weights[m] / total_w) for m in open_rf_months}
    else:
        monthly = remaining / len(open_rf_months)
        return {m: monthly for m in open_rf_months}


def compute_monthly_from_yoy(
    year1_ytd: float,
    yoy_pct: float,
    seasonal_weights: dict | None = None,
) -> dict[int, float]:
    """
    Year 2+ logic: user gives YOY%.
    annual_target = year1_ytd * (1 + yoy_pct)
    Spread annual_target across 12 months (flat or seasonal).
    Returns {period_id: monthly_amount}.
    """
    annual_target = year1_ytd * (1 + yoy_pct)
    if seasonal_weights:
        total_w = sum(seasonal_weights.values())
        if total_w == 0:
            total_w = 12
        return {m: annual_target * (seasonal_weights.get(m, 1.0 / 12) / total_w) for m in range(1, 13)}
    else:
        return {m: annual_target / 12 for m in range(1, 13)}


def expand_year_level_assumption(
    assumption: dict,
    actuals_df: pd.DataFrame | None = None,
    seasonal_weights: dict | None = None,
) -> list[dict]:
    """
    Converts a ytd_target or yoy_pct assumption into per-month delta_value assumptions.

    assumption dict keys:
      scenario_code, assumption_type, assumption_value,
      fiscal_year, entity_id, department_id, coverage_id, account_id,
      input_source, assumption_text, priority_order
    """
    atype = assumption.get("assumption_type")
    fiscal_year = assumption.get("fiscal_year")
    value = float(assumption.get("assumption_value", 0))

    if atype == "ytd_target":
        actuals_ytd = 0.0
        if actuals_df is not None and not actuals_df.empty:
            actuals_ytd = float(actuals_df["ytd_rpt_amount"].fillna(0).sum())

        # Determine open RF months from the scenario code
        scenario_code = assumption.get("scenario_code", "")
        from ml.predict import parse_scenario_code
        try:
            info = parse_scenario_code(scenario_code)
            if info["scenario_type"] == "RF":
                start_month = info["start_month"]
            else:
                start_month = 1
        except Exception:
            start_month = 1

        open_months = list(range(start_month, 13))
        monthly_map = compute_monthly_from_ytd_target(
            ytd_target=value,
            actuals_ytd=actuals_ytd,
            open_rf_months=open_months,
            seasonal_weights=seasonal_weights,
        )

    elif atype == "yoy_pct":
        year1_ytd = 0.0
        if actuals_df is not None and not actuals_df.empty:
            year1_ytd = float(actuals_df["ytd_rpt_amount"].fillna(0).sum())
        monthly_map = compute_monthly_from_yoy(
            year1_ytd=year1_ytd,
            yoy_pct=value,
            seasonal_weights=seasonal_weights,
        )
    else:
        return [assumption]

    expanded = []
    base = {k: v for k, v in assumption.items() if k not in ("assumption_type", "assumption_value")}
    for period_id, monthly_val in monthly_map.items():
        row = base.copy()
        row["assumption_type"] = "delta_value"
        row["assumption_value"] = monthly_val
        row["period_from"] = period_id
        row["period_to"] = period_id
        expanded.append(row)

    return expanded


# ---------------------------------------------------------------------------
# Validation & normalization
# ---------------------------------------------------------------------------

def validate_assumptions_dataframe(assumptions_df: pd.DataFrame):
    required_columns = ["scenario_code", "assumption_type", "assumption_value", "input_source"]
    missing = [c for c in required_columns if c not in assumptions_df.columns]
    if missing:
        raise ValueError(f"assumptions_df is missing required columns: {missing}")

    invalid_types = set(assumptions_df["assumption_type"].dropna().unique()) - ALLOWED_ASSUMPTION_TYPES
    if invalid_types:
        raise ValueError(f"Invalid assumption_type values: {sorted(invalid_types)}")

    invalid_sources = {str(x).upper() for x in assumptions_df["input_source"].dropna().unique()} - ALLOWED_INPUT_SOURCES
    if invalid_sources:
        raise ValueError(f"Invalid input_source values: {sorted(invalid_sources)}")


def normalize_assumptions_dataframe(assumptions_df: pd.DataFrame) -> pd.DataFrame:
    df = assumptions_df.copy()
    optional_defaults = {
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
        "status": "draft",
    }
    for col, default_value in optional_defaults.items():
        if col not in df.columns:
            df[col] = default_value

    df["scenario_code"] = df["scenario_code"].astype(str).str.upper().str.strip()
    df["assumption_type"] = df["assumption_type"].astype(str).str.strip()
    df["input_source"] = df["input_source"].astype(str).str.upper().str.strip()
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_assumptions_to_db(assumptions_df: pd.DataFrame) -> int:
    if assumptions_df.empty:
        raise ValueError("assumptions_df is empty.")

    df = normalize_assumptions_dataframe(assumptions_df)
    validate_assumptions_dataframe(df)

    conn = get_connection()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO dbo.ForecastAssumption (
            scenario_code, entity_id, coverage_id, department_id, segment_id, account_id,
            fiscal_year, period_from, period_to,
            assumption_type, assumption_value, input_source,
            assumption_text, priority_order, is_active, created_at
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


def deactivate_assumption(assumption_id: int) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE dbo.ForecastAssumption SET is_active = 0 WHERE assumption_id = ?",
        assumption_id,
    )
    conn.commit()
    conn.close()


def update_assumption_status(assumption_id: int, status: str) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE dbo.ForecastAssumption SET status = ? WHERE assumption_id = ?",
        status, assumption_id,
    )
    conn.commit()
    conn.close()
