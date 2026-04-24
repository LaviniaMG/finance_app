import pandas as pd
from database.connection import get_connection


def infer_scenario_type(scenario_code: str) -> str:
    scenario_code = scenario_code.upper().strip()

    if scenario_code.startswith("ACT"):
        return "ACT"
    if scenario_code.startswith("RF"):
        return "RF"
    if scenario_code.startswith("BDG"):
        return "BDG"

    return "OTHER"


def get_or_create_scenario(scenario_code: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()

    scenario_code = scenario_code.upper().strip()
    scenario_type = infer_scenario_type(scenario_code)

    select_sql = """
        SELECT scenario_id, scenario_code, scenario_type
        FROM dbo.DimScenario
        WHERE scenario_code = ?
    """
    cursor.execute(select_sql, scenario_code)
    row = cursor.fetchone()

    if row:
        result = {
            "scenario_id": row[0],
            "scenario_code": row[1],
            "scenario_type": row[2],
        }
        conn.close()
        return result

    insert_sql = """
        INSERT INTO dbo.DimScenario (scenario_code, scenario_type)
        VALUES (?, ?)
    """
    cursor.execute(insert_sql, scenario_code, scenario_type)
    conn.commit()

    cursor.execute(select_sql, scenario_code)
    row = cursor.fetchone()

    result = {
        "scenario_id": row[0],
        "scenario_code": row[1],
        "scenario_type": row[2],
    }

    conn.close()
    return result


def create_forecast_run(forecast_request: dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO dbo.ForecastRun
        (
            scenario_code,
            model_name,
            amount_basis,
            mode,
            years_ahead_for_rf,
            budget_months,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, GETDATE())
    """

    cursor.execute(
        insert_sql,
        forecast_request.get("scenario_code"),
        forecast_request.get("model_name"),
        forecast_request.get("amount_basis"),
        forecast_request.get("mode"),
        forecast_request.get("years_ahead_for_rf"),
        forecast_request.get("budget_months"),
    )
    conn.commit()

    cursor.execute("SELECT MAX(forecast_run_id) FROM dbo.ForecastRun")
    forecast_run_id = cursor.fetchone()[0]

    conn.close()
    return forecast_run_id


def validate_forecast_dataframe(forecast_df: pd.DataFrame):
    required_columns = [
        "entity_id",
        "coverage_id",
        "department_id",
        "segment_id",
        "account_id",
        "fiscal_year",
        "period_id",
        "scenario_code",
        "mtd_lcl_amount",
        "mtd_ccy_amount",
        "mtd_rpt_amount",
        "ytd_lcl_amount",
        "ytd_ccy_amount",
        "ytd_rpt_amount",
    ]

    missing = [c for c in required_columns if c not in forecast_df.columns]
    if missing:
        raise ValueError(f"forecast_df is missing required columns: {missing}")


def save_forecast_to_fact(
    forecast_df: pd.DataFrame,
    forecast_request: dict,
    default_category_id: int = 1
) -> int:
    if forecast_df.empty:
        raise ValueError("forecast_df is empty. Nothing to save.")

    validate_forecast_dataframe(forecast_df)

    scenario_code = forecast_request["scenario_code"]
    scenario_info = get_or_create_scenario(scenario_code)
    scenario_id = scenario_info["scenario_id"]

    forecast_run_id = create_forecast_run(forecast_request)

    conn = get_connection()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO dbo.FinanceFact
        (
            entity_id,
            coverage_id,
            department_id,
            segment_id,
            account_id,
            fiscal_year,
            period_id,
            scenario_id,
            scenario_code,
            category_id,
            mtd_lcl_amount,
            mtd_ccy_amount,
            mtd_rpt_amount,
            ytd_lcl_amount,
            ytd_ccy_amount,
            ytd_rpt_amount,
            forecast_run_id,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE(), NULL)
    """

    for _, row in forecast_df.iterrows():
        cursor.execute(
            insert_sql,
            int(row["entity_id"]),
            int(row["coverage_id"]),
            int(row["department_id"]),
            int(row["segment_id"]),
            int(row["account_id"]),
            int(row["fiscal_year"]),
            int(row["period_id"]),
            int(scenario_id),
            str(row["scenario_code"]),
            int(default_category_id),
            float(row["mtd_lcl_amount"]),
            float(row["mtd_ccy_amount"]),
            float(row["mtd_rpt_amount"]),
            float(row["ytd_lcl_amount"]),
            float(row["ytd_ccy_amount"]),
            float(row["ytd_rpt_amount"]),
            int(forecast_run_id),
        )

    conn.commit()
    conn.close()

    return forecast_run_id