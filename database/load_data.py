import pandas as pd
from database.connection import get_connection


# ---------------------------------------------------------------------------
# Dimension loaders
# ---------------------------------------------------------------------------

def load_dim_entities() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM dbo.DimEntity WHERE is_active = 1", conn)
    conn.close()
    return df


def load_dim_accounts() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM dbo.DimAccount WHERE is_active = 1", conn)
    conn.close()
    return df


def load_dim_departments() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM dbo.DimDepartment WHERE is_active = 1", conn)
    conn.close()
    return df


def load_dim_coverages() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM dbo.DimCoverage WHERE is_active = 1", conn)
    conn.close()
    return df


def load_dim_scenarios() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM dbo.DimScenario", conn)
    conn.close()
    return df


def load_dim_periods() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM dbo.DimPeriod ORDER BY sort_order", conn)
    conn.close()
    return df


def load_dim_currencies() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM dbo.DimCurrency WHERE is_active = 1", conn)
    conn.close()
    return df


def load_fx_rates(fiscal_year: int = None) -> pd.DataFrame:
    conn = get_connection()
    if fiscal_year:
        df = pd.read_sql(
            f"SELECT * FROM dbo.FXRates WHERE fiscal_year = {fiscal_year}", conn
        )
    else:
        df = pd.read_sql("SELECT * FROM dbo.FXRates", conn)
    conn.close()
    return df


def load_calculation_rules() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM dbo.CalculationRule WHERE is_active = 1 ORDER BY execution_order",
        conn,
    )
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Fact loaders (generic)
# ---------------------------------------------------------------------------

_FACT_SELECT = """
    SELECT
        ff.FinanceFact_id,
        ff.entity_id,
        ff.coverage_id,
        ff.department_id,
        ff.segment_id,
        ff.account_id,
        ff.period_id,
        ff.fiscal_year,
        ff.scenario_id,
        ff.scenario_code,
        ff.category_id,
        ff.mtd_lcl_amount,
        ff.mtd_ccy_amount,
        ff.mtd_rpt_amount,
        ff.ytd_lcl_amount,
        ff.ytd_ccy_amount,
        ff.ytd_rpt_amount,
        ff.forecast_run_id,
        ff.import_batch_id,
        ff.created_at,
        ff.updated_at,
        da.statement_type,
        da.account_code,
        da.account_name
    FROM dbo.FinanceFact ff
    INNER JOIN dbo.DimAccount da ON ff.account_id = da.account_id
"""


def load_fact_by_scenario_prefix(prefix: str) -> pd.DataFrame:
    conn = get_connection()
    query = _FACT_SELECT + f" WHERE ff.scenario_code LIKE '{prefix}%'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_fact_by_scenario_code(scenario_code: str) -> pd.DataFrame:
    conn = get_connection()
    query = _FACT_SELECT + f" WHERE ff.scenario_code = '{scenario_code}'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_fact_multi_scenario(scenario_codes: list[str]) -> pd.DataFrame:
    if not scenario_codes:
        return pd.DataFrame()
    placeholders = ",".join(f"'{c}'" for c in scenario_codes)
    conn = get_connection()
    query = _FACT_SELECT + f" WHERE ff.scenario_code IN ({placeholders})"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_actuals_for_ml() -> pd.DataFrame:
    return load_fact_by_scenario_prefix("ACT")


def load_finance_fact() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(_FACT_SELECT, conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Operational model loader
# ---------------------------------------------------------------------------

def load_operational_baseline(
    target_scenario: str,
    prev_rf_scenario: str,
    actuals_prefix: str = "ACT",
) -> pd.DataFrame:
    """
    Returns combined dataframe for the operational model:
      - Actuals for all closed periods (source = actuals)
      - Previous RF (prev_rf_scenario) for open/future periods
      - Actuals override prev RF for any period where both exist

    Also attaches prev_rf_amount column so feature engineering can use it.
    """
    actuals_df = load_fact_by_scenario_prefix(actuals_prefix)
    prev_rf_df = load_fact_by_scenario_code(prev_rf_scenario)

    if actuals_df.empty and prev_rf_df.empty:
        return pd.DataFrame()

    key_cols = ["entity_id", "coverage_id", "department_id", "segment_id", "account_id",
                "fiscal_year", "period_id"]

    # Tag prev RF with original amount for use as a feature
    if not prev_rf_df.empty:
        prev_rf_df = prev_rf_df.copy()
        prev_rf_df["prev_rf_mtd_rpt"] = prev_rf_df["mtd_rpt_amount"]
        prev_rf_df["prev_rf_ytd_rpt"] = prev_rf_df["ytd_rpt_amount"]
        prev_rf_df["data_source"] = "PREV_RF"

    if not actuals_df.empty:
        actuals_df = actuals_df.copy()
        actuals_df["prev_rf_mtd_rpt"] = None
        actuals_df["prev_rf_ytd_rpt"] = None
        actuals_df["data_source"] = "ACTUAL"

    # Merge: actuals take priority; prev RF fills gaps
    if actuals_df.empty:
        combined = prev_rf_df.copy()
    elif prev_rf_df.empty:
        combined = actuals_df.copy()
    else:
        actual_keys = set(
            map(tuple, actuals_df[key_cols].values.tolist())
        )
        rf_only = prev_rf_df[
            ~prev_rf_df[key_cols].apply(tuple, axis=1).isin(actual_keys)
        ].copy()

        # For actual rows, attach prev_rf amounts from the prev_rf_df
        merged_actuals = actuals_df.merge(
            prev_rf_df[key_cols + ["prev_rf_mtd_rpt", "prev_rf_ytd_rpt"]],
            on=key_cols,
            how="left",
            suffixes=("", "_from_rf"),
        )
        if "prev_rf_mtd_rpt_from_rf" in merged_actuals.columns:
            merged_actuals["prev_rf_mtd_rpt"] = merged_actuals["prev_rf_mtd_rpt_from_rf"]
            merged_actuals["prev_rf_ytd_rpt"] = merged_actuals["prev_rf_ytd_rpt_from_rf"]
            merged_actuals.drop(
                columns=["prev_rf_mtd_rpt_from_rf", "prev_rf_ytd_rpt_from_rf"], inplace=True
            )

        combined = pd.concat([merged_actuals, rf_only], ignore_index=True)

    combined = combined.sort_values(key_cols + ["fiscal_year"]).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Breakback data loader
# ---------------------------------------------------------------------------

def load_for_breakback(
    scenario_code: str,
    entity_id: int | None,
    department_id: int | None,
    coverage_id: int | None,
    account_id: int | None,
    fiscal_year: int | None,
    period_id: int | None,
    actuals_scenario_prefix: str = "ACT",
) -> dict:
    """
    Loads forecast values + actuals YTD for computing breakback weights.
    Returns:
      - forecast_df: forecast rows matching the filter (any level)
      - actuals_df: actuals for the same filter (for YTD weight computation)
    """
    conditions_base = []
    if fiscal_year:
        conditions_base.append(f"ff.fiscal_year = {fiscal_year}")
    if period_id:
        conditions_base.append(f"ff.period_id = {period_id}")

    def _dim_condition(col, val):
        return f"ff.{col} = {val}" if val is not None else None

    dim_conds = [
        _dim_condition("entity_id", entity_id),
        _dim_condition("department_id", department_id),
        _dim_condition("coverage_id", coverage_id),
        _dim_condition("account_id", account_id),
    ]
    dim_conds = [c for c in dim_conds if c]

    all_conds = conditions_base + dim_conds

    where_forecast = f"ff.scenario_code = '{scenario_code}'"
    where_actuals = f"ff.scenario_code LIKE '{actuals_scenario_prefix}%'"

    if all_conds:
        extra = " AND " + " AND ".join(all_conds)
        where_forecast += extra
        where_actuals += extra

    conn = get_connection()
    forecast_df = pd.read_sql(_FACT_SELECT + f" WHERE {where_forecast}", conn)
    actuals_df = pd.read_sql(_FACT_SELECT + f" WHERE {where_actuals}", conn)
    conn.close()

    return {"forecast": forecast_df, "actuals": actuals_df}


# ---------------------------------------------------------------------------
# Dashboard / reporting helpers
# ---------------------------------------------------------------------------

def load_active_forecast_runs() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT fr.*, ds.scenario_type
        FROM dbo.ForecastRun fr
        LEFT JOIN dbo.DimScenario ds ON fr.scenario_code = ds.scenario_code
        ORDER BY fr.created_at DESC
        """,
        conn,
    )
    conn.close()
    return df


def load_business_warnings(scenario_code: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        f"""
        SELECT fe.*, da.account_name, de.entity_name
        FROM dbo.ForecastExplanation fe
        LEFT JOIN dbo.DimAccount da ON fe.account_id = da.account_id
        LEFT JOIN dbo.DimEntity de ON fe.entity_id = de.entity_id
        WHERE fe.scenario_code = '{scenario_code}'
          AND fe.confidence_level IN ('Low', 'Medium')
        ORDER BY fe.fiscal_year, fe.period_id
        """,
        conn,
    )
    conn.close()
    return df


def load_import_batches() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM dbo.ImportBatch ORDER BY uploaded_at DESC",
        conn,
    )
    conn.close()
    return df
