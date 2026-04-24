import pandas as pd
from database.connection import get_connection


def load_dim_entities():
    conn = get_connection()
    query = "SELECT * FROM dbo.DimEntity"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_fact_by_scenario_prefix(prefix: str):
    conn = get_connection()

    query = f"""
        SELECT
            ff.FinanceFact_id,
            ff.entity_id,
            ff.coverage_id,
            ff.department_id,
            ff.segment_id,
            ff.account_id,
            ff.fiscal_year,
            ff.period_id,
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
            ff.created_at,
            ff.updated_at,
            da.statement_type
        FROM dbo.FinanceFact ff
        INNER JOIN dbo.DimAccount da
            ON ff.account_id = da.account_id
        WHERE ff.scenario_code LIKE '{prefix}%'
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_actuals_for_ml():
    return load_fact_by_scenario_prefix("ACT")


def load_finance_fact():
    conn = get_connection()
    query = "SELECT * FROM dbo.FinanceFact"
    df = pd.read_sql(query, conn)
    conn.close()
    return df