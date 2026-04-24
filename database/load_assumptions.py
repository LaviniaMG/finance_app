import pandas as pd
from database.connection import get_connection


def load_assumptions_by_scenario(scenario_code: str) -> pd.DataFrame:
    conn = get_connection()

    query = """
        SELECT
            assumption_id,
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
        FROM dbo.ForecastAssumption
        WHERE scenario_code = ?
          AND is_active = 1
        ORDER BY priority_order DESC, created_at ASC
    """

    df = pd.read_sql(query, conn, params=[scenario_code.upper().strip()])
    conn.close()
    return df


def load_active_assumptions() -> pd.DataFrame:
    conn = get_connection()

    query = """
        SELECT
            assumption_id,
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
        FROM dbo.ForecastAssumption
        WHERE is_active = 1
        ORDER BY scenario_code, priority_order DESC, created_at ASC
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df