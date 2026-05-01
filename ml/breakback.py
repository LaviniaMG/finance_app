"""
Breakback engine — distributes a user-edited aggregated value down to leaf rows
proportionally based on YTD actuals weights (with fallback to equal split).
"""
import numpy as np
import pandas as pd


SERIES_KEYS = [
    "entity_id",
    "coverage_id",
    "department_id",
    "segment_id",
    "account_id",
]


def identify_leaf_rows(
    forecast_df: pd.DataFrame,
    entity_id: int | None,
    department_id: int | None,
    coverage_id: int | None,
    account_id: int | None,
    fiscal_year: int,
    period_id: int,
) -> pd.DataFrame:
    """
    Returns all rows in forecast_df that match the given dimension filters
    for the specified period. None means 'all values' for that dimension.
    """
    mask = (
        (forecast_df["fiscal_year"] == fiscal_year) &
        (forecast_df["period_id"] == period_id)
    )
    if entity_id is not None:
        mask &= forecast_df["entity_id"] == entity_id
    if department_id is not None:
        mask &= forecast_df["department_id"] == department_id
    if coverage_id is not None:
        mask &= forecast_df["coverage_id"] == coverage_id
    if account_id is not None:
        mask &= forecast_df["account_id"] == account_id

    return forecast_df[mask].copy()


def compute_aggregated_value(
    leaf_rows: pd.DataFrame,
    amount_col: str = "mtd_rpt_amount",
) -> float:
    """Sum of all leaf rows for the given amount column."""
    if leaf_rows.empty:
        return 0.0
    return float(leaf_rows[amount_col].fillna(0).sum())


def compute_ytd_weights(
    leaf_rows: pd.DataFrame,
    actuals_df: pd.DataFrame,
    fiscal_year: int,
) -> pd.Series:
    """
    Computes YTD weight for each leaf row based on actuals.
    Weight = leaf's YTD actuals / total YTD actuals for all leaves.
    Falls back to equal weight if no actuals exist.
    """
    if actuals_df.empty:
        n = len(leaf_rows)
        return pd.Series([1.0 / n] * n, index=leaf_rows.index)

    # Actuals YTD = sum of actuals for the year up to latest period
    actuals_ytd = (
        actuals_df[actuals_df["fiscal_year"] == fiscal_year]
        .groupby(SERIES_KEYS, dropna=False)["ytd_rpt_amount"]
        .max()  # YTD is cumulative; take max = latest period's YTD
        .reset_index()
        .rename(columns={"ytd_rpt_amount": "ytd_actual"})
    )

    merged = leaf_rows.merge(actuals_ytd, on=SERIES_KEYS, how="left")
    merged["ytd_actual"] = merged["ytd_actual"].fillna(0).abs()

    total = merged["ytd_actual"].sum()
    if total == 0:
        n = len(merged)
        merged["weight"] = 1.0 / n if n > 0 else 0.0
    else:
        merged["weight"] = merged["ytd_actual"] / total

    return merged["weight"].values


def apply_breakback(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    entity_id: int | None,
    department_id: int | None,
    coverage_id: int | None,
    account_id: int | None,
    fiscal_year: int,
    period_id: int,
    new_aggregated_value: float,
    amount_col: str = "mtd_rpt_amount",
) -> pd.DataFrame:
    """
    Main breakback function.

    1. Identifies leaf rows matching the dimension filter
    2. Computes current aggregated value
    3. Computes delta = new_aggregated_value - current_aggregated_value
    4. Distributes delta to leaves proportional to YTD actuals weights
    5. Returns updated forecast_df with new values

    Also recalculates YTD amounts for the affected rows.
    """
    leaf_rows = identify_leaf_rows(
        forecast_df, entity_id, department_id, coverage_id, account_id,
        fiscal_year, period_id,
    )

    if leaf_rows.empty:
        return forecast_df

    current_total = compute_aggregated_value(leaf_rows, amount_col)
    delta = new_aggregated_value - current_total

    if abs(delta) < 1e-6:
        return forecast_df

    weights = compute_ytd_weights(leaf_rows, actuals_df, fiscal_year)

    result_df = forecast_df.copy()

    for i, (idx, leaf_row) in enumerate(leaf_rows.iterrows()):
        w = float(weights[i]) if i < len(weights) else 0.0
        leaf_delta = delta * w
        new_val = float(leaf_row[amount_col]) + leaf_delta
        result_df.at[idx, amount_col] = new_val

        # Update companion columns
        if amount_col == "mtd_rpt_amount":
            # Recalculate YTD RPT for this leaf in this year
            same_year_prev = _sum_year_ytd_before_period(
                result_df, leaf_row, fiscal_year, period_id
            )
            result_df.at[idx, "ytd_rpt_amount"] = same_year_prev + new_val

            # Scale LCL and CCY by the same ratio as the original
            if float(leaf_row.get("mtd_rpt_amount", 0) or 0) != 0:
                scale = new_val / float(leaf_row["mtd_rpt_amount"])
            else:
                scale = 1.0

            for col_pair in [("mtd_lcl_amount", "ytd_lcl_amount"),
                             ("mtd_ccy_amount", "ytd_ccy_amount")]:
                mtd_col, ytd_col = col_pair
                if mtd_col in result_df.columns:
                    old_mtd = float(leaf_row.get(mtd_col, 0) or 0)
                    result_df.at[idx, mtd_col] = old_mtd * scale
                if ytd_col in result_df.columns:
                    old_ytd = float(leaf_row.get(ytd_col, 0) or 0)
                    result_df.at[idx, ytd_col] = old_ytd * scale

    return result_df


def _sum_year_ytd_before_period(
    df: pd.DataFrame,
    leaf_row: pd.Series,
    fiscal_year: int,
    period_id: int,
) -> float:
    """Sum of MTD RPT for the same series in the same year, for periods before period_id."""
    mask = (
        (df["entity_id"] == leaf_row["entity_id"]) &
        (df["coverage_id"] == leaf_row["coverage_id"]) &
        (df["department_id"] == leaf_row["department_id"]) &
        (df["segment_id"] == leaf_row["segment_id"]) &
        (df["account_id"] == leaf_row["account_id"]) &
        (df["fiscal_year"] == fiscal_year) &
        (df["period_id"] < period_id)
    )
    subset = df[mask]
    if subset.empty:
        return 0.0
    return float(subset["mtd_rpt_amount"].fillna(0).sum())


def preview_breakback(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    entity_id: int | None,
    department_id: int | None,
    coverage_id: int | None,
    account_id: int | None,
    fiscal_year: int,
    period_id: int,
    new_aggregated_value: float,
    amount_col: str = "mtd_rpt_amount",
) -> pd.DataFrame:
    """
    Returns a preview dataframe showing:
      - leaf dimension keys
      - current ML forecast value
      - YTD weight
      - new value after breakback
      - delta
    Without modifying the original forecast_df.
    """
    leaf_rows = identify_leaf_rows(
        forecast_df, entity_id, department_id, coverage_id, account_id,
        fiscal_year, period_id,
    )

    if leaf_rows.empty:
        return pd.DataFrame()

    current_total = compute_aggregated_value(leaf_rows, amount_col)
    delta = new_aggregated_value - current_total
    weights = compute_ytd_weights(leaf_rows, actuals_df, fiscal_year)

    rows = []
    for i, (_, leaf_row) in enumerate(leaf_rows.iterrows()):
        w = float(weights[i]) if i < len(weights) else 0.0
        leaf_delta = delta * w
        current_val = float(leaf_row.get(amount_col, 0) or 0)
        new_val = current_val + leaf_delta
        rows.append({
            "entity_id": leaf_row.get("entity_id"),
            "coverage_id": leaf_row.get("coverage_id"),
            "department_id": leaf_row.get("department_id"),
            "segment_id": leaf_row.get("segment_id"),
            "account_id": leaf_row.get("account_id"),
            "ytd_weight_pct": round(w * 100, 1),
            "current_value": round(current_val, 0),
            "new_value": round(new_val, 0),
            "delta": round(leaf_delta, 0),
        })

    preview_df = pd.DataFrame(rows)
    totals = {
        "entity_id": "TOTAL",
        "coverage_id": "",
        "department_id": "",
        "segment_id": "",
        "account_id": "",
        "ytd_weight_pct": 100.0,
        "current_value": round(current_total, 0),
        "new_value": round(new_aggregated_value, 0),
        "delta": round(delta, 0),
    }
    preview_df = pd.concat([preview_df, pd.DataFrame([totals])], ignore_index=True)
    return preview_df
