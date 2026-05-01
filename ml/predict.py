import os
import re
import joblib
import numpy as np
import pandas as pd

from config.settings import MODEL_FOLDER
from ml.feature_engineering import get_training_columns


def load_saved_model(model_name: str):
    model_path = os.path.join(MODEL_FOLDER, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Scenario parsing
# ---------------------------------------------------------------------------

def validate_scenario_code(scenario_code: str) -> bool:
    scenario_code = scenario_code.upper().strip()
    rf_match = re.match(r"^RF\d{2}_\d{4}$", scenario_code)
    bdg_match = re.match(r"^BDG_\d{4}$", scenario_code)
    return bool(rf_match or bdg_match)


def parse_scenario_code(scenario_code: str) -> dict:
    scenario_code = scenario_code.upper().strip()
    rf_match = re.match(r"^RF(\d{2})_(\d{4})$", scenario_code)
    bdg_match = re.match(r"^BDG_(\d{4})$", scenario_code)

    if rf_match:
        return {
            "scenario_type": "RF",
            "start_month": int(rf_match.group(1)),
            "base_year": int(rf_match.group(2)),
        }
    if bdg_match:
        return {
            "scenario_type": "BDG",
            "base_year": int(bdg_match.group(1)),
        }
    raise ValueError(f"Invalid scenario format: {scenario_code}")


def get_prev_rf_scenario_code(scenario_code: str) -> str | None:
    """Returns the immediately preceding RF scenario code, or None for BDG."""
    info = parse_scenario_code(scenario_code)
    if info["scenario_type"] != "RF":
        return None
    month = info["start_month"]
    year = info["base_year"]
    if month == 1:
        prev_month = 12
        prev_year = year - 1
    else:
        prev_month = month - 1
        prev_year = year
    return f"RF{prev_month:02d}_{prev_year}"


def build_forecast_window(
    scenario_info: dict,
    years_ahead_for_rf: int = 2,
    budget_months: int = 12,
) -> tuple[tuple[int, int], tuple[int, int]]:
    scenario_type = scenario_info["scenario_type"]
    if scenario_type == "RF":
        start = (scenario_info["base_year"], scenario_info["start_month"])
        end = (scenario_info["base_year"] + years_ahead_for_rf, 12)
        return start, end
    if scenario_type == "BDG":
        start = (scenario_info["base_year"], 1)
        end = (scenario_info["base_year"], budget_months)
        return start, end
    raise ValueError(f"Unsupported scenario type: {scenario_type}")


def next_period(fiscal_year: int, period_id: int) -> tuple[int, int]:
    if period_id == 12:
        return fiscal_year + 1, 1
    return fiscal_year, period_id + 1


def generate_period_range(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    periods = []
    current_year, current_month = start
    end_year, end_month = end
    while (current_year < end_year) or (
        current_year == end_year and current_month <= end_month
    ):
        periods.append((current_year, current_month))
        current_year, current_month = next_period(current_year, current_month)
    return periods


# ---------------------------------------------------------------------------
# Currency helpers
# ---------------------------------------------------------------------------

def _safe_ratio(numerator, denominator):
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return 1.0
    if numerator is None or pd.isna(numerator):
        return 1.0
    return numerator / denominator


def infer_currency_ratios(history_df: pd.DataFrame) -> dict:
    hist = history_df.sort_values(["fiscal_year", "period_id"]).copy()
    latest = hist.iloc[-1]
    return {
        "rpt_to_lcl": _safe_ratio(latest.get("mtd_lcl_amount"), latest.get("mtd_rpt_amount")),
        "rpt_to_ccy": _safe_ratio(latest.get("mtd_ccy_amount"), latest.get("mtd_rpt_amount")),
        "ytd_rpt_to_lcl": _safe_ratio(latest.get("ytd_lcl_amount"), latest.get("ytd_rpt_amount")),
        "ytd_rpt_to_ccy": _safe_ratio(latest.get("ytd_ccy_amount"), latest.get("ytd_rpt_amount")),
    }


# ---------------------------------------------------------------------------
# Feature building for a single future period
# ---------------------------------------------------------------------------

def _validate_history_for_prediction(history_df: pd.DataFrame):
    required_columns = [
        "entity_id", "coverage_id", "department_id", "segment_id", "account_id",
        "fiscal_year", "period_id", "statement_type", "target_amount",
    ]
    missing = [c for c in required_columns if c not in history_df.columns]
    if missing:
        raise ValueError(f"history_df missing columns: {missing}")


def _same_year_ytd_previous(history_df: pd.DataFrame, future_year: int) -> float:
    same_year = history_df[history_df["fiscal_year"] == future_year].copy()
    if same_year.empty:
        return 0.0
    valid = same_year["ytd_rpt_amount"].dropna()
    return float(valid.iloc[-1]) if not valid.empty else 0.0


def _build_single_prediction_row(
    history_df: pd.DataFrame,
    future_year: int,
    future_month: int,
    prev_rf_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    hist = history_df.sort_values(["fiscal_year", "period_id"]).copy()
    if hist.empty:
        raise ValueError("History dataframe is empty.")

    last_row = hist.iloc[-1]
    quarter = ((future_month - 1) // 3) + 1
    is_quarter_end = 1 if future_month in [3, 6, 9, 12] else 0
    is_year_end = 1 if future_month == 12 else 0

    target_series = hist["target_amount"].dropna()

    def _lag(n):
        return float(target_series.iloc[-n]) if len(target_series) >= n else np.nan

    lag_1 = _lag(1)
    lag_2 = _lag(2)
    lag_3 = _lag(3)
    lag_6 = _lag(6)
    lag_12 = _lag(12)

    rolling_mean_3 = float(target_series.tail(3).mean()) if len(target_series) >= 1 else np.nan
    rolling_mean_6 = float(target_series.tail(6).mean()) if len(target_series) >= 1 else np.nan
    rolling_std_3 = float(target_series.tail(3).std()) if len(target_series) >= 2 else np.nan
    rolling_std_6 = float(target_series.tail(6).std()) if len(target_series) >= 2 else np.nan

    mom_change = np.nan
    if pd.notna(lag_1) and pd.notna(lag_2) and lag_2 != 0:
        mom_change = (lag_1 - lag_2) / lag_2

    actual_vs_previous_year_abs = np.nan
    actual_vs_previous_year_pct = np.nan
    if pd.notna(lag_12) and pd.notna(lag_1):
        actual_vs_previous_year_abs = lag_1 - lag_12
        if lag_12 != 0:
            actual_vs_previous_year_pct = (lag_1 - lag_12) / lag_12

    ytd_previous = float(hist["ytd_rpt_amount"].dropna().iloc[-1]) if "ytd_rpt_amount" in hist.columns and not hist["ytd_rpt_amount"].dropna().empty else np.nan

    currency_effect_abs = np.nan
    currency_effect_pct = np.nan
    recent_fx = hist.dropna(subset=["mtd_rpt_amount", "mtd_lcl_amount"]) if "mtd_lcl_amount" in hist.columns else pd.DataFrame()
    if not recent_fx.empty:
        r = recent_fx.iloc[-1]
        currency_effect_abs = float(r["mtd_rpt_amount"] - r["mtd_lcl_amount"])
        if r["mtd_lcl_amount"] != 0:
            currency_effect_pct = float((r["mtd_rpt_amount"] - r["mtd_lcl_amount"]) / r["mtd_lcl_amount"])

    history_count = len(target_series) - 1

    volatility_ratio = np.nan
    if pd.notna(rolling_mean_6) and rolling_mean_6 != 0 and pd.notna(rolling_std_6):
        volatility_ratio = rolling_std_6 / abs(rolling_mean_6)

    # Operational model: prev RF amount for this specific future period
    prev_rf_amount = np.nan
    actual_vs_prev_rf_pct = np.nan
    rf_deviation_rolling_mean_3 = np.nan

    if prev_rf_df is not None and not prev_rf_df.empty:
        match = prev_rf_df[
            (prev_rf_df["fiscal_year"] == future_year) &
            (prev_rf_df["period_id"] == future_month)
        ]
        if not match.empty:
            statement_type = str(last_row["statement_type"]).upper()
            if statement_type == "BS":
                prev_rf_amount = float(match.iloc[0].get("ytd_rpt_amount", np.nan) or np.nan)
            else:
                prev_rf_amount = float(match.iloc[0].get("mtd_rpt_amount", np.nan) or np.nan)

        # Rolling mean of RF deviations from closed periods
        if "actual_vs_prev_rf_pct" in hist.columns:
            rf_dev_series = hist["actual_vs_prev_rf_pct"].dropna()
            if len(rf_dev_series) >= 1:
                rf_deviation_rolling_mean_3 = float(rf_dev_series.tail(3).mean())

    row = {
        "entity_id": last_row["entity_id"],
        "coverage_id": last_row["coverage_id"],
        "department_id": last_row["department_id"],
        "segment_id": last_row["segment_id"],
        "account_id": last_row["account_id"],
        "fiscal_year": future_year,
        "period_id": future_month,
        "statement_type": str(last_row["statement_type"]).upper(),
        "is_pl": int(last_row.get("is_pl", 1)),
        "is_bs": int(last_row.get("is_bs", 0)),
        "month_number": future_month,
        "year_number": future_year,
        "quarter": quarter,
        "is_quarter_end": is_quarter_end,
        "is_year_end": is_year_end,
        "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3,
        "lag_6": lag_6, "lag_12": lag_12,
        "rolling_mean_3": rolling_mean_3, "rolling_mean_6": rolling_mean_6,
        "rolling_std_3": rolling_std_3, "rolling_std_6": rolling_std_6,
        "run_rate_recent": rolling_mean_3,
        "mom_change": mom_change,
        "actual_vs_previous_year_abs": actual_vs_previous_year_abs,
        "actual_vs_previous_year_pct": actual_vs_previous_year_pct,
        "ytd_previous": ytd_previous,
        "currency_effect_abs": currency_effect_abs,
        "currency_effect_pct": currency_effect_pct,
        "has_quarter_history": 1 if history_count >= 3 else 0,
        "has_half_year_history": 1 if history_count >= 6 else 0,
        "has_full_year_history": 1 if history_count >= 12 else 0,
        "volatility_ratio": volatility_ratio,
        # Operational model features
        "prev_rf_amount": prev_rf_amount,
        "actual_vs_prev_rf_pct": actual_vs_prev_rf_pct,
        "rf_deviation_rolling_mean_3": rf_deviation_rolling_mean_3,
    }
    return pd.DataFrame([row])


def _fill_missing_features(pred_row: pd.DataFrame, training_columns: list) -> pd.DataFrame:
    pred = pred_row.copy()
    for col in training_columns:
        if col not in pred.columns:
            pred[col] = 0
        if pred[col].isna().any():
            pred[col] = pred[col].fillna(0)
    return pred[training_columns]


def _derive_amount_columns(
    predicted_target: float,
    statement_type: str,
    working_history: pd.DataFrame,
    future_year: int,
    future_month: int,
    ratios: dict,
) -> dict:
    statement_type = str(statement_type).upper()
    if statement_type == "BS":
        ytd_rpt = float(predicted_target)
        previous_same_year_ytd = _same_year_ytd_previous(working_history, future_year)
        mtd_rpt = ytd_rpt - previous_same_year_ytd
    else:
        mtd_rpt = float(predicted_target)
        if future_month == 1:
            ytd_rpt = mtd_rpt
        else:
            previous_same_year_ytd = _same_year_ytd_previous(working_history, future_year)
            ytd_rpt = previous_same_year_ytd + mtd_rpt

    mtd_lcl = mtd_rpt * ratios["rpt_to_lcl"]
    mtd_ccy = mtd_rpt * ratios["rpt_to_ccy"]
    ytd_lcl = ytd_rpt * ratios["ytd_rpt_to_lcl"]
    ytd_ccy = ytd_rpt * ratios["ytd_rpt_to_ccy"]

    return {
        "mtd_rpt_amount": float(mtd_rpt),
        "mtd_lcl_amount": float(mtd_lcl),
        "mtd_ccy_amount": float(mtd_ccy),
        "ytd_rpt_amount": float(ytd_rpt),
        "ytd_lcl_amount": float(ytd_lcl),
        "ytd_ccy_amount": float(ytd_ccy),
    }


# ---------------------------------------------------------------------------
# Main forecast generators
# ---------------------------------------------------------------------------

def generate_forecast_for_single_series(
    history_df: pd.DataFrame,
    model,
    forecast_request: dict,
    training_columns: list | None = None,
    forecast_run_id: int | None = None,
    prev_rf_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Standard forecast for a single series.

    For the operational model, pass prev_rf_df (the previous RF scenario data).
    The working history will be seeded with prev_rf_df values for open future
    periods so the model context reflects the RF baseline, not just actuals.
    """
    include_prev_rf = prev_rf_df is not None
    if training_columns is None:
        training_columns = get_training_columns(include_prev_rf=include_prev_rf)

    if history_df.empty:
        raise ValueError("history_df is empty.")

    _validate_history_for_prediction(history_df)

    scenario_code = forecast_request["scenario_code"]
    if not validate_scenario_code(scenario_code):
        raise ValueError(f"Invalid scenario code: {scenario_code}")

    years_ahead_for_rf = forecast_request.get("years_ahead_for_rf", 2)
    budget_months = forecast_request.get("budget_months", 12)

    scenario_info = parse_scenario_code(scenario_code)
    start_period, end_period = build_forecast_window(
        scenario_info,
        years_ahead_for_rf=years_ahead_for_rf,
        budget_months=budget_months,
    )

    history_df = history_df.sort_values(["fiscal_year", "period_id"]).reset_index(drop=True).copy()
    future_periods = generate_period_range(start_period, end_period)

    last_actual_year = int(history_df["fiscal_year"].iloc[-1])
    last_actual_month = int(history_df["period_id"].iloc[-1])

    def is_after_last(y, m):
        return (y > last_actual_year) or (y == last_actual_year and m > last_actual_month)

    future_periods = [(y, m) for y, m in future_periods if is_after_last(y, m)]
    if not future_periods:
        raise ValueError("No future periods to forecast. Check scenario and history.")

    ratios = infer_currency_ratios(history_df)
    generated_rows = []
    working_history = history_df.copy()

    for future_year, future_month in future_periods:
        pred_row = _build_single_prediction_row(
            working_history, future_year, future_month, prev_rf_df=prev_rf_df
        )
        X_pred = _fill_missing_features(pred_row, training_columns)
        predicted_target = float(model.predict(X_pred)[0])
        statement_type = str(pred_row.iloc[0]["statement_type"]).upper()

        # For the operational model: if we have a prev_rf value and the ML prediction
        # is implausible (>50% deviation from prev_rf), blend toward prev_rf
        if include_prev_rf and pd.notna(pred_row.iloc[0]["prev_rf_amount"]):
            prev_rf_val = float(pred_row.iloc[0]["prev_rf_amount"])
            if prev_rf_val != 0:
                deviation = abs(predicted_target - prev_rf_val) / abs(prev_rf_val)
                if deviation > 0.5:
                    # Blend: 70% ML, 30% prev RF (keeps ML-driven but anchors to baseline)
                    predicted_target = 0.7 * predicted_target + 0.3 * prev_rf_val

        predicted_amounts = _derive_amount_columns(
            predicted_target=predicted_target,
            statement_type=statement_type,
            working_history=working_history,
            future_year=future_year,
            future_month=future_month,
            ratios=ratios,
        )

        generated = {
            "entity_id": int(pred_row.iloc[0]["entity_id"]),
            "coverage_id": int(pred_row.iloc[0]["coverage_id"]),
            "department_id": int(pred_row.iloc[0]["department_id"]),
            "segment_id": int(pred_row.iloc[0]["segment_id"]),
            "account_id": int(pred_row.iloc[0]["account_id"]),
            "fiscal_year": int(future_year),
            "period_id": int(future_month),
            "statement_type": statement_type,
            "scenario_code": scenario_code,
            "forecast_run_id": forecast_run_id,
            "target_amount": float(predicted_target),
            "run_rate_recent": float(pred_row.iloc[0]["run_rate_recent"]) if pd.notna(pred_row.iloc[0]["run_rate_recent"]) else np.nan,
            "volatility_ratio": float(pred_row.iloc[0]["volatility_ratio"]) if pd.notna(pred_row.iloc[0]["volatility_ratio"]) else np.nan,
            "prev_rf_amount": float(pred_row.iloc[0]["prev_rf_amount"]) if pd.notna(pred_row.iloc[0]["prev_rf_amount"]) else np.nan,
        }
        generated.update(predicted_amounts)
        generated_rows.append(generated)

        append_row = pred_row.copy()
        append_row["target_amount"] = predicted_target
        for col, val in predicted_amounts.items():
            append_row[col] = val
        working_history = pd.concat([working_history, append_row], ignore_index=True)

    return pd.DataFrame(generated_rows)
