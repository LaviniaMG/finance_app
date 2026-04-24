import pandas as pd
import numpy as np


AMOUNT_COLUMNS = {
    "rpt": {"mtd": "mtd_rpt_amount", "ytd": "ytd_rpt_amount"},
    "lcl": {"mtd": "mtd_lcl_amount", "ytd": "ytd_lcl_amount"},
    "ccy": {"mtd": "mtd_ccy_amount", "ytd": "ytd_ccy_amount"},
}

SERIES_KEYS = [
    "entity_id",
    "coverage_id",
    "department_id",
    "segment_id",
    "account_id",
]


def _validate_amount_basis(amount_basis: str) -> None:
    if amount_basis not in AMOUNT_COLUMNS:
        raise ValueError(
            f"amount_basis must be one of {list(AMOUNT_COLUMNS.keys())}, got '{amount_basis}'"
        )


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["month_number"] = df["period_id"]
    df["year_number"] = df["fiscal_year"]
    df["quarter"] = ((df["month_number"] - 1) // 3) + 1

    df["is_quarter_end"] = df["month_number"].isin([3, 6, 9, 12]).astype(int)
    df["is_year_end"] = (df["month_number"] == 12).astype(int)

    return df


def _add_account_type_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "statement_type" not in df.columns:
        raise ValueError(
            "statement_type column is missing. "
            "Join DimAccount and include statement_type='PL'/'BS'."
        )

    df["statement_type"] = df["statement_type"].str.upper().fillna("UNKNOWN")
    df["is_pl"] = (df["statement_type"] == "PL").astype(int)
    df["is_bs"] = (df["statement_type"] == "BS").astype(int)

    return df


def _prepare_target_column(df: pd.DataFrame, amount_basis: str) -> pd.DataFrame:
    df = df.copy()
    _validate_amount_basis(amount_basis)

    mtd_col = AMOUNT_COLUMNS[amount_basis]["mtd"]
    ytd_col = AMOUNT_COLUMNS[amount_basis]["ytd"]

    df["target_amount"] = np.where(
        df["statement_type"] == "BS",
        df[ytd_col],
        df[mtd_col]
    )

    return df


def _add_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(SERIES_KEYS + ["fiscal_year", "period_id"]).reset_index(drop=True)

    df["year_month_key"] = df["fiscal_year"] * 100 + df["period_id"]

    df["lag_1"] = df.groupby(SERIES_KEYS, dropna=False)["target_amount"].shift(1)
    df["lag_2"] = df.groupby(SERIES_KEYS, dropna=False)["target_amount"].shift(2)
    df["lag_3"] = df.groupby(SERIES_KEYS, dropna=False)["target_amount"].shift(3)
    df["lag_6"] = df.groupby(SERIES_KEYS, dropna=False)["target_amount"].shift(6)
    df["lag_12"] = df.groupby(SERIES_KEYS, dropna=False)["target_amount"].shift(12)

    df["rolling_mean_3"] = (
        df.groupby(SERIES_KEYS, dropna=False)["target_amount"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    )

    df["rolling_mean_6"] = (
        df.groupby(SERIES_KEYS, dropna=False)["target_amount"]
        .transform(lambda s: s.shift(1).rolling(window=6, min_periods=1).mean())
    )

    df["rolling_std_3"] = (
        df.groupby(SERIES_KEYS, dropna=False)["target_amount"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=2).std())
    )

    df["rolling_std_6"] = (
        df.groupby(SERIES_KEYS, dropna=False)["target_amount"]
        .transform(lambda s: s.shift(1).rolling(window=6, min_periods=2).std())
    )

    df["run_rate_recent"] = df["rolling_mean_3"]

    df["mom_change"] = np.where(
        df["lag_1"].notna() & (df["lag_1"] != 0),
        (df["target_amount"] - df["lag_1"]) / df["lag_1"],
        np.nan
    )

    df["actual_vs_previous_year_abs"] = np.where(
        df["lag_12"].notna(),
        df["target_amount"] - df["lag_12"],
        np.nan
    )

    df["actual_vs_previous_year_pct"] = np.where(
        df["lag_12"].notna() & (df["lag_12"] != 0),
        (df["target_amount"] - df["lag_12"]) / df["lag_12"],
        np.nan
    )

    return df


def _add_ytd_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ytd_rpt_amount" in df.columns:
        ytd_grouped = df.groupby(SERIES_KEYS, dropna=False)["ytd_rpt_amount"]
        df["ytd_previous"] = ytd_grouped.shift(1)
    else:
        df["ytd_previous"] = np.nan

    return df


def _add_currency_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "mtd_lcl_amount" in df.columns and "mtd_rpt_amount" in df.columns:
        df["currency_effect_abs"] = df["mtd_rpt_amount"] - df["mtd_lcl_amount"]
        df["currency_effect_pct"] = np.where(
            df["mtd_lcl_amount"].notna() & (df["mtd_lcl_amount"] != 0),
            (df["mtd_rpt_amount"] - df["mtd_lcl_amount"]) / df["mtd_lcl_amount"],
            np.nan
        )
    else:
        df["currency_effect_abs"] = np.nan
        df["currency_effect_pct"] = np.nan

    return df


def _add_business_guardrail_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["history_count"] = (
        df.groupby(SERIES_KEYS, dropna=False)["target_amount"].cumcount()
    )

    df["has_quarter_history"] = (df["history_count"] >= 3).astype(int)
    df["has_half_year_history"] = (df["history_count"] >= 6).astype(int)
    df["has_full_year_history"] = (df["history_count"] >= 12).astype(int)

    df["volatility_ratio"] = np.where(
        df["rolling_mean_6"].notna() & (df["rolling_mean_6"] != 0),
        df["rolling_std_6"] / df["rolling_mean_6"].abs(),
        np.nan
    )

    return df


def build_feature_dataset(
    fact_df: pd.DataFrame,
    amount_basis: str = "rpt",
    dropna_target: bool = True,
    min_history: int = 3
) -> pd.DataFrame:
    df = fact_df.copy()

    if df.columns.duplicated().any():
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns found in input dataframe: {duplicated_cols}")

    required_columns = [
        "entity_id",
        "coverage_id",
        "department_id",
        "segment_id",
        "account_id",
        "fiscal_year",
        "period_id",
        "statement_type",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    df = _add_calendar_features(df)
    df = _add_account_type_flags(df)
    df = _prepare_target_column(df, amount_basis)
    df = _add_series_features(df)
    df = _add_ytd_features(df)
    df = _add_currency_features(df)
    df = _add_business_guardrail_features(df)

    if dropna_target:
        df = df[df["target_amount"].notna()].copy()

    if min_history is not None:
        df = df[df["history_count"] >= min_history].copy()

    return df


def get_training_columns() -> list:
    return [
        "month_number",
        "year_number",
        "quarter",
        "is_quarter_end",
        "is_year_end",
        "is_pl",
        "is_bs",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_std_3",
        "rolling_std_6",
        "run_rate_recent",
        "mom_change",
        "actual_vs_previous_year_abs",
        "actual_vs_previous_year_pct",
        "ytd_previous",
        "currency_effect_abs",
        "currency_effect_pct",
        "has_quarter_history",
        "has_half_year_history",
        "has_full_year_history",
        "volatility_ratio",
    ]