import numpy as np
import pandas as pd


def _safe_pct_change(current, reference):
    if pd.isna(current) or pd.isna(reference) or reference == 0:
        return np.nan
    return (current - reference) / abs(reference)


def add_business_rule_flags(
    forecast_df: pd.DataFrame,
    spike_threshold: float = 0.40,
    volatility_threshold: float = 0.25,
    bs_ytd_drop_threshold: float = -0.20
) -> pd.DataFrame:
    """
    Adds business plausibility checks to forecast output.
    """
    df = forecast_df.copy()
    df = df.sort_values(["entity_id", "coverage_id", "department_id", "segment_id", "account_id", "period_id"]).reset_index(drop=True)

    # Required columns fallback
    for col in [
        "statement_type",
        "mtd_rpt_amount",
        "ytd_rpt_amount",
        "run_rate_recent",
        "volatility_ratio"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # 1. Spike vs run rate
    df["spike_vs_run_rate_pct"] = np.where(
        df["run_rate_recent"].notna() & (df["run_rate_recent"] != 0),
        (df["mtd_rpt_amount"] - df["run_rate_recent"]) / df["run_rate_recent"].abs(),
        np.nan
    )

    df["spike_flag"] = (
        df["spike_vs_run_rate_pct"].abs() > spike_threshold
    ).astype(int)

    # 2. Negative values (simple generic guardrail)
    # For now: if statement_type is PL and mtd is negative, flag it
    df["negative_value_flag"] = np.where(
        (df["statement_type"].str.upper() == "PL") & (df["mtd_rpt_amount"] < 0),
        1,
        0
    )

    # 3. High volatility
    df["high_volatility_flag"] = np.where(
        df["volatility_ratio"].notna() & (df["volatility_ratio"] > volatility_threshold),
        1,
        0
    )

    # 4. BS YTD suspicious drop
    group_keys = ["entity_id", "coverage_id", "department_id", "segment_id", "account_id"]
    df["previous_ytd_rpt"] = df.groupby(group_keys, dropna=False)["ytd_rpt_amount"].shift(1)

    df["bs_ytd_change_pct"] = np.where(
        (df["statement_type"].str.upper() == "BS") &
        df["previous_ytd_rpt"].notna() &
        (df["previous_ytd_rpt"] != 0),
        (df["ytd_rpt_amount"] - df["previous_ytd_rpt"]) / abs(df["previous_ytd_rpt"]),
        np.nan
    )

    df["bs_ytd_drop_flag"] = np.where(
        (df["statement_type"].str.upper() == "BS") &
        df["bs_ytd_change_pct"].notna() &
        (df["bs_ytd_change_pct"] < bs_ytd_drop_threshold),
        1,
        0
    )

    return df


def assign_confidence_level(flag_row: pd.Series) -> str:
    """
    Simple rule-based confidence score.
    """
    risk_score = 0

    risk_score += int(flag_row.get("spike_flag", 0))
    risk_score += int(flag_row.get("negative_value_flag", 0))
    risk_score += int(flag_row.get("high_volatility_flag", 0))
    risk_score += int(flag_row.get("bs_ytd_drop_flag", 0))

    if risk_score == 0:
        return "High"
    if risk_score == 1:
        return "Medium"
    return "Low"


def build_business_warning_text(flag_row: pd.Series) -> str:
    warnings = []

    if int(flag_row.get("spike_flag", 0)) == 1:
        warnings.append("forecastul diferă semnificativ față de run-rate-ul recent")

    if int(flag_row.get("negative_value_flag", 0)) == 1:
        warnings.append("valoarea forecastată este negativă pentru un cont de tip P&L")

    if int(flag_row.get("high_volatility_flag", 0)) == 1:
        warnings.append("seria istorică este volatilă, ceea ce reduce stabilitatea predicției")

    if int(flag_row.get("bs_ytd_drop_flag", 0)) == 1:
        warnings.append("YTD pentru contul de bilanț scade neobișnuit de mult față de perioada anterioară")

    if not warnings:
        return "Forecastul respectă regulile de business definite și nu prezintă anomalii majore."

    return "Atenție: " + "; ".join(warnings) + "."


def evaluate_business_rules(
    forecast_df: pd.DataFrame,
    spike_threshold: float = 0.40,
    volatility_threshold: float = 0.25,
    bs_ytd_drop_threshold: float = -0.20
) -> pd.DataFrame:
    """
    Main function: add flags + confidence + business warning text.
    """
    df = add_business_rule_flags(
        forecast_df=forecast_df,
        spike_threshold=spike_threshold,
        volatility_threshold=volatility_threshold,
        bs_ytd_drop_threshold=bs_ytd_drop_threshold
    )

    df["confidence_level"] = df.apply(assign_confidence_level, axis=1)
    df["business_warning_text"] = df.apply(build_business_warning_text, axis=1)

    return df