import os
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config.settings import MODEL_FOLDER, RANDOM_STATE, TEST_SIZE
from ml.feature_engineering import get_training_columns


# ----------------------------
# Basic helpers
# ----------------------------

def _ensure_model_folder():
    os.makedirs(MODEL_FOLDER, exist_ok=True)


def _safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def _safe_smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def _safe_wape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan

    return np.sum(np.abs(y_true - y_pred)) / denominator * 100


def _forecast_bias(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan

    return np.sum(y_pred - y_true) / denominator * 100


def calculate_metrics(y_true, y_pred):
    r2 = np.nan
    if len(y_true) >= 2:
        try:
            r2 = r2_score(y_true, y_pred)
        except Exception:
            r2 = np.nan

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": _safe_mape(y_true, y_pred),
        "sMAPE": _safe_smape(y_true, y_pred),
        "WAPE": _safe_wape(y_true, y_pred),
        "Bias": _forecast_bias(y_true, y_pred),
        "R2": r2,
    }


# ----------------------------
# Data preparation
# ----------------------------

def prepare_training_data(feature_df: pd.DataFrame, min_non_null_ratio: float = 0.6):
    training_columns = get_training_columns()

    required_columns = training_columns + ["target_amount", "fiscal_year", "period_id"]
    missing = [c for c in required_columns if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing columns for training: {missing}")

    df = feature_df.copy()
    df = df[df["target_amount"].notna()].copy()

    if df.empty:
        raise ValueError("Training dataset is empty because target_amount is null.")

    usable_training_columns = []
    for col in training_columns:
        non_null_ratio = df[col].notna().mean()
        if non_null_ratio >= min_non_null_ratio:
            usable_training_columns.append(col)

    if not usable_training_columns:
        raise ValueError("No usable training columns available after null filtering.")

    for col in usable_training_columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.sort_values(["fiscal_year", "period_id"]).reset_index(drop=True)
    X = df[usable_training_columns].copy()
    y = df["target_amount"].copy()

    return df, X, y, usable_training_columns

def time_based_train_test_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE):
    df = df.sort_values(["fiscal_year", "period_id"]).reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    split_index = int(len(df) * (1 - test_size))

    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Not enough data for time-based train/test split.")

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

# ----------------------------
# Models
# ----------------------------

def get_available_models():
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=2,
            random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }


def get_advanced_model(model_name: str, params: dict):
    if model_name == "linear_regression":
        return LinearRegression(**params)

    if model_name == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, **params)

    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)

    raise ValueError(f"Unsupported model_name: {model_name}")


def get_model_explainability_score(model_name: str) -> int:
    """
    Higher = more explainable
    """
    scores = {
        "linear_regression": 3,
        "random_forest": 2,
        "gradient_boosting": 1,
    }
    return scores.get(model_name, 0)


# ----------------------------
# Training
# ----------------------------

def train_single_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)

    return model, predictions, metrics


def walk_forward_validation_for_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    min_train_size: int = 3,
    test_window: int = 1
):
    """
    Simple expanding-window backtest.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if len(X) < (min_train_size + test_window):
        raise ValueError("Not enough rows for walk-forward validation.")

    fold_results = []

    for split_end in range(min_train_size, len(X) - test_window + 1):
        X_train = X.iloc[:split_end]
        y_train = y.iloc[:split_end]

        X_test = X.iloc[split_end:split_end + test_window]
        y_test = y.iloc[split_end:split_end + test_window]

        model_instance = clone(model)
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)

        fold_metrics = calculate_metrics(y_test, y_pred)
        fold_metrics["train_size"] = len(X_train)
        fold_metrics["test_size"] = len(X_test)
        fold_results.append(fold_metrics)

    return pd.DataFrame(fold_results)


def aggregate_backtest_metrics(backtest_df: pd.DataFrame) -> dict:
    if backtest_df.empty:
        raise ValueError("backtest_df is empty.")

    metric_cols = ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "Bias", "R2"]

    aggregated = {}
    for col in metric_cols:
        aggregated[f"{col}_mean"] = backtest_df[col].mean(skipna=True)
        aggregated[f"{col}_std"] = backtest_df[col].std(skipna=True)

    aggregated["n_folds"] = len(backtest_df)

    return aggregated


def recommend_best_model(results_df: pd.DataFrame) -> str:
    """
    Recommendation logic:
    1. prioritize lower RMSE_mean
    2. if models are very close, prefer more explainable one
    """
    df = results_df.copy()

    if "RMSE_mean" not in df.columns:
        raise ValueError("results_df must contain RMSE_mean.")

    df["explainability_score"] = df["model_name"].apply(get_model_explainability_score)

    best_rmse = df["RMSE_mean"].min()

    # consider models within 5% of best RMSE as "close"
    threshold = best_rmse * 1.05
    candidates = df[df["RMSE_mean"] <= threshold].copy()

    # among close models, prefer higher explainability
    candidates = candidates.sort_values(
        by=["explainability_score", "RMSE_mean"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return candidates.iloc[0]["model_name"]


# ----------------------------
# Standard mode
# ----------------------------

def standard_mode_training(feature_df: pd.DataFrame):
    df, X, y, training_columns = prepare_training_data(feature_df)

    results = []
    trained_models = {}
    backtest_outputs = {}

    for model_name, model in get_available_models().items():
        # backtesting
        backtest_df = walk_forward_validation_for_model(
            model=model,
            X=X,
            y=y,
            min_train_size=max(3, int(len(X) * 0.5)),
            test_window=1
        )

        aggregated = aggregate_backtest_metrics(backtest_df)
        aggregated["model_name"] = model_name
        results.append(aggregated)
        backtest_outputs[model_name] = backtest_df

        # final full training on all available rows
        final_model = clone(model)
        final_model.fit(X, y)
        trained_models[model_name] = final_model

    results_df = pd.DataFrame(results).sort_values(by="RMSE_mean", ascending=True).reset_index(drop=True)

    recommended_model_name = recommend_best_model(results_df)
    recommended_model = trained_models[recommended_model_name]

    return {
        "results_table": results_df,
        "recommended_model_name": recommended_model_name,
        "recommended_model": recommended_model,
        "training_columns": training_columns,
        "n_training_rows": len(X),
        "backtest_outputs": backtest_outputs,
    }


# ----------------------------
# Advanced mode
# ----------------------------

def advanced_mode_training(feature_df: pd.DataFrame, model_name: str, model_params: dict | None = None):
    if model_params is None:
        model_params = {}

    df, X, y, training_columns = prepare_training_data(feature_df)

    model = get_advanced_model(model_name, model_params)

    backtest_df = walk_forward_validation_for_model(
        model=model,
        X=X,
        y=y,
        min_train_size=max(3, int(len(X) * 0.5)),
        test_window=1
    )

    aggregated = aggregate_backtest_metrics(backtest_df)

    final_model = clone(model)
    final_model.fit(X, y)

    return {
        "model_name": model_name,
        "model": final_model,
        "metrics": aggregated,
        "training_columns": training_columns,
        "n_training_rows": len(X),
        "backtest_output": backtest_df,
    }


# ----------------------------
# Save model
# ----------------------------

def save_trained_model(model, model_name: str):
    _ensure_model_folder()
    model_path = os.path.join(MODEL_FOLDER, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    return model_path