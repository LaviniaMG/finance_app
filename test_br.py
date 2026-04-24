from database.load_data import load_actuals_for_ml
from ml.feature_engineering import build_feature_dataset
from ml.train_model import standard_mode_training
from ml.predict import generate_forecast_for_single_series
from ml.business_rules import evaluate_business_rules

# Load actuals
df = load_actuals_for_ml()

# Build features
features_df = build_feature_dataset(df, amount_basis="rpt", min_history=1)

# Train model
result = standard_mode_training(features_df)
model = result["recommended_model"]
training_columns = result["training_columns"]

# Forecast request
forecast_request = {
    "scenario_code": "RF07_2025",
    "years_ahead_for_rf": 2,
    "budget_months": 12,
    "model_name": result["recommended_model_name"],
    "amount_basis": "rpt",
    "mode": "standard"
}

# Generate forecast
history_df = features_df.sort_values("period_id").copy()

forecast_df = generate_forecast_for_single_series(
    history_df=history_df,
    model=model,
    forecast_request=forecast_request,
    training_columns=training_columns,
    forecast_run_id=1
)

# Apply business rules
checked_df = evaluate_business_rules(forecast_df)

print(
    checked_df[
        [
            "period_id",
            "statement_type",
            "mtd_rpt_amount",
            "ytd_rpt_amount",
            "run_rate_recent",
            "volatility_ratio",
            "spike_flag",
            "negative_value_flag",
            "high_volatility_flag",
            "bs_ytd_drop_flag",
            "confidence_level",
            "business_warning_text"
        ]
    ].head(20)
)