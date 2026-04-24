from database.load_data import load_actuals_for_ml
from ml.feature_engineering import build_feature_dataset
from ml.train_model import standard_mode_training
from ml.predict import generate_forecast_for_single_series
from ml.business_rules import evaluate_business_rules
from ml.explainability import explain_forecast_output

# 1. Load actuals
df = load_actuals_for_ml()

# 2. Build features
features_df = build_feature_dataset(df, amount_basis="rpt", min_history=1)

# 3. Train models
result = standard_mode_training(features_df)

model_name = result["recommended_model_name"]
model = result["recommended_model"]
training_columns = result["training_columns"]
results_table = result["results_table"]

# 4. Forecast request
forecast_request = {
    "scenario_code": "RF07_2025",
    "years_ahead_for_rf": 2,
    "budget_months": 12,
    "model_name": model_name,
    "amount_basis": "rpt",
    "mode": "standard"
}

# 5. Generate forecast
history_df = features_df.sort_values("period_id").copy()

forecast_df = generate_forecast_for_single_series(
    history_df=history_df,
    model=model,
    forecast_request=forecast_request,
    training_columns=training_columns,
    forecast_run_id=1
)

# 6. Add business rules
forecast_checked_df = evaluate_business_rules(forecast_df)

# 7. Explain forecast
explain_output = explain_forecast_output(
    forecast_df=forecast_checked_df,
    model=model,
    model_name=model_name,
    training_columns=training_columns,
    results_table=results_table,
    top_n_features=5
)

print("=== Feature importance ===")
print(explain_output["feature_importance_df"].head(10))

print("\n=== Top features ===")
print(explain_output["top_features_df"])

print("\n=== Model recommendation ===")
print(explain_output["model_recommendation_text"])

print("\n=== Row level explanations ===")
print(explain_output["row_level_explanations_df"].head(5))