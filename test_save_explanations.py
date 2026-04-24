from database.load_data import load_actuals_for_ml
from ml.feature_engineering import build_feature_dataset
from ml.train_model import standard_mode_training
from ml.predict import generate_forecast_for_single_series
from ml.business_rules import evaluate_business_rules
from ml.explainability import explain_forecast_output
from database.save_forecast import save_forecast_to_fact
from database.save_explanations import save_explanations_to_db

# 1. Load actuals
df = load_actuals_for_ml()

# 2. Build features
features_df = build_feature_dataset(df, amount_basis="rpt", min_history=1)

# 3. Train
result = standard_mode_training(features_df)
model = result["recommended_model"]
model_name = result["recommended_model_name"]
training_columns = result["training_columns"]
results_table = result["results_table"]

# 4. Request
forecast_request = {
    "scenario_code": "RF07_2025",
    "years_ahead_for_rf": 2,
    "budget_months": 12,
    "model_name": model_name,
    "amount_basis": "rpt",
    "mode": "standard"
}

# 5. Predict
history_df = features_df.sort_values(["fiscal_year", "period_id"]).copy()

forecast_df = generate_forecast_for_single_series(
    history_df=history_df,
    model=model,
    forecast_request=forecast_request,
    training_columns=training_columns,
    forecast_run_id=None
)

# 6. Business rules
forecast_checked_df = evaluate_business_rules(forecast_df)

# 7. Explainability
explain_output = explain_forecast_output(
    forecast_df=forecast_checked_df,
    model=model,
    model_name=model_name,
    training_columns=training_columns,
    results_table=results_table,
    top_n_features=5
)

row_level_explanations_df = explain_output["row_level_explanations_df"]
top_features_df = explain_output["top_features_df"]

# 8. Save forecast
forecast_run_id = save_forecast_to_fact(
    forecast_df=forecast_df,
    forecast_request=forecast_request,
    default_category_id=1
)

# 9. Save explanations
rows_saved = save_explanations_to_db(
    explanations_df=row_level_explanations_df,
    top_features_df=top_features_df,
    forecast_run_id=forecast_run_id
)

print("Forecast run saved:", forecast_run_id)
print("Explanation rows saved:", rows_saved)
