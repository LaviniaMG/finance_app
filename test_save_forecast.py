from database.load_data import load_actuals_for_ml
from ml.feature_engineering import build_feature_dataset
from ml.train_model import standard_mode_training, save_trained_model
from ml.predict import generate_forecast_for_single_series

df = load_actuals_for_ml()
features_df = build_feature_dataset(df, amount_basis="rpt", min_history=1)

result = standard_mode_training(features_df)

model_name = result["recommended_model_name"]
model = result["recommended_model"]
training_columns = result["training_columns"]

save_trained_model(model, model_name)

forecast_request = {
    "scenario_code": "RF07_2025",
    "years_ahead_for_rf": 2,
    "budget_months": 12,
    "model_name": model_name,
    "amount_basis": "rpt",
    "mode": "standard"
}

history_df = features_df.sort_values(["fiscal_year", "period_id"]).copy()

forecast_df = generate_forecast_for_single_series(
    history_df=history_df,
    model=model,
    forecast_request=forecast_request,
    training_columns=training_columns,
    forecast_run_id=1
)

print("=== Forecast result ===")
print(
    forecast_df[
        [
            "fiscal_year",
            "period_id",
            "scenario_code",
            "statement_type",
            "mtd_rpt_amount",
            "ytd_rpt_amount"
        ]
    ].head(30)
)
print("\nShape:", forecast_df.shape)