from database.load_data import load_actuals_for_ml
from ml.feature_engineering import build_feature_dataset
from ml.train_model import standard_mode_training, save_trained_model

df = load_actuals_for_ml()
features_df = build_feature_dataset(df, amount_basis="rpt", min_history=1)

result = standard_mode_training(features_df)

print("=== Model comparison ===")
print(result["results_table"])

print("\nRecommended model:")
print(result["recommended_model_name"])

print("\nTraining rows:", result["n_training_rows"])

model_path = save_trained_model(
    result["recommended_model"],
    result["recommended_model_name"]
)

print("\nSaved model to:", model_path)