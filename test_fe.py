from database.load_data import load_actuals_for_ml
from ml.feature_engineering import build_feature_dataset, get_training_columns

df = load_actuals_for_ml()

print("Raw columns:")
print(df.columns.tolist())

features_df = build_feature_dataset(df, amount_basis="rpt", min_history=1)

print(features_df.head(10))
print("\nShape:", features_df.shape)
print("\nTraining columns:")
print(get_training_columns())