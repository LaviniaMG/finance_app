from database.load_data import load_dim_entities, load_financial_fact, load_actuals_for_ml

print("=== DimEntity ===")
print(load_dim_entities().head())

print("\n=== FinancialFact ===")
print(load_financial_fact().head())

print("\n=== Actuals for ML ===")
print(load_actuals_for_ml().head())