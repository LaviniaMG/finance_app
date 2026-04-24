import pandas as pd
from database.save_assumptions import save_assumptions_to_db
from database.load_assumptions import load_assumptions_by_scenario

assumptions_df = pd.DataFrame([
    {
        "scenario_code": "RF07_2025",
        "entity_id": 1,
        "coverage_id": 1,
        "department_id": 1,
        "segment_id": 1,
        "account_id": 1,
        "fiscal_year": 2026,
        "period_from": 1,
        "period_to": 12,
        "assumption_type": "growth_pct",
        "assumption_value": 0.02,
        "input_source": "MANUAL",
        "assumption_text": "Increase this account by 2% for 2026",
        "priority_order": 10,
        "is_active": 1,
    },
    {
        "scenario_code": "RF07_2025",
        "entity_id": 1,
        "coverage_id": 1,
        "department_id": 1,
        "segment_id": 1,
        "account_id": 1,
        "fiscal_year": 2026,
        "period_from": 1,
        "period_to": 12,
        "assumption_type": "inflation_pct",
        "assumption_value": 0.05,
        "input_source": "MANUAL",
        "assumption_text": "Apply inflation of 5% for 2026",
        "priority_order": 5,
        "is_active": 1,
    }
])

rows_saved = save_assumptions_to_db(assumptions_df)
print("Rows saved:", rows_saved)

loaded_df = load_assumptions_by_scenario("RF07_2025")
print(loaded_df)