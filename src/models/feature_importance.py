
import joblib
import pandas as pd

MODEL_PATH = "models/lightgbm_demand.pkl"

FEATURES = [
    "price",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_7",
    "rolling_28",
    "wday",
    "month",
    "year",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]

model = joblib.load(MODEL_PATH)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
})

importance["importance_pct"] = (
    importance["importance"]
    / importance["importance"].sum()
    * 100
)

importance = importance.sort_values(
    "importance", ascending=False
)

print("\nLightGBM Feature Importance")
print("=" * 55)
print(importance.round(2).to_string(index=False))
