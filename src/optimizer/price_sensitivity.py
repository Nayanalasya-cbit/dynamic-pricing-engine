import numpy as np
import pandas as pd
import joblib


DATA_PATH = "data/processed/m5_features.csv"

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
    "snap_WI"
]


def price_sensitivity(product_id, candidate_prices):

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    product_data = df[df["id"] == product_id].copy()

    if product_data.empty:
        raise ValueError(f"Product {product_id} not found")

    latest = product_data.iloc[-1]

    model = joblib.load("models/quantile_p50.pkl")

    rows = []

    for price in candidate_prices:

        row = latest.copy()
        row["price"] = price

        X = pd.DataFrame([row])[FEATURES]

        predicted_demand = model.predict(X)[0]
        predicted_demand = max(0, predicted_demand)

        rows.append({
            "price": price,
            "predicted_demand": predicted_demand
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    product_id = "FOODS_3_090_CA_3_validation"

    current_price = 1.60

    candidate_prices = [
        round(current_price * x, 2)
        for x in [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    ]

    result = price_sensitivity(
        product_id,
        candidate_prices
    )

    print("\nPrice Sensitivity Analysis")
    print("=" * 50)

    print(result.to_string(index=False))

    print("\nDemand Change")
    print("=" * 50)

    base_demand = result.iloc[4]["predicted_demand"]

    for _, row in result.iterrows():

        demand_change = (
            (row["predicted_demand"] - base_demand)
            / base_demand
        ) * 100

        print(
            f"Price {row['price']:.2f} | "
            f"Demand {row['predicted_demand']:.2f} | "
            f"Change {demand_change:.2f}%"
        )
        
    valid = result[result["predicted_demand"] > 0]

    if len(valid) >= 2:
        elasticity = np.polyfit(
            np.log(valid["price"]),
            np.log(valid["predicted_demand"]),
            1
        )[0]

        print("\nModel-Implied Price Sensitivity")
        print("=" * 40)
        print(f"Estimated elasticity: {elasticity:.4f}")
        print("Interpretation: model-implied, not causal.")