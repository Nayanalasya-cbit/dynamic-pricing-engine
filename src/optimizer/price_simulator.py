import pandas as pd
import joblib

DATA_PATH = "data/processed/m5_features.csv"
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
    "snap_WI"
]


def simulate_prices(product_id, prices, cost_ratio=0.70):

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    product_data = df[df["id"] == product_id].copy()

    if product_data.empty:
        raise ValueError(f"Product {product_id} not found")

    latest = product_data.iloc[-1]

    rows = []

    for price in prices:
        row = latest.copy()
        row["price"] = price
        rows.append(row)

    simulation_data = pd.DataFrame(rows).reset_index(drop=True)

    model = joblib.load(MODEL_PATH)

    X_simulation = simulation_data[FEATURES]

    predictions = model.predict(X_simulation)

    simulation_data["predicted_demand"] = predictions

    simulation_data["revenue"] = (
        simulation_data["price"] *
        simulation_data["predicted_demand"]
    )

    simulation_data["estimated_cost"] = (
        latest["price"] * cost_ratio
    )

    simulation_data["profit"] = (
        simulation_data["price"] -
        simulation_data["estimated_cost"]
    ) * simulation_data["predicted_demand"]

    return simulation_data[
        [
            "id",
            "price",
            "predicted_demand",
            "revenue",
            "estimated_cost",
            "profit"
        ]
    ]
def optimize_price(
    product_id,
    current_price,
    cost_ratio=0.70,
    max_price_change=0.20
):

    min_price = current_price * (1 - max_price_change)
    max_price = current_price * (1 + max_price_change)

    prices = [
        round(min_price + i * 0.04, 2)
        for i in range(
            int(round((max_price - min_price) / 0.04)) + 1
        )
    ]

    simulation = simulate_prices(
        product_id,
        prices,
        cost_ratio
    )

    best_row = simulation.loc[
        simulation["profit"].idxmax()
    ]

    return best_row, simulation

if __name__ == "__main__":

    current_price = 1.60

    best_price, simulation = optimize_price(
        "FOODS_3_090_CA_3_validation",
        current_price,
        cost_ratio=0.70,
        max_price_change=0.10
    )

    print("\nPRICE SIMULATION")
    print("----------------")
    print(simulation.to_string(index=False))

    print("\nPRICE CONSTRAINT")
    print("-----------------")
    print("Current price      :", current_price)
    print("Minimum allowed    :", round(current_price * 0.80, 2))
    print("Maximum allowed    :", round(current_price * 1.20, 2))

    print("\nRECOMMENDED PRICE")
    print("-----------------")
    print("Price :", best_price["price"])
    print("Demand:", best_price["predicted_demand"])
    print("Revenue:", best_price["revenue"])
    print("Profit :", best_price["profit"])