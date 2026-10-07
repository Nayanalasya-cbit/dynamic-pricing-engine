import pandas as pd
import numpy as np
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


def simulate_risk(
    product_id,
    candidate_prices,
    cost_ratio=0.70,
    risk_aversion=0.5
):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    product_data = df[df["id"] == product_id].copy()

    if product_data.empty:
        raise ValueError(f"Product {product_id} not found")

    latest = product_data.iloc[-1]

    current_price = latest["price"]
    unit_cost = current_price * cost_ratio

    p10_model = joblib.load("models/quantile_p10.pkl")
    p50_model = joblib.load("models/quantile_p50.pkl")
    p90_model = joblib.load("models/quantile_p90.pkl")

    rows = []

    for price in candidate_prices:

        row = latest.copy()
        row["price"] = price

        X = pd.DataFrame([row])[FEATURES]

        p10 = p10_model.predict(X)[0]
        p50 = p50_model.predict(X)[0]
        p90 = p90_model.predict(X)[0]

        # Demand cannot be negative
        p10 = max(0, p10)
        p50 = max(0, p50)
        p90 = max(0, p90)

        # Ensure P10 <= P50 <= P90
        p10, p50, p90 = sorted([p10, p50, p90])

        p10_profit = (price - unit_cost) * p10
        p50_profit = (price - unit_cost) * p50
        p90_profit = (price - unit_cost) * p90

        downside = p50_profit - p10_profit

        risk_adjusted_profit = (
            p50_profit - risk_aversion * downside
        )

        rows.append({
            "product_id": product_id,
            "price": price,
            "p10_demand": p10,
            "p50_demand": p50,
            "p90_demand": p90,
            "p10_profit": p10_profit,
            "p50_profit": p50_profit,
            "p90_profit": p90_profit,
            "downside_risk": downside,
            "risk_adjusted_profit": risk_adjusted_profit
        })

    return pd.DataFrame(rows)


def optimize_risk_aware_price(
    product_id,
    current_price,
    cost_ratio=0.70,
    max_price_change=0.20,
    risk_aversion=0.5
):

    min_price = current_price * (1 - max_price_change)
    max_price = current_price * (1 + max_price_change)

    prices = [
        round(min_price + i * 0.04, 2)
        for i in range(
            int(round((max_price - min_price) / 0.04)) + 1
        )
    ]

    simulation = simulate_risk(
        product_id,
        prices,
        cost_ratio,
        risk_aversion
    )

    best_row = simulation.loc[
        simulation["risk_adjusted_profit"].idxmax()
    ]

    return best_row, simulation


if __name__ == "__main__":

    product_id = "FOODS_3_090_CA_3_validation"
    current_price = 1.60

    best_price, simulation = optimize_risk_aware_price(
        product_id=product_id,
        current_price=current_price,
        cost_ratio=0.70,
        max_price_change=0.20,
        risk_aversion=0.5
    )

    print("\nRisk-Aware Price Simulation")
    print("=" * 70)

    print(
        simulation[
            [
                "price",
                "p10_demand",
                "p50_demand",
                "p90_demand",
                "p50_profit",
                "downside_risk",
                "risk_adjusted_profit"
            ]
        ].to_string(index=False)
    )

    print("\nRecommended Price")
    print("=" * 70)
    print("Current Price:", current_price)
    print("Recommended Price:", best_price["price"])
    print("P50 Demand:", best_price["p50_demand"])
    print("P50 Profit:", best_price["p50_profit"])
    print("Downside Risk:", best_price["downside_risk"])
    print("Risk-Adjusted Profit:", best_price["risk_adjusted_profit"])