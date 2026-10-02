import pandas as pd


INPUT_PATH = "data/processed/m5_pricing_data.csv"
OUTPUT_PATH = "data/processed/m5_features.csv"


def build_features():

    print("Loading processed M5 data...")

    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])

    # Sort each item-store series chronologically
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    print("Creating lag features...")

    # Previous-day demand
    df["lag_1"] = (
        df.groupby("id")["units_sold"]
        .shift(1)
    )

    # Demand 7 days ago
    df["lag_7"] = (
        df.groupby("id")["units_sold"]
        .shift(7)
    )

    # Demand 28 days ago
    df["lag_28"] = (
        df.groupby("id")["units_sold"]
        .shift(28)
    )

    print("Creating rolling features...")

    # Shift first so today's demand is never included
    previous_sales = (
        df.groupby("id")["units_sold"]
        .shift(1)
    )

    df["rolling_7"] = (
        previous_sales
        .groupby(df["id"])
        .transform(lambda x: x.rolling(7).mean())
    )

    df["rolling_28"] = (
        previous_sales
        .groupby(df["id"])
        .transform(lambda x: x.rolling(28).mean())
    )

    print("Removing rows with incomplete history...")

    df = df.dropna(
        subset=[
            "lag_1",
            "lag_7",
            "lag_28",
            "rolling_7",
            "rolling_28"
        ]
    )

    print("Saving feature dataset...")

    df.to_csv(OUTPUT_PATH, index=False)

    print("Feature engineering complete.")
    print("Final shape:", df.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    build_features()