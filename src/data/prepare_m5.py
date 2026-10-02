import pandas as pd

SALES_PATH = "data/raw/m5/sales_train_validation.csv"
CALENDAR_PATH = "data/raw/m5/calendar.csv"
PRICES_PATH = "data/raw/m5/sell_prices.csv"

TOP_N_SERIES = 500


def prepare_m5():

    print("Loading sales data...")
    sales = pd.read_csv(SALES_PATH)

    day_columns = [col for col in sales.columns if col.startswith("d_")]

    print(f"Total series: {len(sales)}")
    print(f"Total days: {len(day_columns)}")

    print("Selecting top series...")

    sales["total_sales"] = sales[day_columns].sum(axis=1)

    sales = sales.sort_values(
        "total_sales",
        ascending=False
    ).head(TOP_N_SERIES)

    sales = sales.drop(columns=["total_sales"])

    print(f"Selected series: {len(sales)}")

    print("Converting sales from wide to long format...")

    id_columns = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id"
    ]

    sales_long = sales.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="units_sold"
    )

    print(f"Long sales shape: {sales_long.shape}")

    print("Loading calendar...")

    calendar = pd.read_csv(CALENDAR_PATH)

    calendar = calendar[
        [
            "d",
            "date",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI"
        ]
    ]

    print("Joining calendar...")

    data = sales_long.merge(
        calendar,
        on="d",
        how="left"
    )

    print(f"After calendar join: {data.shape}")

    print("Loading prices...")

    prices = pd.read_csv(PRICES_PATH)

    print("Joining prices...")

    data = data.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    data = data.rename(
        columns={"sell_price": "price"}
    )

    print(f"After price join: {data.shape}")

    print("Removing rows without price...")

    data = data.dropna(
        subset=["price"]
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    output_path = "data/processed/m5_pricing_data.csv"

    data.to_csv(
        output_path,
        index=False
    )

    print(f"Saved dataset to: {output_path}")
    print(f"Final shape: {data.shape}")


if __name__ == "__main__":
    prepare_m5()