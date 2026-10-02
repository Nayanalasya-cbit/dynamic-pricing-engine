from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
M5_DIR = PROJECT_ROOT / "data" / "raw" / "m5"


def load_m5_data():
    calendar = pd.read_csv(M5_DIR / "calendar.csv")
    sales = pd.read_csv(M5_DIR / "sales_train_validation.csv")
    prices = pd.read_csv(M5_DIR / "sell_prices.csv")

    return calendar, sales, prices


if __name__ == "__main__":
    calendar, sales, prices = load_m5_data()

    print("Calendar shape:", calendar.shape)
    print("Sales shape:", sales.shape)
    print("Prices shape:", prices.shape)

    print("\nCalendar columns:")
    print(calendar.columns.tolist())

    print("\nSales columns:")
    print(sales.columns.tolist()[:10], "...")

    print("\nPrice columns:")
    print(prices.columns.tolist())
