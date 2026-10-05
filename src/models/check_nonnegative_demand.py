import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_PATH = "data/processed/m5_features.csv"

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

TARGET = "units_sold"


def main():

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    X = df[FEATURES]
    y = df[TARGET]

    split_date = df["date"].quantile(0.8)

    train_mask = df["date"] <= split_date
    test_mask = df["date"] > split_date

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    negative_count = np.sum(predictions < 0)
    negative_percentage = negative_count / len(predictions) * 100

    clipped_predictions = np.maximum(predictions, 0)

    raw_mae = mean_absolute_error(y_test, predictions)
    raw_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    clipped_mae = mean_absolute_error(y_test, clipped_predictions)
    clipped_rmse = np.sqrt(mean_squared_error(y_test, clipped_predictions))

    print("\nNEGATIVE DEMAND CHECK")
    print("=" * 40)

    print(f"Total predictions : {len(predictions)}")
    print(f"Negative values   : {negative_count}")
    print(f"Negative %        : {negative_percentage:.4f}%")

    print("\nPrediction range")
    print("-" * 40)
    print(f"Minimum : {predictions.min():.4f}")
    print(f"Maximum : {predictions.max():.4f}")

    print("\nRAW MODEL")
    print("-" * 40)
    print(f"MAE  : {raw_mae:.6f}")
    print(f"RMSE : {raw_rmse:.6f}")

    print("\nAFTER CLIPPING AT ZERO")
    print("-" * 40)
    print(f"MAE  : {clipped_mae:.6f}")
    print(f"RMSE : {clipped_rmse:.6f}")

    print("\nCHANGE")
    print("-" * 40)
    print(f"MAE change  : {clipped_mae - raw_mae:.6f}")
    print(f"RMSE change : {clipped_rmse - raw_rmse:.6f}")


if __name__ == "__main__":
    main()