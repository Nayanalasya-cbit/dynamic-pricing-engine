import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_PATH = "data/processed/m5_features.csv"


def train_lightgbm():

    print("Loading feature dataset...")

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print("Dataset shape:", df.shape)

    features = [
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

    target = "units_sold"

    X = df[features]
    y = df[target]

    # Same time-based split as the baseline
    split_date = df["date"].quantile(0.8)

    train_mask = df["date"] <= split_date
    test_mask = df["date"] > split_date

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Split date:", split_date)

    print("\nTraining LightGBM...")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\nLightGBM Results")
    print("----------------")
    print("MAE :", mae)
    print("RMSE:", rmse)


if __name__ == "__main__":
    train_lightgbm()