import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_PATH = "data/processed/m5_features.csv"
OUTPUT_PATH = "data/processed/quantile_predictions.csv"


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


def pinball_loss(y_true, y_pred, alpha):
    error = y_true - y_pred

    loss = np.where(
        error >= 0,
        alpha * error,
        (alpha - 1) * error
    )

    return np.mean(loss)


def train_quantile_model(X_train, y_train, X_test, alpha):

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, predictions


def train_quantile_models():

    print("Loading feature dataset...")

    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])

    print("Dataset shape:", df.shape)

    X = df[FEATURES]
    y = df[TARGET]

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

    print("\nTraining P10 model...")

    p10_model, p10_predictions = train_quantile_model(
        X_train,
        y_train,
        X_test,
        0.10
    )

    print("P10 model complete.")

    print("\nTraining P50 model...")

    p50_model, p50_predictions = train_quantile_model(
        X_train,
        y_train,
        X_test,
        0.50
    )

    print("P50 model complete.")

    print("\nTraining P90 model...")

    p90_model, p90_predictions = train_quantile_model(
        X_train,
        y_train,
        X_test,
        0.90
    )

    print("P90 model complete.")

    print("\nCalculating metrics...")

    for name, predictions, alpha in [
        ("P10", p10_predictions, 0.10),
        ("P50", p50_predictions, 0.50),
        ("P90", p90_predictions, 0.90)
    ]:

        pinball = pinball_loss(
            y_test.values,
            predictions,
            alpha
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        print(f"\n{name}")
        print("----------------")
        print("Pinball Loss:", pinball)
        print("MAE         :", mae)

    print("\nChecking prediction interval...")

    interval_coverage = np.mean(
        (y_test.values >= p10_predictions) &
        (y_test.values <= p90_predictions)
    )

    print(
        "P10-P90 empirical coverage:",
        interval_coverage
    )

    crossing_count = np.sum(
        (p10_predictions > p50_predictions) |
        (p50_predictions > p90_predictions)
    )

    print(
        "Quantile crossing count:",
        crossing_count
    )

    results = df.loc[test_mask, [
        "id",
        "item_id",
        "store_id",
        "date",
        "price",
        "units_sold"
    ]].copy()

    results["p10"] = p10_predictions
    results["p50"] = p50_predictions
    results["p90"] = p90_predictions

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nPredictions saved to:")
    print(OUTPUT_PATH)

    print("\nQuantile forecasting complete.")


if __name__ == "__main__":
    train_quantile_models()