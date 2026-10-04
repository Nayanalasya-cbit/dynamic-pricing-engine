import pandas as pd
import numpy as np
import lightgbm as lgb

from mapie.regression import ConformalizedQuantileRegressor
from mapie.metrics.regression import regression_coverage_score
from mapie.metrics.regression import regression_mean_width_score

INPUT_PATH = "data/processed/m5_features.csv"
OUTPUT_PATH = "data/processed/conformal_predictions.csv"


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


def train_conformal_model():

    print("Loading feature dataset...")

    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    print("Dataset shape:", df.shape)

    train_end = df["date"].quantile(0.70)
    calibration_end = df["date"].quantile(0.85)

    train_mask = df["date"] <= train_end

    calibration_mask = (
        (df["date"] > train_end) &
        (df["date"] <= calibration_end)
    )

    test_mask = df["date"] > calibration_end

    train = df[train_mask]
    calibration = df[calibration_mask]
    test = df[test_mask]

    print("\nData split")
    print("----------------")
    print("Training:", len(train))
    print("Calibration:", len(calibration))
    print("Testing:", len(test))

    print("\nDates")
    print("----------------")
    print("Training end:", train_end)
    print("Calibration end:", calibration_end)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_calibration = calibration[FEATURES]
    y_calibration = calibration[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("\nTraining quantile model...")

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    print("Creating conformalized quantile regressor...")

    mapie = ConformalizedQuantileRegressor(
        estimator=model,
        confidence_level=0.80
    )

    print("\nFitting model...")

    mapie.fit(
        X_train,
        y_train
    )

    print("Model fitted.")

    print("\nConformalizing...")

    mapie.conformalize(
        X_calibration,
        y_calibration
    )

    print("Calibration complete.")

    print("\nGenerating prediction intervals...")

    predictions, intervals = mapie.predict_interval(
        X_test
    )

    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]

    coverage = regression_coverage_score(
        y_test.values,
        intervals
    )[0]

    mean_width = regression_mean_width_score(
        intervals
    )[0]

    print("\nConformal Results")
    print("----------------")
    print("Target coverage :", 0.80)
    print("Actual coverage :", coverage)
    print("Mean interval width:", mean_width)

    results = test[
        [
            "id",
            "item_id",
            "store_id",
            "date",
            "price",
            "units_sold"
        ]
    ].copy()

    results["prediction"] = predictions
    results["lower_bound"] = lower
    results["upper_bound"] = upper

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved predictions to:")
    print(OUTPUT_PATH)

    print("\nConformal calibration complete.")


if __name__ == "__main__":
    train_conformal_model()