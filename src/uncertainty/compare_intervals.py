import pandas as pd
import numpy as np
import lightgbm as lgb

from mapie.regression import ConformalizedQuantileRegressor
from mapie.metrics.regression import (
    regression_coverage_score,
    regression_mean_width_score
)

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

    df = df.sort_values("date").reset_index(drop=True)

    # 70% Train / 15% Calibration / 15% Test
    train_end = df["date"].quantile(0.70)
    calibration_end = df["date"].quantile(0.85)

    train = df[df["date"] <= train_end]

    calibration = df[
        (df["date"] > train_end) &
        (df["date"] <= calibration_end)
    ]

    test = df[df["date"] > calibration_end]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_calibration = calibration[FEATURES]
    y_calibration = calibration[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET].values

    print("Train:", len(train))
    print("Calibration:", len(calibration))
    print("Test:", len(test))

    # ==================================================
    # RAW QUANTILE MODELS
    # ==================================================

    print("\nTraining raw quantile models...")

    raw_models = {}

    for alpha in [0.10, 0.50, 0.90]:

        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        model.fit(X_train, y_train)

        raw_models[alpha] = model

    # Generate raw predictions

    raw_p10 = raw_models[0.10].predict(X_test)
    raw_p50 = raw_models[0.50].predict(X_test)
    raw_p90 = raw_models[0.90].predict(X_test)

    # ==================================================
    # QUANTILE ORDERING CHECK
    # ==================================================

    print("\n========================================")
    print("QUANTILE ORDERING CHECK")
    print("========================================")

    print("P10 > P50:", np.mean(raw_p10 > raw_p50))
    print("P50 > P90:", np.mean(raw_p50 > raw_p90))
    print("P10 > P90:", np.mean(raw_p10 > raw_p90))

    print("\nMinimum predictions")
    print("----------------")
    print("P10:", raw_p10.min())
    print("P50:", raw_p50.min())
    print("P90:", raw_p90.min())

    print("\nMaximum predictions")
    print("----------------")
    print("P10:", raw_p10.max())
    print("P50:", raw_p50.max())
    print("P90:", raw_p90.max())

    # ==================================================
    # RAW INTERVAL
    # ==================================================

    raw_lower = np.minimum(raw_p10, raw_p90)
    raw_upper = np.maximum(raw_p10, raw_p90)

    raw_intervals = np.column_stack(
        (raw_lower, raw_upper)
    )

    # MAPIE expects interval shape (n_samples, 2, 1)
    raw_intervals = raw_intervals[:, :, np.newaxis]

    # ==================================================
    # RAW METRICS
    # ==================================================

    raw_coverage = regression_coverage_score(
        y_test,
        raw_intervals
    )[0]

    raw_width = regression_mean_width_score(
        raw_intervals
    )[0]

    # ==================================================
    # CONFORMAL CQR
    # ==================================================

    print("\nTraining conformal model...")

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

    conformal_model = ConformalizedQuantileRegressor(
        estimator=model,
        confidence_level=0.80
    )

    # Train conformal model

    conformal_model.fit(
        X_train,
        y_train
    )

    # Calibrate using separate calibration data

    conformal_model.conformalize(
        X_calibration,
        y_calibration
    )

    # Generate prediction intervals

    predictions, intervals = conformal_model.predict_interval(
    X_test,
    symmetric_correction=True
)

    # ==================================================
    # CONFORMAL METRICS
    # ==================================================

    conformal_coverage = regression_coverage_score(
        y_test,
        intervals
    )[0]

    conformal_width = regression_mean_width_score(
        intervals
    )[0]

    # ==================================================
    # COMPARISON
    # ==================================================

    coverage_change = (
        conformal_coverage - raw_coverage
    )

    width_change = (
        conformal_width - raw_width
    )

    width_change_percent = (
        width_change / raw_width
    ) * 100

    # ==================================================
    # FINAL RESULTS
    # ==================================================

    print("\n========================================")
    print("RAW vs CONFORMAL COMPARISON")
    print("========================================")

    print("\nRaw P10-P90")
    print("----------------")
    print("Coverage :", raw_coverage)
    print("Width    :", raw_width)

    print("\nConformal CQR")
    print("----------------")
    print("Coverage :", conformal_coverage)
    print("Width    :", conformal_width)

    print("\nDifference")
    print("----------------")
    print("Coverage change :", coverage_change)
    print("Width change    :", width_change)
    print("Width change %  :", width_change_percent)

    print("\nTarget coverage: 0.80")


if __name__ == "__main__":
    main()