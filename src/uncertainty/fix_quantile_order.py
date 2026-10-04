import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/quantile_predictions.csv"


def main():

    df = pd.read_csv(INPUT_PATH)

    p10 = df["p10"].values
    p50 = df["p50"].values
    p90 = df["p90"].values
    y = df["units_sold"].values

    print("========================================")
    print("BEFORE ORDERING FIX")
    print("========================================")

    print("P10 > P50:", np.sum(p10 > p50))
    print("P50 > P90:", np.sum(p50 > p90))
    print("P10 > P90:", np.sum(p10 > p90))

    print("\nNegative predictions")
    print("----------------")

    print("P10:", np.sum(p10 < 0))
    print("P50:", np.sum(p50 < 0))
    print("P90:", np.sum(p90 < 0))

    # --------------------------------------------------
    # NON-CROSSING ORDER
    # --------------------------------------------------

    ordered = np.sort(
        np.column_stack((p10, p50, p90)),
        axis=1
    )

    fixed_p10 = ordered[:, 0]
    fixed_p50 = ordered[:, 1]
    fixed_p90 = ordered[:, 2]

    print("\n========================================")
    print("AFTER ORDERING FIX")
    print("========================================")

    print("P10 > P50:", np.sum(fixed_p10 > fixed_p50))
    print("P50 > P90:", np.sum(fixed_p50 > fixed_p90))
    print("P10 > P90:", np.sum(fixed_p10 > fixed_p90))

    print("\nNegative predictions")
    print("----------------")

    print("P10:", np.sum(fixed_p10 < 0))
    print("P50:", np.sum(fixed_p50 < 0))
    print("P90:", np.sum(fixed_p90 < 0))

    # --------------------------------------------------
    # COVERAGE COMPARISON
    # --------------------------------------------------

    raw_coverage = np.mean(
        (y >= p10) &
        (y <= p90)
    )

    fixed_coverage = np.mean(
        (y >= fixed_p10) &
        (y <= fixed_p90)
    )

    print("\n========================================")
    print("COVERAGE COMPARISON")
    print("========================================")

    print("Raw coverage  :", raw_coverage)
    print("Fixed coverage:", fixed_coverage)

    print("\nCoverage change:")
    print(fixed_coverage - raw_coverage)

    # --------------------------------------------------
    # INTERVAL WIDTH
    # --------------------------------------------------

    raw_width = np.mean(p90 - p10)

    fixed_width = np.mean(
        fixed_p90 - fixed_p10
    )

    print("\n========================================")
    print("INTERVAL WIDTH")
    print("========================================")

    print("Raw width  :", raw_width)
    print("Fixed width:", fixed_width)

    print("\nWidth change:")
    print(fixed_width - raw_width)


if __name__ == "__main__":
    main()