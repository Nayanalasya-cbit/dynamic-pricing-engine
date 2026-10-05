import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

INPUT_PATH = "data/processed/m5_features.csv"


def main():

    df = pd.read_csv(INPUT_PATH)

    df = df[
        (df["price"] > 0) &
        (df["units_sold"] > 0)
    ].copy()

    df["log_price"] = np.log(df["price"])
    df["log_demand"] = np.log(df["units_sold"])

    formula = """
        log_demand ~ log_price
        + C(wday)
        + C(month)
        + C(year)
        + snap_CA
        + snap_TX
        + snap_WI
    """

    model = smf.ols(
        formula=formula,
        data=df
    ).fit(
        cov_type="HC3"
    )

    elasticity = model.params["log_price"]
    p_value = model.pvalues["log_price"]
    r_squared = model.rsquared

    print("\nCONTROLLED PRICE ELASTICITY")
    print("=" * 40)

    print(f"Elasticity : {elasticity:.6f}")
    print(f"P-value    : {p_value:.6f}")
    print(f"R-squared  : {r_squared:.6f}")

    print("\nELASTICITY CONFIDENCE INTERVAL")
    print("=" * 40)

    print(
        model.conf_int().loc["log_price"].to_string()
    )

    print("\nMODEL SUMMARY")
    print("=" * 40)

    print(model.summary())


if __name__ == "__main__":
    main()