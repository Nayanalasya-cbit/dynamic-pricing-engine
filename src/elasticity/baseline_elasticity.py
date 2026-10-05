import pandas as pd
import numpy as np
import statsmodels.api as sm

INPUT_PATH = "data/processed/m5_features.csv"

def main():

    df = pd.read_csv(INPUT_PATH)

    df = df[
        (df["price"] > 0) &
        (df["units_sold"] > 0)
    ].copy()

    df["log_price"] = np.log(df["price"])
    df["log_demand"] = np.log(df["units_sold"])

    X = sm.add_constant(df["log_price"])
    y = df["log_demand"]

    model = sm.OLS(y, X).fit()

    elasticity = model.params["log_price"]
    p_value = model.pvalues["log_price"]
    r_squared = model.rsquared

    print("\nBASELINE PRICE ELASTICITY")
    print("=" * 40)

    print(f"Elasticity : {elasticity:.6f}")
    print(f"P-value    : {p_value:.6f}")
    print(f"R-squared  : {r_squared:.6f}")

    print("\nMODEL SUMMARY")
    print("=" * 40)
    print(model.summary())


if __name__ == "__main__":
    main()