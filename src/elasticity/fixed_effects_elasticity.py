import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression

INPUT_PATH = "data/processed/m5_features.csv"


def main():

    df = pd.read_csv(INPUT_PATH)

    df = df[
        (df["price"] > 0) &
        (df["units_sold"] > 0)
    ].copy()

    df["log_price"] = np.log(df["price"])
    df["log_demand"] = np.log(df["units_sold"])

    categorical = pd.get_dummies(
        df[["id", "wday", "month", "year"]],
        drop_first=True,
        dtype=float
    )

    numeric = df[
        ["log_price", "snap_CA", "snap_TX", "snap_WI"]
    ].to_numpy()

    X = sparse.hstack([
        sparse.csr_matrix(numeric),
        sparse.csr_matrix(categorical.to_numpy())
    ])

    y = df["log_demand"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    elasticity = model.coef_[0]
    r_squared = model.score(X, y)

    print("\nITEM-STORE FIXED EFFECTS ELASTICITY")
    print("=" * 45)

    print(f"Observations       : {len(df)}")
    print(f"Item-store groups  : {df['id'].nunique()}")
    print(f"Elasticity         : {elasticity:.6f}")
    print(f"R-squared          : {r_squared:.6f}")

    print("\nMODEL")
    print("=" * 45)
    print(
        "log_demand ~ log_price + item-store FE"
        " + weekday + month + year + SNAP"
    )


if __name__ == "__main__":
    main()