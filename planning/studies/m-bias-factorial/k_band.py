"""
X intervals that carry stage 1's k uncertainty (planning/deep-coverage.md
step 2).

    python k_band.py stage2_single_seed1 pc1/plus pc1/minus --out x_kband.csv

The first directory is the stage 2 fit at stage 1's k and m; the other two
are stage 2 refits with k moved one jackknife SD up and down along the
first principal component of k's jackknife covariance (jackknife_summary.py
--pc1_tables; it carries most of k's variance). Per genotype and
concentration, X's k-driven SD is |X(plus) - X(minus)| / 2 (the medians),
added in quadrature to the stage 2 Laplace SD ((q0.975 - q0.025) / 3.92).
Writes genotype, titrant_conc, q0.025, q0.5 (the first fit's), q0.975 and
both SDs, for score_x.py.
"""

import argparse
import os

import pandas as pd

COLS = ["genotype", "titrant_name", "titrant_conc", "q0.025", "q0.5", "q0.975"]
KEY = ["genotype", "titrant_name", "titrant_conc"]


def pred(d):
    return pd.read_csv(os.path.join(d, "tfs_pred_theta.csv"), usecols=COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("plus")
    ap.add_argument("minus")
    ap.add_argument("--out", default="x_kband.csv")
    args = ap.parse_args()
    base = pred(args.base)
    plus = pred(args.plus)[KEY + ["q0.5"]].rename(columns={"q0.5": "x_plus"})
    minus = pred(args.minus)[KEY + ["q0.5"]].rename(columns={"q0.5": "x_minus"})
    df = base.merge(plus, on=KEY).merge(minus, on=KEY)
    df["sd_laplace"] = (df["q0.975"] - df["q0.025"]) / 3.92
    df["sd_k"] = (df["x_plus"] - df["x_minus"]).abs() / 2
    sd = (df["sd_laplace"] ** 2 + df["sd_k"] ** 2) ** 0.5
    df["q0.025"] = df["q0.5"] - 1.96 * sd
    df["q0.975"] = df["q0.5"] + 1.96 * sd
    df[KEY + ["q0.025", "q0.5", "q0.975", "sd_laplace", "sd_k"]].to_csv(
        args.out, index=False)
    print(f"{len(df)} rows; median sd_laplace {df['sd_laplace'].median():.4f}, "
          f"median sd_k {df['sd_k'].median():.4f}")


if __name__ == "__main__":
    main()
