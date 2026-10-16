"""
X against the simulation's truth by read depth (the tables in README.md).

    python score_x.py depth.csv held_s2 km_draws/pooled_aggregate.csv \
        --truth held_s2/summary/tfs_summarize_theta_corr_test.csv

Each input is a run directory (its summary/tfs_summarize_theta_corr_test.csv,
which carries the truth as `ref`) or a quantile table (q0.025, q0.5, q0.975
by genotype and titrant_conc, such as tfs-compare-runs' aggregate), scored
against --truth. Doubles only unless --all. Per depth bin: r, the slope of
truth on fit, RMSE, 95% coverage, median 95% width, and the mean error at
the lowest and highest concentration.
"""

import argparse
import os

import numpy as np
import pandas as pd

BINS = [0, 1e2, 1e3, 1e4, 1e5, 1e13]
LABELS = ["<=1e2", "1e2-1e3", "1e3-1e4", "1e4-1e5", ">1e5"]
COLS = ["genotype", "titrant_conc", "q0.025", "q0.5", "q0.975"]


def load(path, truth):
    if os.path.isdir(path):
        return pd.read_csv(os.path.join(path, "summary",
                                        "tfs_summarize_theta_corr_test.csv"),
                           usecols=COLS + ["ref"])
    if truth is None:
        raise SystemExit(f"{path} is a table; give --truth for its reference")
    df = pd.read_csv(path, usecols=COLS)
    return df.merge(truth, on=["genotype", "titrant_conc"])


def score(df, depth, doubles_only=True):
    g = df["genotype"].astype(str)
    df = df[g != "wt"]
    if doubles_only:
        df = df[df["genotype"].astype(str).str.count("/") == 1]
    df = df.assign(bin=pd.cut(df["genotype"].map(depth), BINS, labels=LABELS))
    lo, hi = df["titrant_conc"].min(), df["titrant_conc"].max()
    rows = []
    for b, d in df.groupby("bin", observed=True):
        x, y = d["q0.5"].to_numpy(), d["ref"].to_numpy()
        err = d["q0.5"] - d["ref"]
        rows.append(dict(
            bin=b, n_geno=d["genotype"].nunique(), r=np.corrcoef(x, y)[0, 1],
            slope=np.polyfit(x, y, 1)[0], rmse=np.sqrt(np.mean((x - y) ** 2)),
            cover95=np.mean((d["q0.025"] <= y) & (y <= d["q0.975"])),
            width95=np.median(d["q0.975"] - d["q0.025"]),
            err_lo=err[d["titrant_conc"] == lo].mean(),
            err_hi=err[d["titrant_conc"] == hi].mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("depth_csv")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--truth", default=None,
                    help="a run's summary/tfs_summarize_theta_corr_test.csv")
    ap.add_argument("--all", action="store_true", help="singles too")
    args = ap.parse_args()
    depth = pd.read_csv(args.depth_csv).set_index("genotype")["reads"]
    truth = None
    if args.truth:
        truth = pd.read_csv(args.truth,
                            usecols=["genotype", "titrant_conc", "ref"])
    for path in args.inputs:
        print(path)
        print(score(load(path, truth), depth, not args.all)
              .round(3).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
