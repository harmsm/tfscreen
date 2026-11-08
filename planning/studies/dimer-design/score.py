"""
Score the dimer-design fits against the simulation truth.

    python score.py            # from this directory, after the fits

Reads each fit's tfs-summarize-fit tables (summary/, posterior quantiles
with the truth in `ref`) and writes results/constants.csv (one row per
run x arm x per-mutation constant), results/theta.csv (one row per run x
arm) and results/by_arm.csv (constants pooled over terms), and prints
the pooled tables.

Constants are scored on the library mutations every run shares, so the
arms are compared on the same mutations; a background's own mutations
are left out. Errors are in kcal/mol (RT at 310.15 K, the truth's
temperature) so they read against the narrative's predicted standard
errors of 0.09-0.19 kcal/mol. `se` is the posterior half-width
(q0.841 - q0.159) / 2, the analogue of a standard error.
"""

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "results")
RT = 0.0019872 * 310.15  # kcal/mol per kT
ARMS = ("PK", "PddG_s1", "PddG_dial")


def _fits():
    for run in sorted(os.listdir(RUNS)):
        for arm in ARMS:
            d = os.path.join(RUNS, run, f"fit_{arm}", "summary")
            if glob.glob(os.path.join(d, "tfs_summarize_params_d_ln_K_*.csv")):
                yield run, arm, d


def _score(df, scale=1.0):
    est, ref = df["q0.5"].to_numpy() * scale, df["ref"].to_numpy() * scale
    lo, hi = df["q0.025"].to_numpy() * scale, df["q0.975"].to_numpy() * scale
    ok = np.isfinite(est) & np.isfinite(ref)
    est, ref, lo, hi = est[ok], ref[ok], lo[ok], hi[ok]
    se = (df["q0.841"].to_numpy() - df["q0.159"].to_numpy())[ok] * scale / 2
    return {"n": int(ok.sum()),
            "rmse": float(np.sqrt(np.mean((est - ref) ** 2))),
            "truth_sd": float(np.std(ref)),
            "r": float(np.corrcoef(est, ref)[0, 1]) if ok.sum() > 2 else np.nan,
            "cov95": float(np.mean((ref >= lo) & (ref <= hi))),
            "median_se": float(np.median(se)),
            "z_rms": float(np.sqrt(np.mean(((est - ref) / np.maximum(se, 1e-9)) ** 2)))}


def _table(path):
    df = pd.read_csv(path)
    if "titrant_name" in df.columns:
        df = df.groupby("mutation", as_index=False).first()
    return df.set_index("mutation")


def main():
    fits = list(_fits())
    tables = {}
    for run, arm, d in fits:
        for f in glob.glob(os.path.join(d, "tfs_summarize_params_d_ln_K_*.csv")):
            if f.endswith(("_pit.csv", "_calibration_curve.csv")):
                continue
            term = os.path.basename(f)[len("tfs_summarize_params_"):-4]
            tables[(run, arm, term)] = _table(f)
    shared = None
    for t in tables.values():
        shared = set(t.index) if shared is None else shared & set(t.index)
    shared = sorted(shared)

    rows = []
    for (run, arm, term), t in sorted(tables.items()):
        rows.append({"run": run, "arm": arm, "term": term,
                     **_score(t.loc[shared], RT)})
    const = pd.DataFrame(rows)

    theta_rows = []
    for run, arm, d in fits:
        f = os.path.join(d, "tfs_summarize_theta_corr_test.csv")
        if os.path.exists(f):
            theta_rows.append({"run": run, "arm": arm,
                               **_score(pd.read_csv(f))})
    theta = pd.DataFrame(theta_rows)

    pooled = (const.groupby(["run", "arm"])
              .agg(terms=("term", "size"), rmse=("rmse", "median"),
                   r=("r", "median"), cov95=("cov95", "median"),
                   median_se=("median_se", "median"), z_rms=("z_rms", "median"))
              .reset_index())

    os.makedirs(OUT, exist_ok=True)
    const.to_csv(os.path.join(OUT, "constants.csv"), index=False)
    theta.to_csv(os.path.join(OUT, "theta.csv"), index=False)
    pooled.to_csv(os.path.join(OUT, "by_arm.csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"{len(fits)} fits, {len(shared)} shared mutations, kcal/mol\n")
    print("constants, median over terms:")
    print(pooled.round(3).to_string(index=False))
    print("\nper term, posterior RMSE (kcal/mol):")
    print(const.pivot_table(index="term", columns=["run", "arm"], values="rmse")
          .round(2).T.to_string())
    print("\ntheta:")
    print(theta.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
