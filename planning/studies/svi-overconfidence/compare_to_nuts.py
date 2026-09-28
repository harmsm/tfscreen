"""
Compare each inference arm's posterior marginals with NUTS on the same data.

For every (seed, fit) the grid has a NUTS run and the other inference arms,
all on one simulated data set. Per quantity and arm this reports:

- width_ratio: the arm's central 68% interval width over NUTS's, median
  over the quantity's elements (1 = right width, < 1 = overconfident).
- shift_z: |arm median - NUTS median| in NUTS SDs (half the 68% width),
  median over elements (0 = same center).
- coverage_68/95: how often the arm's 68%/95% interval holds the NUTS
  median (a truth-free coverage; NUTS itself scores about 1.0/1.0).

Truth-based coverage for every arm, NUTS included, comes from
tfs-summarize-calibration.

Usage (from the study directory):
    python compare_to_nuts.py svi_overconfidence --out_prefix calib/svi_overconfidence
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# summary file -> key columns
QUANTITIES = {
    "growth_k": ("summary/tfs_summarize_params_growth_k.csv", ["condition_rep"]),
    "growth_m": ("summary/tfs_summarize_params_growth_m.csv", ["condition_rep"]),
    "dk_geno": ("summary/tfs_summarize_params_dk_geno.csv", ["genotype"]),
    "log_hill_K": ("summary/tfs_summarize_params_log_hill_K.csv", ["genotype"]),
    "hill_n": ("summary/tfs_summarize_params_hill_n.csv", ["genotype"]),
    "theta": ("summary/tfs_summarize_theta_corr_test.csv",
              ["genotype", "titrant_name", "titrant_conc"]),
}


def _runs(grid_dir):
    """One row per run: directory, seed, fit, inference."""
    with open(os.path.join(grid_dir, "grid_summary.json")) as f:
        summary = json.load(f)
    rows = []
    for run in summary["runs"]:
        rows.append(dict(run=run["run"],
                         path=os.path.join(grid_dir, run["run"]),
                         seed=run["simulate"]["seed"],
                         fit=run["template"]["fit"],
                         inference=run["template"]["inference"]))
    return pd.DataFrame(rows)


def _read(path, rel, keys):
    f = os.path.join(path, rel)
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    df = df.drop_duplicates(keys)
    df["sd"] = (df["q0.841"] - df["q0.159"]) / 2
    return df


def compare(grid_dir):
    runs = _runs(grid_dir)
    out = []
    for (seed, fit), group in runs.groupby(["seed", "fit"]):
        ref = group[group.inference == "nuts"]
        if ref.empty:
            continue
        ref_path = ref.path.iloc[0]
        for quantity, (rel, keys) in QUANTITIES.items():
            nuts = _read(ref_path, rel, keys)
            if nuts is None:
                continue
            for _, arm in group[group.inference != "nuts"].iterrows():
                other = _read(arm.path, rel, keys)
                if other is None:
                    continue
                m = nuts.merge(other, on=keys, suffixes=("_nuts", "_arm"))
                m = m[m.sd_nuts > 0]
                width = (m.sd_arm / m.sd_nuts)
                shift = (m["q0.5_arm"] - m["q0.5_nuts"]).abs() / m.sd_nuts
                c68 = ((m["q0.159_arm"] <= m["q0.5_nuts"])
                       & (m["q0.5_nuts"] <= m["q0.841_arm"]))
                c95 = ((m["q0.025_arm"] <= m["q0.5_nuts"])
                       & (m["q0.5_nuts"] <= m["q0.975_arm"]))
                out.append(dict(seed=seed, fit=fit, inference=arm.inference,
                                quantity=quantity, n=len(m),
                                width_ratio=width.median(),
                                shift_z=shift.median(),
                                coverage_68=c68.mean(),
                                coverage_95=c95.mean()))
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("grid_dir")
    parser.add_argument("--out_prefix", default="compare_to_nuts")
    args = parser.parse_args()

    per_run = compare(args.grid_dir)
    if per_run.empty:
        raise SystemExit("No (seed, fit) pairs with a NUTS run and another arm.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    per_run.to_csv(f"{args.out_prefix}_per_run.csv", index=False)

    arms = (per_run.groupby(["fit", "inference", "quantity"])
            [["width_ratio", "shift_z", "coverage_68", "coverage_95"]]
            .mean().reset_index())
    arms.to_csv(f"{args.out_prefix}_arms.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(arms.round(3).to_string(index=False))
    print(f"Wrote {args.out_prefix}_per_run.csv and {args.out_prefix}_arms.csv")


if __name__ == "__main__":
    main()
