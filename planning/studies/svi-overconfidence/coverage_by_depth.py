"""
Per-arm calibration against truth where the relative-fit grid showed the gap.

For every run of the grid:

- theta (X for the relative fit) test coverage at 50% and 95%, by the
  genotype's median reads per tube (<= 5, 6-20, 21-100, 101-1000, > 1000).
  A guide that leaves out shared uncertainty covers the low-read genotypes
  and misses the high-read ones.
- growth_m and growth_k: error in the arm's own SDs (z) and the arm's SD,
  per condition.
- the guide diagnostics (Pareto k-hat, log-weight SD), where present.

Writes {out_prefix}_depth.csv, {out_prefix}_growth.csv and
{out_prefix}_guides.csv, and prints per-arm means.

Usage (from the study directory):
    python coverage_by_depth.py svi_overconfidence --out_prefix calib/svi_overconfidence
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

DEPTH_BINS = [-1, 5, 20, 100, 1000, np.inf]
DEPTH_LABELS = ["<=5", "6-20", "21-100", "101-1000", ">1000"]


def _runs(grid_dir):
    with open(os.path.join(grid_dir, "grid_summary.json")) as f:
        summary = json.load(f)
    return [dict(run=r["run"], path=os.path.join(grid_dir, r["run"]),
                 seed=r["simulate"]["seed"], fit=r["template"]["fit"],
                 inference=r["template"]["inference"])
            for r in summary["runs"]]


def _depth(run):
    theta = os.path.join(run["path"], "summary",
                         "tfs_summarize_theta_corr_test.csv")
    growth = os.path.join(run["path"], "tfs_sim_growth.csv")
    if not (os.path.exists(theta) and os.path.exists(growth)):
        return None
    d = pd.read_csv(theta)
    reads = pd.read_csv(growth).groupby("genotype")["counts"].median()
    d["reads"] = d["genotype"].map(reads)
    d["depth"] = pd.cut(d["reads"], DEPTH_BINS, labels=DEPTH_LABELS)
    d["c50"] = (d["q0.25"] <= d["ref"]) & (d["ref"] <= d["q0.75"])
    d["c95"] = (d["q0.025"] <= d["ref"]) & (d["ref"] <= d["q0.975"])
    out = (d.groupby("depth", observed=True)
           .agg(n=("c95", "size"), coverage_50=("c50", "mean"),
                coverage_95=("c95", "mean"))
           .reset_index())
    return out.assign(**{k: run[k] for k in ("run", "seed", "fit", "inference")})


def _growth(run):
    rows = []
    for name in ("growth_k", "growth_m"):
        f = os.path.join(run["path"], "summary",
                         f"tfs_summarize_params_{name}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        sd = (d["q0.841"] - d["q0.159"]) / 2
        rows.append(pd.DataFrame(dict(
            parameter=name, condition_rep=d["condition_rep"],
            error=d["q0.5"] - d["ref"], sd=sd,
            z=(d["q0.5"] - d["ref"]) / sd,
            covered_95=(d["q0.025"] <= d["ref"]) & (d["ref"] <= d["q0.975"]))))
    if not rows:
        return None
    return pd.concat(rows).assign(
        **{k: run[k] for k in ("run", "seed", "fit", "inference")})


def _guide(run):
    f = os.path.join(run["path"], "tfs_guide_diagnostics.json")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        d = json.load(fh)
    return dict(d, **{k: run[k] for k in ("run", "seed", "fit", "inference")})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("grid_dir")
    parser.add_argument("--out_prefix", default="coverage_by_depth")
    args = parser.parse_args()

    runs = _runs(args.grid_dir)
    depth = [x for x in map(_depth, runs) if x is not None]
    growth = [x for x in map(_growth, runs) if x is not None]
    guides = [x for x in map(_guide, runs) if x is not None]
    if not depth:
        raise SystemExit("No run has a theta test summary yet.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    depth = pd.concat(depth)
    depth.to_csv(f"{args.out_prefix}_depth.csv", index=False)
    arm = ["fit", "inference"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("theta/X coverage by median reads per tube (mean over seeds):")
        print(depth.groupby(arm + ["depth"], observed=True)
              [["n", "coverage_50", "coverage_95"]].mean().round(3).to_string())
        if growth:
            growth = pd.concat(growth)
            growth.to_csv(f"{args.out_prefix}_growth.csv", index=False)
            print("\ngrowth_k/growth_m: median |z| and 95% coverage over "
                  "seeds and conditions:")
            print(growth.assign(abs_z=growth.z.abs())
                  .groupby(arm + ["parameter"])
                  .agg(median_abs_z=("abs_z", "median"),
                       coverage_95=("covered_95", "mean"),
                       median_sd=("sd", "median")).round(4).to_string())
        if guides:
            guides = pd.DataFrame(guides)
            guides.to_csv(f"{args.out_prefix}_guides.csv", index=False)
            print("\nguide diagnostics (median over seeds):")
            print(guides.groupby(arm)[["khat", "log_w_sd", "ess"]]
                  .median().round(2).to_string())


if __name__ == "__main__":
    main()
