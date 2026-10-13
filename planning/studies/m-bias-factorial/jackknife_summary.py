"""
k and m standard errors from a delete-d jackknife of a stage-1 fit.

    python jackknife_summary.py jk_dir --full ../m_subsets/no_doubles_mix \
        --truth ../sim_realistic_s2 --out jackknife_km.csv

jk_dir holds jk_NN/ from run_jackknife.srun (each with resample.log and
tfs_params_growth_{k,m}.csv). For n tubes, d dropped per subsample and the
subsample estimates t_i with mean t_bar, the jackknife variance is
(n - d) / d * mean((t_i - t_bar)^2). Prints, per condition, the full fit's
k and m (--full), the jackknife SE and, with --truth (a simulation
directory holding tfs_sim_parameters.csv and tfs_sim_growth_parameters.csv),
the truth on the X scale and the error in jackknife SEs.

--pc1_tables DIR also writes DIR/plus/ and DIR/minus/growth_priors_held.csv:
k at the full fit's value plus and minus one jackknife SD along the first
principal component of the k covariance (m at the full fit's value), for
two stage 2 refits that measure how X moves with k. It prints the share
of k's variance on that component.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import yaml


def read_km(d):
    out = {}
    for p in ("k", "m"):
        f = os.path.join(d, f"tfs_params_growth_{p}.csv")
        out[p] = pd.read_csv(f).groupby("condition_rep")["q0.5"].mean()
    return pd.DataFrame(out)


def n_and_d(d):
    text = open(os.path.join(d, "resample.log")).read()
    m = re.search(r"n (\d+) tubes, dropped d (\d+)", text)
    return int(m.group(1)), int(m.group(2))


def truth_x(sim_dir, config):
    """k and m on the fit's X scale (wt's true curve at the gauge)."""
    cfg = yaml.safe_load(open(config))
    c_lo, c_hi = (float(x) for x in cfg["components"]["theta_gauge_conc"])
    par = pd.read_csv(os.path.join(sim_dir, "tfs_sim_parameters.csv"))
    wt = par[par["genotype"] == "wt"].iloc[0]

    def th(c):
        if c <= 0:
            return wt["theta_low"]
        occ = 1 / (1 + np.exp(-wt["hill_n"] * (np.log(c) - wt["log_hill_K"])))
        return wt["theta_low"] + (wt["theta_high"] - wt["theta_low"]) * occ

    tr = pd.read_csv(os.path.join(sim_dir, "tfs_sim_growth_parameters.csv"))
    tr = tr.set_index("condition_rep")
    m = tr["growth_m"] * (th(c_lo) - th(c_hi))
    k = tr["growth_k"] + tr["growth_m"] * th(c_hi)
    return pd.DataFrame({"k": k, "m": m})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jk_dir")
    ap.add_argument("--full", default=None, help="the full stage-1 fit's dir")
    ap.add_argument("--truth", default=None, help="simulation dir")
    ap.add_argument("--out", default="jackknife_km.csv")
    ap.add_argument("--pc1_tables", default=None,
                    help="write plus/ and minus/ pinned tables here (needs --full)")
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.jk_dir, "jk_*"))
                  if os.path.exists(os.path.join(d, "tfs_params_growth_m.csv")))
    if len(dirs) < 2:
        raise SystemExit(f"{len(dirs)} finished subsamples in {args.jk_dir}")
    nd = {n_and_d(d) for d in dirs}
    if len(nd) != 1:
        raise SystemExit(f"subsamples differ in n and d: {sorted(nd)}")
    n, d = nd.pop()
    est = pd.concat({os.path.basename(x): read_km(x) for x in dirs})
    est.index.names = ["subsample", "condition_rep"]

    rows = []
    for c, g in est.groupby(level="condition_rep"):
        row = {"condition_rep": c, "n_sub": len(g)}
        for p in ("k", "m"):
            t = g[p].to_numpy()
            row[f"{p}_jk_mean"] = t.mean()
            row[f"{p}_jk_se"] = np.sqrt((n - d) / d * np.mean((t - t.mean()) ** 2))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("condition_rep")

    if args.full:
        full = read_km(args.full)
        for p in ("k", "m"):
            out[f"{p}_full"] = full[p]
        if args.truth:
            truth = truth_x(args.truth,
                            os.path.join(args.full, "tfs_configure_config.yaml"))
            for p in ("k", "m"):
                out[f"{p}_truth"] = truth[p]
                out[f"{p}_err_in_se"] = (out[f"{p}_full"] - out[f"{p}_truth"]) \
                    / out[f"{p}_jk_se"]
    if args.pc1_tables:
        if not args.full:
            raise SystemExit("--pc1_tables needs --full")
        k = est["k"].unstack("condition_rep")
        dev = (k - k.mean()) * np.sqrt((n - d) / d)
        cov = np.cov(dev.to_numpy().T)
        w, v = np.linalg.eigh(cov)
        top = int(np.argmax(w))
        step = pd.Series(v[:, top] * np.sqrt(w[top]), index=k.columns)
        full = read_km(args.full)
        print(f"k variance on the first component: {w[top] / w.sum():.3f}; "
              f"one-SD step {step.round(5).to_dict()}")
        for name, sign in (("plus", 1), ("minus", -1)):
            dd = os.path.join(args.pc1_tables, name)
            os.makedirs(dd, exist_ok=True)
            table = pd.DataFrame({
                "condition_rep": full.index,
                "k_loc": full["k"] + sign * step.reindex(full.index),
                "k_scale": 0.002, "m_loc": full["m"], "m_scale": 0.002})
            table.to_csv(os.path.join(dd, "growth_priors_held.csv"), index=False)
    out.to_csv(args.out)
    print(f"{len(dirs)} subsamples, n {n} tubes, d {d} dropped "
          f"(variance factor {(n - d) / d:.2f})")
    print(out.to_string(float_format="%.5f"))


if __name__ == "__main__":
    main()
