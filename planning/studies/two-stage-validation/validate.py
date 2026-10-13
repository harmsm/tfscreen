"""
Check a tfs-fit-two-stage run against the m-bias study's numbers
(planning/studies/two-stage-validation/README.md).

    python validate.py run_seed1 --depth depth.csv \
        --truth ../m-bias-factorial/stage2_single_seed1/summary/tfs_summarize_theta_corr_test.csv \
        --sim_dir ../m-bias-factorial/m_subsets/no_doubles_mix \
        --reference ../m-bias-factorial/stage2_single_seed1/tfs_pred_theta.csv \
        --study_stage1 ../m-bias-factorial/m_subsets/no_doubles_mix

Prints stage 1's k and m (against the truth on the X scale when --sim_dir
holds the simulation's tfs_sim_parameters.csv and
tfs_sim_growth_parameters.csv, and against the study's stage 1 with
--study_stage1), the jackknife summary, the doubles' X by read depth
without the shift (stage2/) and with it (predict/) when --truth is given,
agreement with --reference (the study's matching fit), and the share of
genotypes the consistency step flags, overall and above 1e4 reads.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

BINS = [0, 1e2, 1e3, 1e4, 1e5, 1e13]
LABELS = ["<=1e2", "1e2-1e3", "1e3-1e4", "1e4-1e5", ">1e5"]
KEY = ["genotype", "titrant_conc"]
COLS = KEY + ["q0.025", "q0.5", "q0.975"]


def km(d):
    k = pd.read_csv(os.path.join(d, "tfs_params_growth_k.csv"))
    m = pd.read_csv(os.path.join(d, "tfs_params_growth_m.csv"))
    return pd.DataFrame({"k": k.groupby("condition_rep")["q0.5"].mean(),
                         "m": m.groupby("condition_rep")["q0.5"].mean()})


def truth_km(sim_dir, config):
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
    return pd.DataFrame({"k": tr["growth_k"] + tr["growth_m"] * th(c_hi),
                         "m": tr["growth_m"] * (th(c_lo) - th(c_hi))})


def score(pred, truth, depth):
    df = pred[COLS].merge(truth, on=KEY)
    df = df[df["genotype"].astype(str).str.count("/") == 1]
    df = df.assign(bin=pd.cut(df["genotype"].map(depth), BINS, labels=LABELS))
    rows = []
    for b, d in df.groupby("bin", observed=True):
        x, y = d["q0.5"].to_numpy(), d["ref"].to_numpy()
        rows.append(dict(bin=b, n_geno=d["genotype"].nunique(),
                         slope=np.polyfit(x, y, 1)[0],
                         rmse=np.sqrt(np.mean((x - y) ** 2)),
                         cover95=np.mean((d["q0.025"] <= y) & (y <= d["q0.975"])),
                         width95=np.median(d["q0.975"] - d["q0.025"])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--depth", default=None, help="genotype,reads CSV")
    ap.add_argument("--truth", default=None,
                    help="a simulation fit's summary/tfs_summarize_theta_corr_test.csv")
    ap.add_argument("--sim_dir", default=None,
                    help="directory with tfs_sim_parameters.csv and "
                         "tfs_sim_growth_parameters.csv")
    ap.add_argument("--reference", default=None,
                    help="the study's matching tfs_pred_theta.csv")
    ap.add_argument("--study_stage1", default=None,
                    help="the study's stage 1 directory (tfs_params_growth_*)")
    args = ap.parse_args()
    run = args.run_dir
    depth = (pd.read_csv(args.depth).set_index("genotype")["reads"]
             if args.depth else None)

    s1 = km(os.path.join(run, "stage1"))
    table = s1.add_prefix("run_")
    if args.study_stage1:
        table = table.join(km(args.study_stage1).add_prefix("study_"))
    if args.sim_dir:
        t = truth_km(args.sim_dir, os.path.join(run, "stage1",
                                                "tfs_configure_config.yaml"))
        table = table.join(t.add_prefix("truth_"))
        table["m_ratio"] = table["run_m"] / table["truth_m"]
    print("== stage 1 k and m")
    print(table.to_string(float_format="%.5f"))

    jk = os.path.join(run, "jackknife")
    if os.path.exists(os.path.join(jk, "jackknife.json")):
        print("\n== jackknife", json.load(open(os.path.join(jk, "jackknife.json"))))
        print(pd.read_csv(os.path.join(jk, "jackknife_se.csv"),
                          index_col=0).to_string(float_format="%.5f"))

    preds = {}
    for name, f in (("stage2 (no shift)", "stage2/tfs_pred_theta.csv"),
                    ("predict (shift)", "predict/tfs_pred_theta.csv")):
        p = os.path.join(run, f)
        if os.path.exists(p):
            preds[name] = pd.read_csv(p, usecols=COLS)
    if args.truth and depth is not None:
        truth = pd.read_csv(args.truth, usecols=KEY + ["ref"])
        for name, p in preds.items():
            print(f"\n== doubles' X against the truth: {name}")
            print(score(p, truth, depth).round(3).to_string(index=False))

    if args.reference and preds:
        ref = pd.read_csv(args.reference, usecols=COLS)
        p = preds.get("predict (shift)", next(iter(preds.values())))
        m = p.merge(ref, on=KEY, suffixes=("_run", "_ref"))
        m = m[m["genotype"] != "wt"]
        r = np.corrcoef(m["q0.5_run"], m["q0.5_ref"])[0, 1]
        slope = np.polyfit(m["q0.5_ref"], m["q0.5_run"], 1)[0]
        print(f"\n== against the reference: r {r:.4f}, slope {slope:.3f}, "
              f"mean shift {(m['q0.5_run'] - m['q0.5_ref']).mean():+.4f}")

    c = os.path.join(run, "consistency", "consistency.csv")
    if os.path.exists(c):
        cons = pd.read_csv(c, index_col="genotype")
        line = (f"\n== consistency: {cons['inconsistent'].mean():.3f} of "
                f"{len(cons)} genotypes flagged")
        if depth is not None:
            deep = cons[cons.index.map(depth) >= 1e4]
            line += (f"; {deep['inconsistent'].mean():.3f} of {len(deep)} "
                     "above 1e4 reads")
        print(line)


if __name__ == "__main__":
    main()
