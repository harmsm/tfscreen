"""
Each genotype's within-tube swing (change from 0 to 1 mM IPTG of the
slope of ln(genotype / M42I/H74A/K84L)), observed from counts and
predicted by a stage 1 fit, in both selection markers (README).

    python swing_ppc.py runs/rel_real/stage1 --out out
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "within-tube-m"))
from within_tube_m import slopes_by_conc, swing  # noqa: E402

REF = "M42I/H74A/K84L"
KEY = ["replicate", "condition_sel", "titrant_conc", "t_sel"]
COND = ["kanR+kan", "pheS+4CP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage1_dir")
    ap.add_argument("--min_reads", type=float, default=1e5)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    obs = pd.read_csv(os.path.join(args.stage1_dir, "tfs_growth.csv"),
                      usecols=KEY + ["genotype", "counts"])
    pred = pd.read_csv(os.path.join(args.stage1_dir, "pred_growth.csv"),
                       usecols=KEY + ["genotype", "q0.5"])
    m = obs.merge(pred, on=KEY + ["genotype"])
    m = m[m["condition_sel"].isin(COND)]
    reads = obs.groupby("genotype")["counts"].sum()
    deep = set(reads[reads >= args.min_reads].index)
    ref = m[m["genotype"] == REF].set_index(KEY)[["counts", "q0.5"]]
    rows = []
    for g, d in m[(m["genotype"] != REF) & m["genotype"].isin(deep)].groupby("genotype"):
        rec = {"genotype": g}
        for cond in COND:
            t = d[d["condition_sel"] == cond].set_index(KEY)[["counts", "q0.5"]].join(
                ref, rsuffix="_ref", how="inner").reset_index()
            t = t[(t["counts"] > 0) & (t["counts_ref"] > 0)]
            if t["titrant_conc"].isin([0.0, 1.0]).sum() < 4:
                continue
            t["lr"] = np.log(t["counts"] / t["counts_ref"])
            t["v"] = 1 / t["counts"] + 1 / t["counts_ref"]
            t["lp"] = t["q0.5"] - t["q0.5_ref"]
            try:
                c, b, cov, _ = slopes_by_conc(t, "lr", "v")
                rec[f"obs_{cond}"], rec[f"obs_{cond}_se"] = swing(c, b, cov)
                c, b, cov, _ = slopes_by_conc(t, "lp")
                rec[f"pred_{cond}"] = swing(c, b, cov)[0]
            except (ValueError, np.linalg.LinAlgError):
                pass
        rows.append(rec)
    s = pd.DataFrame(rows).dropna()
    s.to_csv(os.path.join(args.out, "swing_ppc.csv"), index=False)
    singles = s[~s["genotype"].str.contains("/") & (s["genotype"] != "wt")]
    print(f"{len(singles)} deep singles")
    for cond in COND:
        o, p = singles[f"obs_{cond}"], singles[f"pred_{cond}"]
        slope = np.polyfit(o, p, 1)[0]
        print(f"{cond}: predicted on observed swing slope {slope:+.3f}, r "
              f"{np.corrcoef(o, p)[0, 1]:.3f}, median |obs| {o.abs().median():.4f}, "
              f"median |pred| {p.abs().median():.4f}")
    for kind in ("obs", "pred"):
        a, b = singles[f"{kind}_kanR+kan"], singles[f"{kind}_pheS+4CP"]
        print(f"{kind}: pheS on kanR swing, line through 0: "
              f"{np.sum(a * b) / np.sum(a * a):+.3f}")
    print(s[s["genotype"].isin(["wt", "H74A/K84L", "M42I/H74A", "M42I/K84L"])].to_string(
        index=False, float_format="%+.4f"))


if __name__ == "__main__":
    main()
