"""
Apparent IPTG of every selection tube from its genotype composition
(README, run 3): a tube whose label and contents disagree stands out.

Within each replicate x time row, each tube's log frequencies are
centered on the row and projected on the axis from the 0-IPTG tubes' mean
composition to the 1 mM tubes' (computed without the tube itself), then
scaled so 0-IPTG tubes average 0 and 1 mM tubes 1. Rows missing some
concentrations are shifted by the centering.

    python tube_qc.py ../within-tube-m/out/marker_rows.csv --out out
"""

import argparse
import os

import numpy as np
import pandas as pd


def apparent_iptg(rows, cond):
    d = rows[(rows["condition_sel"] == cond) & (rows["counts"] > 0)].copy()
    d["tube"] = (d["replicate"].astype(str) + "|" + d["titrant_conc"].astype(str)
                 + "|" + d["t_sel"].astype(str))
    P = d.pivot_table(index="genotype", columns="tube", values="counts").dropna()
    L = np.log(P / P.sum())
    meta = pd.DataFrame([(k, int(k.split("|")[0]), float(k.split("|")[1]),
                          float(k.split("|")[2])) for k in L.columns],
                        columns=["tube", "replicate", "titrant_conc", "t_sel"]).set_index("tube")
    for _, g in meta.groupby(["replicate", "t_sel"]):
        L[g.index] = L[g.index].sub(L[g.index].mean(axis=1), axis=0)
    score = {}
    for k in L.columns:
        o = meta.drop(k)
        axis = (L[o.index[o["titrant_conc"] == 1.0]].mean(axis=1)
                - L[o.index[o["titrant_conc"] == 0.0]].mean(axis=1))
        score[k] = float((L[k] * axis).sum() / np.sqrt((axis ** 2).sum()))
    meta["score"] = pd.Series(score)
    z0 = meta.loc[meta["titrant_conc"] == 0.0, "score"].mean()
    z1 = meta.loc[meta["titrant_conc"] == 1.0, "score"].mean()
    meta["apparent"] = (meta["score"] - z0) / (z1 - z0)
    meta["condition_sel"] = cond
    return meta.reset_index(), len(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = pd.read_csv(args.rows_csv)
    parts = []
    for cond in sorted(rows["condition_sel"].unique()):
        t, n = apparent_iptg(rows, cond)
        parts.append(t)
        tab = t.pivot_table(index=["replicate", "t_sel"], columns="titrant_conc",
                            values="apparent")
        tab.columns = [f"{c * 1e3:g}uM" for c in tab.columns]
        print(f"\n== {cond} ({n} genotypes in every tube): apparent IPTG index")
        print(tab.round(2).to_string())
    pd.concat(parts).to_csv(os.path.join(args.out, "tube_qc.csv"), index=False)


if __name__ == "__main__":
    main()
