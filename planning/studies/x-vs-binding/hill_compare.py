"""
Hill midpoints and steepness of growth X against binding, per genotype
(README, run 2): does M42I's inverted order survive on the two-stage,
kanR-only X?

    python hill_compare.py ../../dev-data/binding/2026-10-15_binding.csv ../two-stage-validation/run_real --out out

Fits a four-parameter Hill curve (top, bottom, K, n; n in [0.2, 8]) to
each genotype's predicted X (median, 8 IPTG concentrations) from the
two-stage run's stage 2 (both libraries) and consistency/ (kanR alone),
and to its mean binding theta at 2 and 20 uM protein. Writes out/hill.csv
and prints K (with its ratio to wt's), n and amplitude. K is trustworthy
only where the curve has amplitude (here >= 0.35).
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

GENOTYPES = ["wt", "M42I", "H74A", "K84L", "M42I/H74A", "H74A/K84L", "M42I/K84L",
             "M42I/H74A/K84L", "D88A", "G65H", "V95N"]
FITS = {"X both": "stage2", "X kanR": "consistency"}
MIN_AMPLITUDE = 0.35


def fit_hill(c, y):
    c, y = np.asarray(c, float), np.asarray(y, float)
    lc = np.log(np.where(c == 0, 1e-9, c))

    def resid(p):
        top, bottom, log_k, log_n = p
        return top + (bottom - top) / (1 + np.exp(-np.exp(log_n) * (lc - log_k))) - y

    best = None
    for k0 in np.log([1e-4, 1e-3, 1e-2, 1e-1]):
        r = least_squares(resid, [y[0], y[-1], k0, 0.0],
                          bounds=([-2, -2, np.log(1e-6), np.log(0.2)],
                                  [3, 3, np.log(10), np.log(8)]))
        if best is None or r.cost < best.cost:
            best = r
    top, bottom, log_k, log_n = best.x
    return dict(K_uM=1e3 * np.exp(log_k), n=np.exp(log_n), amplitude=top - bottom,
                rms=float(np.sqrt(np.mean(best.fun ** 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binding_csv")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for name, sub in FITS.items():
        p = pd.read_csv(os.path.join(args.run_dir, sub, "tfs_pred_theta.csv"),
                        usecols=["genotype", "titrant_conc", "q0.5"])
        p["genotype"] = p["genotype"].astype(str)
        for g in GENOTYPES:
            x = p[p["genotype"] == g].sort_values("titrant_conc")
            if len(x) >= 5:
                rows.append(dict(source=name, genotype=g,
                                 **fit_hill(x["titrant_conc"], x["q0.5"])))
    b = pd.read_csv(args.binding_csv).dropna(subset=["theta"])
    b["protein_uM"] = (b["protein_dimer_conc_molar"] * 1e6).round(3)
    b["c"] = (b["iptg_conc_molar"] * 1e3).round(9)
    for protein in (2.0, 20.0):
        for g in GENOTYPES:
            x = b[(b["genotype"] == g) & np.isclose(b["protein_uM"], protein)]
            if x["c"].nunique() >= 5:
                m = x.groupby("c")["theta"].mean().reset_index()
                rows.append(dict(source=f"binding {protein:g} uM", genotype=g,
                                 **fit_hill(m["c"], m["theta"])))
    r = pd.DataFrame(rows)
    wt = r[r["genotype"] == "wt"].set_index("source")["K_uM"]
    r["K_over_wt"] = r["K_uM"] / r["source"].map(wt)
    r["trusted"] = r["amplitude"] >= MIN_AMPLITUDE
    r.to_csv(os.path.join(args.out, "hill.csv"), index=False)
    pd.set_option("display.width", 200)
    for col, fmt in (("K_over_wt", "{:.2g}"), ("K_uM", "{:.3g}"), ("n", "{:.2f}"),
                     ("amplitude", "{:.2f}")):
        t = r.pivot(index="genotype", columns="source", values=col)
        print(f"\n{col}")
        print(t.map(lambda v: "" if pd.isna(v) else fmt.format(v)).to_string())
    print(f"\ntrusted (amplitude >= {MIN_AMPLITUDE} in every source with data):")
    ok = r.groupby("genotype")["trusted"].all()
    print(", ".join(ok[ok].index))


if __name__ == "__main__":
    main()
