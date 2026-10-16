"""
Each genotype's within-tube swing in each selection marker: the change,
from 0 to 1 mM IPTG, of the slope of ln(genotype / M42I/H74A/K84L) over
the selection (README, run 3). With one X per genotype shared by both
markers, swing = m_marker (X(0) - X(1 mM)), so across genotypes the pheS
swing against the kanR swing is a line through 0 with slope m_pheS /
m_kanR. wt is one point on it.

    python marker_ratio.py ../../dev-data/processed ../m-bias-factorial/real_depth/depth.csv --out out
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from within_tube_m import slopes_by_conc, swing  # noqa: E402

plt.rc("pdf", fonttype=42)
plt.rc("font", size=14)
plt.rc("axes", titlesize=14, labelsize=16)

REF = "M42I/H74A/K84L"
COND = {"kanR": "kanR+kan", "pheS": "pheS+4CP"}
COLS = ["genotype", "counts", "replicate", "condition_sel", "t_sel", "titrant_conc"]


def read(processed_dir, genotypes, cache):
    if os.path.exists(cache):
        return pd.read_csv(cache)
    parts = []
    for lib, cond in COND.items():
        for ch in pd.read_csv(os.path.join(processed_dir, f"growth_{lib}.csv.gz"),
                              usecols=COLS, chunksize=2_000_000):
            parts.append(ch[(ch["condition_sel"] == cond) & ch["genotype"].isin(genotypes)])
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(cache, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed_dir")
    ap.add_argument("depth_csv")
    ap.add_argument("--min_reads", type=float, default=1e5)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    depth = pd.read_csv(args.depth_csv).set_index("genotype")["reads"]
    singles = [g for g, r in depth.items()
               if r >= args.min_reads and isinstance(g, str) and "/" not in g
               and g not in ("wt", "__unknown__")]
    keep = set(singles) | {"wt", REF, "H74A/K84L", "M42I/H74A", "M42I", "H74A",
                           "K84L", "D88A", "M42I/K84L"}
    rows = read(args.processed_dir, keep, os.path.join(args.out, "marker_rows.csv"))
    key = ["replicate", "condition_sel", "titrant_conc", "t_sel"]
    ref = rows[rows["genotype"] == REF].set_index(key)["counts"].rename("c_ref")
    out = []
    for g, d in rows[rows["genotype"] != REF].groupby("genotype"):
        rec = {"genotype": g, "reads": depth.get(g, np.nan)}
        for lib, cond in COND.items():
            t = d[d["condition_sel"] == cond].set_index(key)[["counts"]].join(ref, how="inner")
            t = t[(t["counts"] > 0) & (t["c_ref"] > 0)].reset_index()
            if t["titrant_conc"].isin([0.0, 1.0]).sum() < 4:
                continue
            t["lr"] = np.log(t["counts"] / t["c_ref"])
            t["v"] = 1.0 / t["counts"] + 1.0 / t["c_ref"]
            try:
                c, b, cov, _ = slopes_by_conc(t, "lr", "v")
                rec[f"swing_{lib}"], rec[f"swing_{lib}_se"] = swing(c, b, cov)
            except (ValueError, np.linalg.LinAlgError):
                pass
        out.append(rec)
    s = pd.DataFrame(out).dropna(subset=["swing_kanR", "swing_pheS"])
    s.to_csv(os.path.join(args.out, "marker_swings.csv"), index=False)

    lib = s[~s["genotype"].isin(["wt"]) & ~s["genotype"].str.contains("/")]
    w = 1 / (lib["swing_pheS_se"] ** 2 + 1e-12)
    # weighted line through 0 (pheS on kanR), and a robust median ratio
    big = lib[lib["swing_kanR"].abs() > 2 * lib["swing_kanR_se"]]
    slope0 = np.sum(w * lib.swing_kanR * lib.swing_pheS) / np.sum(w * lib.swing_kanR ** 2)
    print(f"{len(lib)} singles with >= {args.min_reads:g} reads")
    print(f"pheS swing on kanR swing, weighted line through 0: slope {slope0:+.3f}")
    print(f"median pheS/kanR ratio, singles whose kanR swing is > 2 SE ({len(big)}): "
          f"{np.median(big.swing_pheS / big.swing_kanR):+.3f}")
    print("stage 1 m_pheS / m_kanR: 0.0070 / -0.0148 = -0.47")
    print(s[~s["genotype"].isin(lib["genotype"])].to_string(index=False, float_format="%+.4f"))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(lib.swing_kanR, lib.swing_pheS, xerr=lib.swing_kanR_se, yerr=lib.swing_pheS_se,
                fmt="o", ms=3, color="#56B4E9", alpha=0.4, lw=0.5, label="singles")
    sp = s[~s["genotype"].isin(lib["genotype"])]
    for _, r in sp.iterrows():
        col = "#000000" if r.genotype == "wt" else "#D55E00"
        ax.errorbar(r.swing_kanR, r.swing_pheS, xerr=r.swing_kanR_se, yerr=r.swing_pheS_se,
                    fmt="s", color=col, ms=7)
        ax.annotate(r.genotype, (r.swing_kanR, r.swing_pheS), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    x = np.array([-0.03, 0.01])
    ax.plot(x, slope0 * x, color="#0072B2", lw=1, label=f"singles, slope {slope0:+.2f}")
    ax.plot(x, -0.47 * x, color="0.5", lw=1, ls=":", label="stage 1 m ratio -0.47")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("kanR+kan swing vs M42I/H74A/K84L")
    ax.set_ylabel("pheS+4CP swing vs M42I/H74A/K84L")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "marker_ratio.pdf"))


if __name__ == "__main__":
    main()
