"""
Does each selection marker respond at the same IPTG? (README, run 2.)

For every deep genotype and marker, the within-tube curve y(c) = b(c) -
b(1 mM), b(c) the slope over t_sel of ln(genotype / M42I/H74A/K84L) at
IPTG c, is fit with y = A (1 - s(c)), s = sigmoid(n (ln c - ln K)), s(0) =
0. Genotypes whose swing is beyond 3 SE in both markers are compared on
ln K. Also the mean normalized curve (y / y(0)) per marker.

    python midpoints.py ../within-tube-m/out/marker_rows.csv --out out
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "within-tube-m"))
from within_tube_m import slopes_by_conc, ZERO_X  # noqa: E402

plt.rc("pdf", fonttype=42)
plt.rc("font", size=14)
plt.rc("axes", titlesize=14, labelsize=16)

REF = "M42I/H74A/K84L"
KEY = ["replicate", "condition_sel", "titrant_conc", "t_sel"]
COND = {"kanR+kan": "#0072B2", "pheS+4CP": "#D55E00"}


def hill_drop(c, A, lnK, n):
    s = np.where(c > 0, 1 / (1 + np.exp(-n * (np.log(np.maximum(c, 1e-12)) - lnK))), 0.0)
    return A * (1 - s)


def curve(t):
    c, b, cov, _ = slopes_by_conc(t, "lr", "v")
    j = list(c).index(1.0)
    se = np.sqrt(np.clip(np.diag(cov) + cov[j, j] - 2 * cov[:, j], 0, None))
    return c, b - b[j], se


def fit(c, y, se):
    se = np.where(se > 0, se, np.median(se[se > 0]))
    p0 = [y[0], np.log(0.01), 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, cov = curve_fit(hill_drop, c, y, p0=p0, sigma=se, absolute_sigma=True,
                           bounds=([-1, np.log(1e-5), 0.3], [1, np.log(10), 4]),
                           maxfev=20000)
    return p, np.sqrt(np.diag(cov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = pd.read_csv(args.rows_csv)
    ref = rows[rows["genotype"] == REF].set_index(KEY)["counts"].rename("c_ref")
    out, curves = [], []
    for g, d in rows[rows["genotype"] != REF].groupby("genotype"):
        rec = {"genotype": g}
        for cond in COND:
            t = d[d["condition_sel"] == cond].set_index(KEY)[["counts"]].join(
                ref, how="inner").reset_index()
            t = t[(t["counts"] > 0) & (t["c_ref"] > 0)]
            if t["titrant_conc"].nunique() < 6:
                continue
            t["lr"] = np.log(t["counts"] / t["c_ref"])
            t["v"] = 1 / t["counts"] + 1 / t["c_ref"]
            try:
                c, y, se = curve(t)
                p, pse = fit(c, y, se)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
            rec.update({f"y0_{cond}": y[0], f"y0_se_{cond}": se[0],
                        f"lnK_{cond}": p[1], f"lnK_se_{cond}": pse[1],
                        f"n_{cond}": p[2]})
            curves.append(pd.DataFrame(dict(genotype=g, cond=cond, c=c, y=y, se=se)))
        out.append(rec)
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(args.out, "midpoints.csv"), index=False)
    cv = pd.concat(curves, ignore_index=True)

    ok = s.dropna(subset=["lnK_kanR+kan", "lnK_pheS+4CP"]).copy()
    for cond in COND:
        ok = ok[(ok[f"y0_{cond}"] / ok[f"y0_se_{cond}"]).abs() > args.z]
    singles = ok[~ok["genotype"].str.contains("/") & (ok["genotype"] != "wt")]
    d = singles["lnK_pheS+4CP"] - singles["lnK_kanR+kan"]
    print(f"{len(singles)} singles with a swing beyond {args.z} SE in both markers")
    print(f"ln K, pheS minus kanR: median {d.median():+.2f} (x{np.exp(d.median()):.2f} "
          f"in IPTG), IQR {d.quantile(.25):+.2f} to {d.quantile(.75):+.2f}, "
          f"{(d < 0).mean():.0%} below 0")
    for cond in COND:
        k = np.exp(singles[f"lnK_{cond}"]) * 1e3
        print(f"  {cond}: median K {k.median():.1f} uM, median n {singles[f'n_{cond}'].median():.2f}")
    show = s[s["genotype"].isin(["wt", "M42I/H74A", "M42I/K84L", "M42I", "H74A", "K84L", "D88A"])]
    cols = ["genotype"] + [f"{v}_{c}" for c in COND for v in ("y0", "lnK", "n")]
    print(show[cols].assign(**{f"K_uM_{c}": np.exp(show[f"lnK_{c}"]) * 1e3 for c in COND})
          .to_string(index=False, float_format="%+.4f"))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sel = cv[cv["genotype"].isin(singles["genotype"])]
    for cond, col in COND.items():
        x = sel[sel["cond"] == cond].copy()
        y0 = x[x["c"] == 0.0].set_index("genotype")["y"]
        x["yn"] = x["y"] / x["genotype"].map(y0)
        m = x.groupby("c")["yn"].median()
        ax[0].plot(np.where(m.index > 0, m.index, ZERO_X), m.values, "o-", color=col,
                   label=f"{cond} (median of {len(y0)} singles)")
        w = cv[(cv["genotype"] == "wt") & (cv["cond"] == cond)]
        ax[0].plot(np.where(w.c > 0, w.c, ZERO_X), w.y / w.y.iloc[0], "s--", color=col,
                   mfc="white", label=f"{cond}, wt")
    ax[0].set_xscale("log")
    ax[0].axhline(0, color="0.85", lw=0.8)
    ax[0].axhline(1, color="0.85", lw=0.8)
    ax[0].set_xlabel("IPTG (mM; 0 at left)")
    ax[0].set_ylabel("within-tube curve / its value at 0")
    ax[0].legend(fontsize=10)
    ax[1].hist(d, bins=40, color="#999999")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_xlabel("ln K, pheS minus kanR")
    ax[1].set_ylabel("singles")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "midpoints.pdf"))


if __name__ == "__main__":
    main()
