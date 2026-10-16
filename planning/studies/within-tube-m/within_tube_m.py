"""
Each selection condition's growth slope m, measured two ways from the
same data (planning/studies/within-tube-m/README.md):

- within tube: the slope over t_sel of ln(counts_wt / counts_ref), for a
  reference genotype whose X is flat in IPTG (pure spikes H74A/K84L and
  M42I/H74A/K84L). With growth = k + dk + m X and wt gauged to X = 1 at
  0 IPTG and 0 at 1 mM, that slope is (dk_wt - dk_ref) + m (X_wt - X_ref),
  so its change from 0 to 1 mM is m. Tube totals, offsets and dk cancel.
- absolute: the same change in wt's own ln_cfu slope, which goes through
  the tube totals (OD600).

Also the reference's own absolute slope change (0 under the model, since
its X is flat), wt's within-tube curve against IPTG, and m from the early
and late time intervals. Compared with the wt monoculture and a
two-stage run's stage 1.

    python within_tube_m.py PROCESSED_DIR --stage1_km RUN/stage1/km.csv --out out
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

plt.rc("pdf", fonttype=42)
plt.rc("font", size=14)
plt.rc("axes", titlesize=14, labelsize=16)
plt.rc("legend", fontsize=12)

REFS = ["H74A/K84L", "M42I/H74A/K84L"]
REF_COLOR = {"H74A/K84L": "#0072B2", "M42I/H74A/K84L": "#009E73"}
SELECTION = ["kanR+kan", "pheS+4CP"]
ZERO_X = 3e-5
COLS = ["sample", "genotype", "counts", "replicate", "condition_sel", "t_sel",
        "titrant_conc", "ln_cfu"]


def read_rows(processed_dir, genotypes, cache):
    if os.path.exists(cache):
        return pd.read_csv(cache)
    parts = []
    for lib in ("kanR", "pheS"):
        path = os.path.join(processed_dir, f"growth_{lib}.csv.gz")
        for ch in pd.read_csv(path, usecols=COLS, chunksize=2_000_000):
            parts.append(ch[ch["genotype"].isin(genotypes)])
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(cache, index=False)
    return out


def slopes_by_conc(df, y, var=None):
    """
    Fit y = a[conc, replicate] + b[conc] * t_sel by weighted least squares
    (weights 1/var, or unweighted), scaling the covariance by the reduced
    chi-square when it exceeds 1. Returns (concs, b, cov, chi2/dof).
    """
    df = df.dropna(subset=[y]).copy()
    concs = np.sort(df["titrant_conc"].unique())
    groups = df.groupby(["titrant_conc", "replicate"]).ngroup().to_numpy()
    n_g = groups.max() + 1
    ci = np.searchsorted(concs, df["titrant_conc"].to_numpy())
    X = np.zeros((len(df), n_g + len(concs)))
    X[np.arange(len(df)), groups] = 1.0
    X[np.arange(len(df)), n_g + ci] = df["t_sel"].to_numpy()
    w = np.ones(len(df)) if var is None else 1.0 / df[var].to_numpy()
    sw = np.sqrt(w)
    A, yy = X * sw[:, None], df[y].to_numpy() * sw
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ beta
    dof = max(len(df) - X.shape[1], 1)
    chi2 = float(resid @ resid) / dof
    cov = np.linalg.pinv(A.T @ A)
    scale = max(chi2, 1.0) if var is not None else chi2
    cov = cov * scale
    k = slice(n_g, n_g + len(concs))
    return concs, beta[k], cov[k, k], chi2


def swing(concs, b, cov, lo=0.0, hi=1.0):
    """b(lo) - b(hi) and its SE."""
    i, j = list(concs).index(lo), list(concs).index(hi)
    v = cov[i, i] + cov[j, j] - 2 * cov[i, j]
    return b[i] - b[j], np.sqrt(max(v, 0.0))


def pair_table(rows, ref):
    """One row per tube with wt's and ref's counts and ln_cfu."""
    key = ["sample", "replicate", "condition_sel", "t_sel", "titrant_conc"]
    a = rows[rows["genotype"] == "wt"].set_index(key)[["counts", "ln_cfu"]]
    b = rows[rows["genotype"] == ref].set_index(key)[["counts", "ln_cfu"]]
    t = a.join(b, lsuffix="_wt", rsuffix="_ref", how="inner").reset_index()
    t = t[(t["counts_wt"] > 0) & (t["counts_ref"] > 0)]
    t["lr"] = np.log(t["counts_wt"] / t["counts_ref"])
    t["lr_var"] = 1.0 / t["counts_wt"] + 1.0 / t["counts_ref"]
    return t


def interval_swings(t):
    """m from the early and the late interval of each (conc, replicate)
    series: per-interval slopes of lr, averaged over replicates."""
    out = {}
    rows = []
    for (c, r), d in t.groupby(["titrant_conc", "replicate"]):
        d = d.sort_values("t_sel")
        if len(d) < 3:
            continue
        tt, yy = d["t_sel"].to_numpy(), d["lr"].to_numpy()
        for name, (i, j) in (("early", (0, 1)), ("late", (len(d) - 2, len(d) - 1))):
            dt = tt[j] - tt[i]
            if dt <= 0:
                continue
            rows.append(dict(titrant_conc=c, replicate=r, interval=name,
                             slope=(yy[j] - yy[i]) / dt))
    s = pd.DataFrame(rows)
    if s.empty:
        return out
    g = s.groupby(["interval", "titrant_conc"])["slope"].agg(["mean", "std", "count"])
    for name in ("early", "late"):
        if (name, 0.0) in g.index and (name, 1.0) in g.index:
            a, b = g.loc[(name, 0.0)], g.loc[(name, 1.0)]
            se = np.sqrt((a["std"] ** 2) / a["count"] + (b["std"] ** 2) / b["count"])
            out[name] = (a["mean"] - b["mean"], se)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed_dir")
    ap.add_argument("--stage1_km", default=None, help="a two-stage run's stage1/km.csv")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = read_rows(args.processed_dir, ["wt"] + REFS,
                     os.path.join(args.out, "rows.csv"))
    mono = pd.read_csv(os.path.join(args.processed_dir, "wt_control_rate_summary.csv"))
    km = (pd.read_csv(args.stage1_km, index_col="condition_rep")
          if args.stage1_km else None)

    summary, curves = [], []
    for cond in sorted(rows["condition_sel"].unique()):
        r = rows[rows["condition_sel"] == cond]
        mo = mono[mono["condition_sel"] == cond].set_index("titrant_conc")
        m_mono = (mo.loc[0.0, "rate_mean"] - mo.loc[1.0, "rate_mean"]
                  if {0.0, 1.0} <= set(mo.index) else np.nan)
        m_mono_se = (np.hypot(mo.loc[0.0, "rate_sd"] / np.sqrt(mo.loc[0.0, "num_replicates"]),
                              mo.loc[1.0, "rate_sd"] / np.sqrt(mo.loc[1.0, "num_replicates"]))
                     if {0.0, 1.0} <= set(mo.index) else np.nan)
        # stage 1's m is the X-scale slope: growth = k + m X, so b(0) - b(1 mM) = m
        m_s1 = km.loc[cond, "m"] if km is not None and cond in km.index else np.nan

        wt = r[r["genotype"] == "wt"]
        c, b, cov, chi2 = slopes_by_conc(wt, "ln_cfu")
        m_abs, m_abs_se = swing(c, b, cov)
        base = dict(condition_sel=cond, m_stage1=m_s1, m_mono=m_mono,
                    m_mono_se=m_mono_se, m_abs_wt=m_abs, m_abs_wt_se=m_abs_se)
        for ref in REFS:
            t = pair_table(r, ref)
            c, b, cov, chi2 = slopes_by_conc(t, "lr", "lr_var")
            m_rel, m_rel_se = swing(c, b, cov)
            rr = r[r["genotype"] == ref]
            c2, b2, cov2, _ = slopes_by_conc(rr, "ln_cfu")
            ref_abs, ref_abs_se = swing(c2, b2, cov2)
            iv = interval_swings(t)
            summary.append(dict(base, reference=ref, m_within=m_rel,
                                m_within_se=m_rel_se, within_chi2=chi2,
                                ref_abs_change=ref_abs, ref_abs_change_se=ref_abs_se,
                                m_within_early=iv.get("early", (np.nan,))[0],
                                m_within_early_se=iv.get("early", (np.nan, np.nan))[1],
                                m_within_late=iv.get("late", (np.nan,))[0],
                                m_within_late_se=iv.get("late", (np.nan, np.nan))[1]))
            j = list(c).index(1.0)
            se = np.sqrt(np.clip(np.diag(cov) + cov[j, j] - 2 * cov[:, j], 0, None))
            curves.append(pd.DataFrame(dict(condition_sel=cond, reference=ref,
                                            titrant_conc=c, curve=b - b[j],
                                            curve_se=se)))
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(args.out, "m_estimates.csv"), index=False)
    cv = pd.concat(curves, ignore_index=True)
    cv.to_csv(os.path.join(args.out, "wt_curves.csv"), index=False)
    show = ["condition_sel", "reference", "m_within", "m_within_se", "within_chi2",
            "m_within_early", "m_within_late", "m_abs_wt", "m_abs_wt_se",
            "ref_abs_change", "ref_abs_change_se", "m_mono", "m_mono_se", "m_stage1"]
    print(s[show].to_string(index=False, float_format="%.4f"))

    stamp = datetime.date.today().isoformat()
    with PdfPages(os.path.join(args.out, f"{stamp}_within-tube-m.pdf")) as pdf:
        fig, axes = plt.subplots(1, len(SELECTION), figsize=(13, 5), squeeze=False)
        for ax, cond in zip(axes[0], SELECTION):
            for ref in REFS:
                d = cv[(cv.condition_sel == cond) & (cv.reference == ref)]
                x = np.where(d.titrant_conc > 0, d.titrant_conc, ZERO_X)
                ax.errorbar(x, d.curve, yerr=1.96 * d.curve_se, fmt="o-",
                            color=REF_COLOR[ref], ms=6, capsize=2,
                            label=f"wt vs {ref}, within tube")
            mo = mono[mono["condition_sel"] == cond].sort_values("titrant_conc")
            if len(mo):
                hi = mo.set_index("titrant_conc").loc[1.0, "rate_mean"]
                x = np.where(mo.titrant_conc > 0, mo.titrant_conc, ZERO_X)
                ax.errorbar(x, mo.rate_mean - hi, yerr=mo.rate_sd, fmt="s--",
                            color="#E69F00", mfc="white", ms=6, capsize=2,
                            label="wt monoculture")
            if km is not None and cond in km.index:
                ax.axhline(km.loc[cond, "m"], color="#000000", lw=1, ls=":",
                           label="stage 1 m")
            ax.axhline(0, color="0.8", lw=0.8, zorder=0)
            ax.set_xscale("log")
            ax.set_title(cond)
            ax.set_xlabel("IPTG (mM; 0 at left)")
            ax.set_ylabel("wt's growth minus at 1 mM (1/min)")
        axes[0, 0].legend(fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
