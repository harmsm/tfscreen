"""
wt against the library over the selection, observed and predicted by two
stage 1 fits (README, run 6): ln(wt / library) per tube, the library the
geometric mean of the deep singles present in every tube, against t_sel
for each condition and IPTG. Tube totals and offsets cancel.

    python wt_trajectory.py runs/warp_real_attrfree/stage1 runs/warp_real_attrpinned/stage1 \
        --labels free pinned --out out

Each folder needs tfs_growth.csv (observed counts) and pred_growth.csv
(tfs-predict-growth at the MAP; q0.5 = predicted ln_cfu).
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
plt.rc("font", size=12)

KEY = ["replicate", "condition_sel", "titrant_conc", "t_sel"]
CONDS = ["kanR+kan", "kanR-kan", "pheS+4CP", "pheS-4CP"]
REP_COLOR = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
FIT_STYLE = ["-", "--", ":"]


def wt_vs_library(stage1, min_reads):
    counts = pd.read_csv(os.path.join(stage1, "tfs_growth.csv"),
                         usecols=KEY + ["genotype", "counts"])
    reads = counts.groupby("genotype")["counts"].sum()
    lib = [g for g, r in reads.items() if r >= min_reads and "/" not in g and g != "wt"]
    obs_w = counts.pivot_table(index=KEY, columns="genotype", values="counts")
    pred = pd.read_csv(os.path.join(stage1, "pred_growth.csv"),
                       usecols=KEY + ["genotype", "q0.5"])
    P = pred.pivot_table(index=KEY, columns="genotype", values="q0.5").reindex(obs_w.index)
    keep = [g for g in lib if g in obs_w.columns and (obs_w[g] > 0).all() and g in P.columns]
    out = pd.DataFrame(index=obs_w.index)
    out["obs"] = np.log(obs_w["wt"]) - np.log(obs_w[keep]).mean(axis=1)
    out["obs_se"] = np.sqrt(1 / obs_w["wt"])
    out["pred"] = P["wt"] - P[keep].mean(axis=1)
    return out.reset_index(), len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage1_dirs", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--min_reads", type=float, default=1e5)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(os.path.dirname(os.path.abspath(d)))
                             for d in args.stage1_dirs]
    fits = []
    for d, lab in zip(args.stage1_dirs, labels):
        t, n = wt_vs_library(d, args.min_reads)
        print(f"{lab}: library = geometric mean of {n} deep singles")
        fits.append(t.assign(fit=lab))
    obs = fits[0][KEY + ["obs", "obs_se"]]
    concs = sorted(obs["titrant_conc"].unique())

    rows = []
    for t in fits:
        r = t.assign(resid=t["obs"] - t["pred"])
        for (cond, c), g in r.groupby(["condition_sel", "titrant_conc"]):
            tt = g["t_sel"] - g.groupby("replicate")["t_sel"].transform("mean")
            rr = g["resid"] - g.groupby("replicate")["resid"].transform("mean")
            slope = np.polyfit(tt, rr, 1)[0] if tt.abs().sum() > 0 else np.nan
            rows.append(dict(fit=t["fit"].iloc[0], condition_sel=cond, titrant_conc=c,
                             mean_resid=g["resid"].mean(), resid_trend=slope,
                             rms_resid=np.sqrt((g["resid"] ** 2).mean())))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(args.out, "wt_trajectory_residuals.csv"), index=False)
    for what in ("mean_resid", "resid_trend"):
        print(f"\n== observed - predicted ln(wt / library): {what}"
              f"{' (per minute, within replicate)' if what == 'resid_trend' else ''}")
        tab = s.pivot_table(index=["condition_sel", "fit"], columns="titrant_conc", values=what)
        tab.columns = [f"{c * 1e3:g}uM" for c in tab.columns]
        print(tab.to_string(float_format=lambda v: f"{v:+.4f}"))

    stamp = datetime.date.today().isoformat()
    with PdfPages(os.path.join(args.out, f"{stamp}_wt-trajectory.pdf")) as pdf:
        fig, axes = plt.subplots(len(CONDS), len(concs), figsize=(2.6 * len(concs), 2.6 * len(CONDS)),
                                 sharex="row", squeeze=False)
        for i, cond in enumerate(CONDS):
            for j, c in enumerate(concs):
                ax = axes[i, j]
                o = obs[(obs.condition_sel == cond) & (obs.titrant_conc == c)]
                if o.empty:
                    ax.axis("off")
                    continue
                for rep, g in o.groupby("replicate"):
                    g = g.sort_values("t_sel")
                    ax.errorbar(g.t_sel, g.obs, yerr=1.96 * g.obs_se, fmt="o",
                                color=REP_COLOR.get(rep, "k"), ms=4)
                    for k, t in enumerate(fits):
                        p = t[(t.condition_sel == cond) & (t.titrant_conc == c)
                              & (t.replicate == rep)].sort_values("t_sel")
                        ax.plot(p.t_sel, p.pred, FIT_STYLE[k % 3], color=REP_COLOR.get(rep, "k"),
                                lw=1.2, label=f"{t['fit'].iloc[0]}" if rep == 1 else None)
                ax.set_title(f"{cond}, {c * 1e3:g} uM", fontsize=9)
                if j == 0:
                    ax.set_ylabel("ln(wt / library)")
                if i == len(CONDS) - 1:
                    ax.set_xlabel("t_sel (min)")
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
