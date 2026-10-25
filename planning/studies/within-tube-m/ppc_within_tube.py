"""
Posterior-predictive check of the within-tube statistic (README, run 3):
does a fit reproduce the change, between 0 and 1 mM IPTG, of the slope of
ln(wt / reference) for the flat pure spikes?

    python ppc_within_tube.py runs/ppc/run_real runs/ppc/ktitrant_real_loglinear runs/ppc/run_seed1 --out out

Each RUN_DIR is a stage 1 folder with tfs_growth.csv (the observed counts
and ln_cfu) and pred_growth.csv (tfs-predict-growth at the MAP; q0.5 is
the predicted ln_cfu). The observed ratio comes from counts, the
predicted one from the difference of predicted ln_cfu in the same tube
(tube totals and offsets cancel in both). Also the absolute ln_cfu slopes
of wt and each reference, observed and predicted.
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

from within_tube_m import REFS, ZERO_X, slopes_by_conc, swing  # noqa: E402

plt.rc("pdf", fonttype=42)
plt.rc("font", size=14)
plt.rc("axes", titlesize=14, labelsize=16)
plt.rc("legend", fontsize=11)

KEY = ["replicate", "condition_sel", "titrant_conc", "t_sel"]
SELECTION = ["kanR+kan", "pheS+4CP"]


def load(run_dir):
    keep = ["wt"] + REFS
    obs = pd.read_csv(os.path.join(run_dir, "tfs_growth.csv"),
                      usecols=KEY + ["genotype", "counts", "ln_cfu"])
    obs = obs[obs["genotype"].isin(keep)]
    pred = pd.read_csv(os.path.join(run_dir, "pred_growth.csv"),
                       usecols=KEY + ["genotype", "q0.5"])
    pred = pred[pred["genotype"].isin(keep)]
    m = obs.merge(pred, on=KEY + ["genotype"], how="inner")
    return m.rename(columns={"q0.5": "ln_cfu_pred"})


def pairs(m, cond, ref):
    d = m[m["condition_sel"] == cond]
    a = d[d["genotype"] == "wt"].set_index(KEY)
    b = d[d["genotype"] == ref].set_index(KEY)
    t = a.join(b, lsuffix="_wt", rsuffix="_ref", how="inner").reset_index()
    t = t[(t["counts_wt"] > 0) & (t["counts_ref"] > 0)].copy()
    t["lr_obs"] = np.log(t["counts_wt"] / t["counts_ref"])
    t["lr_var"] = 1.0 / t["counts_wt"] + 1.0 / t["counts_ref"]
    t["lr_pred"] = t["ln_cfu_pred_wt"] - t["ln_cfu_pred_ref"]
    return t


def curve(t, col, var=None):
    c, b, cov, _ = slopes_by_conc(t, col, var)
    j = list(c).index(1.0)
    se = np.sqrt(np.clip(np.diag(cov) + cov[j, j] - 2 * cov[:, j], 0, None))
    return c, b - b[j], se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, curves = [], []
    for run_dir in args.run_dirs:
        name = os.path.basename(os.path.normpath(run_dir))
        m = load(run_dir)
        for cond in SELECTION:
            for ref in REFS:
                t = pairs(m, cond, ref)
                if t.empty:
                    continue
                c, b, cov, chi2 = slopes_by_conc(t, "lr_obs", "lr_var")
                obs, obs_se = swing(c, b, cov)
                c, b, cov, _ = slopes_by_conc(t, "lr_pred")
                pred, _ = swing(c, b, cov)
                absolute = {}
                for g, col in (("wt", "wt"), ("ref", "ref")):
                    for kind, y in (("obs", f"ln_cfu_{col}"), ("pred", f"ln_cfu_pred_{col}")):
                        c2, b2, cov2, _ = slopes_by_conc(t, y)
                        absolute[f"{g}_abs_{kind}"] = swing(c2, b2, cov2)[0]
                rows.append(dict(run=name, condition_sel=cond, reference=ref,
                                 within_obs=obs, within_obs_se=obs_se,
                                 within_pred=pred, **absolute))
                for kind, col, var in (("observed", "lr_obs", "lr_var"),
                                       ("predicted", "lr_pred", None)):
                    cc, yy, ss = curve(t, col, var)
                    curves.append(pd.DataFrame(dict(run=name, condition_sel=cond,
                                                    reference=ref, kind=kind,
                                                    titrant_conc=cc, curve=yy, se=ss)))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(args.out, "ppc_within_tube.csv"), index=False)
    print("Change from 0 to 1 mM of each slope (1/min). within: ln(wt/ref);"
          " *_abs: the genotype's own ln_cfu.")
    print(s.to_string(index=False, float_format="%+.4f"))
    cv = pd.concat(curves, ignore_index=True)
    cv.to_csv(os.path.join(args.out, "ppc_curves.csv"), index=False)

    stamp = datetime.date.today().isoformat()
    runs = list(dict.fromkeys(cv["run"]))
    with PdfPages(os.path.join(args.out, f"{stamp}_ppc-within-tube.pdf")) as pdf:
        fig, axes = plt.subplots(len(runs), len(SELECTION), squeeze=False,
                                 figsize=(13, 4.5 * len(runs)))
        for i, run in enumerate(runs):
            for j, cond in enumerate(SELECTION):
                ax = axes[i, j]
                for ref, col in zip(REFS, ("#0072B2", "#009E73")):
                    for kind, mk, ls in (("observed", "o", "-"), ("predicted", "s", "--")):
                        d = cv[(cv.run == run) & (cv.condition_sel == cond)
                               & (cv.reference == ref) & (cv.kind == kind)]
                        if d.empty:
                            continue
                        x = np.where(d.titrant_conc > 0, d.titrant_conc, ZERO_X)
                        ax.errorbar(x, d.curve, yerr=1.96 * d.se if kind == "observed" else None,
                                    fmt=mk + ls, color=col, ms=5, capsize=2,
                                    mfc="white" if kind == "predicted" else col,
                                    label=f"{kind}, vs {ref}")
                ax.axhline(0, color="0.8", lw=0.8, zorder=0)
                ax.set_xscale("log")
                ax.set_title(f"{run}: {cond}", fontsize=12)
                ax.set_xlabel("IPTG (mM; 0 at left)")
                ax.set_ylabel("wt/ref slope minus at 1 mM")
        axes[0, 0].legend(fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
