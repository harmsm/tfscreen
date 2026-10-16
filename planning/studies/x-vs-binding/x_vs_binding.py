"""
Growth X against in vitro binding theta for the binding genotypes
(planning/studies/x-vs-binding/README.md).

    python x_vs_binding.py BINDING_CSV RUN_DIR PROCESSED_DIR --out out

BINDING_CSV   wells: genotype, biorep, date, protein_dimer_conc_molar,
              iptg_conc_molar, theta (dev-data/binding/2026-10-15_binding.csv)
RUN_DIR       a tfs-fit-two-stage run with stage2/, consistency/ (kanR
              alone), consistency_pheS/ (pheS alone) and, when present,
              consistency_pheS_wtheld/ (pheS alone, wt's curve held)
PROCESSED_DIR growth_{kanR,pheS}.csv.gz and wt_control_rate_summary.csv

Binding theta goes onto the X gauge through wt at the same protein
concentration: (theta - theta_wt(1 mM)) / (theta_wt(0) - theta_wt(1 mM)).
Writes out/x_vs_binding.csv (one row per genotype, protein, IPTG and fit),
out/summary.csv (gauged and affine agreement per protein and fit),
out/wt_rates.csv (wt's growth rate in the library, model-free, against the
monoculture) and out/YYYY-MM-DD_x-vs-binding.pdf.
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

FITS = {"both": "stage2", "kanR": "consistency", "pheS": "consistency_pheS",
        "pheS_wtheld": "consistency_pheS_wtheld"}
FIT_COLOR = {"both": "#000000", "kanR": "#0072B2", "pheS": "#D55E00",
             "pheS_wtheld": "#E69F00"}
PROTEIN_MARK = {2.0: ("o", "#009E73"), 20.0: ("s", "#CC79A7")}
SCREEN_IPTG = [0.0, 1e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 1.0]   # mM
ZERO_X = 3e-5   # where 0 IPTG is drawn on the log axis (mM)


def read_binding(path):
    """Mean theta per genotype, protein (uM dimer) and IPTG (mM)."""
    b = pd.read_csv(path).dropna(subset=["theta"])
    b["protein_uM"] = (b["protein_dimer_conc_molar"] * 1e6).round(3)
    b["titrant_conc"] = (b["iptg_conc_molar"] * 1e3).round(7)
    b = b[b["titrant_conc"].isin(SCREEN_IPTG)]
    out = b.groupby(["genotype", "protein_uM", "titrant_conc"])["theta"].agg(
        theta="mean", theta_sd="std", n_wells="size").reset_index()
    out["theta_se"] = out["theta_sd"] / np.sqrt(out["n_wells"])
    return out


def gauge(b):
    """Put theta on the X gauge through wt at each protein concentration."""
    rows = []
    for p, d in b.groupby("protein_uM"):
        w = d[d["genotype"] == "wt"].set_index("titrant_conc")["theta"]
        if 0.0 not in w.index or 1.0 not in w.index:
            continue
        span = w[0.0] - w[1.0]
        rows.append(d.assign(theta_gauged=(d["theta"] - w[1.0]) / span,
                             theta_gauged_se=d["theta_se"] / abs(span)))
    return pd.concat(rows, ignore_index=True)


def read_x(run_dir, genotypes):
    rows = []
    for fit, sub in FITS.items():
        p = os.path.join(run_dir, sub, "tfs_pred_theta.csv")
        if not os.path.exists(p):
            print(f"no {p}; skipping {fit}")
            continue
        t = pd.read_csv(p, usecols=["genotype", "titrant_conc", "q0.025",
                                    "q0.5", "q0.975"])
        t = t[t["genotype"].isin(genotypes)].assign(fit=fit)
        t["titrant_conc"] = t["titrant_conc"].round(7)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def agreement(m):
    rows = []
    for (p, fit), d in m[m["genotype"] != "wt"].groupby(["protein_uM", "fit"]):
        x, t = d["q0.5"].to_numpy(), d["theta_gauged"].to_numpy()
        slope, icpt = np.polyfit(t, x, 1)
        rows.append(dict(protein_uM=p, fit=fit, n=len(d),
                         n_genotypes=d["genotype"].nunique(),
                         gauged_mean=np.mean(x - t),
                         gauged_rms=np.sqrt(np.mean((x - t) ** 2)),
                         affine_intercept=icpt, affine_slope=slope,
                         affine_rms=np.sqrt(np.mean((x - icpt - slope * t) ** 2)),
                         r=np.corrcoef(t, x)[0, 1]))
    return pd.DataFrame(rows)


def wt_rates(processed_dir, cache):
    """wt's growth rate per condition and IPTG in the library (slope of its
    ln_cfu over t_sel, averaged over replicates), with the monoculture's."""
    if os.path.exists(cache):
        return pd.read_csv(cache)
    keep = ["genotype", "replicate", "condition_sel", "t_sel", "titrant_conc",
            "ln_cfu", "sample_ln_cfu"]
    parts = []
    for lib in ("kanR", "pheS"):
        for ch in pd.read_csv(os.path.join(processed_dir, f"growth_{lib}.csv.gz"),
                              usecols=keep, chunksize=2_000_000):
            parts.append(ch[ch["genotype"] == "wt"])
    w = pd.concat(parts)

    def slopes(g):
        if g["t_sel"].nunique() < 2:
            return pd.Series({"library_rate": np.nan, "population_rate": np.nan})
        return pd.Series({
            "library_rate": np.polyfit(g["t_sel"], g["ln_cfu"], 1)[0],
            "population_rate": np.polyfit(g["t_sel"], g["sample_ln_cfu"], 1)[0]})

    r = (w.groupby(["condition_sel", "titrant_conc", "replicate"])
         .apply(slopes, include_groups=False)
         .groupby(["condition_sel", "titrant_conc"])
         .agg(["mean", "std"]))
    r.columns = [f"{a}_{'sd' if b == 'std' else b}" for a, b in r.columns]
    mono = pd.read_csv(os.path.join(processed_dir, "wt_control_rate_summary.csv"))
    mono = mono.set_index(["condition_sel", "titrant_conc"])[["rate_mean", "rate_sd"]]
    mono.columns = ["mono_rate_mean", "mono_rate_sd"]
    out = r.join(mono, how="outer").reset_index()
    out.to_csv(cache, index=False)
    return out


def xpos(c):
    return np.where(c > 0, c, ZERO_X)


def plot_curves(pdf, m, b, spiked):
    genos = ["wt"] + sorted(g for g in m["genotype"].unique() if g != "wt")
    n = len(genos)
    ncol = 3
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(15, 4.2 * np.ceil(n / ncol)),
                             sharex=True, sharey=True)
    for ax, g in zip(axes.flat, genos):
        x = m[m["genotype"] == g].drop_duplicates(["fit", "titrant_conc"])
        for fit, d in x.groupby("fit"):
            d = d.sort_values("titrant_conc")
            ax.fill_between(xpos(d["titrant_conc"]), d["q0.025"], d["q0.975"],
                            color=FIT_COLOR[fit], alpha=0.15, lw=0)
            ax.plot(xpos(d["titrant_conc"]), d["q0.5"], "-", color=FIT_COLOR[fit],
                    label=f"X, {fit}")
        bb = b[b["genotype"] == g]
        for p, (mk, col) in PROTEIN_MARK.items():
            d = bb[bb["protein_uM"] == p].sort_values("titrant_conc")
            if len(d):
                ax.errorbar(xpos(d["titrant_conc"]), d["theta_gauged"],
                            yerr=1.96 * d["theta_gauged_se"].fillna(0), fmt=mk + "--",
                            color=col, mfc="white", ms=7, lw=1,
                            label=f"theta, {p:g} uM")
        ax.set_xscale("log")
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.axhline(1, color="0.8", lw=0.8, zorder=0)
        ax.set_title(f"{g} ({'spike' if g in spiked else 'bulk only'})")
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("IPTG (mM; 0 at left)")
    for ax in axes[:, 0]:
        ax.set_ylabel("X, gauged theta")
    axes.flat[0].legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_scatter(pdf, m):
    ps = sorted(m["protein_uM"].unique())
    fits = [f for f in FITS if f in set(m["fit"])]
    fig, axes = plt.subplots(len(ps), len(fits), figsize=(5 * len(fits), 5 * len(ps)),
                             squeeze=False, sharex=True, sharey=True)
    for i, p in enumerate(ps):
        for j, fit in enumerate(fits):
            ax = axes[i, j]
            d = m[(m["protein_uM"] == p) & (m["fit"] == fit) & (m["genotype"] != "wt")]
            ax.plot([-0.5, 1.5], [-0.5, 1.5], color="0.7", lw=1)
            ax.errorbar(d["theta_gauged"], d["q0.5"],
                        yerr=[d["q0.5"] - d["q0.025"], d["q0.975"] - d["q0.5"]],
                        fmt="o", color=FIT_COLOR[fit], ms=5, alpha=0.7, lw=0.8)
            ax.set_title(f"{fit}, {p:g} uM protein")
            if i == len(ps) - 1:
                ax.set_xlabel("gauged binding theta")
            if j == 0:
                ax.set_ylabel("growth X")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_wt_rates(pdf, r):
    conds = [c for c in ("kanR+kan", "pheS+4CP") if c in set(r["condition_sel"])]
    fig, axes = plt.subplots(1, len(conds), figsize=(6.5 * len(conds), 5), squeeze=False)
    for ax, c in zip(axes[0], conds):
        d = r[r["condition_sel"] == c].sort_values("titrant_conc")
        for col, lab, colr in (("library_rate", "wt in the library", "#0072B2"),
                               ("mono_rate", "wt monoculture", "#E69F00"),
                               ("population_rate", "library population", "#000000")):
            dd = d.dropna(subset=[f"{col}_mean"])
            ax.errorbar(xpos(dd["titrant_conc"]), dd[f"{col}_mean"], yerr=dd[f"{col}_sd"],
                        fmt="o-", color=colr, label=lab, ms=6, lw=1.2, capsize=2)
        ax.set_xscale("log")
        ax.set_title(c)
        ax.set_xlabel("IPTG (mM; 0 at left)")
        ax.set_ylabel("growth rate (1/min)")
    axes[0, 0].legend(fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binding_csv")
    ap.add_argument("run_dir")
    ap.add_argument("processed_dir")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    b = gauge(read_binding(args.binding_csv))
    lib = pd.read_csv(os.path.join(args.run_dir, "stage2", "tfs_configure_library.csv"))
    spiked = set(lib.loc[lib["in_spiked_origin"].astype(bool), "genotype"])
    x = read_x(args.run_dir, set(b["genotype"]))
    m = b.merge(x, on=["genotype", "titrant_conc"])
    m["spiked"] = m["genotype"].isin(spiked)
    m.to_csv(os.path.join(args.out, "x_vs_binding.csv"), index=False)

    s = agreement(m)
    s.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print(s.round(3).to_string(index=False))

    r = wt_rates(args.processed_dir, os.path.join(args.out, "wt_rates.csv"))
    print(r.round(4).to_string(index=False))

    stamp = datetime.date.today().isoformat()
    with PdfPages(os.path.join(args.out, f"{stamp}_x-vs-binding.pdf")) as pdf:
        plot_curves(pdf, m, b, spiked)
        plot_scatter(pdf, m)
        plot_wt_rates(pdf, r)


if __name__ == "__main__":
    main()
