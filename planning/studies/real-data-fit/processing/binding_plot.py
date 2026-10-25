"""Per-genotype check of binding_fit.py: well theta, fitted curves, 10 uM."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from binding_fit import Model, OUT

plt.rc('pdf', fonttype=42)
plt.rc('font', size=14)
plt.rc('axes', titlesize=14, labelsize=16)
COLORS = {0.02: "#CC79A7", 0.2: "#56B4E9", 2.0: "#E69F00", 20.0: "#0072B2",
          80.0: "#009E73"}

w = pd.read_csv(os.path.join(OUT, "binding_wells_theta.csv"))
w = w[~w["free"]]
curves = pd.read_csv(os.path.join(OUT, "binding_curves.csv"))
interp = pd.read_csv(os.path.join(OUT, "binding_curves_10uM.csv"))
genos = ["wt", "M42I", "H74A", "K84L", "M42I/H74A", "H74A/K84L", "G65H", "V95N"]
c = np.logspace(-6.5, 0.8, 200)
xfloor = 3e-7

fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
for ax, g in zip(axes.ravel(), genos):
    for p, d in w[w["genotype"] == g].groupby("protein_uM_dimer"):
        x = np.where(d["iptg_mM"] > 0, d["iptg_mM"], xfloor)
        ax.scatter(x, d["theta"], s=14, color=COLORS[p], alpha=0.7,
                   label=f"{p:g} uM")
    for _, r in curves[curves["genotype"] == g].iterrows():
        y = Model.hill(r.theta_lo, r.theta_hi, np.exp(r.log_K_mM), r.hill_n, c)
        ax.plot(c, y, color=COLORS[r.protein_uM_dimer], lw=1.5)
    for _, r in interp[interp["genotype"] == g].iterrows():
        y = Model.hill(r.theta_lo, r.theta_hi, np.exp(r.log_K_mM), r.hill_n, c)
        ax.plot(c, y, color="#000000", lw=2, ls="--", label="10 uM")
    ax.set_xscale("log")
    ax.set_title(g)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axhline(1, color="0.7", lw=0.8)
for ax in axes[1]:
    ax.set_xlabel("IPTG (mM); 0 at left edge")
for ax in axes[:, 0]:
    ax.set_ylabel("theta (fraction bound)")
handles, labels = [], []
for ax in axes.ravel():
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h); labels.append(l)
fig.legend(handles, labels, loc="upper right", ncol=6, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(OUT, "binding_fit.pdf"))
print("wrote", os.path.join(OUT, "binding_fit.pdf"))
