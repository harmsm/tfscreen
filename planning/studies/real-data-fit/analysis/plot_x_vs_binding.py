"""
Growth-only X (rel_off_n05; 95% band from rel_off_n05_blap, the
per-genotype Laplace, when present) against binding theta at 10 uM protein for the
seven binding genotypes, both against IPTG. Binding theta is put on the X
gauge through wt (theta -> (theta - theta_wt(1 mM)) / (theta_wt(0) -
theta_wt(1 mM))), so wt's two curves share their ends.

    python plot_x_vs_binding.py   ->  resid/2026-10-02_x-vs-binding.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rc("pdf", fonttype=42)
plt.rc("font", size=14)
plt.rc("axes", titlesize=14, labelsize=16)
OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
ORDER = ["wt", "M42I", "M42I/H74A", "G65H", "H74A", "K84L", "H74A/K84L"]
ZERO_X = 3e-5   # where 0 IPTG is drawn on the log axis (mM)

j = pd.read_csv("resid/X_vs_binding_off_n05.csv")
try:
    band = pd.read_parquet("resid/blap_X.parquet")[["genotype", "titrant_conc",
                                                   "q0.025", "q0.975"]]
    j = j.merge(band, on=["genotype", "titrant_conc"], how="left")
except FileNotFoundError:
    j["q0.025"] = j["q0.975"] = np.nan
w = j[j.genotype == "wt"].set_index("titrant_conc").theta_obs
j["theta_gauged"] = (j.theta_obs - w[1.0]) / (w[0.0] - w[1.0])
j["x_plot"] = j.titrant_conc.where(j.titrant_conc > 0, ZERO_X)

fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True, sharey=True)
for ax, g, c in zip(axes.flat, ORDER, OKABE):
    d = j[j.genotype == g].sort_values("x_plot")
    ax.fill_between(d.x_plot, d["q0.025"], d["q0.975"], color=c, alpha=0.2,
                    lw=0, label="X 95% (block Laplace)")
    ax.plot(d.x_plot, d.X, "o-", color=c, label="growth X")
    ax.plot(d.x_plot, d.theta_gauged, "s--", color=c, mfc="white",
            label="binding θ (wt gauge)")
    ax.axhline(0, color="0.8", lw=1, zorder=0)
    ax.axhline(1, color="0.8", lw=1, zorder=0)
    ax.set_xscale("log")
    ax.set_title(g)
axes.flat[-1].axis("off")
axes.flat[0].legend(frameon=False, fontsize=11, loc="lower left")
for ax in axes[1]:
    ax.set_xlabel("IPTG (mM; 0 drawn at 3e-5)")
for ax in axes[:, 0]:
    ax.set_ylabel("X")
fig.tight_layout()
fig.savefig("resid/2026-10-02_x-vs-binding.pdf")
print("wrote resid/2026-10-02_x-vs-binding.pdf")
