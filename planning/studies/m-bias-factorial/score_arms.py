"""
Score the factorial arms: k and m against the truth on the X scale.

    python score_arms.py m_bias/*_s*

Truth on the fit's X scale (theta: hill_relative gauges wt to X = 1 at the
low and 0 at the high gauge concentration): m_X = m (theta_wt(c_lo) -
theta_wt(c_hi)), k_X = b + m theta_wt(c_hi), with wt's true Hill curve. Also
the median dk_geno error over genotypes. Writes factorial_scores.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
import yaml


def wt_theta(p, conc):
    lo, hi = p["theta_low"], p["theta_high"]
    if conc <= 0:
        return lo
    occ = 1.0 / (1.0 + np.exp(-p["hill_n"] * (np.log(conc) - p["log_hill_K"])))
    return lo + (hi - lo) * occ


rows = []
for d in sys.argv[1:]:
    if not os.path.exists(os.path.join(d, "tfs_params_growth_m.csv")):
        print("skip (not done):", d)
        continue
    cfg = yaml.safe_load(open(os.path.join(d, "tfs_configure_config.yaml")))
    gauge = None
    for key in ("theta_gauge_conc",):
        gauge = cfg.get("components", {}).get(key) or cfg.get(key)
    c_lo, c_hi = (float(x) for x in gauge)
    par = pd.read_csv(os.path.join(d, "tfs_sim_parameters.csv"))
    wt = par[par.genotype == "wt"].iloc[0]
    t_lo, t_hi = wt_theta(wt, c_lo), wt_theta(wt, c_hi)
    truth = pd.read_csv(os.path.join(d, "tfs_sim_growth_parameters.csv")).set_index("condition_rep")
    k = pd.read_csv(os.path.join(d, "tfs_params_growth_k.csv")).groupby("condition_rep")["q0.5"].mean()
    m = pd.read_csv(os.path.join(d, "tfs_params_growth_m.csv")).groupby("condition_rep")["q0.5"].mean()
    dk = pd.read_csv(os.path.join(d, "tfs_params_dk_geno.csv"))[["genotype", "q0.5"]].merge(
        par[["genotype", "dk_geno"]], on="genotype")
    arm = os.path.basename(os.path.normpath(d))
    for c in ("kanR+kan", "pheS+4CP"):
        b, mm = truth.loc[c, "growth_k"], truth.loc[c, "growth_m"]
        m_x, k_x = mm * (t_lo - t_hi), b + mm * t_hi
        rows.append(dict(arm=arm.rsplit("_s", 1)[0], seed=arm.rsplit("_s", 1)[1],
                         condition=c, k_true=k_x, k_fit=k[c], m_true=m_x, m_fit=m[c],
                         m_ratio=m[c] / m_x, dk_bias=float(np.median(dk["q0.5"] - dk["dk_geno"]))))
df = pd.DataFrame(rows)
df.to_csv("factorial_scores.csv", index=False)
print(df.round(4).to_string(index=False))
print()
print(df.groupby(["arm", "condition"])[["m_ratio", "dk_bias"]].mean().round(3).to_string())
