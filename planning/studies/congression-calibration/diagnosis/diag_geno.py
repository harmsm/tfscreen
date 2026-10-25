"""
Per-genotype growth log-likelihood gained from the congressed classes in
grid run 0002 (lambda 0, mixture, matched prior), at the fit's median latents.
Writes geno_gain.csv. Run from this directory.
"""
import os
import sys
sys.argv = [sys.argv[0]]
from diag_lam0 import *
import pandas as pd

(d2, o2, s2, st2) = load("run_0002")
lat2, _ = median_latents(o2, s2, st2)
off = dict(lat2); off["transformation_lam"] = jnp.array(1e-8)
on_lp, off_lp = per_genotype(o2, lat2, off)
drop = (on_lp - off_lp).reshape(-1, on_lp.shape[-1]).sum(axis=0)
labels = list(o2.growth_tm.tensor_dim_labels[o2.growth_tm.tensor_dim_names.index("genotype")])
df = pd.DataFrame({"genotype": labels, "loglik_gain_from_congressed": drop})
par = pd.read_csv(glob.glob(d2 + "/summary/*_params_dk_geno.csv")[0])[["genotype", "q0.5", "ref"]]
par.columns = ["genotype", "dk_fit", "dk_true"]
df = df.merge(par, on="genotype", how="left").sort_values("loglik_gain_from_congressed", ascending=False)
print("total gain", df.loglik_gain_from_congressed.sum().round(1))
print("share of gain in top 10 / top 50 genotypes:",
      (df.loglik_gain_from_congressed.head(10).sum() / df.loglik_gain_from_congressed.sum()).round(2),
      (df.loglik_gain_from_congressed.head(50).sum() / df.loglik_gain_from_congressed.sum()).round(2))
print(df.head(12).round(4).to_string(index=False))
print("\nall genotypes: dk_fit - dk_true median %.4f" % (df.dk_fit - df.dk_true).median())
df.to_csv("geno_gain.csv", index=False)
