"""
Compare follow-up runs (grid_masked.yaml: rows with < 5 reads dropped;
grid_bw1.yaml: binding_weight 1) with their twins in grid.yaml (same
simulated data). Run from the study directory; arguments name the follow-up
grid directories (default: congression_calibration_masked).
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

PAIRS = {"0001": "0001", "0002": "0002", "0003": "0003",
         "0004": "0037", "0005": "0038", "0006": "0039"}


def summarize(d):
    c = json.load(open(f"{d}/combo.json"))
    s, t = c["simulate"], c["template"]
    out = dict(lam_true=s["transformation_poisson_lambda"],
               arm=f"{t['fit']}/{t['lam_prior']}")
    lam = glob.glob(f"{d}/summary/*_params_lam.csv")
    if lam:
        r = pd.read_csv(lam[0]).iloc[0]
        out["lam_q50"], out["lam_q025"], out["lam_q975"] = r["q0.5"], r["q0.025"], r["q0.975"]
    k = pd.read_csv(glob.glob(f"{d}/summary/*_params_growth_k.csv")[0])
    dk = pd.read_csv(glob.glob(f"{d}/summary/*_params_dk_geno.csv")[0])
    out["k_off"] = (k["q0.5"] - k["ref"]).mean()
    out["dk_off"] = (dk["q0.5"] - dk["ref"]).mean()
    out["dk_rmse_centered"] = ((dk["q0.5"] - dk["ref"]) - out["dk_off"]).pow(2).mean() ** 0.5

    th = pd.read_csv(glob.glob(f"{d}/summary/*_theta_corr_test.csv")[0])
    lib = pd.read_csv(glob.glob(f"{d}/*_sim_library.csv")[0])
    binding = set(pd.read_csv(glob.glob(f"{d}/*_sim_binding.csv")[0]).genotype.astype(str))
    spiked = set(lib.loc[lib.library_origin == "spiked", "genotype"].astype(str))
    th["genotype"] = th.genotype.astype(str)
    bulk_nobind = th[~th.genotype.isin(spiked | binding)]
    lo, hi = bulk_nobind["q0.025"], bulk_nobind["q0.975"]
    out["theta_rmse"] = ((bulk_nobind["q0.5"] - bulk_nobind["ref"]) ** 2).mean() ** 0.5
    out["theta_cov95"] = ((lo <= bulk_nobind.ref) & (bulk_nobind.ref <= hi)).mean()
    return out


GRIDS = sys.argv[1:] or ["congression_calibration_masked"]
rows = []
for m, o in PAIRS.items():
    runs = [("original", glob.glob(f"congression_calibration/run_{o}_*")[0])]
    runs += [(g.replace("congression_calibration_", ""), glob.glob(f"{g}/run_{m}_*")[0])
             for g in GRIDS]
    for tag, d in runs:
        r = summarize(d)
        r["data"] = tag
        rows.append(r)

df = pd.DataFrame(rows)
cols = ["lam_true", "arm", "data", "lam_q025", "lam_q50", "lam_q975",
        "k_off", "dk_off", "dk_rmse_centered", "theta_rmse", "theta_cov95"]
pd.set_option("display.width", 220)
print(df[cols].sort_values(["lam_true", "arm", "data"]).round(4).to_string(index=False))
