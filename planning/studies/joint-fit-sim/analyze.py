"""
Score joint stage 1 fits against the simulation's truth (README).

    python analyze.py RUN_DIR:SIM_DIR [RUN_DIR:SIM_DIR ...] --out out

For each fit: k, m, the warps, log s, the binding offsets and their scale
against the truth files, and theta (in binding units, from the fitted
curves at the growth concentrations) against the truth Hill curves for
wt, the spikes and the singles: RMSE of the median and 95% coverage.
Writes out/params.csv and out/theta.csv and prints both.
"""

import argparse
import os

import numpy as np
import pandas as pd

from tfscreen.tfmodel import two_stage as ts

Q_LO, Q_MID, Q_HI = "q0.025", "q0.5", "q0.975"


def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def hill(p, conc):
    c = np.where(np.asarray(conc) == 0, 1e-20, conc)
    occ = 1 / (1 + np.exp(-p["hill_n"] * (np.log(c) - p["log_hill_K"])))
    return p["theta_low"] + (p["theta_high"] - p["theta_low"]) * occ


def param_rows(run, sim, name):
    rows = []

    def add(what, key, est, truth):
        if est is None:
            return
        for _, r in est.iterrows():
            t = truth(r) if callable(truth) else truth
            rows.append(dict(run=name, parameter=what, key=key(r), truth=t,
                             q0_5=r[Q_MID], lo=r[Q_LO], hi=r[Q_HI],
                             covered=bool(r[Q_LO] <= t <= r[Q_HI])))

    gp = pd.read_csv(os.path.join(sim, "tfs_sim_growth_parameters.csv")).set_index("condition_rep")
    for p, col in (("growth_k", "growth_k"), ("growth_m", "growth_m"),
                   ("warp_log_a", "warp_log_a"), ("warp_x0", "warp_x0")):
        if col not in gp.columns:
            continue
        add(p, lambda r: r["condition_rep"],
            _read(os.path.join(run, f"tfs_params_{p}.csv")),
            lambda r, col=col: gp.loc[r["condition_rep"], col])
    scale = _read(os.path.join(sim, "tfs_sim_growth_titrant_scale.csv"))
    log_s = 0.0 if scale is None else float(scale["ref"].iloc[0])
    add("log_s", lambda r: "global",
        _read(os.path.join(run, "tfs_params_growth_log_conc_shift.csv")), log_s)
    offs = _read(os.path.join(sim, "tfs_sim_binding_offsets.csv"))
    off_truth = {} if offs is None else dict(zip(offs["genotype"], offs["offset"]))
    add("binding_offset", lambda r: r["genotype"],
        _read(os.path.join(run, "tfs_params_offset.csv")),
        lambda r: off_truth.get(r["genotype"], 0.0))
    import yaml
    with open(os.path.join(sim, "simulate_config.yaml")) as fh:
        tau = float((yaml.safe_load(fh).get("binding_data") or {})
                    .get("genotype_offset_sd", 0.0))
    add("binding_offset_scale", lambda r: "global",
        _read(os.path.join(run, "tfs_params_offset_scale.csv")), tau)
    return rows


def theta_rows(run, sim, name):
    pred = pd.read_csv(os.path.join(run, "tfs_pred_theta.csv"))
    truth = pd.read_csv(os.path.join(sim, "tfs_sim_parameters.csv"))
    truth = truth.assign(genotype=truth["genotype"].astype(str)).set_index("genotype")
    pred["genotype"] = pred["genotype"].astype(str)
    pred = pred[pred["genotype"].isin(truth.index)].copy()
    t = truth.loc[pred["genotype"]]
    pred["truth"] = hill(t[["theta_low", "theta_high", "log_hill_K", "hill_n"]]
                         .reset_index(drop=True),
                         pred["titrant_conc"].to_numpy()).to_numpy()
    lib = pd.read_csv(os.path.join(run, "tfs_configure_library.csv"))
    spiked = set(lib.loc[lib["in_spiked_origin"].astype(bool), "genotype"].astype(str))
    bfile = os.path.join(run, "binding.csv")
    bound = (set(pd.read_csv(bfile)["genotype"].astype(str)) - spiked
             if os.path.exists(bfile) else set())
    pred["group"] = np.where(pred["genotype"] == "wt", "wt",
                             np.where(pred["genotype"].isin(spiked), "spike",
                                      np.where(pred["genotype"].isin(bound),
                                               "single, binding",
                                               np.where(pred["genotype"].map(ts.is_single),
                                                        "single", "other"))))
    pred["err"] = pred[Q_MID] - pred["truth"]
    pred["covered"] = (pred[Q_LO] <= pred["truth"]) & (pred["truth"] <= pred[Q_HI])
    out = (pred.groupby("group")
           .agg(n_genotypes=("genotype", "nunique"),
                rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                bias=("err", "mean"),
                coverage95=("covered", "mean"))
           .reset_index())
    out.insert(0, "run", name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="RUN_DIR:SIM_DIR")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    params, thetas = [], []
    for pair in args.pairs:
        run, sim = pair.split(":")
        name = os.path.basename(os.path.normpath(run))
        params.extend(param_rows(run, sim, name))
        thetas.append(theta_rows(run, sim, name))
    p = pd.DataFrame(params)
    th = pd.concat(thetas, ignore_index=True)
    p.to_csv(os.path.join(args.out, "params.csv"), index=False)
    th.to_csv(os.path.join(args.out, "theta.csv"), index=False)
    pd.set_option("display.width", 200)
    print(p.to_string(index=False, float_format="%+.4f"))
    print()
    print(th.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
