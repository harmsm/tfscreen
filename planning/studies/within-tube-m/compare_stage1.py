"""
Stage 1's k and m with and without a titrant-dependent k, and the fitted
shift (README, run 2).

    python compare_stage1.py runs ../two-stage-validation

RUNS_DIR holds ktitrant_{run}_{mode}/stage1/ (km.csv, tfs_configure_config.yaml,
tfs_fit_model_params.npz); VALIDATION_DIR holds run_{run}/stage1/km.csv,
the same stage 1 without the term.
"""

import argparse
import os

import numpy as np
import pandas as pd

WITHIN_TUBE = {"kanR+kan": -0.0049, "pheS+4CP": 0.0178}


def shift_table(stage1_dir):
    """
    The fitted shift from the stage's MAP: a (selection condition x nonzero
    IPTG) table for the free form, a slope per selection condition for the
    log-linear one. Selection conditions in condition_rep order (the '+'
    rows of km.csv), IPTG ascending.
    """
    z = np.load(os.path.join(stage1_dir, "tfs_fit_model_params.npz"))
    km = pd.read_csv(os.path.join(stage1_dir, "km.csv"), index_col="condition_rep")
    sel_conds = [c for c in km.index if "+" in c]
    if "condition_growth_k_titrant_auto_loc" in z.files:
        v = np.asarray(z["condition_growth_k_titrant_auto_loc"])
        return pd.DataFrame(v, index=sel_conds), "free"
    if "condition_growth_k_titrant_slope_auto_loc" in z.files:
        v = np.asarray(z["condition_growth_k_titrant_slope_auto_loc"])
        return pd.Series(v, index=sel_conds), "loglinear"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("validation_dir")
    args = ap.parse_args()
    concs = np.array([0.0, 1e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 1.0])
    L = np.log(np.maximum(concs, 1e-5) / 1e-5)
    for run in ("real", "seed1"):
        base = pd.read_csv(os.path.join(args.validation_dir, f"run_{run}", "stage1",
                                        "km.csv"), index_col="condition_rep")
        table = base.add_prefix("none_")
        for mode in ("free", "loglinear"):
            d = os.path.join(args.runs_dir, f"ktitrant_{run}_{mode}", "stage1")
            if not os.path.exists(os.path.join(d, "km.csv")):
                print(f"{d}: not there yet")
                continue
            table = table.join(pd.read_csv(os.path.join(d, "km.csv"),
                                           index_col="condition_rep").add_prefix(f"{mode}_"))
            sh, kind = shift_table(d)
            if kind == "free":
                print(f"\n{run} {mode}: k shift per IPTG (1/min), 0 at no IPTG")
                sh.columns = concs[:sh.shape[1]]
                print(sh.to_string(float_format="%+.4f"))
            elif kind == "loglinear":
                print(f"\n{run} {mode}: slope per log IPTG; shift at 1 mM")
                print(pd.DataFrame({"slope": sh, "shift_1mM": sh * L[-1]}).to_string(
                    float_format="%+.5f"))
        if run == "real":
            table["within_tube_m"] = pd.Series(WITHIN_TUBE)
        print(f"\n== {run}: stage 1 k and m")
        print(table.to_string(float_format="%+.5f"))


if __name__ == "__main__":
    main()
