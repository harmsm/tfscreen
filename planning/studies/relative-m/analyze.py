"""
Relative stage 1 against the absolute stage 1, the simulations' truth and
the real data's within-tube m (README).

    python analyze.py runs ../two-stage-validation

RUNS_DIR holds rel_{run}/stage1/ (km.csv, tfs_configure_config.yaml);
VALIDATION_DIR holds run_{run}/stage1/km.csv. Truth on the X gauge comes
from the m-bias study's local copies of each simulation's parameters.
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "two-stage-validation"))
from validate import truth_km  # noqa: E402

SIM_DIRS = {"seed1": "../m-bias-factorial/m_subsets/no_doubles_mix",
            "seed2": "../m-bias-factorial/sim_realistic_seed2"}
WITHIN_TUBE = {"kanR+kan": -0.0049, "pheS+4CP": 0.0178}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("validation_dir")
    args = ap.parse_args()
    for run in ("real", "seed1", "seed2"):
        rel = os.path.join(args.runs_dir, f"rel_{run}", "stage1")
        if not os.path.exists(os.path.join(rel, "km.csv")):
            print(f"{rel}: not there yet")
            continue
        t = pd.read_csv(os.path.join(rel, "km.csv"), index_col="condition_rep")
        t = t.rename(columns={"k": "k_pinned", "m": "m_relative"})
        a = pd.read_csv(os.path.join(args.validation_dir, f"run_{run}", "stage1",
                                     "km.csv"), index_col="condition_rep")
        t["m_absolute"] = a["m"]
        if run in SIM_DIRS:
            tr = truth_km(os.path.join(HERE, SIM_DIRS[run]),
                          os.path.join(rel, "tfs_configure_config.yaml"))
            t["m_truth"] = tr["m"]
            t["relative/truth"] = t["m_relative"] / t["m_truth"]
            t["absolute/truth"] = t["m_absolute"] / t["m_truth"]
        else:
            t["m_within_tube"] = pd.Series(WITHIN_TUBE)
        sel = [c for c in t.index if "+" in c]
        if len(sel) == 2:
            k, p = sorted(sel)
            print(f"\n== {run}: pheS/kanR m ratio, relative "
                  f"{t.loc[p, 'm_relative'] / t.loc[k, 'm_relative']:+.2f}, absolute "
                  f"{t.loc[p, 'm_absolute'] / t.loc[k, 'm_absolute']:+.2f}")
        print(t.to_string(float_format="%+.5f"))


if __name__ == "__main__":
    main()
