"""
Growth priors that hold k and m at a subset fit's MAP (m-bias factorial).

    python held_priors.py no_doubles --out growth_priors_held.csv

Reads the arm's tfs_params_growth_k.csv and tfs_params_growth_m.csv (one
row per condition_rep with --growth_shares_replicates) and writes a
--growth_priors table with those values as k_loc and m_loc. Configure with
--set_priors m_pinned=1 k_pinned=1 to clamp them; the scales are then
unused.
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--out", default="growth_priors_held.csv")
    args = ap.parse_args()
    km = {}
    for p in ("k", "m"):
        df = pd.read_csv(os.path.join(args.arm_dir, f"tfs_params_growth_{p}.csv"))
        km[p] = df.groupby("condition_rep")["q0.5"].mean()
    out = pd.DataFrame({"k_loc": km["k"], "k_scale": 0.002,
                        "m_loc": km["m"], "m_scale": 0.002})
    out.index.name = "condition_rep"
    out.to_csv(args.out)
    print(out.to_string())


if __name__ == "__main__":
    main()
