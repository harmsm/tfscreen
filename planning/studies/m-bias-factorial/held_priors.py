"""
Growth priors that hold k and m at a subset fit's MAP (m-bias factorial).

    python held_priors.py no_doubles --out growth_priors_held.csv
    python held_priors.py no_doubles --hold kanR --loose growth_priors_loose.csv \
        --out growth_priors_held.csv

Reads the arm's tfs_params_growth_k.csv and tfs_params_growth_m.csv (one
row per condition_rep with --growth_shares_replicates) and writes a
--growth_priors table with those values as k_loc and m_loc.

Without --hold every condition is held: configure with --set_priors
m_pinned=1 k_pinned=1 to clamp them (the scales are then unused). With
--hold, only conditions whose name starts with one of the given prefixes
are held, clamped through the table's k_pinned/m_pinned columns (linear's
per-condition pins); every other condition takes its loc and scale from
--loose and is fitted. Do not add the --set_priors pins then: they act on
every condition. (Round 4 held kanR by a tight prior instead, SD 1e-4; it
leaked about 11% of the doubles' pull.)
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--out", default="growth_priors_held.csv")
    ap.add_argument("--hold", nargs="+", default=None,
                    help="condition_rep prefixes to hold (default: all)")
    ap.add_argument("--loose", default=None,
                    help="growth priors for the conditions not held")
    args = ap.parse_args()
    km = {}
    for p in ("k", "m"):
        df = pd.read_csv(os.path.join(args.arm_dir, f"tfs_params_growth_{p}.csv"))
        km[p] = df.groupby("condition_rep")["q0.5"].mean()
    out = pd.DataFrame({"k_loc": km["k"], "k_scale": 0.002,
                        "m_loc": km["m"], "m_scale": 0.002})
    out.index.name = "condition_rep"
    if args.hold is not None:
        if args.loose is None:
            ap.error("--hold needs --loose for the conditions not held")
        loose = pd.read_csv(args.loose).set_index("condition_rep")
        held = [c for c in out.index if any(c.startswith(h) for h in args.hold)]
        if not held:
            ap.error(f"no condition starts with {args.hold}: {list(out.index)}")
        free = [c for c in out.index if c not in held]
        missing = [c for c in free if c not in loose.index]
        if missing:
            ap.error(f"{args.loose} has no row for {missing}")
        out = pd.concat([out.loc[held], loose.loc[free, out.columns]])
        out["k_pinned"] = [int(c in held) for c in out.index]
        out["m_pinned"] = out["k_pinned"]
    out.to_csv(args.out)
    print(out.to_string())


if __name__ == "__main__":
    main()
