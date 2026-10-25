"""
Stage 1 of the two-stage fit with only relative information
(planning/relative-m.md, step 1): the per-tube level offsets nearly free
(sigma_fixed 10, so they absorb each tube's total), k pinned per condition
at a reference stage 1's values (it drops out), m free under a loose
prior (SD 0.05).

    python stage1_relative.py ../two-stage-validation/two_stage_real.yaml \
        ../two-stage-validation/run_real/stage1/km.csv rel_real

Writes OUT_DIR/stage1/ (km.csv, the MAP and its checkpoint) and
OUT_DIR/growth_priors_relative.csv.
"""

import argparse
import os

import pandas as pd

from tfscreen.tfmodel.scripts import fit_two_stage_cli as cli

SIGMA_FIXED = 10.0
M_SCALE = 0.05


def relative_priors(loose_csv, km_csv):
    """The loose stage-1 table with each k pinned at km.csv's and m free."""
    t = pd.read_csv(loose_csv).set_index("condition_rep")
    km = pd.read_csv(km_csv, index_col="condition_rep")
    missing = set(t.index) - set(km.index)
    if missing:
        raise ValueError(f"{km_csv} lacks conditions {sorted(missing)}")
    t["k_loc"] = km.loc[t.index, "k"]
    t["m_scale"] = M_SCALE
    t["k_pinned"] = 1
    t["m_pinned"] = 0
    return t.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_file")
    ap.add_argument("km_csv", help="a stage 1 km.csv; its k values are pinned")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    cfg, settings = cli.read_two_stage_config(args.config_file)
    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    pri = os.path.join(out, "growth_priors_relative.csv")
    if not os.path.exists(pri):
        relative_priors(cfg["growth_priors"], args.km_csv).to_csv(pri, index=False)
    cfg["growth_priors"] = pri
    cfg["set_priors"] = [p for p in (cfg.get("set_priors") or [])
                         if not p.split("=")[0].strip().endswith("sigma_fixed")]
    cfg["set_priors"].append(f"sigma_fixed={SIGMA_FIXED}")
    cli.run_stage1(out, cfg, settings)
    cli._say("done")


if __name__ == "__main__":
    main()
