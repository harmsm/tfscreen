"""
Stage 1 of the two-stage fit with a titrant-dependent k in the selection
conditions (linear growth's k_titrant_loglinear_scale), to see whether
stage 1's m moves to the within-tube m (README).

    python stage1_titrant.py ../two-stage-validation/two_stage_real.yaml ktitrant_real_loglinear loglinear

MODE loglinear: one slope in log IPTG per selection condition, prior SD
0.005. Run 2 also had a free form (one k shift per IPTG, prior SD 0.02);
it was not identified and was removed from the code afterwards. Writes
OUT_DIR/stage1/ (km.csv and the MAP's tfs_fit_model_params.npz, which
holds the slope as condition_growth_k_titrant_slope_auto_loc).
"""

import argparse
import os

from tfscreen.tfmodel.scripts import fit_two_stage_cli as cli

PRIORS = {"loglinear": "k_titrant_loglinear_scale=0.005"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_file")
    ap.add_argument("out_dir")
    ap.add_argument("mode", choices=sorted(PRIORS))
    args = ap.parse_args()
    cfg, settings = cli.read_two_stage_config(args.config_file)
    cfg["set_priors"] = list(cfg.get("set_priors") or []) + [PRIORS[args.mode]]
    cli.run_stage1(os.path.abspath(args.out_dir), cfg, settings)
    cli._say("done")


if __name__ == "__main__":
    main()
