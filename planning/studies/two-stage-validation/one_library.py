"""
Stage 2 on one library's tubes alone, k and m pinned at stage 1: the
consistency step with another library. The run's consistency/ is kanR
alone; this gives pheS alone for the X-vs-binding comparison (does each
marker's X follow binding theta the same way?).

    python one_library.py two_stage_real.yaml run_real pheS
    python one_library.py two_stage_real.yaml run_real pheS --wt_hill 0.0238 0.753

Writes run_dir/consistency_{library}/ (tfs_pred_theta.csv, and
consistency.csv against the full stage 2). --wt_hill K N holds wt's Hill K
(titrant units) and n there (theta_wt_hill_K_fixed / _n_fixed) and writes
consistency_{library}_wtheld/ instead: the pheS-only fit collapsed wt's
curve below 1 uM IPTG (planning/studies/x-vs-binding/).
"""

import argparse
import os

from tfscreen.tfmodel.scripts import fit_two_stage_cli as cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_file")
    ap.add_argument("run_dir")
    ap.add_argument("library")
    ap.add_argument("--wt_hill", nargs=2, type=float, metavar=("K", "N"),
                    default=None, help="hold wt's Hill K and n")
    args = ap.parse_args()
    cfg, settings = cli.read_two_stage_config(args.config_file)
    settings["consistency_library"] = args.library
    name = f"consistency_{args.library}"
    if args.wt_hill is not None:
        k, n = args.wt_hill
        cfg["set_priors"] = list(cfg.get("set_priors") or []) + [
            f"theta_wt_hill_K_fixed={k}", f"theta_wt_hill_n_fixed={n}"]
        name += "_wtheld"
    cli.run_consistency(os.path.abspath(args.run_dir), cfg, settings, name=name)
    cli._say("done")


if __name__ == "__main__":
    main()
