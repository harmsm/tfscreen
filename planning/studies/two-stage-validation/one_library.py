"""
Stage 2 on one library's tubes alone, k and m pinned at stage 1: the
consistency step with another library. The run's consistency/ is kanR
alone; this gives pheS alone for the X-vs-binding comparison (does each
marker's X follow binding theta the same way?).

    python one_library.py two_stage_real.yaml run_real pheS

Writes run_dir/consistency_{library}/ (tfs_pred_theta.csv, and
consistency.csv against the full stage 2).
"""

import argparse
import os

from tfscreen.tfmodel.scripts import fit_two_stage_cli as cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_file")
    ap.add_argument("run_dir")
    ap.add_argument("library")
    args = ap.parse_args()
    cfg, settings = cli.read_two_stage_config(args.config_file)
    settings["consistency_library"] = args.library
    cli.run_consistency(os.path.abspath(args.run_dir), cfg, settings,
                        name=f"consistency_{args.library}")
    cli._say("done")


if __name__ == "__main__":
    main()
