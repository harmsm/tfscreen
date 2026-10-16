"""
The k-shift control: refit stage 2 exactly as the k-shift step does
(start at stage 2's MAP, same settings) but with k left at stage 1's values.
If X drifts as far as in kshift/plus and kshift/minus, the refit procedure
moves X, not k. Also extracts the parameters (tube offsets included) of all
three refits, which the k-shift step does not.

    python control_refit.py two_stage_real.yaml run_real

Writes run_dir/kshift/zero/ (fit, tfs_pred_theta.csv at the MAP,
tfs_params_*) and tfs_params_* in kshift/plus and kshift/minus.
"""

import argparse
import os

import pandas as pd

from tfscreen.tfmodel import two_stage as ts
from tfscreen.tfmodel.scripts import fit_two_stage_cli as cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_file")
    ap.add_argument("run_dir")
    args = ap.parse_args()
    cfg, settings = cli.read_two_stage_config(args.config_file)
    run = os.path.abspath(args.run_dir)
    init = os.path.join(run, "stage2", cli.PARAMS)
    km = pd.read_csv(os.path.join(run, "stage1", "km.csv"),
                     index_col="condition_rep")

    with cli._in_dir(os.path.join(run, "kshift", "zero")):
        cli._say("control refit: k at stage 1")
        ts.pinned_growth_priors(km).to_csv("growth_priors_pinned.csv", index=False)
        cli._configure(cfg, cfg["growth_df"], "growth_priors_pinned.csv",
                       drop_wt_rates=True)
        cli._fit(settings, init_from=init)
        if not os.path.exists(cli.PRED):
            cli.predict_theta(cli.CONFIG, cli.CHECKPOINT, out_prefix="tfs_pred_theta")

    for name in ("zero", "plus", "minus"):
        with cli._in_dir(os.path.join(run, "kshift", name)):
            if not os.path.exists("tfs_params_sample_offset_offset.csv"):
                cli._say(f"extract {name}")
                cli.extract_params(cli.CONFIG, cli.CHECKPOINT, out_prefix="tfs_params")
    cli._say("done")


if __name__ == "__main__":
    main()
