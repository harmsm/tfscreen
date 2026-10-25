"""
Trace a fit's guide parameters across its epoch checkpoints, to see whether
they had stopped moving when SVI declared convergence.

For each run, one row per checkpoint (checkpoints/*_checkpoint.pkl every 1000
epochs, then the final tfs_fit_model_checkpoint.pkl): the per-condition
growth_k guide loc, dk_geno's hyper-shift loc and median per-genotype offset
guide scale, the guide medians of nu, sigma_k and lambda, and the smoothed
loss. A truth row comes from tfs_sim_growth_parameters.csv. The stop line
gives the convergence bookkeeping stored in the final checkpoint and the
10-epoch loss change below which a tolerance of 1e-4 (run.srun) stops.

Run from this directory. GRID names a pulled grid directory (default: the
anchored grid inside the congression-calibration study); the arguments are
run-number prefixes (default: run_0001 run_0002 run_0005). The default runs
take about 20 s on a laptop CPU.
"""

import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import dill
import numpy as np
import pandas as pd

from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.run_inference import RunInference

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.environ.get("GRID", os.path.join(
    HERE, "..", "congression-calibration", "congression_calibration_anchor"))
RUNS = sys.argv[1:] or ["run_0001", "run_0002", "run_0005"]
TOLERANCE = 1e-4


def smoothed_loss(run_dir):
    losses = pd.read_csv(os.path.join(run_dir, "tfs_fit_model_losses.txt"),
                         header=None, names=["epoch", "loss", "change"])
    losses = losses[pd.to_numeric(losses.epoch, errors="coerce").notna()]
    losses = losses.astype(float).set_index("epoch").loss
    return losses.rolling(50, center=True, min_periods=1).median()


def condition_order(run_dir):
    priors = pd.read_csv(os.path.join(run_dir, "tfs_configure_priors.csv"))
    k = priors[priors.parameter == "growth.condition_growth.k_loc"]
    return list(k.sort_values("flat_index").condition_rep)


def guide_median(params, name):
    """Median of a LogNormal guide site with loc parameter ``name``."""
    if name not in params:
        return np.nan
    return float(np.exp(np.ravel(np.asarray(params[name]))[0]))


def trace(run_dir):
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        orch, init_params = read_configuration("tfs_configure_config.yaml")
        ri = RunInference(orch, 0)
        files = sorted(glob.glob("checkpoints/*_checkpoint.pkl"))
        files.append("tfs_fit_model_checkpoint.pkl")
        loss = smoothed_loss(".")
        conditions = condition_order(".")

        rows = []
        for f in files:
            svi, state = ri.restore_svi_from_checkpoint(
                f, init_params=init_params)
            p = svi.get_params(state)
            with open(f, "rb") as handle:
                stored = dill.load(handle)
            epoch = stored["current_step"] // ri._iterations_per_epoch
            row = {"epoch": epoch}
            k = np.ravel(np.asarray(p["condition_growth_k_locs"]))
            row.update({f"k[{c}]": v for c, v in zip(conditions, k)})
            row["dk_shift"] = float(np.ravel(p["dk_geno_hyper_shift_loc"])[0])
            row["dk_scale_med"] = float(np.median(p["dk_geno_offset_scales"]))
            row["nu"] = guide_median(p, "growth_nu_loc")
            row["sigma_k"] = guide_median(p, "growth_noise_sigma_k_loc")
            row["lam"] = guide_median(p, "transformation_lam_loc")
            row["loss"] = float(loss.iloc[np.argmin(np.abs(loss.index - epoch))])
            rows.append(row)

        truth = pd.read_csv("tfs_sim_growth_parameters.csv")
        truth = truth.set_index("condition_rep").growth_k
        truth_row = {"epoch": "truth"}
        truth_row.update({f"k[{c}]": truth[c] for c in conditions})
        rows.append(truth_row)

        denom = stored["loss_start"] - stored["loss_best"]
        stop = (f"stop: loss_start {stored['loss_start']:.3g}, loss_best "
                f"{stored['loss_best']:.3g}; tolerance {TOLERANCE:g} stops once "
                f"the 10-epoch change in mean loss is below "
                f"{TOLERANCE * denom:,.0f}")
    finally:
        os.chdir(cwd)
    return pd.DataFrame(rows), stop


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for run in RUNS:
        run_dir = glob.glob(os.path.join(GRID, run + "_*"))[0]
        df, stop = trace(run_dir)
        print(f"\n=== {os.path.basename(run_dir)}")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(stop)
