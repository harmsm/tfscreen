import os
import dill
import numpy as np
import h5py
from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.run_inference import RunInference
from tfscreen.tfmodel.analysis.extraction import extract_parameters
from tfscreen.util.cli import generalized_main


def extract_params(config_file, param_file, out_prefix="tfs_params"):
    """
    Extract parameters from a checkpoint or posterior file and write to CSV.

    If ``param_file`` ends with ``.pkl`` it is treated as a MAP (AutoDelta)
    checkpoint: parameters are converted to constrained space and written with
    a single ``point_est`` column.

    Otherwise it is treated as a posterior samples ``.h5`` or ``.npz`` file
    produced by tfs-sample-posterior, and the default quantile set (min,
    lower_95, lower_std, lower_quartile, median, upper_quartile, upper_std,
    upper_95, max) is computed.

    Writes one CSV per parameter group named ``{out_prefix}_{param_name}.csv``
    (e.g. tfs_params_activity.csv, tfs_params_theta.csv).

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file used when fitting the model.
    param_file : str
        Path to a MAP checkpoint ``.pkl`` file, or a posterior ``.h5``/``.npz``
        file produced by tfs-sample-posterior.
    out_prefix : str, optional
        Prefix for output CSV files. Default ``'tfs_params'``.
    """
    gm, _ = read_configuration(config_file)

    if param_file.endswith(".pkl"):
        _extract_from_checkpoint(gm, param_file, out_prefix)
    else:
        _extract_from_posteriors(gm, param_file, out_prefix)


def _extract_from_checkpoint(gm, ckpt_path, out_prefix):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: '{ckpt_path}'")

    print(f"Loading {ckpt_path}...", flush=True)
    with open(ckpt_path, "rb") as f:
        chk_data = dill.load(f)

    if "mcmc_samples" in chk_data:
        raise ValueError(
            f"'{ckpt_path}' is a NUTS checkpoint. "
            "tfs-extract-params only supports MAP (AutoDelta) checkpoints."
        )

    ri = RunInference(gm, seed=0)
    temp_svi = ri.setup_svi(guide_type="delta")
    map_params = temp_svi.optim.get_params(chk_data["svi_state"].optim_state)

    if not any(k.endswith("_auto_loc") for k in map_params):
        raise ValueError(
            f"'{ckpt_path}' is an SVI (variational) checkpoint. "
            "tfs-extract-params only supports MAP (AutoDelta) checkpoints."
        )

    constrained = ri._map_params_to_constrained(map_params)
    posteriors = {k: np.expand_dims(np.asarray(v), 0) for k, v in constrained.items()}

    print(f"Writing parameter CSVs to {out_prefix}_*.csv...", flush=True)
    param_dfs = extract_parameters(gm, posteriors, q_to_get={"point_est": 0.5})
    for p_name, df in param_dfs.items():
        out_file = f"{out_prefix}_{p_name}.csv"
        df.to_csv(out_file, index=False)
        print(f"  Wrote {out_file}", flush=True)
    print("Done.", flush=True)


def _extract_from_posteriors(gm, posterior_file, out_prefix):
    if not os.path.exists(posterior_file):
        if os.path.exists(f"{posterior_file}.h5"):
            posterior_file = f"{posterior_file}.h5"
        elif os.path.exists(f"{posterior_file}_posterior.h5"):
            posterior_file = f"{posterior_file}_posterior.h5"
        elif os.path.exists(f"{posterior_file}_posterior.npz"):
            posterior_file = f"{posterior_file}_posterior.npz"
        else:
            raise FileNotFoundError(f"Posterior file not found: {posterior_file}")

    print(f"Loading posteriors from {posterior_file}...", flush=True)
    if posterior_file.endswith(".h5") or posterior_file.endswith(".hdf5"):
        context_manager = h5py.File(posterior_file, "r")
    else:
        context_manager = np.load(posterior_file)

    with context_manager as posteriors:
        if any(k.endswith("_auto_loc") for k in posteriors.keys()):
            print("WARNING: The provided file appears to contain guide parameters "
                  "rather than posterior samples. This usually happens if you provide "
                  "the '_params.npz' file instead of the posterior .h5 file. "
                  "Extraction of natural parameters may fail.", flush=True)

        print(f"Extracting parameters to {out_prefix}_*.csv...", flush=True)
        params = extract_parameters(gm, posteriors)
        for p_name, p_df in params.items():
            p_df.to_csv(f"{out_prefix}_{p_name}.csv", index=False)

    print("Done.", flush=True)


def main():
    generalized_main(extract_params)


if __name__ == "__main__":
    main()
