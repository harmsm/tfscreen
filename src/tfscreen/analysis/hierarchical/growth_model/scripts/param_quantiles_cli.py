import numpy as np
import h5py
import os
from tfscreen.analysis.hierarchical.growth_model.configuration_io import read_configuration
from tfscreen.analysis.hierarchical.growth_model.extraction import extract_parameters
from tfscreen.util.cli.generalized_main import generalized_main


def param_quantiles(config_file,
                    posterior_file,
                    out_prefix="tfs_param"):
    """
    Extract posterior parameter quantiles and write to CSV files.

    Reads the posterior samples produced by tfs-sample-posterior and writes
    one CSV per parameter group to {out_prefix}_{param_name}.csv.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    posterior_file : str
        Path to the _posterior.h5 file containing posterior samples.
    out_prefix : str, optional
        Prefix for output files. Default 'tfs_param'.
    """
    gm, _ = read_configuration(config_file)

    if not os.path.exists(posterior_file):
        if os.path.exists(f"{posterior_file}_posterior.h5"):
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
        is_params_file = any(k.endswith("_auto_loc") for k in posteriors.keys())
        if is_params_file:
            print("WARNING: The provided file appears to contain guide parameters "
                  "rather than posterior samples. This usually happens if you provide "
                  "the '_params.npz' file instead of the '_posterior.h5' file. "
                  "Extraction of natural parameters may fail.", flush=True)

        print(f"Extracting parameters to {out_prefix}_*.csv...", flush=True)
        params = extract_parameters(gm, posteriors)
        for p_name, p_df in params.items():
            p_df.to_csv(f"{out_prefix}_{p_name}.csv", index=False)

    print("Done.", flush=True)


def main():
    generalized_main(param_quantiles)


if __name__ == "__main__":
    main()
