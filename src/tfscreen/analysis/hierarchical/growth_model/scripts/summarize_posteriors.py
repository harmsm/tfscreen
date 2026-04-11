import pandas as pd
import numpy as np
import h5py
import os
import yaml
from tfscreen.analysis.hierarchical.growth_model import (
    GrowthModel,
    extract_parameters,
    extract_growth_predictions,
    extract_theta_curves
)
from tfscreen.analysis.hierarchical.growth_model.configuration_io import read_configuration
from tfscreen.util.cli.generalized_main import generalized_main

def summarize_posteriors(config_file,
                        posterior_file,
                        out_root="tfs",
                        q_to_get=None,
                        num_samples=100):
    """
    Summarize posterior samples and write to CSV files.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    posterior_file : str
        Path to the .npz or .h5 file containing posterior samples.
    out_root : str, optional
        Root filename for output files.
    q_to_get : dict, optional
        Dictionary mapping column names to quantiles.
    num_samples : int, optional
        If provided, randomly select this many joint posterior samples and
        write them as ``sample_0``, ``sample_1``, ... columns alongside the
        quantile columns in the growth predictions and theta curves CSVs.
        Useful for visualizing individual trajectories without marginalization.
        Sampling is with replacement when ``num_samples`` exceeds the total
        number of posterior samples. Default: 100

    Returns
    -------
    None
    """

    # Load configuration
    gm, _ = read_configuration(config_file)

    # Load posteriors
    if not os.path.exists(posterior_file):
        # Check if it's a root name and we can find .h5 or .npz
        if os.path.exists(f"{posterior_file}_posterior.h5"):
            posterior_file = f"{posterior_file}_posterior.h5"
        elif os.path.exists(f"{posterior_file}_posterior.npz"):
            posterior_file = f"{posterior_file}_posterior.npz"
        else:
            raise FileNotFoundError(f"Posterior file not found: {posterior_file}")
    
    print(f"Loading posteriors from {posterior_file}...", flush=True)
    if posterior_file.endswith(".h5") or posterior_file.endswith(".hdf5"):
        context_manager = h5py.File(posterior_file, 'r')
    else:
        context_manager = np.load(posterior_file)

    with context_manager as posteriors:

        # Check for common params-only keys that suggest this is not the sample file
        is_params_file = False
        for k in posteriors.keys():
            if k.endswith("_auto_loc"):
                is_params_file = True
                break
        
        if is_params_file:
            print("WARNING: The provided file appears to contain guide parameters "
                  "rather than posterior samples. This usually happens if you provide "
                  "the '_params.npz' file instead of the '_posterior.h5' file. "
                  "Extraction of natural parameters may fail.", flush=True)

        # Extract and save parameters
        print(f"Extracting parameters to {out_root}_*.csv...", flush=True)
        params = extract_parameters(gm, posteriors)
        for p_name, p_df in params.items():
            p_df.to_csv(f"{out_root}_{p_name}.csv", index=False)

        # Extract and save growth predictions
        print(f"Extracting growth predictions to {out_root}_growth_pred.csv...", flush=True)
        growth_pred_df = extract_growth_predictions(gm, posteriors, num_samples=num_samples)
        growth_pred_df.to_csv(f"{out_root}_growth_pred.csv", index=False)

        # Extract and save theta curves (if applicable)
        if gm.settings["theta"] == "hill":
            print(f"Extracting theta curves to {out_root}_theta_curves.csv...", flush=True)
            theta_curves_df = extract_theta_curves(gm, posteriors, num_samples=num_samples)
            theta_curves_df.to_csv(f"{out_root}_theta_curves.csv", index=False)

    print("Summarization complete.", flush=True)

def main():
    """CLI entry point for summarizing posteriors."""
    generalized_main(summarize_posteriors, manual_arg_types={"num_samples": int})

if __name__ == "__main__":
    main()
