import pandas as pd
import numpy as np
import os
import yaml
from tfscreen.analysis.hierarchical.growth_model import GrowthModel
from tfscreen.util.cli.generalized_main import generalized_main

def summarize_posteriors(posterior_file,
                         config_file,
                         out_root="tfs"):
    """
    Summarize posterior samples from a hierarchical growth model run.

    This function loads the model configuration and posterior samples,
    extracts various parameter estimates and predictions, and saves them
    as CSV files.

    Parameters
    ----------
    posterior_file : str
        Path to the .npz file containing posterior samples.
    config_file : str
        Path to the YAML configuration file.
    out_root : str, optional
        Root filename for output CSV files (default "tfs").
    """

    # Load configuration
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    settings = config["settings"]
    growth_df = config["growth_df"]
    binding_df = config["binding_df"]

    # Initialize the model with the settings from the config
    gm = GrowthModel(growth_df,
                     binding_df,
                     condition_growth=settings["condition_growth"],
                     ln_cfu0=settings["ln_cfu0"],
                     dk_geno=settings["dk_geno"],
                     activity=settings["activity"],
                     theta=settings["theta"],
                     transformation=settings["transformation"],
                     theta_growth_noise=settings["theta_growth_noise"],
                     theta_binding_noise=settings["theta_binding_noise"],
                     spiked_genotypes=settings["spiked_genotypes"])

    # Load posteriors
    if not os.path.exists(posterior_file):
        raise FileNotFoundError(f"Posterior file not found: {posterior_file}")
    
    with np.load(posterior_file) as posteriors:

        # Extract and save parameters
        print(f"Extracting parameters to {out_root}_*.csv...", flush=True)
        params = gm.extract_parameters(posteriors)
        for p_name, p_df in params.items():
            p_df.to_csv(f"{out_root}_{p_name}.csv", index=False)

        # Extract and save growth predictions
        print(f"Extracting growth predictions to {out_root}_growth_pred.csv...", flush=True)
        growth_pred_df = gm.extract_growth_predictions(posteriors)
        growth_pred_df.to_csv(f"{out_root}_growth_pred.csv", index=False)

        # Extract and save theta curves (if applicable)
        if settings["theta"] == "hill":
            print(f"Extracting theta curves to {out_root}_theta_curves.csv...", flush=True)
            theta_curves_df = gm.extract_theta_curves(posteriors)
            theta_curves_df.to_csv(f"{out_root}_theta_curves.csv", index=False)

    print("Summarization complete.", flush=True)

def main():
    """CLI entry point for summarizing posteriors."""
    generalized_main(summarize_posteriors)

if __name__ == "__main__":
    main()
