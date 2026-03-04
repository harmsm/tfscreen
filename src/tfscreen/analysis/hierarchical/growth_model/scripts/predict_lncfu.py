import pandas as pd
import numpy as np
import os
from tfscreen.analysis.hierarchical.growth_model.configuration_io import read_configuration
from tfscreen.analysis.hierarchical.growth_model.prediction import predict_lncfu
from tfscreen.util.cli.generalized_main import generalized_main

def predict_lncfu_cli(config_file,
                      posterior_file,
                      output_file="tfs_prediction.csv",
                      num_samples=100,
                      t_pre=None,
                      t_sel=None,
                      titrant_conc=None,
                      genotypes=None,
                      condition_pre=None,
                      condition_sel=None,
                      titrant_name=None,
                      replicate=None):
    """
    Predict ln_cfu using model configuration and posterior samples.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    posterior_file : str
        Path to the .npz or .h5 file containing posterior samples.
    output_file : str, optional
        Path to the CSV file where predictions will be saved.
    num_samples : int, optional
        Number of posterior samples to use for prediction.
    t_pre : float, optional
        A single timepoint for pre-growth.
    t_sel : list, optional
        List of timepoints for selection.
    titrant_conc : list, optional
        List of titrant concentrations.
    genotypes : list, optional
        List of genotypes to include.
    condition_pre : list, optional
        List of pre-growth conditions to include.
    condition_sel : list, optional
        List of selection conditions to include.
    titrant_name : list, optional
        List of titrant names to include.
    replicate : list, optional
        List of replicates to include.
    """
    
    # Load model class
    print(f"Loading configuration from {config_file}...", flush=True)
    gm, _ = read_configuration(config_file)
    
    # Run prediction
    print(f"Running prediction using {num_samples} samples...", flush=True)
    df = predict_lncfu(model_class=gm,
                       param_posteriors=posterior_file,
                       num_samples=num_samples,
                       t_pre=t_pre,
                       t_sel=t_sel,
                       titrant_conc=titrant_conc,
                       genotypes=genotypes,
                       condition_pre=condition_pre,
                       condition_sel=condition_sel,
                       titrant_name=titrant_name,
                       replicate=replicate)
    
    # Save output
    df.to_csv(output_file, index=False)
    print(f"Wrote predictions to {output_file}", flush=True)

def main():
    """CLI entry point for predicting cell counts."""
    
    # Define manual overrides for generalized_main
    manual_arg_types = {
        "t_sel": float,
        "titrant_conc": float,
        "genotypes": str,
        "condition_pre": str,
        "condition_sel": str,
        "titrant_name": str,
        "replicate": str
    }
    manual_arg_nargs = {
        "t_sel": "+",
        "titrant_conc": "+",
        "genotypes": "+",
        "condition_pre": "+",
        "condition_sel": "+",
        "titrant_name": "+",
        "replicate": "+"
    }
    
    generalized_main(predict_lncfu_cli,
                     manual_arg_types=manual_arg_types,
                     manual_arg_nargs=manual_arg_nargs)

if __name__ == "__main__":
    main()
