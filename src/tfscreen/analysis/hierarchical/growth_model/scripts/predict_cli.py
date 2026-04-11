import pandas as pd
import numpy as np
import os
from tfscreen.analysis.hierarchical.growth_model.configuration_io import read_configuration
from tfscreen.analysis.hierarchical.growth_model.prediction import predict
from tfscreen.util.cli.generalized_main import generalized_main

def predict_cli(config_file,
                posterior_file,
                out_prefix="tfs",
                predict_sites=None,
                num_samples=None,
                num_marginal_samples=None,
                t_pre=None,
                t_sel=None,
                titrant_conc=None,
                genotypes=None):
    """
    Predict model sites using configuration and posterior samples.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    posterior_file : str
        Path to the .h5 file containing posterior samples.
    out_prefix : str, optional
        Prefix for the output CSV files. Each site will be saved to
        {out_prefix}_{site_name}.csv. Default "tfs".
    predict_sites : list, optional
        List of strings specifying the sites to predict. If None,
        defaults to ["growth_pred"].
    num_samples : int or None, optional
        Randomly select this many joint draws from the predicted samples and
        write them as ``sample_0`` … ``sample_{N-1}`` columns alongside the
        quantile columns in the output CSVs. Set to ``None`` to suppress
        sample columns and return only quantiles. Default 100.
    num_marginal_samples : int or None, optional
        Number of posterior samples to run through the model for computing
        quantile predictions. If None, uses all available samples.
    t_pre : list, optional
        List of timepoints for pre-growth.
    t_sel : list, optional
        List of timepoints for selection.
    titrant_conc : list, optional
        List of titrant concentrations.
    genotypes : list, optional
        List of genotypes to include.
    """
    
    if predict_sites is None:
        predict_sites = ["growth_pred"]

    # Load model class
    print(f"Loading configuration from {config_file}...", flush=True)
    gm, _ = read_configuration(config_file)
    
    # Run prediction
    print(f"Running prediction for {predict_sites} using {num_samples} samples...", flush=True)
    results = predict(model_class=gm,
                      param_posteriors=posterior_file,
                      predict_sites=predict_sites,
                      num_samples=num_samples,
                      num_marginal_samples=num_marginal_samples,
                      t_pre=t_pre,
                      t_sel=t_sel,
                      titrant_conc=titrant_conc,
                      genotypes=genotypes)
    
    # Save output(s)
    if isinstance(results, pd.DataFrame):
        site_name = predict_sites[0]
        output_file = f"{out_prefix}_{site_name}.csv"
        results.to_csv(output_file, index=False)
        print(f"Wrote predictions for '{site_name}' to {output_file}", flush=True)
    else:
        for site_name, df in results.items():
            output_file = f"{out_prefix}_{site_name}.csv"
            df.to_csv(output_file, index=False)
            print(f"Wrote predictions for '{site_name}' to {output_file}", flush=True)

def main():
    """CLI entry point for predicting model sites."""
    
    # Define manual overrides for generalized_main
    manual_arg_types = {
        "predict_sites": str,
        "num_samples": int,
        "num_marginal_samples": int,
        "t_pre": float,
        "t_sel": float,
        "titrant_conc": float,
        "genotypes": str
    }
    manual_arg_nargs = {
        "predict_sites": "+",
        "t_pre": "+",
        "t_sel": "+",
        "titrant_conc": "+",
        "genotypes": "+"
    }
    
    generalized_main(predict_cli,
                     manual_arg_types=manual_arg_types,
                     manual_arg_nargs=manual_arg_nargs)

if __name__ == "__main__":
    main()
