import pandas as pd
import os
from tfscreen.analysis.hierarchical.growth_model.configuration_io import read_configuration
from tfscreen.analysis.hierarchical.growth_model.extraction import extract_theta_unmeasured
from tfscreen.util.cli.generalized_main import generalized_main


def predict_unmeasured_cli(config_file,
                           posterior_file,
                           titrant_name,
                           titrant_conc,
                           genotypes=None,
                           out_prefix="tfs_theta_pred"):
    """
    Predict theta (operator occupancy) for unmeasured genotypes.

    Assembles per-genotype Hill or thermodynamic parameters from additive
    per-mutation effects stored in the posterior, then evaluates theta over the
    supplied concentration grid.  Genotypes containing any mutation not seen
    during training are returned as NaN.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file used when fitting the model.
    posterior_file : str
        Path to the .h5 or .npz file containing posterior samples.
    titrant_name : list of str
        Titrant name(s) for the prediction grid.  Supply one name to apply it
        to every concentration in titrant_conc, or supply a list the same
        length as titrant_conc to specify each row individually.
    titrant_conc : list of float
        Effector concentrations for the prediction grid (same units as the
        training data).
    genotypes : list of str, optional
        Genotype strings to predict (slash-separated mutations, e.g.
        "M42I/K84L", or "wt").  If None, predicts all genotypes present in
        the training data.
    out_prefix : str, optional
        Prefix for the output CSV file. Output is written to
        {out_prefix}_theta_unmeasured.csv. Default "tfs_theta_pred".
    """

    # Build titrant DataFrame
    if len(titrant_name) == 1:
        titrant_name = titrant_name * len(titrant_conc)
    elif len(titrant_name) != len(titrant_conc):
        raise ValueError(
            f"titrant_name has {len(titrant_name)} elements but titrant_conc "
            f"has {len(titrant_conc)}.  Supply either one titrant_name (applied "
            f"to all concentrations) or one per concentration."
        )
    manual_titrant_df = pd.DataFrame({
        "titrant_name": titrant_name,
        "titrant_conc": titrant_conc,
    })

    # Load model
    print(f"Loading configuration from {config_file}...", flush=True)
    gm, _ = read_configuration(config_file)

    # Default to all training genotypes when none specified
    if genotypes is None:
        titrant_dim   = gm.growth_tm.tensor_dim_names.index("titrant_name")
        geno_dim      = gm.growth_tm.tensor_dim_names.index("genotype")
        genotypes     = list(gm.growth_tm.tensor_dim_labels[geno_dim])
        print(f"No genotypes specified; predicting all {len(genotypes)} "
              f"training genotypes.", flush=True)

    print(f"Predicting theta for {len(genotypes)} genotypes over "
          f"{len(manual_titrant_df)} concentration points...", flush=True)

    result_df = extract_theta_unmeasured(
        model=gm,
        posteriors=posterior_file,
        target_genotypes=genotypes,
        manual_titrant_df=manual_titrant_df,
    )

    output_file = f"{out_prefix}_theta_unmeasured.csv"
    result_df.to_csv(output_file, index=False)
    print(f"Wrote theta predictions to {output_file}", flush=True)


def main():
    """CLI entry point for predicting theta for unmeasured genotypes."""
    generalized_main(
        predict_unmeasured_cli,
        manual_arg_types={
            "titrant_name": str,
            "titrant_conc": float,
            "genotypes":    str,
        },
        manual_arg_nargs={
            "titrant_name": "+",
            "titrant_conc": "+",
            "genotypes":    "+",
        },
    )


if __name__ == "__main__":
    main()
