from tfscreen.analysis.hierarchical.growth_model.configuration_io import read_configuration
from tfscreen.analysis.hierarchical.growth_model.prediction import predict
from tfscreen.util.cli import generalized_main, read_lines


def predict_growth(config_file,
                   posterior_file,
                   out_prefix="tfs_growth_pred",
                   genotypes_file=None,
                   titrant_names_file=None,
                   titrant_concs_file=None,
                   num_samples=100,
                   num_marginal_samples=None):
    """
    Predict growth signal (ln_cfu) from a fitted hierarchical model.

    By default predicts at all (genotype, titrant_name, titrant_conc,
    replicate, t_pre, t_sel) combinations present in the training data.
    Optional files can restrict genotypes and concentrations; titrant_names_file
    filters the output rows to the named titrants.

    A boolean column 'in_training_data' is added to the output: 1 if the
    (genotype, titrant_name, titrant_conc) triple was in the training data,
    0 otherwise.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    posterior_file : str
        Path to the _posterior.h5 file produced by tfs-sample-posterior.
    out_prefix : str, optional
        Prefix for the output CSV file. Written to {out_prefix}.csv.
        Default 'tfs_growth_pred'.
    genotypes_file : str or None, optional
        Plain-text file with one genotype per line (slash-separated mutations,
        e.g. 'M42I/K84L', or 'wt'). Must be a subset of training genotypes.
        If None, all training genotypes are used.
    titrant_names_file : str or None, optional
        Plain-text file with one titrant name per line. Used to filter output
        rows; does not change which concentrations are predicted.
        If None, all titrant names are included.
    titrant_concs_file : str or None, optional
        Plain-text file with one concentration per line. Predictions are made
        at these concentrations for the selected genotypes. If None, all
        training concentrations are used.
    num_samples : int or None, optional
        Number of joint posterior samples to include as sample_0 … sample_N-1
        columns alongside the quantile columns. Set to None for quantiles only.
        Default 100.
    num_marginal_samples : int or None, optional
        Number of posterior samples to run through the model when computing
        quantiles. If None, all available samples are used.
    """
    genotypes = read_lines(genotypes_file) if genotypes_file else None
    titrant_names = read_lines(titrant_names_file) if titrant_names_file else None
    titrant_concs = [float(x) for x in read_lines(titrant_concs_file)] if titrant_concs_file else None

    print(f"Loading configuration from {config_file}...", flush=True)
    gm, _ = read_configuration(config_file)

    # Build training-data membership set for in_training_data column.
    training_tuples = set(
        zip(gm.growth_df["genotype"],
            gm.growth_df["titrant_name"],
            gm.growth_df["titrant_conc"])
    )

    print("Running growth predictions...", flush=True)
    result_df = predict(model_class=gm,
                        param_posteriors=posterior_file,
                        predict_sites=["growth_pred"],
                        num_samples=num_samples,
                        num_marginal_samples=num_marginal_samples,
                        titrant_conc=titrant_concs,
                        genotypes=genotypes)

    # Apply titrant_name filter post-prediction.
    if titrant_names is not None:
        result_df = result_df[result_df["titrant_name"].isin(titrant_names)].reset_index(drop=True)

    result_df["in_training_data"] = result_df.apply(
        lambda row: int((row["genotype"], row["titrant_name"], row["titrant_conc"])
                        in training_tuples),
        axis=1,
    )

    out_file = f"{out_prefix}.csv"
    result_df.to_csv(out_file, index=False)
    print(f"Wrote {len(result_df)} rows to {out_file}", flush=True)


def main():
    generalized_main(predict_growth,
                     manual_arg_types={"genotypes_file": str,
                                       "titrant_names_file": str,
                                       "titrant_concs_file": str,
                                       "num_marginal_samples": int})


if __name__ == "__main__":
    main()
