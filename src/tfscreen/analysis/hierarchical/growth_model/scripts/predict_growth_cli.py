from tfscreen.analysis.hierarchical.growth_model.configuration_io import read_configuration
from tfscreen.analysis.hierarchical.growth_model.checkpoint_io import resolve_param_file
from tfscreen.analysis.hierarchical.growth_model.prediction import predict
from tfscreen.util.cli import generalized_main, read_lines


def predict_growth(config_file,
                   param_file,
                   out_prefix="tfs_growth_pred",
                   genotypes_file=None,
                   titrant_names_file=None,
                   titrant_concs_file=None,
                   only_files=False,
                   num_samples=0,
                   num_marginal_samples=None):
    """
    Predict growth signal (ln_cfu) from a fitted hierarchical model.

    By default predicts at all (genotype, titrant_name, titrant_conc,
    replicate, t_pre, t_sel) combinations present in the training data, unioned
    with any genotypes or concentrations supplied via file arguments.  Pass
    --only_files to predict exclusively at the file-specified inputs and skip
    training-data combinations.

    titrant_names_file is a post-prediction row filter (restrict-only): it
    narrows which titrant names appear in the output but does not affect which
    concentrations are predicted.  It is not subject to union semantics.

    A boolean column 'in_training_data' is added to the output: 1 if the
    (genotype, titrant_name, titrant_conc) triple was in the training data,
    0 otherwise.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    param_file : str
        Path to a posterior .h5 file produced by tfs-sample-posterior, or a
        MAP checkpoint .pkl file produced by tfs-fit-model.  When a
        .pkl file is supplied the MAP point estimate is used: a 1-sample
        posterior is written to {out_prefix}_map_posterior.h5 and predictions
        are made at that single parameter set.  NUTS and SVI checkpoints are
        not supported directly; run tfs-sample-posterior first.
    out_prefix : str, optional
        Prefix for the output CSV file. Written to {out_prefix}.csv.
        Default 'tfs_growth_pred'.
    genotypes_file : str or None, optional
        Plain-text file with one genotype per line (slash-separated mutations,
        e.g. 'M42I/K84L', or 'wt'). These genotypes are unioned with all
        training genotypes unless --only_files is set. Default None.
    titrant_names_file : str or None, optional
        Plain-text file with one titrant name per line. Filters output rows to
        the named titrants (restrict-only; does not affect which concentrations
        are predicted). If None, all titrant names are included.
    titrant_concs_file : str or None, optional
        Plain-text file with one concentration per line. These concentrations
        are unioned with all training concentrations unless --only_files is set.
        Default None.
    only_files : bool, optional
        If True, predict only at the genotypes and concentrations supplied via
        file arguments, ignoring training-data combinations. Default False.
    num_samples : int or None, optional
        Number of joint posterior samples to include as sample_0 … sample_N-1
        columns alongside the quantile columns. Set to None for quantiles only.
        Default 0.
    num_marginal_samples : int or None, optional
        Number of posterior samples to run through the model when computing
        quantiles. If None, all available samples are used.
    """
    file_genotypes = read_lines(genotypes_file) if genotypes_file else []
    titrant_names = read_lines(titrant_names_file) if titrant_names_file else None
    file_concs = [float(x) for x in read_lines(titrant_concs_file)] if titrant_concs_file else []

    print(f"Loading configuration from {config_file}...", flush=True)
    gm, _ = read_configuration(config_file)
    param_file = resolve_param_file(param_file, gm, out_prefix)

    # Build training-data membership set for in_training_data column.
    training_tuples = set(
        zip(gm.growth_df["genotype"],
            gm.growth_df["titrant_name"],
            gm.growth_df["titrant_conc"])
    )

    if only_files:
        genotypes = file_genotypes if file_genotypes else None
        titrant_concs = file_concs if file_concs else None
    else:
        training_genotypes = list(gm.growth_df["genotype"].unique())
        genotypes = list(dict.fromkeys(training_genotypes + file_genotypes)) if file_genotypes else None
        training_concs = list(gm.growth_df["titrant_conc"].unique())
        titrant_concs = sorted(set(training_concs) | set(file_concs)) if file_concs else None

    print("Running growth predictions...", flush=True)
    result_df = predict(model_class=gm,
                        param_posteriors=param_file,
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
                                       "num_marginal_samples": int,
                                       "only_files": bool})


if __name__ == "__main__":
    main()
