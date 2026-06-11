import pandas as pd
from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.checkpoint_io import resolve_param_file
from tfscreen.tfmodel.analysis.extraction import (
    extract_theta_curves,
    extract_theta_unmeasured,
)
from tfscreen.tfmodel.generative.registry import model_registry
from tfscreen.util.cli import generalized_main, read_lines


def predict_theta(config_file,
                  param_file,
                  out_prefix="tfs_theta_pred",
                  genotypes_file=None,
                  titrant_names_file=None,
                  titrant_concs_file=None,
                  only_files=False,
                  num_samples=0,
                  genotype_batch_size=2000):
    """
    Predict operator occupancy (theta) from a fitted hierarchical model.

    By default predicts theta at all (genotype, titrant_name, titrant_conc)
    combinations present in the training data, unioned with any genotypes or
    titrant (name, conc) pairs supplied via file arguments.  Pass --only_files
    to predict exclusively at the file-specified inputs and skip training-data
    combinations.

    titrant_names_file and titrant_concs_file together define a paired
    (name, conc) grid that is unioned with the training titrant grid (or
    replaces it when --only_files is set).  They must always be provided
    together or both omitted.

    When any requested genotype was not seen during training,
    extract_theta_unmeasured is used.

    A boolean column 'in_training_data' is added to the output: 1 if the
    (genotype, titrant_name, titrant_conc) triple was in the training data,
    0 otherwise.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    param_file : str
        Path to a posterior .h5 file produced by tfs-sample-posterior, or a
        MAP checkpoint .pkl file produced by tfs-fit-model.

        When a .pkl file is supplied the output contains a single ``point_est``
        column with no uncertainty information.

        To obtain uncertainty estimates from a MAP fit, first run
        tfs-sample-posterior on the .pkl checkpoint; it will construct a
        Laplace (Hessian-based) posterior approximation and write a .h5 file.
        Passing that .h5 here produces the full quantile columns (median,
        lower_95, upper_95, etc.).

        NUTS and SVI checkpoints are not supported directly; run
        tfs-sample-posterior first.
    out_prefix : str, optional
        Prefix for the output CSV file. Written to {out_prefix}.csv.
        Default 'tfs_theta_pred'.
    genotypes_file : str or None, optional
        Plain-text file with one genotype per line (slash-separated mutations,
        e.g. 'M42I/K84L', or 'wt'). These genotypes are unioned with all
        training genotypes unless --only_files is set. May include genotypes
        not seen during training when the theta component supports
        predict_unmeasured. Default None.
    titrant_names_file : str or None, optional
        Plain-text file with one titrant name per line. Must be provided
        together with titrant_concs_file (or both omitted). The resulting
        (name, conc) pairs are unioned with the training titrant grid unless
        --only_files is set. A single name is broadcast across all
        concentrations in titrant_concs_file. Default None.
    titrant_concs_file : str or None, optional
        Plain-text file with one concentration per line. Must be provided
        together with titrant_names_file (or both omitted). Default None.
    only_files : bool, optional
        If True, predict only at the genotypes and titrant (name, conc) pairs
        supplied via file arguments, ignoring training-data combinations.
        Default False.
    num_samples : int or None, optional
        Number of joint posterior samples to include as sample_0 … sample_N-1
        columns alongside the quantile columns. Set to None for quantiles only.
        Default 0.
    genotype_batch_size : int, optional
        When predicting unmeasured genotypes, process this many at a time to
        cap the memory used by the epistasis pair-indicator matrix
        (batch_size × N_pair × 4 bytes).  Smaller values reduce peak memory
        at the cost of more iterations.  Default 2000.
    """
    if (titrant_names_file is None) != (titrant_concs_file is None):
        raise ValueError(
            "titrant_names_file and titrant_concs_file must be provided "
            "together (or both omitted)."
        )

    print(f"Loading configuration from {config_file}...", flush=True)
    orchestrator, _ = read_configuration(config_file)
    is_map = param_file.endswith(".pkl")
    param_file = resolve_param_file(param_file, orchestrator, out_prefix)

    # Determine training genotypes and (genotype, titrant_name, titrant_conc) set.
    # growth_tm is preferred (more genotypes); binding_tm is the fallback for
    # binding-only runs where growth_tm is None.
    training_genotypes = set(orchestrator.training_tm.df["genotype"].unique())
    training_tuples = set(
        zip(orchestrator.training_tm.df["genotype"],
            orchestrator.training_tm.df["titrant_name"],
            orchestrator.training_tm.df["titrant_conc"])
    )

    # Resolve requested genotypes.
    file_genotypes = read_lines(genotypes_file) if genotypes_file else []
    if only_files:
        requested_genotypes = file_genotypes if file_genotypes else list(training_genotypes)
    else:
        requested_genotypes = list(dict.fromkeys(
            list(training_genotypes) + file_genotypes
        ))

    # Resolve requested titrant grid.
    if titrant_names_file is not None:
        titrant_names = read_lines(titrant_names_file)
        titrant_concs = [float(x) for x in read_lines(titrant_concs_file)]
        if len(titrant_names) == 1:
            titrant_names = titrant_names * len(titrant_concs)
        elif len(titrant_names) != len(titrant_concs):
            raise ValueError(
                f"titrant_names_file has {len(titrant_names)} entries but "
                f"titrant_concs_file has {len(titrant_concs)}. Supply either "
                "one name (broadcast to all concentrations) or one per "
                "concentration."
            )
        file_titrant_df = pd.DataFrame({
            "titrant_name": titrant_names,
            "titrant_conc": titrant_concs,
        })
        if only_files:
            manual_titrant_df = file_titrant_df
        else:
            training_titrant_df = (
                orchestrator.training_tm.df[["titrant_name", "titrant_conc"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            manual_titrant_df = (
                pd.concat([training_titrant_df, file_titrant_df])
                .drop_duplicates()
                .reset_index(drop=True)
            )
    else:
        manual_titrant_df = None

    # Decide extraction path.
    out_of_training = [g for g in requested_genotypes if g not in training_genotypes]

    if out_of_training:
        module = model_registry.get("theta", {}).get(orchestrator._theta)
        if module is None or not hasattr(module, "predict_unmeasured"):
            raise ValueError(
                f"The theta component '{orchestrator._theta}' does not support prediction "
                "for genotypes not seen during training. Remove out-of-training "
                "genotypes from genotypes_file, or use a theta component that "
                "implements predict_unmeasured (e.g. 'hill')."
            )
        print(f"Predicting theta via additive mutation assembly "
              f"({len(out_of_training)} out-of-training genotype(s))...", flush=True)
        if manual_titrant_df is None:
            # Use unique (titrant_name, titrant_conc) pairs from training data.
            manual_titrant_df = (
                orchestrator.training_tm.df[["titrant_name", "titrant_conc"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
        q_to_get = [0.5] if is_map else None
        result_df = extract_theta_unmeasured(
            orchestrator=orchestrator,
            posteriors=param_file,
            target_genotypes=requested_genotypes,
            manual_titrant_df=manual_titrant_df,
            genotype_batch_size=genotype_batch_size,
            q_to_get=q_to_get,
        )
    else:
        print(f"Predicting theta for {len(requested_genotypes)} training genotype(s)...",
              flush=True)
        q_to_get = [0.5] if is_map else None
        result_df = extract_theta_curves(
            orchestrator=orchestrator,
            posteriors=param_file,
            manual_titrant_df=manual_titrant_df,
            num_samples=num_samples,
            q_to_get=q_to_get,
        )
        # extract_theta_curves returns all training genotypes when manual_titrant_df
        # has no 'genotype' column; filter to the requested subset.
        if genotypes_file is not None:
            result_df = result_df[result_df["genotype"].isin(requested_genotypes)].reset_index(drop=True)

    result_df["in_training_data"] = result_df.apply(
        lambda row: int((row["genotype"], row["titrant_name"], row["titrant_conc"])
                        in training_tuples),
        axis=1,
    )

    out_file = f"{out_prefix}.csv"
    result_df.to_csv(out_file, index=False)
    print(f"Wrote {len(result_df)} rows to {out_file}", flush=True)


def main():
    generalized_main(predict_theta,
                     manual_arg_types={"genotypes_file": str,
                                       "titrant_names_file": str,
                                       "titrant_concs_file": str,
                                       "only_files": bool,
                                       "num_samples": int,
                                       "genotype_batch_size": int})


if __name__ == "__main__":
    main()
