import pandas as pd
from tfscreen.analysis.hierarchical.growth_model.configuration_io import read_configuration
from tfscreen.analysis.hierarchical.growth_model.extraction import (
    extract_theta_curves,
    extract_theta_unmeasured,
)
from tfscreen.analysis.hierarchical.growth_model.registry import model_registry
from tfscreen.util.cli import generalized_main, read_lines


def predict_theta(config_file,
                  posterior_file,
                  out_prefix="tfs_theta_pred",
                  genotypes_file=None,
                  titrant_names_file=None,
                  titrant_concs_file=None):
    """
    Predict operator occupancy (theta) from a fitted hierarchical model.

    By default predicts theta at all (genotype, titrant_name, titrant_conc)
    combinations present in the training data. Optional files can specify
    different genotypes and/or concentrations.

    When the requested genotypes are all present in the training data,
    extract_theta_curves is used. When any requested genotype was not seen
    during training, extract_theta_unmeasured is used (requires the theta
    component to implement predict_unmeasured; raises an error for the
    'categorical' component).

    A boolean column 'in_training_data' is added to the output: 1 if the
    (genotype, titrant_name, titrant_conc) triple was in the training data,
    0 otherwise.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    posterior_file : str
        Path to the .h5 file produced by tfs-sample-posterior.
    out_prefix : str, optional
        Prefix for the output CSV file. Written to {out_prefix}.csv.
        Default 'tfs_theta_pred'.
    genotypes_file : str or None, optional
        Plain-text file with one genotype per line (slash-separated mutations,
        e.g. 'M42I/K84L', or 'wt'). May include genotypes not seen during
        training when the theta component supports predict_unmeasured.
        If None, all training genotypes are used.
    titrant_names_file : str or None, optional
        Plain-text file with one titrant name per line. Must be provided
        together with titrant_concs_file (or both omitted).
        If None, all training (titrant_name, titrant_conc) pairs are used.
    titrant_concs_file : str or None, optional
        Plain-text file with one concentration per line. Must be provided
        together with titrant_names_file (or both omitted). When both are
        given, their rows are paired one-to-one, or a single titrant name is
        broadcast across all concentrations.
    """
    if (titrant_names_file is None) != (titrant_concs_file is None):
        raise ValueError(
            "titrant_names_file and titrant_concs_file must be provided "
            "together (or both omitted)."
        )

    print(f"Loading configuration from {config_file}...", flush=True)
    gm, _ = read_configuration(config_file)

    # Determine training genotypes and (genotype, titrant_name, titrant_conc) set.
    training_genotypes = set(gm.growth_tm.df["genotype"].unique())
    training_tuples = set(
        zip(gm.growth_tm.df["genotype"],
            gm.growth_tm.df["titrant_name"],
            gm.growth_tm.df["titrant_conc"])
    )

    # Resolve requested genotypes.
    if genotypes_file is not None:
        requested_genotypes = read_lines(genotypes_file)
    else:
        requested_genotypes = list(training_genotypes)

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
        manual_titrant_df = pd.DataFrame({
            "titrant_name": titrant_names,
            "titrant_conc": titrant_concs,
        })
    else:
        manual_titrant_df = None

    # Decide extraction path.
    out_of_training = [g for g in requested_genotypes if g not in training_genotypes]

    if out_of_training:
        module = model_registry.get("theta", {}).get(gm._theta)
        if module is None or not hasattr(module, "predict_unmeasured"):
            raise ValueError(
                f"The theta component '{gm._theta}' does not support prediction "
                "for genotypes not seen during training. Remove out-of-training "
                "genotypes from genotypes_file, or use a theta component that "
                "implements predict_unmeasured (e.g. 'hill')."
            )
        print(f"Predicting theta via additive mutation assembly "
              f"({len(out_of_training)} out-of-training genotype(s))...", flush=True)
        if manual_titrant_df is None:
            # Use unique (titrant_name, titrant_conc) pairs from training data.
            manual_titrant_df = (
                gm.growth_tm.df[["titrant_name", "titrant_conc"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
        result_df = extract_theta_unmeasured(
            model=gm,
            posteriors=posterior_file,
            target_genotypes=requested_genotypes,
            manual_titrant_df=manual_titrant_df,
        )
    else:
        print(f"Predicting theta for {len(requested_genotypes)} training genotype(s)...",
              flush=True)
        result_df = extract_theta_curves(
            model=gm,
            posteriors=posterior_file,
            manual_titrant_df=manual_titrant_df,
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
                                       "titrant_concs_file": str})


if __name__ == "__main__":
    main()
