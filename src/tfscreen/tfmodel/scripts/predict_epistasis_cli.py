import pandas as pd
from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.checkpoint_io import resolve_param_file
from tfscreen.tfmodel.analysis.extraction import extract_theta_epistasis
from tfscreen.util.cli import generalized_main, read_lines


def predict_epistasis(config_file,
                      param_file,
                      out_prefix="tfs_pred_epistasis",
                      genotypes_file=None,
                      titrant_names_file=None,
                      titrant_concs_file=None,
                      only_files=False,
                      scale="logit",
                      scale_constant=1.0,
                      regime_eps=0.01,
                      regime_ci=0.95):
    """
    Predict second-order epistasis from the joint posterior of a fitted model.

    For every double mutant, epistasis is calculated on its mutant cycle (the
    double, its two single-mutant parents, and the wildtype).  In contrast to
    running tfs-extract-epistasis on a tfs-predict-theta table -- which uses the
    per-genotype marginal theta estimates and assumes the four corners are
    independent -- this command draws theta for all genotypes from the same
    posterior sample, computes epistasis within each draw, and then reports
    quantiles across draws.  The uncertainty therefore reflects the true
    posterior covariance between the corners of each cycle.

    Epistasis is computed independently at each condition (titrant_name,
    titrant_conc).  Output columns are 'genotype', 'titrant_name',
    'titrant_conc', one 'q<level>' column per quantile (e.g. 'q0.5', 'q0.025',
    'q0.975' -- the library-wide quantile-output convention), and a trailing
    'in_regime' flag (int 0/1), written to {out_prefix}.csv.

    'in_regime' marks whether the estimate is backed by data or by model
    extrapolation.  It is 1 only when all four corners of the mutant cycle (wt,
    both singles, double) have their theta posterior (central 95% interval by
    default) inside the resolvable band [regime_eps, 1 - regime_eps].  When a
    corner's theta is near saturation (0 or 1), logit(theta) is compressed and
    the growth data constrain it weakly, so the epistasis there leans on the
    theta-model's functional form and cross-genotype posterior covariance --
    treat in_regime == 0 rows as model-conditional.  (This checks posterior mass
    only; it does not separately test whether the growth signal exceeds the
    growth noise.)

    Only genotypes seen during training are supported (the joint sample matrix
    is built from the training theta curves); requesting an out-of-training
    genotype is an error.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    param_file : str
        Path to a posterior .h5 file produced by tfs-sample-posterior, or a MAP
        checkpoint .pkl file produced by tfs-fit-model.  A .pkl provides a
        single point estimate, so every ep_<quantile> column collapses to that
        value (no uncertainty); run tfs-sample-posterior first to obtain a
        Laplace posterior and real quantiles.
    out_prefix : str, optional
        Prefix for the output CSV file. Written to {out_prefix}.csv.
        Default 'tfs_pred_epistasis'.
    genotypes_file : str or None, optional
        Plain-text file with one genotype per line (slash-separated mutations,
        e.g. 'M42I/K84L', or 'wt'). Restricts the analysis to cycles whose
        genotypes appear here (unioned with all training genotypes unless
        --only_files is set). All genotypes must have been seen during
        training. Default None.
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
        If True, use only the genotypes and titrant (name, conc) pairs supplied
        via file arguments, ignoring training-data combinations. Default False.
    scale : {"logit", "add", "mult"}, optional
        Epistatic scale. 'logit' (default) computes additive epistasis of
        logit(theta) -- the natural scale for an occupancy in [0, 1]. 'add':
        (Y11 - Y10) - (Y01 - Y00). 'mult': (Y11 / Y10) / (Y01 / Y00).
    scale_constant : float, optional
        Constant applied to the transform before the difference-of-differences;
        multiplies the reported epistasis. Default 1.0. Mainly a unit
        conversion for scale='logit': since logit(theta) = -dG/RT, passing -RT
        (e.g. -0.6159 for kcal/mol at 310.15 K) reports epistasis as an
        interaction free energy. Has no effect on (and is rejected for)
        scale='mult'.
    regime_eps : float, optional
        Theta resolution floor for the 'in_regime' flag. A cycle corner counts
        as resolvable when its theta posterior sits in [regime_eps,
        1 - regime_eps]. Default 0.01 (matching a 1%-in-theta resolution, whose
        logit band is +/- ~4.6). Must be in [0, 0.5).
    regime_ci : float, optional
        Central posterior-mass fraction required inside the band for a corner to
        count as in-regime. Default 0.95 (checks the theta 2.5-97.5% interval).
        Must be in (0, 1).
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

    training_genotypes = set(orchestrator.training_tm.df["genotype"].unique())

    # Resolve requested genotypes and fail fast on any out-of-training genotype:
    # the joint sample matrix only covers training genotypes.
    file_genotypes = read_lines(genotypes_file) if genotypes_file else []
    out_of_training = [g for g in file_genotypes if g not in training_genotypes]
    if out_of_training:
        raise ValueError(
            "tfs-predict-epistasis only supports genotypes seen during "
            "training (joint epistasis needs a joint theta sample for every "
            "cycle corner). The following requested genotype(s) were not in "
            f"the training data: {out_of_training}."
        )

    if only_files and file_genotypes:
        requested_genotypes = set(file_genotypes)
    elif file_genotypes:
        requested_genotypes = training_genotypes | set(file_genotypes)
    else:
        requested_genotypes = training_genotypes

    # Resolve requested titrant grid (mirrors tfs-predict-theta).
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

    print(f"Computing joint epistasis (scale='{scale}') over "
          f"{len(requested_genotypes)} genotype(s)...", flush=True)

    q_to_get = [0.5] if is_map else None
    result_df = extract_theta_epistasis(
        orchestrator=orchestrator,
        posteriors=param_file,
        q_to_get=q_to_get,
        manual_titrant_df=manual_titrant_df,
        scale=scale,
        scale_constant=scale_constant,
        regime_eps=regime_eps,
        regime_ci=regime_ci,
    )

    # Restrict cycles to those whose double mutant is a requested genotype.
    if not result_df.empty:
        result_df = result_df[
            result_df["genotype"].isin(requested_genotypes)
        ].reset_index(drop=True)

    out_file = f"{out_prefix}.csv"
    result_df.to_csv(out_file, index=False)
    print(f"Wrote {len(result_df)} rows to {out_file}", flush=True)


def main():
    generalized_main(predict_epistasis,
                     manual_arg_types={"genotypes_file": str,
                                       "titrant_names_file": str,
                                       "titrant_concs_file": str,
                                       "only_files": bool,
                                       "scale": str})


if __name__ == "__main__":
    main()
