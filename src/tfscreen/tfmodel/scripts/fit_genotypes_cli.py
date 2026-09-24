"""
tfs-fit-genotypes: per-genotype MLE fits of the growth model against real data.

A non-Bayesian, per-genotype alternative to the joint ``tfs-fit-model``: given
processed ``ln_cfu`` data and a *frozen* per-condition growth calibration
(``k``/``m`` from ``tfs-prefit-calibration``), it independently fits each
genotype's phenotype ``(dk_geno, theta_low, theta_high, log_hill_K, hill_n)``
by nonlinear least squares (see :mod:`tfscreen.tfmodel.genotype_fit.fit`).

Outputs (under ``--out_prefix``)
--------------------------------
* ``<prefix>_params.csv`` — one row per (genotype, titrant_name): the MLE
  fit (natural + transformed params, transformed-space std).
* ``<prefix>_theta.csv`` — long form ``[genotype, titrant_name, titrant_conc,
  theta_raw]``: the fitted Hill curves evaluated on each titrant's
  concentration grid.

The fits are not corrected for congression (co-transformation): the bulk
genotypes' theta curves carry that bias.  The old theta-only de-attenuation
(``--congression_lambda``) was retired with the move to the observable-level
congression mixture in ``tfs-fit-model``.
"""

import os

from tfscreen.tfmodel.genotype_fit.fit import (
    fit_phenotypes, predict_theta,
)
from tfscreen.util.io import read_dataframe
from tfscreen.util.cli.generalized_main import generalized_main


def fit_genotypes(growth_file,
                  calibration_file,
                  out_prefix="tfs_mle",
                  intercept_cols="replicate",
                  dk_geno_prior_sd=1.0,
                  min_obs=None,
                  num_workers=1):
    """
    Fit the growth model to each genotype independently (frozen calibration).

    Parameters
    ----------
    growth_file : str
        Processed ``ln_cfu`` CSV (``tfs-process-counts`` output).
    calibration_file : str
        Frozen per-condition growth calibration: a ``tfs-prefit-calibration``
        priors CSV, or a wide ``condition_rep,growth_k,growth_m`` CSV.
    out_prefix : str
        Output prefix (see the module docstring for the files written).
    intercept_cols : str
        Comma-separated columns whose unique combinations each get a nuisance
        ``ln_cfu0`` (default ``"replicate"``; empty string -> single intercept).
    dk_geno_prior_sd : float
        Std of the weak Normal prior on ``dk_geno`` (<=0 disables it).
    min_obs : int, optional
        Skip genotypes with fewer than this many usable observations.
    num_workers : int
        Parallelize the per-genotype fits over a process pool: ``1`` (default)
        serial; ``-1`` uses ``os.cpu_count() - 1``; ``N`` uses ``N``.
    """
    growth_df = read_dataframe(growth_file)

    icols = [c.strip() for c in str(intercept_cols).split(",") if c.strip()]

    dk_prior = None
    if dk_geno_prior_sd is not None and float(dk_geno_prior_sd) > 0:
        dk_prior = (0.0, float(dk_geno_prior_sd))

    # --- Per-genotype MLE fits. -------------------------------------------
    print("Fitting each genotype independently (frozen calibration)...",
          flush=True)
    results_df, fits = fit_phenotypes(
        growth_df, calibration_file, intercept_cols=icols,
        dk_geno_prior=dk_prior, min_obs=min_obs, num_workers=num_workers)

    theta_df = predict_theta(fits, growth_df, theta_col="theta_raw")

    # --- Write outputs. ---------------------------------------------------
    params_path = os.path.abspath(f"{out_prefix}_params.csv")
    results_df.to_csv(params_path, index=False)

    theta_path = os.path.abspath(f"{out_prefix}_theta.csv")
    theta_df.to_csv(theta_path, index=False)

    bar = "=" * 72
    print(f"\n{bar}")
    print(f"Fit {len(fits)} (genotype, titrant_name) groups.")
    print("\n  Per-genotype parameters (MLE fit):")
    print(f"    {params_path}")
    print("  Predicted theta vs (genotype, titrant_name, titrant_conc):")
    print(f"    {theta_path}")
    print(bar)


def main():
    return generalized_main(
        fit_genotypes,
        manual_arg_types={"min_obs": int, "num_workers": int})


if __name__ == "__main__":
    main()
