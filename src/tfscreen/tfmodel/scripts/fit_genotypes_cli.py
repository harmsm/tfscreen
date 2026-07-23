"""
tfs-fit-genotypes: per-genotype MLE fits of the growth model against real data.

A non-Bayesian, per-genotype alternative to the joint ``tfs-fit-model``: given
processed ``ln_cfu`` data and a *frozen* per-condition growth calibration
(``k``/``m`` from ``tfs-prefit-calibration``), it independently fits each
genotype's phenotype ``(dk_geno, theta_low, theta_high, log_hill_K, hill_n)``
by nonlinear least squares (see :mod:`tfscreen.tfmodel.genotype_fit.fit`).

An optional congression de-attenuation pass (``--congression_lambda``) removes
the co-transformation bias from the *bulk* genotypes' theta curves via the
iterative fixed point (predict theta -> de-attenuate against the population ->
refit Hill).  It is inherently a population operation: the correction couples
all bulk genotypes through their shared background occupancy distribution.
Spiked genotypes (``--spiked_file``) are congression-free and pass through
untouched.

Outputs (under ``--out_prefix``)
--------------------------------
* ``<prefix>_params.csv`` — one row per (genotype, titrant_name): the RAW MLE
  fit (natural + transformed params, transformed-space std).
* ``<prefix>_params_deattenuated.csv`` — same schema for the de-attenuated
  fits (only when ``--congression_lambda`` is given).
* ``<prefix>_theta.csv`` — long form ``[genotype, titrant_name, titrant_conc,
  theta_raw]`` (plus ``theta_deattenuated`` when congression ran): the fitted
  Hill curves evaluated on each titrant's concentration grid.
* ``<prefix>_theta_history.csv`` — long form ``[genotype, titrant_name,
  titrant_conc, iter, theta]``: the congression fixed-point trajectory (only
  with ``--congression_lambda`` and ``--save_theta_history``).
"""

import os

from tfscreen.tfmodel.genotype_fit.fit import (
    fit_phenotypes, fits_to_results_df, predict_theta,
)
from tfscreen.util.io import read_dataframe
from tfscreen.util.cli import read_lines
from tfscreen.util.cli.generalized_main import generalized_main


def fit_genotypes(growth_file,
                  calibration_file,
                  out_prefix="tfs_mle",
                  congression_lambda=None,
                  spiked_file=None,
                  intercept_cols="replicate",
                  dk_geno_prior_sd=1.0,
                  min_obs=None,
                  num_workers=1,
                  save_theta_history=False):
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
    congression_lambda : float, optional
        Zero-truncated Poisson congression rate (the same lambda as the
        simulator's ``transformation_poisson_lambda``).  When given, run the
        de-attenuation pass; omit for raw MLE fits only.
    spiked_file : str, optional
        Text file of congression-free (spiked) genotype names, one per line.
        These are excluded from the de-attenuation correction and background.
        Only relevant with ``--congression_lambda``.
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
    save_theta_history : bool
        Also write the congression fixed-point theta trajectory
        (``<prefix>_theta_history.csv``).  Only meaningful with
        ``--congression_lambda``.
    """
    growth_df = read_dataframe(growth_file)
    spiked = read_lines(spiked_file) if spiked_file else None

    icols = [c.strip() for c in str(intercept_cols).split(",") if c.strip()]

    dk_prior = None
    if dk_geno_prior_sd is not None and float(dk_geno_prior_sd) > 0:
        dk_prior = (0.0, float(dk_geno_prior_sd))

    # --- Raw per-genotype MLE fits. ---------------------------------------
    print("Fitting each genotype independently (frozen calibration)...",
          flush=True)
    results_df, fits = fit_phenotypes(
        growth_df, calibration_file, intercept_cols=icols,
        dk_geno_prior=dk_prior, min_obs=min_obs, num_workers=num_workers)

    theta_df = predict_theta(fits, growth_df, theta_col="theta_raw")

    # --- Optional congression de-attenuation. -----------------------------
    deatt_df = None
    history_df = None
    do_congression = (congression_lambda is not None
                      and float(congression_lambda) > 0)
    if do_congression:
        from tfscreen.tfmodel.genotype_fit.congression import (
            deattenuate_congression,
        )
        print(f"De-attenuating congression "
              f"(lambda={float(congression_lambda):g})...", flush=True)
        result = deattenuate_congression(
            fits, growth_df, float(congression_lambda), spiked=spiked,
            return_theta_history=save_theta_history)
        if save_theta_history:
            corrected_fits, history_df = result
        else:
            corrected_fits = result

        deatt_df = fits_to_results_df(corrected_fits)
        theta_deatt = predict_theta(corrected_fits, growth_df,
                                    theta_col="theta_deattenuated")
        theta_df = theta_df.merge(
            theta_deatt, on=["genotype", "titrant_name", "titrant_conc"],
            how="left")

    # --- Write outputs. ---------------------------------------------------
    params_path = os.path.abspath(f"{out_prefix}_params.csv")
    results_df.to_csv(params_path, index=False)

    theta_path = os.path.abspath(f"{out_prefix}_theta.csv")
    theta_df.to_csv(theta_path, index=False)

    deatt_path = None
    if deatt_df is not None:
        deatt_path = os.path.abspath(f"{out_prefix}_params_deattenuated.csv")
        deatt_df.to_csv(deatt_path, index=False)

    history_path = None
    if history_df is not None:
        history_path = os.path.abspath(f"{out_prefix}_theta_history.csv")
        history_df.to_csv(history_path, index=False)

    bar = "=" * 72
    print(f"\n{bar}")
    print(f"Fit {len(fits)} (genotype, titrant_name) groups.")
    print("\n  Per-genotype parameters (raw MLE fit):")
    print(f"    {params_path}")
    if deatt_path is not None:
        print("  Per-genotype parameters (congression-de-attenuated):")
        print(f"    {deatt_path}")
    print("  Predicted theta vs (genotype, titrant_name, titrant_conc):")
    print(f"    {theta_path}")
    if history_path is not None:
        print("  Congression fixed-point theta trajectory:")
        print(f"    {history_path}")
    print(bar)


def main():
    return generalized_main(
        fit_genotypes,
        manual_arg_types={"spiked_file": str, "congression_lambda": float,
                          "min_obs": int, "num_workers": int})


if __name__ == "__main__":
    main()
