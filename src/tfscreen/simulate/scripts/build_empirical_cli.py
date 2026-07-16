"""
tfs-build-empirical: fit real screen data to an empirical
phenotype-generating distribution (the empirical pipeline, one command).

Takes the **experimental inputs** (real growth + binding data) and internally
runs the whole pipeline: it configures a linear/hill_geno model
(``tfs-configure-model`` defaults — there are no meaningful choices here for
this use case) and MAP-calibrates the per-condition growth linkage
(``tfs-prefit-calibration``) to get frozen ``k``/``m``, then

  * Stage 1 — per-genotype MLE of the growth model (frozen calibration) →
    per-genotype ``(dk_geno, theta_low, theta_high, log_hill_K, hill_n)`` with
    covariances;
  * Stage 2 — measurement-error deconvolution into a ``PopulationModel``
    (with wt's actual fitted phenotype embedded as ``wt_ref``).

The saved model is what ``tfs-simulate`` resamples from via
``phenotype_source: empirical``.

The configure/prefit intermediates are written under ``<out_prefix>_configure_*``
and ``<out_prefix>_prefit_*`` so they can be inspected and reused.  The MAP
prefit is the slow step; pass ``--calibration_file`` (a prefit priors CSV or a
wide ``condition_rep,growth_k,growth_m`` CSV) to skip configure+prefit and
iterate on the fast Stage-1/2 knobs.
"""

import os
import warnings

from tfscreen.tfmodel.genotype_fit.fit import (
    fit_phenotypes, fits_to_results_df, _natural_from_transformed,
)
from tfscreen.simulate.empirical.population import fit_population
from tfscreen.util.io import read_dataframe
from tfscreen.util.cli import read_lines
from tfscreen.util.cli.generalized_main import generalized_main

_WT_REF_COLS = ["theta_low", "theta_high", "log_hill_K", "hill_n"]


def _run_configure_and_prefit(growth_file, binding_file, spiked,
                              base_growth_file, thermo_data, out_prefix, seed):
    """Configure a linear/hill_geno model and MAP-calibrate k/m.

    Returns the path to the calibrated priors CSV.  Imports the (heavy)
    inference stack lazily so the ``--calibration_file`` path stays light.
    """
    if binding_file is None:
        raise ValueError(
            "binding data is required to calibrate the growth linkage; pass "
            "--binding_file, or pass --calibration_file to skip configure+prefit.")

    from tfscreen.tfmodel.scripts.configure_model_cli import configure_model
    from tfscreen.tfmodel.scripts.prefit_calibration_cli import (
        run_prefit_calibration,
    )

    configure_prefix = f"{out_prefix}_configure"

    print("Configuring model (linear growth, hill_geno theta)...", flush=True)
    configure_model(
        binding_df=binding_file,
        growth_df=growth_file,
        base_growth_df=base_growth_file,
        spiked=spiked,
        thermo_data=thermo_data,
        out_prefix=configure_prefix,
    )

    print("Pre-fit calibrating per-condition growth k/m (MAP)...", flush=True)
    run_prefit_calibration(
        f"{configure_prefix}_config.yaml",
        seed=seed,
        out_prefix=f"{out_prefix}_prefit",
    )

    # prefit updates the config's priors CSV in place.
    return f"{configure_prefix}_priors.csv"


def build_empirical(growth_file,
                    seed,
                    binding_file=None,
                    out_prefix="tfs_empirical",
                    calibration_file=None,
                    spiked_file=None,
                    base_growth_file=None,
                    thermo_data=None,
                    congression_lambda=None,
                    intercept_cols="replicate",
                    dk_geno_prior_sd=1.0,
                    min_obs=None,
                    drop_railed=True,
                    num_workers=1):
    """
    Fit real screen data to an empirical phenotype generating distribution.

    Parameters
    ----------
    growth_file : str
        Processed ``ln_cfu`` CSV (``tfs-process-counts`` output) for real data.
    seed : int
        Random seed for the pre-fit MAP calibration (positional / required for
        reproducibility; ignored when ``--calibration_file`` is supplied).
    binding_file : str, optional
        Real binding CSV.  Required unless ``calibration_file`` is given; it
        identifies the growth slope ``m`` from data (the in-library anchors).
    out_prefix : str
        Output prefix.  Writes ``<prefix>_model.npz`` (+ ``.names.json``),
        ``<prefix>_stage1_fits.csv`` (the RAW per-genotype fits), and — when
        ``congression_lambda`` is given — ``<prefix>_stage1p5_fits.csv`` (the
        de-attenuated fits that feed Stage 2).  Unless ``calibration_file`` is
        given, also the ``<prefix>_configure_*`` / ``<prefix>_prefit_*`` intermediates.
    calibration_file : str, optional
        Skip the configure+prefit step and use this calibration directly — a
        prefit priors CSV or a wide ``condition_rep,growth_k,growth_m`` CSV.
        Use this to iterate on the Stage-1/2 knobs without re-running the MAP.
    spiked_file : str, optional
        Text file of spiked genotype names (one per line) forwarded to
        ``configure_model``.
    base_growth_file : str, optional
        Direct growth-rate calibration CSV forwarded to ``configure_model``.
    thermo_data : str, optional
        Thermodynamic data path forwarded to ``configure_model`` (unused by
        the default hill_geno theta).
    congression_lambda : float, optional
        Zero-truncated Poisson congression rate (the same lambda as the
        simulator's ``transformation_poisson_lambda``).  When given, run Stage
        1.5: de-attenuate the bulk theta curves for co-transformation before
        building the distribution (spiked genotypes are congression-free and
        left alone).  Omit for no correction.
    intercept_cols : str
        Comma-separated columns whose unique combinations each get a nuisance
        ``ln_cfu0`` (default ``"replicate"``; empty string -> single intercept).
    dk_geno_prior_sd : float
        Std of the weak Normal prior on ``dk_geno`` (<=0 disables it).
    min_obs : int, optional
        Skip genotypes with fewer than this many usable observations.
    drop_railed : bool
        Drop theta-logit-railed Stage-1 fits before the Stage-2 fit.
    num_workers : int
        Parallelize the Stage-1 per-genotype fits over a process pool: ``1``
        (default) serial; ``-1`` uses ``os.cpu_count() - 1``; ``N`` uses ``N``.
        Recommended for large libraries (the fits are embarrassingly parallel).
    """
    growth_df = read_dataframe(growth_file)
    spiked = read_lines(spiked_file) if spiked_file else None

    icols = [c.strip() for c in str(intercept_cols).split(",") if c.strip()]

    dk_prior = None
    if dk_geno_prior_sd is not None and float(dk_geno_prior_sd) > 0:
        dk_prior = (0.0, float(dk_geno_prior_sd))

    # Calibration: reuse a supplied one, or configure+prefit to produce it.
    if calibration_file is None:
        calibration_file = _run_configure_and_prefit(
            growth_file, binding_file, spiked, base_growth_file,
            thermo_data, out_prefix, seed)
    else:
        print(f"Using supplied calibration: {calibration_file}", flush=True)

    # Stage 1: fit every measured genotype independently (frozen calibration).
    # The calibration path is normalized (prefit priors CSV or wide) inside.
    print("Stage 1: fitting each measured genotype independently...", flush=True)
    results_df, fits = fit_phenotypes(
        growth_df, calibration_file, intercept_cols=icols,
        dk_geno_prior=dk_prior, min_obs=min_obs, num_workers=num_workers)

    # Stage 1.5 (optional): de-attenuate the bulk theta curves for congression.
    # The raw Stage-1 table (results_df) is left as-is; the corrected fits are
    # written to their own table so both are available and unambiguous.
    stage1p5_df = None
    if congression_lambda is not None and float(congression_lambda) > 0:
        from tfscreen.tfmodel.genotype_fit.congression import (
            deattenuate_congression,
        )
        print(f"Stage 1.5: de-attenuating congression "
              f"(lambda={float(congression_lambda):g})...", flush=True)
        fits = deattenuate_congression(
            fits, growth_df, float(congression_lambda), spiked=spiked)
        stage1p5_df = fits_to_results_df(fits)

    # Stage 2: turn the per-genotype fits into ONE generating distribution
    # (deconvolving estimation noise).  This distribution is the deliverable.
    print(f"Stage 2: building the generating distribution from "
          f"{len(fits)} per-genotype fits...", flush=True)
    model = fit_population(fits, drop_railed=drop_railed)

    # Embed wt's actual (congression-corrected, if applied) phenotype so the
    # simulation pins wt to its real value rather than a resampled/mean draw.
    wt_keys = [k for k in fits if k[0] == "wt"]
    if wt_keys:
        nat = _natural_from_transformed(fits[wt_keys[0]].est_t,
                                        fits[wt_keys[0]].pheno_slice)
        model.wt_ref = {c: float(nat[c]) for c in _WT_REF_COLS}
    else:
        warnings.warn(
            "no 'wt' genotype in the Stage-1 fits; wt_ref not set (the "
            "simulation will default wt to the population mean).")

    # The model is the ONLY file the simulation needs -- one self-contained JSON.
    model_path = os.path.abspath(model.save(f"{out_prefix}_phenotype_model"))
    fits_path = os.path.abspath(f"{out_prefix}_stage1_fits.csv")
    results_df.to_csv(fits_path, index=False)

    # The de-attenuated (Stage-1.5) fits, when congression was applied.  The
    # Stage-1 table above is always the *raw* (pre-de-attenuation) fit.
    stage1p5_path = None
    if stage1p5_df is not None:
        stage1p5_path = os.path.abspath(f"{out_prefix}_stage1p5_fits.csv")
        stage1p5_df.to_csv(stage1p5_path, index=False)

    bar = "=" * 72
    print(f"\n{bar}")
    print(f"Fit the generating distribution from {model.n_used} genotypes "
          f"(loglik={model.loglik:.4g}, {model.n_iter} EM iters).")
    print("\n  Phenotype model  (the distribution tfs-simulate samples from):")
    print(f"    {model_path}")
    print("  Per-genotype Stage-1 fits, RAW / pre-de-attenuation "
          "(diagnostic only, not used downstream):")
    print(f"    {fits_path}")
    if stage1p5_path is not None:
        print("  Per-genotype Stage-1.5 fits, congression-de-attenuated "
              "(what feeds Stage 2):")
        print(f"    {stage1p5_path}")
    print("\nTo simulate from it, add to your tfs-simulate config:")
    print("    phenotype_source: empirical")
    print("    empirical:")
    print(f"      phenotype_model: {model_path}")
    print("(An absolute path resolves from any directory you run tfs-simulate in.)")
    print(bar)


def main():
    return generalized_main(
        build_empirical,
        manual_arg_types={"binding_file": str, "calibration_file": str,
                          "spiked_file": str, "base_growth_file": str,
                          "thermo_data": str, "seed": int, "min_obs": int,
                          "congression_lambda": float, "num_workers": int})
