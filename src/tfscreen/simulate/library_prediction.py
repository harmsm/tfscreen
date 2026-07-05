
import tfscreen

from tfscreen.simulate import (
    build_sample_dataframes,
    thermo_to_growth,
)
from tfscreen.simulate.sim_data_class import build_sim_data
from tfscreen.simulate.sample_theta import sample_theta_stratified, sample_theta_prior
from tfscreen.simulate.binding_params import (
    SUPPORTED_COMPONENTS as _BINDING_PARAMS_SUPPORTED,
    read_binding_genotype_params,
    build_theta_gc_override_hill_geno,
    build_theta_gc_override_hill_mut,
    build_binding_theta_from_params,
    _wt_params_from_sim_priors,
)
from tfscreen.genetics import library_manager
from tfscreen.genetics import standardize_genotypes

import jax
import numpy as np
import pandas as pd

from typing import Any, Dict, Union
from pathlib import Path

def library_prediction(cf: Union[Dict[str, Any], str, Path],
                       override_keys: dict=None):
    """
    Predict the "ground-truth" phenotypes for a transcription factor screen
    experiment given a library, thermodynamic model, and map between that model
    and bacterial growth rate.

    Parameters
    ----------
    cf : dict or str or pathlib.Path
        The configuration for the simulation. Can be a dictionary or a path
        to a YAML file containing the configuration parameters.
    override_keys : dict, default=None
        after reading the configuration file, replace keys in the configuration
        with the key/value pairs in override keys. No error checking is done
        on these keys; the user is responsible for checking their sanity.

    Returns
    -------
    library_df : pandas.DataFrame
        dataframe holding genotypes one would get when using the degenerate
        codons and sequences specified in the configuration
    phenotype_df : pandas.DataFrame
        dataframe with predicted fractional occupancy and growth rates for each
        of the genotypes in each of the conditions specified in the configuration
    genotype_theta_df : pandas.DataFrame
        long-form dataframe with one row per (genotype, titrant_name,
        titrant_conc) combination; columns: genotype, titrant_name,
        titrant_conc, theta
    parameters_df : pandas.DataFrame
        one row per unique genotype; columns ``dk_geno``, ``activity``, and
        any scalar per-genotype fields from the theta component (e.g.
        ``theta_low``, ``theta_high``, ``log_hill_K``, ``hill_n``)
    binding_theta_df : pandas.DataFrame or None
        Present when ``binding_data`` is in the config.  One row per
        (genotype, titrant_conc) for the calibration genotypes listed in
        ``binding_data.genotypes``, with columns ``genotype``,
        ``titrant_name``, ``titrant_conc``, ``theta_true``.  Theta values
        here are stratified (greedy maximin) across the binding concentrations
        and are consistent with the growth phenotypes in ``phenotype_df``.
    """

    # -------------------------------------------------------------------------
    # Read inputs and set up simulation

    cf = tfscreen.util.read_yaml(cf, override_keys=override_keys)

    # -------------------------------------------------------------------------
    # Do main calculation

    # Build library_df
    lm = library_manager.LibraryManager(cf)
    library_df = lm.build_library_df()

    # Build sample_df (holds all conditions for the experiment)
    sample_df = build_sample_dataframes(
        cf['condition_blocks'],
        replicate=1
    )

    # Build SimData: lightweight container for the theta model
    sim_data = build_sim_data(
        library_df=library_df,
        sample_df=sample_df,
        thermo_data=cf.get('thermo_data'),
    )

    # Derive both RNG objects from the single seed.  They are independent
    # objects with independent state; using the same integer for both introduces
    # no correlation between the two streams.
    seed = cf.get('seed', None)
    theta_rng_key = jax.random.PRNGKey(seed if seed is not None else 0)
    rng = np.random.default_rng(seed)

    dk_geno_zero = cf.get('dk_geno_zero', False)
    if dk_geno_zero:
        dk_geno_hyper_loc = cf.get('dk_geno_hyper_loc', -3.5)
        dk_geno_hyper_scale = cf.get('dk_geno_hyper_scale', 1.0)
        dk_geno_hyper_shift = cf.get('dk_geno_hyper_shift', 0.02)
    else:
        dk_geno_hyper_loc = cf['dk_geno_hyper_loc']
        dk_geno_hyper_scale = cf['dk_geno_hyper_scale']
        dk_geno_hyper_shift = cf['dk_geno_hyper_shift']

    # -------------------------------------------------------------------------
    # Stratified theta sampling for binding calibration genotypes

    binding_cfg = cf.get('binding_data')
    theta_gc_override = {}
    theta_params_override = {}
    binding_theta_df = None

    if binding_cfg is not None:
        binding_concs = binding_cfg['titrant_conc']
        titrant_name = binding_cfg['titrant_name']
        binding_noise = binding_cfg.get('noise', 0.0)

        binding_sample_df = pd.DataFrame({"titrant_conc": binding_concs})
        sorted_binding_concs = np.sort(np.unique(binding_concs))
        conc_to_col = {float(c): j for j, c in enumerate(sorted_binding_concs)}

        rows = []  # accumulated rows for binding_theta_df

        # --- Section 1: Simulated genotypes (existing 'genotypes' key) ---
        binding_genotypes = list(standardize_genotypes(binding_cfg.get('genotypes', [])))

        # wt gets its natural unperturbed reference parameters — not a random
        # draw from the pool.  Only non-wt binding genotypes are stratified.
        non_wt_genotypes = [g for g in binding_genotypes if g != "wt"]

        # wt reference curve (if wt is a simulated binding genotype)
        wt_binding_gc = None
        if "wt" in binding_genotypes:
            single_wt_df = pd.DataFrame({"genotype": ["wt"]})
            binding_wt_sim = build_sim_data(single_wt_df, binding_sample_df,
                                            thermo_data=cf.get('thermo_data'),
                                            skip_pairs=True)
            # Perturbation path for wt (M=0): always returns the fixed sim-priors
            # reference curve regardless of rng_key, so the binding data matches
            # the growth simulation exactly.
            wt_binding_gc, _ = sample_theta_prior(
                cf['theta_component'], binding_wt_sim, theta_rng_key,
                sim_priors_overrides=cf.get('theta_sim_priors'),
            )

        # Stratified curves for non-wt simulated binding genotypes
        selected_binding_gc = None
        selected_growth_gc = None
        if non_wt_genotypes:
            selected_binding_gc, selected_growth_gc = sample_theta_stratified(
                component_name=cf['theta_component'],
                binding_sample_df=binding_sample_df,
                growth_sample_df=sample_df,
                rng_key=theta_rng_key,
                n_select=len(non_wt_genotypes),
                thermo_data=cf.get('thermo_data'),
                pool_size=cf.get('binding_stratify_pool_size', 500),
                priors_overrides=cf.get('theta_priors'),
                sim_priors_overrides=cf.get('theta_sim_priors'),
            )

        # Override theta at growth concentrations for non-wt binding genotypes only.
        # wt is deliberately excluded so thermo_to_growth uses its natural curve.
        if non_wt_genotypes and selected_growth_gc is not None:
            theta_gc_override.update({
                g: selected_growth_gc[i]
                for i, g in enumerate(non_wt_genotypes)
            })

        # Accumulate binding rows for simulated genotypes
        if "wt" in binding_genotypes and wt_binding_gc is not None:
            for conc in binding_concs:
                rows.append({
                    "genotype": "wt",
                    "titrant_name": titrant_name,
                    "titrant_conc": float(conc),
                    "theta_true": float(wt_binding_gc[0, conc_to_col[float(conc)]]),
                })
        for i, g in enumerate(non_wt_genotypes):
            for conc in binding_concs:
                rows.append({
                    "genotype": g,
                    "titrant_name": titrant_name,
                    "titrant_conc": float(conc),
                    "theta_true": float(selected_binding_gc[i, conc_to_col[float(conc)]]),
                })

        # --- Section 2: Measured genotype params from CSV ---
        params_file = binding_cfg.get('genotype_params_file')
        if params_file is not None:
            theta_component = cf['theta_component']
            if theta_component not in _BINDING_PARAMS_SUPPORTED:
                raise ValueError(
                    f"genotype_params_file is only supported with Hill-based theta "
                    f"components: {sorted(_BINDING_PARAMS_SUPPORTED)}. "
                    f"Got: '{theta_component}'."
                )

            # Build sim_priors for WT reference / NaN filling
            from tfscreen.tfmodel.generative.registry import model_registry
            theta_module = model_registry["theta"][theta_component]
            sim_params = theta_module.get_sim_hyperparameters()
            if cf.get('theta_sim_priors'):
                sim_params.update(cf['theta_sim_priors'])
            sim_priors = theta_module.SimPriors(**sim_params)

            wt_params = _wt_params_from_sim_priors(sim_priors)

            params_dict = read_binding_genotype_params(params_file)

            # Compute theta overrides for measured genotypes
            log_conc_growth = np.array(sim_data.log_titrant_conc)

            if theta_component == "hill_geno":
                measured_override, measured_params = build_theta_gc_override_hill_geno(
                    params_dict, log_conc_growth, wt_params
                )
            else:  # hill_mut
                measured_override, measured_params = build_theta_gc_override_hill_mut(
                    params_dict=params_dict,
                    library_genotypes=library_df["genotype"].tolist(),
                    sim_data=sim_data,
                    sim_priors=sim_priors,
                    log_conc=log_conc_growth,
                    rng=rng,
                )

            # Measured data takes precedence over stratified simulated data
            theta_gc_override.update(measured_override)
            theta_params_override.update(measured_params)

            # Accumulate binding rows for measured genotypes (overrides simulated rows)
            # Store noise-free theta_true; noise is applied later by
            # binding_data.generate_binding_df, consistent with the
            # simulated-genotypes path.
            measured_binding_df = build_binding_theta_from_params(
                params_dict=params_dict,
                binding_concs=binding_concs,
                titrant_name=titrant_name,
                noise=0.0,
                rng=rng,
                wt_params=wt_params,
            )
            # Remove any existing rows for genotypes now covered by measured data
            measured_genotypes = set(params_dict.keys())
            rows = [r for r in rows if r["genotype"] not in measured_genotypes]
            rows.extend(measured_binding_df.to_dict("records"))

        binding_theta_df = pd.DataFrame(rows) if rows else None

    # -------------------------------------------------------------------------
    # Calculate phenotype for each genotype across all conditions in sample_df

    phenotype_df, genotype_theta_df, parameters_df = thermo_to_growth(
        genotypes=library_df["genotype"],
        sim_data=sim_data,
        sample_df=sample_df,
        theta_component=cf['theta_component'],
        theta_rng_key=theta_rng_key,
        growth_params=cf['growth'],
        theta_priors_overrides=cf.get('theta_priors'),
        theta_sim_priors_overrides=cf.get('theta_sim_priors'),
        dk_geno_hyper_loc=dk_geno_hyper_loc,
        dk_geno_hyper_scale=dk_geno_hyper_scale,
        dk_geno_hyper_shift=dk_geno_hyper_shift,
        dk_geno_zero=dk_geno_zero,
        activity_wt=cf.get('activity_wt', 1.0),
        activity_mut_scale=cf.get('activity_mut_scale', 0.0),
        rng=rng,
        activity_component=cf.get('activity_component', 'fixed'),
        activity_priors_overrides=cf.get('activity_priors'),
        theta_rescale=cf.get('theta_rescale', 'passthrough'),
        theta_gc_override=theta_gc_override,
        theta_params_override=theta_params_override or None,
    )

    return library_df, phenotype_df, genotype_theta_df, parameters_df, binding_theta_df
