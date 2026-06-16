
import tfscreen

from tfscreen.simulate import (
    build_sample_dataframes,
    thermo_to_growth,
)
from tfscreen.simulate.sim_data_class import build_sim_data
from tfscreen.simulate.sample_theta import sample_theta_stratified
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
    theta_gc_override = None
    binding_theta_df = None

    if binding_cfg is not None:
        binding_genotypes = list(standardize_genotypes(binding_cfg['genotypes']))
        binding_concs = binding_cfg['titrant_conc']
        titrant_name = binding_cfg['titrant_name']

        binding_sample_df = pd.DataFrame({"titrant_conc": binding_concs})

        selected_binding_gc, selected_growth_gc = sample_theta_stratified(
            component_name=cf['theta_component'],
            binding_sample_df=binding_sample_df,
            growth_sample_df=sample_df,
            rng_key=theta_rng_key,
            n_select=len(binding_genotypes),
            thermo_data=cf.get('thermo_data'),
            pool_size=cf.get('binding_stratify_pool_size', 500),
            priors_overrides=cf.get('theta_priors'),
            sim_priors_overrides=cf.get('theta_sim_priors'),
        )

        # Build override dict for thermo_to_growth (theta at growth concentrations)
        theta_gc_override = {
            g: selected_growth_gc[i]
            for i, g in enumerate(binding_genotypes)
        }

        # Build binding_theta_df (theta at binding concentrations, pre-noise)
        sorted_binding_concs = np.sort(np.unique(binding_concs))
        conc_to_col = {float(c): j for j, c in enumerate(sorted_binding_concs)}
        rows = []
        for i, g in enumerate(binding_genotypes):
            for conc in binding_concs:
                col_idx = conc_to_col[float(conc)]
                rows.append({
                    "genotype": g,
                    "titrant_name": titrant_name,
                    "titrant_conc": float(conc),
                    "theta_true": float(selected_binding_gc[i, col_idx]),
                })
        binding_theta_df = pd.DataFrame(rows)

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
    )

    return library_df, phenotype_df, genotype_theta_df, parameters_df, binding_theta_df
