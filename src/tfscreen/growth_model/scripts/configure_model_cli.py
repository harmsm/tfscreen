import os
import yaml
import pandas as pd
import numpy as np
import tfscreen
from tfscreen.growth_model.model_class import ModelClass as GrowthModel
from tfscreen.growth_model.configuration_io import write_configuration
from tfscreen.util.cli.generalized_main import generalized_main

# Maps (condition_growth, theta_rescale) pairs that are fundamentally incompatible
# to a human-readable explanation of why.
#
# power uses theta**n (non-integer exponent): negative theta from logit → NaN.
# saturation uses theta/(1+theta): logit can produce theta near -1 → pole/NaN.
INCOMPATIBLE_CG_TR = {
    ("power", "logit"): (
        "The 'power' growth model raises theta to a non-integer exponent (theta**n). "
        "The 'logit' rescale maps theta to all of R, producing negative values that "
        "cause NaN when raised to a non-integer power."
    ),
    ("saturation", "logit"): (
        "The 'saturation' growth model evaluates theta/(1+theta). "
        "The 'logit' rescale can map theta to values near -1, causing a "
        "division-by-zero singularity."
    ),
}


def check_component_compatibility(condition_growth_model, theta_rescale_model):
    """Raise ValueError if the (condition_growth, theta_rescale) pair is incompatible.

    Parameters
    ----------
    condition_growth_model : str
        Name of the condition_growth component.
    theta_rescale_model : str
        Name of the theta_rescale component.

    Raises
    ------
    ValueError
        If the pair is listed in INCOMPATIBLE_CG_TR.
    """
    key = (condition_growth_model, theta_rescale_model)
    if key in INCOMPATIBLE_CG_TR:
        raise ValueError(
            f"Incompatible model components: "
            f"condition_growth='{condition_growth_model}' and "
            f"theta_rescale='{theta_rescale_model}'. "
            f"{INCOMPATIBLE_CG_TR[key]}"
        )


def configure_model(binding_df,
                    growth_df=None,
                    out_prefix="tfs_configure",
                    condition_growth_model="linear",
                    growth_transition_model="instant",
                    ln_cfu0_model="hierarchical",
                    dk_geno_model="hierarchical",
                    activity_model="horseshoe",
                    theta_model="hill",
                    transformation_model="empirical",
                    theta_rescale_model="passthrough",
                    theta_growth_noise_model="zero",
                    theta_binding_noise_model="zero",
                    spiked=None,
                    growth_shares_replicates=False,
                    epistasis=False,
                    struct_ensemble_path=None,
                    batch_size=1024,
                    binding_weight=None):
    """
    Build and write the YAML configuration files needed by tfs-fit-model.

    Constructs a GrowthModel from the supplied data and model-component choices,
    then writes three files: {out_prefix}_config.yaml (the main configuration),
    {out_prefix}_priors.csv (prior distributions for all parameters), and
    {out_prefix}_guesses.csv (initial-value guesses for array parameters).

    When only binding_df is provided (no growth_df), a binding-only model is
    configured that infers theta directly from observed binding measurements
    rather than from bacterial growth data.

    Parameters
    ----------
    binding_df : str
        Path to the binding data CSV file (theta vs. titrant measurements per
        genotype). Required.
    growth_df : str, optional
        Path to the growth data CSV file (ln_cfu measurements per genotype,
        replicate, and timepoint). When omitted, a binding-only model is
        configured.
    out_prefix : str, optional
        Prefix for the three output files ({out_prefix}_config.yaml,
        {out_prefix}_priors.csv, {out_prefix}_guesses.csv).
        Default 'tfs_configure'.
    condition_growth_model: str, optional
        Model to use to describe growth under different conditions (e.g.,
        pheS+4CP). Allowed values are 'linear' (default), 'linear_independent',
        'linear_fixed', 'power', or 'saturation'.
    growth_transition_model : str, optional
        Model to use to describe the transition between the pre-selection
        and selection phases. Allowed values are 'instant' (default), 'memory',
        or 'baranyi'.
    ln_cfu0_model : str, optional
        Model to use to describe ln_cfu0, the initial populations of genotypes
        in each replicate. Only 'hierarchical' is allowed at this point.
    dk_geno_model : str, optional
        Model to use to describe dk_geno, the pleiotropic effect of a genotype
        on growth, independent of occupancy. Allowed values are 'hierarchical'
        (default), 'fixed', or 'hierarchical_mut'.
    activity_model : str, optional
        Model to use to describe activity, a scalar multiplied against
        occupancy that defines how strongly a genotype alters transcription
        given its occupancy. Allowed values are 'fixed' (default), 'hierarchical',
        'horseshoe', or 'hierarchical_mut'.
    theta_model : str, optional
        Model to use to describe theta, the fractional occupancy of a genotype
        on the transcription factor binding site. Allowed values are 'hill'
        (default), 'categorical', 'hill_mut', 'lac_dimer_lnK_mut',
        'lac_dimer_lnK_nn_prior', 'mwc_dimer_lnK_mut', 'mwc_dimer_lnK_nn_prior',
        or 'mwc_dimer_lnK_ddG_prior'.
    transformation_model : str, optional
        Model for transformation correction. Allowed values are 'single',
        'empirical', or 'logit_norm'. Default 'empirical'.
    theta_rescale_model : str, optional
        Rescaling applied to theta before it enters the growth model. Allowed
        values are 'passthrough' (default, identity) or 'logit' (maps theta to
        log(theta/(1-theta)), expanding the dynamic range at both extremes).
    theta_growth_noise_model : str, optional
        Model to use for stochastic experimental noise in theta measured by
        bacterial growth. Allowed values are 'beta' (default) or 'zero'.
    theta_binding_noise_model : str, optional
        Model to use for stochastic experimental noise in theta measured by
        binding. Allowed values are 'beta' (default) or 'zero'.
    spiked : list or str, optional
        Names of genotypes that should be excluded from congression
        correction.
    growth_shares_replicates : bool, optional
        Whether replicates should share the same parameters for the growth and
        growth transition models. Default is False.
    epistasis : bool, optional
        Whether to model pairwise epistatic interactions between mutations in
        any mutation-level model (``hill_mut``, ``hierarchical_mut``). When
        True, each pair of mutations present in the same genotype gets an
        independent epistasis term. When False (default), effects are purely
        additive at the mutation level.
    struct_ensemble_path : str, optional
        Path to the structural data file.  Required when ``theta_model`` is a
        struct-based model; ignored otherwise.  For ``lac_dimer_lnK_nn_prior``
        and ``mwc_dimer_lnK_nn_prior`` this must be the HDF5 file produced by
        ``scripts/generate_struct_ensemble.py``.  For
        ``mwc_dimer_lnK_ddG_prior`` this must be a CSV file with a ``mut``
        column and one column per structure (``H``, ``HO``, ``L``, ``LO``,
        ``HE2``, ``LE2``) containing pre-computed ΔΔG prior means.
    batch_size : int, optional
        Mini-batch size for SVI. Defaults to 1024. Set to None to use the full
        dataset as a single batch.
    binding_weight : float, optional
        Multiplicative scale applied to the binding log-likelihood at every SVI
        step.  Because growth data typically outnumber binding observations by
        several orders of magnitude, the binding likelihood contributes a
        negligible gradient signal unless it is upweighted.  When None
        (default), the weight is auto-computed as
        ``N_growth_rows / N_binding_rows`` so that each binding observation
        contributes the same total weight as the average growth observation.
        Pass an explicit positive float to override this heuristic.  The
        resolved value (never None) is saved in the YAML so that
        ``tfs-fit-model`` applies the same weight without recomputing it.

    Returns
    -------
    None
    """
    if binding_df is None:
        raise ValueError("binding_df must be provided")

    binding_only = growth_df is None
    if not binding_only:
        check_component_compatibility(condition_growth_model, theta_rescale_model)

    # Initialize model to build mappings and get guesses
    gm = GrowthModel(growth_df,
                     binding_df,
                     binding_only=binding_only,
                     condition_growth=condition_growth_model,
                     growth_transition=growth_transition_model,
                     ln_cfu0=ln_cfu0_model,
                     dk_geno=dk_geno_model,
                     activity=activity_model,
                     theta=theta_model,
                     transformation=transformation_model,
                     theta_rescale=theta_rescale_model,
                     theta_growth_noise=theta_growth_noise_model,
                     theta_binding_noise=theta_binding_noise_model,
                     spiked_genotypes=spiked,
                     growth_shares_replicates=growth_shares_replicates,
                     epistasis=epistasis,
                     struct_ensemble_path=struct_ensemble_path,
                     batch_size=batch_size,
                     binding_weight=binding_weight)

    # Write the model configuration to a file. This includes the model component
    # names, the data file paths, and the parameter guesses/priors.
    growth_path = None if binding_only else (growth_df if isinstance(growth_df, str) else "growth.csv")
    write_configuration(gm=gm,
                        out_prefix=out_prefix,
                        growth_df_path=growth_path,
                        binding_df_path=binding_df if isinstance(binding_df, str) else "binding.csv")

def main():
    return generalized_main(configure_model,
                            manual_arg_types={"binding_df":str,
                                              "growth_df":str,
                                              "spiked":list,
                                              "struct_ensemble_path":str,
                                              "batch_size":int,
                                              "binding_weight":float},
                            manual_arg_nargs={"spiked":"+"})

if __name__ == "__main__":
    main()
