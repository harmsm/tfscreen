"""
Prior-predictive theta sampling from any registered theta component.

sample_theta_prior draws one sample of ground-truth theta values from the
prior of a registered theta component.  This replaces the old ddG-spreadsheet
path: instead of reading fixed thermodynamic parameters from a file, we draw
parameters from the model prior, making the simulation consistent with the
model family being inferred.
"""

import numpy as np
from numpyro import handlers

from tfscreen.tfmodel.generative.registry import model_registry

# Components excluded from simulation (calibration-only, no generative prior).
_EXCLUDED = frozenset({"simple"})


def sample_theta_prior(component_name,
                       sim_data,
                       rng_key,
                       priors_overrides=None):
    """
    Draw one sample of theta from the prior of a registered theta component.

    Parameters
    ----------
    component_name : str
        A key in ``model_registry["theta"]`` (e.g. ``"hill"``,
        ``"mwc_dimer_lnK_mut"``).  ``"simple"`` is not supported.
    sim_data : SimData
        Built by ``build_sim_data``.  Must have all fields required by the
        chosen component.
    rng_key : jax.random.PRNGKey
        Seed for the NumPyro sampler.
    priors_overrides : dict or None
        Key-value overrides applied to ``get_hyperparameters()`` before
        constructing the priors object.  Use to set physical constants such
        as ``theta_tf_total_M`` or concentration unit scales without
        modifying the component defaults.

    Returns
    -------
    theta : np.ndarray, shape (num_genotype, num_titrant_conc)
        Ground-truth fractional occupancy for each genotype at each unique
        effector concentration.  Store alongside phenotype_df to enable
        comparison with fitted posteriors.
    theta_param : ThetaParam
        Sampled parameter pytree (structure is model-specific).  Contains
        the underlying equilibrium constants, Hill parameters, etc. that
        generated ``theta``.

    Raises
    ------
    ValueError
        If ``component_name`` is not in the registry or is excluded.
    """
    theta_registry = model_registry["theta"]

    if component_name not in theta_registry:
        raise ValueError(
            f"theta component '{component_name}' not found in model_registry['theta']. "
            f"Available: {sorted(theta_registry)}"
        )
    if component_name in _EXCLUDED:
        raise ValueError(
            f"'{component_name}' cannot be used for simulation "
            f"(it is a calibration-only component with no generative prior)."
        )

    module = theta_registry[component_name]

    # Build priors: start from module defaults, then apply any overrides.
    params = module.get_hyperparameters()
    if priors_overrides:
        params.update(priors_overrides)
    priors = module.ModelPriors(**params)

    # Seed the NumPyro trace and call define_model directly.
    # handlers.seed gives every pyro.sample site a reproducible draw
    # without needing a full Predictive wrapper.  pyro.param sites return
    # their init_value (module default) when no param_store is active,
    # which is the correct behaviour for prior-predictive simulation.
    with handlers.seed(rng_seed=rng_key):
        theta_param = module.define_model("theta", sim_data, priors)

    # run_model returns (T=1, C, G) for simulation (scatter_theta=0).
    theta_tcg = module.run_model(theta_param, sim_data)
    theta_gc = np.array(theta_tcg[0]).T   # (G, C)

    return theta_gc, theta_param
