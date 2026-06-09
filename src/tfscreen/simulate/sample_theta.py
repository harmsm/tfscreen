"""
Prior-predictive theta sampling from any registered theta component.

sample_theta_prior draws one sample of ground-truth theta values from the
prior of a registered theta component.  This replaces the old ddG-spreadsheet
path: instead of reading fixed thermodynamic parameters from a file, we draw
parameters from the model prior, making the simulation consistent with the
model family being inferred.
"""

import inspect
import numpy as np
from numpyro import handlers

from tfscreen.tfmodel.generative.registry import model_registry

# Components excluded from simulation (calibration-only, no generative prior).
_EXCLUDED = frozenset({"_simple"})


def sample_theta_prior(component_name,
                       sim_data,
                       rng_key,
                       priors_overrides=None,
                       sim_priors_overrides=None):
    """
    Draw one sample of theta from a registered theta component.

    If the component defines a ``simulate`` function (detected via
    ``inspect.isfunction``), the perturbation-based path is used: wildtype
    reference parameters are drawn from ``SimPriors`` built from
    ``get_sim_hyperparameters()`` plus any ``sim_priors_overrides``.

    Otherwise the prior-predictive path is used: ``define_model`` is called
    inside a seeded NumPyro trace, and ``run_model`` produces ``theta_gc``.

    Parameters
    ----------
    component_name : str
        A key in ``model_registry["theta"]`` (e.g. ``"hill_geno"``,
        ``"mwc_dimer_lnK_mut"``).  ``"_simple"`` is not supported.
    sim_data : SimData
        Built by ``build_sim_data``.  Must have all fields required by the
        chosen component.
    rng_key : jax.random.PRNGKey
        Seed for sampling (NumPyro or NumPy depending on the path taken).
    priors_overrides : dict or None
        Overrides for the inference ``ModelPriors`` (prior-predictive path
        only).  Ignored when the component provides ``simulate``.
    sim_priors_overrides : dict or None
        Overrides for ``SimPriors`` (perturbation path only).  Ignored when
        the component does not provide ``simulate``.

    Returns
    -------
    theta : np.ndarray, shape (num_genotype, num_titrant_conc)
        Ground-truth fractional occupancy for each genotype at each unique
        effector concentration.
    theta_param : ThetaParam
        Sampled parameter pytree.  When the perturbation path is used,
        ``mu`` and ``sigma`` fields are ``None``.

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

    # Perturbation-based path: component defines a simulate() function.
    if inspect.isfunction(getattr(module, "simulate", None)):
        sim_params = module.get_sim_hyperparameters()
        if sim_priors_overrides:
            sim_params.update(sim_priors_overrides)
        sim_priors = module.SimPriors(**sim_params)
        return module.simulate("theta", sim_data, sim_priors, rng_key)

    # Prior-predictive fallback: seed the NumPyro trace and call define_model.
    # handlers.seed gives every pyro.sample site a reproducible draw
    # without needing a full Predictive wrapper.  pyro.param sites return
    # their init_value (module default) when no param_store is active,
    # which is the correct behaviour for prior-predictive simulation.
    params = module.get_hyperparameters()
    if priors_overrides:
        params.update(priors_overrides)
    priors = module.ModelPriors(**params)

    with handlers.seed(rng_seed=rng_key):
        theta_param = module.define_model("theta", sim_data, priors)

    # run_model returns (T=1, C, G) for simulation (scatter_theta=0).
    theta_tcg = module.run_model(theta_param, sim_data)
    theta_gc = np.array(theta_tcg[0]).T   # (G, C)

    return theta_gc, theta_param
