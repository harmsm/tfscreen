"""
Prior-predictive activity sampling from any registered activity component.

sample_activity_prior draws one sample of per-genotype TF activity values
from the prior of a registered activity component.  This keeps the
simulation pipeline consistent with the model family used during inference:
if the horseshoe activity component is used at inference time, the same
component can generate ground-truth activity values during simulation.

The "fixed" component (activity = 1.0 for all genotypes) is the default
and is appropriate for repressor libraries where mutation effects are
captured entirely by changes in operator occupancy (theta).  Non-fixed
components (horseshoe, hierarchical, …) are available for activator
experiments or any context where per-genotype variation in intrinsic TF
strength is expected.
"""

import numpy as np
from numpyro import handlers

from tfscreen.tfmodel.generative.registry import model_registry


def sample_activity_prior(component_name,
                          sim_data,
                          rng_key,
                          priors_overrides=None):
    """
    Draw one sample of per-genotype TF activity from the prior of a
    registered activity component.

    Parameters
    ----------
    component_name : str
        A key in ``model_registry["activity"]``.  Valid options are
        ``"fixed"``, ``"hierarchical"``, ``"horseshoe"``,
        ``"hierarchical_mut"``, and ``"horseshoe_mut"``.
    sim_data : SimData
        Built by ``build_sim_data``.  For genotype-level components
        (``"horseshoe"``, ``"hierarchical"``) the object must have
        ``scale_vector`` and ``wt_indexes`` populated; ``build_sim_data``
        always satisfies this.  Mutation-decomposed components
        (``"horseshoe_mut"``, ``"hierarchical_mut"``) additionally require
        the ``mut_nnz_*`` fields, which ``build_sim_data`` also provides.
    rng_key : jax.random.PRNGKey or int
        Seed for the NumPyro sampler.  For ``"fixed"`` the component has no
        stochastic sites so the key is accepted but unused.
    priors_overrides : dict or None
        Key-value pairs merged into ``get_hyperparameters()`` before
        constructing the priors object.  Use to tighten or loosen the prior
        without editing the component source.

    Returns
    -------
    activity : numpy.ndarray, shape (num_genotype,)
        Per-genotype TF activity values.  Always positive.  Wild-type
        genotype(s) receive 1.0 for all components that honour
        ``sim_data.wt_indexes`` (i.e. all non-mut-decomposed components).
        Mutation-decomposed components implicitly give wt activity = 1.0
        because the wild-type carries no mutations.

    Raises
    ------
    ValueError
        If ``component_name`` is not registered in
        ``model_registry["activity"]``.
    """

    activity_registry = model_registry["activity"]

    if component_name not in activity_registry:
        raise ValueError(
            f"activity component '{component_name}' not found in "
            f"model_registry['activity'].  "
            f"Available: {sorted(activity_registry)}"
        )

    module = activity_registry[component_name]

    # Build priors: start from module defaults, apply any overrides.
    params = module.get_hyperparameters()
    if priors_overrides:
        params.update(priors_overrides)
    priors = module.ModelPriors(**params)

    # Seed the NumPyro trace and call define_model directly.
    # handlers.seed intercepts pyro.sample sites and gives them
    # reproducible draws.  pyro.deterministic sites (used by "fixed") are
    # unaffected.
    with handlers.seed(rng_seed=rng_key):
        activity_7d = module.define_model("activity", sim_data, priors)

    # activity components return shape (1, 1, 1, 1, 1, 1, G) — a 7-D
    # broadcast tensor matching the tfmodel observation layout.  Strip the
    # six leading singleton dimensions to recover the per-genotype 1-D array.
    return np.array(activity_7d[0, 0, 0, 0, 0, 0, :])
