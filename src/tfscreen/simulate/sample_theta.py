"""
Prior-predictive theta sampling from any registered theta component.

sample_theta_prior draws one sample of ground-truth theta values from the
prior of a registered theta component.  This replaces the old ddG-spreadsheet
path: instead of reading fixed thermodynamic parameters from a file, we draw
parameters from the model prior, making the simulation consistent with the
model family being inferred.

sample_theta_stratified draws a large pool of independent theta curves from
the prior at the binding concentrations, selects n_select curves that
maximally span the theta-curve space (greedy maximin), then returns the
selected curves at both binding and growth concentrations.  This ensures
calibration genotypes cover diverse binding behaviors rather than clustering
by chance.
"""

import inspect
import numpy as np
import pandas as pd
import tqdm
import jax
from numpyro import handlers

from tfscreen.tfmodel.generative.registry import model_registry
from tfscreen.simulate.sim_data_class import build_sim_data

# Components excluded from simulation (calibration-only, no generative prior).
_EXCLUDED = frozenset({"_simple"})


def sample_theta_prior(component_name,
                       sim_data,
                       rng_key,
                       priors_overrides=None,
                       sim_priors_overrides=None,
                       force_prior_predictive=False):
    """
    Draw one sample of theta from a registered theta component.

    If the component defines a ``simulate`` function (detected via
    ``inspect.isfunction``) and ``force_prior_predictive`` is False, the
    perturbation-based path is used: wildtype reference parameters are drawn
    from ``SimPriors`` built from ``get_sim_hyperparameters()`` plus any
    ``sim_priors_overrides``.

    Otherwise the prior-predictive path is used: ``define_model`` is called
    inside a seeded NumPyro trace, and ``run_model`` produces ``theta_gc``.
    Set ``force_prior_predictive=True`` to always use this path — this is
    required for mutation-decomposed components (e.g. ``hill_mut``) when the
    pool library has only the wt genotype and M=0 mutations, because the
    perturbation path would produce zero deltas and a homogeneous pool.

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
    force_prior_predictive : bool, default False
        When True, always use the prior-predictive path (``define_model`` +
        ``handlers.seed``), even if the component provides ``simulate``.

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
    if not force_prior_predictive and inspect.isfunction(getattr(module, "simulate", None)):
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


def _greedy_maximin(theta_gc, n_select):
    """
    Select n_select rows from theta_gc that maximally span the theta-curve space.

    Uses greedy farthest-point selection (maximin): starts with the row of
    greatest dynamic range (max - min across concentrations), then repeatedly
    adds the row that maximises the minimum Euclidean distance to the
    already-selected set.

    Parameters
    ----------
    theta_gc : np.ndarray, shape (pool_size, n_conc)
    n_select : int

    Returns
    -------
    np.ndarray of int, shape (n_select,)
        Indices into theta_gc of the selected rows, in selection order.
    """
    pool_size = theta_gc.shape[0]
    if n_select >= pool_size:
        return np.arange(pool_size, dtype=int)

    theta_range = theta_gc.max(axis=1) - theta_gc.min(axis=1)
    selected = [int(np.argmax(theta_range))]

    while len(selected) < n_select:
        selected_arr = theta_gc[selected]                      # (k, C)
        # Squared distance from every pool member to its nearest selected point
        diff = theta_gc[:, None, :] - selected_arr[None, :, :]  # (pool, k, C)
        sq_dist = (diff ** 2).sum(axis=-1)                     # (pool, k)
        min_sq_dist = sq_dist.min(axis=1)                      # (pool,)
        min_sq_dist[selected] = -1.0                           # exclude already selected
        selected.append(int(np.argmax(min_sq_dist)))

    return np.array(selected, dtype=int)


def sample_theta_stratified(component_name,
                             binding_sample_df,
                             growth_sample_df,
                             rng_key,
                             n_select,
                             thermo_data=None,
                             pool_size=500,
                             priors_overrides=None,
                             sim_priors_overrides=None,
                             select_mode="stratified"):
    """
    Sample a pool of theta curves from the prior and select n_select that
    maximally span the theta-curve space (greedy maximin on binding concentrations).

    Builds an internal pool library of pool_size "wt" genotypes.  For per-genotype
    theta components (e.g. hill_geno), each pool member receives an independent
    i.i.d. draw from the prior.  Mutation-decomposed components are not supported.

    The same JAX rng_key is used for both the binding-concentration and
    growth-concentration sampling calls.  Because per-genotype parameter draws
    in the NumPyro trace depend only on plate sizes (not on concentration values),
    both calls produce the same underlying parameter sets evaluated at their
    respective concentrations, so the returned curves are mutually consistent.

    Parameters
    ----------
    component_name : str
        A key in model_registry["theta"].
    binding_sample_df : pd.DataFrame
        Must contain a "titrant_conc" column (mM).  These concentrations are
        used to select which parameter sets maximally span the binding space.
    growth_sample_df : pd.DataFrame
        Must contain a "titrant_conc" column (mM).  The selected parameter sets
        are also evaluated here so they can be injected into the growth simulation.
    rng_key : jax.random.PRNGKey
    n_select : int
        Number of genotypes to select (= number of calibration binding genotypes).
    thermo_data : str or None
        Forwarded to build_sim_data.
    pool_size : int, default 500
        Number of candidate parameter sets to draw before selection.
    priors_overrides : dict or None
        Forwarded to sample_theta_prior (prior-predictive path).
    sim_priors_overrides : dict or None
        Forwarded to sample_theta_prior (perturbation path).

    Returns
    -------
    binding_theta_gc : np.ndarray, shape (n_select, n_binding_concs)
        Selected theta curves at binding concentrations (sorted ascending).
    growth_theta_gc : np.ndarray, shape (n_select, n_growth_concs)
        Same selected parameter sets evaluated at growth concentrations.

    Raises
    ------
    ValueError
        If n_select > pool_size.
    """
    if n_select > pool_size:
        raise ValueError(
            f"n_select ({n_select}) must not exceed pool_size ({pool_size}). "
            f"Increase pool_size or reduce the number of binding genotypes."
        )

    if isinstance(rng_key, int):
        rng_key = jax.random.PRNGKey(rng_key)

    # Build sim_data for a single genotype at each concentration set.
    # The pool is built by calling sample_theta_prior once per member with a
    # unique key derived via fold_in.  This gives each member its own
    # hyperprior draw (loc, scale, …) rather than sharing one draw across all
    # pool members — which would concentrate the pool around a single binding
    # regime regardless of pool_size.
    single_library_df = pd.DataFrame({"genotype": ["wt"]})
    binding_sim_data = build_sim_data(single_library_df, binding_sample_df,
                                      thermo_data=thermo_data, skip_pairs=True)
    growth_sim_data = build_sim_data(single_library_df, growth_sample_df,
                                     thermo_data=thermo_data, skip_pairs=True)

    # Always use the prior-predictive path for pool building.  The perturbation
    # path (simulate()) requires mutations to generate diversity; with a wt-only
    # single-genotype pool library (M=0), it produces zero deltas and an
    # entirely homogeneous pool regardless of pool_size.  The prior-predictive
    # path samples WT-level parameters (log_K_wt, logit_low_wt, …) from their
    # broad Normal priors, giving real diversity across pool members.
    print(f"Building stratified pool ({pool_size} candidates)... ",
          end="", flush=True)
    pool_binding_rows = []
    pool_growth_rows = []
    for i in tqdm.tqdm(range(pool_size)):
        key_i = jax.random.fold_in(rng_key, i)
        theta_b, _ = sample_theta_prior(component_name, binding_sim_data, key_i,
                                        priors_overrides, sim_priors_overrides,
                                        force_prior_predictive=True)
        theta_g, _ = sample_theta_prior(component_name, growth_sim_data, key_i,
                                        priors_overrides, sim_priors_overrides,
                                        force_prior_predictive=True)
        pool_binding_rows.append(theta_b[0])  # (C_b,) — single genotype
        pool_growth_rows.append(theta_g[0])   # (C_g,)
    print("Done.", flush=True)

    pool_binding_gc = np.stack(pool_binding_rows)  # (pool_size, C_b)
    pool_growth_gc  = np.stack(pool_growth_rows)   # (pool_size, C_g)

    if select_mode == "stratified":
        selected_indices = _greedy_maximin(pool_binding_gc, n_select)
    elif select_mode == "random":
        # random draw from the prior pool (no maximin spread)
        rng = np.random.default_rng(int(jax.random.randint(
            rng_key, (), 0, 2**31 - 1)))
        selected_indices = rng.choice(pool_size, size=n_select, replace=False)
    else:
        raise ValueError(
            f"Unknown select_mode '{select_mode}' (expected 'stratified' or 'random')."
        )

    return pool_binding_gc[selected_indices], pool_growth_gc[selected_indices]
