"""
Per-mutation per-structure ΔΔG sampling with NN-predicted prior means.

This module provides the MODEL side of the ΔΔG prior.  Each ΔΔG[m, s] is
a latent variable sampled as:

    ΔΔG[m, s] = nn_means[m, s]  +  σ_s[s] · offset[m, s]
    offset[m, s] ~ Normal(0, 1)

where σ_s is a per-structure trust scale learned via ``pyro.param``.

Interpretation of σ_s
----------------------
σ_s controls how tightly the prior is centred on the NN prediction:
  σ_s → 0  :  ΔΔG collapses to the deterministic NN output
  σ_s → ∞  :  prior becomes flat; posterior is driven by the likelihood
              alone (like lnK_mut with flat sigma prior)

Initialising σ_s = 1.0 places the model between these extremes and lets
the data learn how predictive each structure's LigandMPNN features are.

σ_s (and the PnnC NN weights) are empirical-Bayes point estimates: after a
fit, posterior sampling, the Laplace and predictions hold them at their
fitted values (``tfscreen.tfmodel.inference.model_params``); the Laplace
does not include them in the Hessian.

The guide (in lnK_nn_prior.py) must provide matching ``pyro.sample`` sites
for all ``{name}_ddG_offset`` variables sampled here.

Recorded sites
--------------
``{name}_ddG`` (M, S) is the sampled ΔΔG and ``{name}_ddG_prior_mean``
(M, S) its prior mean (the CSV values for PddG, the MLP output for PnnC),
both with the structure axis in the order of the ``struct_names`` passed in.
Every PddG/PnnC model passes its ``STRUCTURE_NAMES`` order (PnnC C4/U1
requires the data in that order; the others permute the data's columns into
it), the order the simulator labels its ``ddG_{structure}`` truth with.
:func:`ddG_extract_specs` turns both into one ``tfs-extract-params`` table per
structure.
"""

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints


def sample_ddG(name, struct_names, num_mut, nn_means):
    """
    Sample per-mutation per-structure ΔΔG values with NN-predicted prior means.

    The per-structure trust scale σ_s is learned via ``pyro.param`` and is
    shared across all mutations within a structure (but differs between
    structures).

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sites and params.
    struct_names : sequence of str, length S
        Structure identifiers (used only to determine S).
    num_mut : int
        Number of mutations M.
    nn_means : jnp.ndarray, shape (M, S)
        NN-predicted prior means for each mutation and structure.

    Returns
    -------
    jnp.ndarray, shape (M, S)
        Sampled ΔΔG values.
    """
    S = len(struct_names)

    # Per-structure trust scale — learned, not sampled.
    # Initialised to 1.0 so the model starts between deterministic-NN and flat.
    sigma_s = pyro.param(
        f"{name}_ddG_sigma_s",
        jnp.ones(S),
        constraint=constraints.positive,
    )  # (S,)

    # Sample offsets in nested plates → shape (S, M).  A wt-only library
    # (num_mut == 0, e.g. a stratified binding pool) skips the site, since
    # numpyro.plate needs size > 0; the guides skip it the same way.
    if num_mut > 0:
        with pyro.plate(f"{name}_struct_plate", S, dim=-2):
            with pyro.plate(f"{name}_mut_plate", num_mut, dim=-1):
                offsets = pyro.sample(
                    f"{name}_ddG_offset",
                    dist.Normal(0.0, 1.0),
                )  # (S, M)
    else:
        offsets = jnp.zeros((S, 0))

    # Apply per-structure scale and add NN prior means
    # sigma_s[:, None] broadcasts (S,1) over (S, M)
    # nn_means.T converts (M, S) → (S, M) for consistent (S, M) arithmetic
    ddG_SM = nn_means.T + sigma_s[:, None] * offsets   # (S, M)

    pyro.deterministic(f"{name}_ddG", ddG_SM.T)
    pyro.deterministic(f"{name}_ddG_prior_mean", nn_means)
    return ddG_SM.T   # (M, S)


def ddG_extract_specs(ctx, struct_names, name="theta"):
    """
    ``get_extract_specs`` entries for the per-structure ΔΔG sites.

    The (M, S) sites ``{name}_ddG`` and ``{name}_ddG_prior_mean`` become one
    table per structure, ``ddG_{structure}`` and
    ``ddG_prior_mean_{structure}`` (``tfs-extract-params`` writes
    ``{out_prefix}_ddG_H.csv`` and so on), keyed by ``mutation``.  The names
    match the simulator's ``ddG_{structure}`` truth columns in
    ``*_sim_mutation_parameters.csv``, which ``tfs-summarize-fit`` joins on
    ``mutation``.

    Parameters
    ----------
    ctx : ExtractionContext
        Supplies ``mut_labels``.
    struct_names : sequence of str
        The structure axis of the sites, i.e. the model's
        ``STRUCTURE_NAMES`` (the order every PddG/PnnC model samples in).
    name : str, default "theta"
        Site-name prefix.
    """
    import pandas as pd

    mut_labels = list(ctx.mut_labels)
    num_mut, S = len(mut_labels), len(struct_names)
    specs = []
    for s, struct in enumerate(struct_names):
        # Row-major flattening of (M, S): element (m, s) is m * S + s.
        mut_df = pd.DataFrame({
            "mutation": mut_labels,
            "map_ddG": [m * S + s for m in range(num_mut)],
        })
        # Separate specs, so a posterior file written before the prior-mean
        # site existed still gives the ddG tables.
        for site in ("ddG", "ddG_prior_mean"):
            specs.append(dict(
                input_df=mut_df,
                params_to_get=[site],
                out_names=[f"{site}_{struct}"],
                map_column="map_ddG",
                get_columns=["mutation"],
                in_run_prefix=f"{name}_",
            ))
    return specs
