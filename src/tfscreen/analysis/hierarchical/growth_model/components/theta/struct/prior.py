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
  σ_s → 0  :  ΔΔG collapses to the deterministic NN output (like nn_mut)
  σ_s → ∞  :  prior becomes flat; posterior is driven by the likelihood
              alone (like lnK_mut with flat sigma prior)

Initialising σ_s = 1.0 places the model between these extremes and lets
the data learn how predictive each structure's LigandMPNN features are.

The guide (in lnK_nn_prior.py) must provide matching ``pyro.sample`` sites
for all ``{name}_ddG_offset`` variables sampled here.
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

    # Sample offsets in nested plates → shape (S, M)
    with pyro.plate(f"{name}_struct_plate", S, dim=-2):
        with pyro.plate(f"{name}_mut_plate", num_mut, dim=-1):
            offsets = pyro.sample(
                f"{name}_ddG_offset",
                dist.Normal(0.0, 1.0),
            )  # (S, M)

    # Apply per-structure scale and add NN prior means
    # sigma_s[:, None] broadcasts (S,1) over (S, M)
    # nn_means.T converts (M, S) → (S, M) for consistent (S, M) arithmetic
    ddG_SM = nn_means.T + sigma_s[:, None] * offsets   # (S, M)

    pyro.deterministic(f"{name}_ddG", ddG_SM.T)
    return ddG_SM.T   # (M, S)
