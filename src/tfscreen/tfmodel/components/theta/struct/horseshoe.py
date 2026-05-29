"""
Distance-dependent regularised horseshoe prior for per-structure pairwise
epistasis ΔΔG values.

For each mutation pair p and structure s, the epistasis ΔΔG is:

    epi[p, s] = offset[p, s] · τ · λ̃[p, s]

where τ is a global scale (shared across all structures), λ̃ is the
regularised local scale:

    λ̃[p, s] = sqrt( c² · λ²[p,s]  /  (c² + τ² · λ²[p,s]) )

and the prior on the local scale is distance-dependent:

    λ[p, s] ~ HalfCauchy( exp(-dist[p, s] / d₀) )

Pairs with large inter-residue distance (dist[p, s] → ∞) have
scale ≈ 0, strongly shrinking their epistasis toward zero.  Close contacts
(dist[p, s] ≈ 0) have scale ≈ 1, allowing large epistasis if the data
support it.

The guide (in lnK_nn_prior.py) must provide matching ``pyro.sample`` sites
for all variables sampled here.

Reference
---------
Piironen & Vehtari, 2017. "Sparsity information and regularization in the
horseshoe and other shrinkage priors." Electronic Journal of Statistics.
"""

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

_DEFAULT_D0 = 8.0  # Å — residues within ~8 Å Cα-Cα are considered contacts


def sample_pair_ddG(name, struct_names, contact_distances,
                    tau_scale=0.1, slab_scale=2.0, slab_df=4.0,
                    d0=_DEFAULT_D0):
    """
    Sample pairwise epistasis ΔΔG with a distance-dependent horseshoe prior.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sites.
    struct_names : sequence of str, length S
        Structure identifiers (used to determine S).
    contact_distances : jnp.ndarray, shape (P, S)
        Min Cα-Cα distance in Å per pair per structure.
        999.0 indicates no contact in that structure.
    tau_scale : float
        HalfCauchy scale for the global shrinkage parameter τ.
    slab_scale : float
        Typical magnitude of a large epistasis effect (slab scale c).
    slab_df : float
        InverseGamma shape ν for the slab variance c² prior.
    d0 : float
        Distance scale in Å.  Contacts closer than d₀ receive near-unit
        prior scale on λ; contacts farther than d₀ are increasingly shrunk.

    Returns
    -------
    jnp.ndarray, shape (P, S)
        Sampled epistasis ΔΔG per pair per structure.
    """
    S = len(struct_names)
    dists = jnp.array(contact_distances)   # (P, S)
    P = dists.shape[0]

    # Global shrinkage scale and slab variance (shared across all structures)
    tau = pyro.sample(f"{name}_epi_tau",
                      dist.HalfCauchy(tau_scale))
    c2  = pyro.sample(
        f"{name}_epi_c2",
        dist.InverseGamma(slab_df / 2.0,
                          slab_df * slab_scale ** 2 / 2.0),
    )

    # Distance-dependent prior scale for local shrinkage:
    # close contacts (small dist) → scale ≈ 1; distant → scale → 0
    lam_scale = jnp.exp(-dists / d0)    # (P, S)

    # Sample per-(pair, struct) local scales and offsets.
    # Plates: struct dim=-2 (outer), pair dim=-1 (inner) → shapes (S, P).
    # lam_scale is (P, S); transposing to (S, P) aligns with plate layout.
    with pyro.plate(f"{name}_struct_epi_plate", S, dim=-2):
        with pyro.plate(f"{name}_pair_plate", P, dim=-1):
            lam    = pyro.sample(
                f"{name}_epi_lambda",
                dist.HalfCauchy(lam_scale.T),   # (S, P)
            )
            offset = pyro.sample(
                f"{name}_epi_offset",
                dist.Normal(0.0, 1.0),           # (S, P)
            )

    # Regularised horseshoe: clamp variance contribution from τ
    lam_tilde = jnp.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))  # (S, P)
    epi_SP    = offset * tau * lam_tilde                                  # (S, P)

    pyro.deterministic(f"{name}_epi_ddG", epi_SP.T)
    return epi_SP.T   # (P, S)
