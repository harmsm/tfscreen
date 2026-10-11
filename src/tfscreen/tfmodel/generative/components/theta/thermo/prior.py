"""
Structural prior on the per-mutation equilibrium-constant shifts Δln K.

This module provides the MODEL side of the PddG/PnnC structural prior.  The
structural input is a per-state ΔΔG prior mean (the ΔΔG table for PddG, the
structure MLP's output for PnnC), and the error model is per state: the true
ΔΔG of state s is its prior mean plus Normal(0, σ_s).  The data see only the
constants, Δln K = P · ΔΔG, with P the model's fixed projection (C × S; rows
are constants, columns are states, ``thermo/mwc.py``).  So the prior is put
on the constants, as the distribution that per-state model induces:

    σ_s[s]     ~ LogNormal(sigma_loc, sigma_scale)            (one per state)
    Δln K[m,:] ~ MultivariateNormal(P · prior_mean[m,:], P · diag(σ_s²) · Pᵀ)

``Δln K`` is the sampled per-mutation latent (site ``{name}_dlnK``, shape
M × C, constants in the model's order).  P includes every row the model
uses: the per-site effector rows (ΔΔG_X − ΔΔG_XE2)/2 and, in the U1 models,
the unfolding row Δln K_U = ΔΔG_H (ΔΔG_U = 0).

Why the constants and not the states
------------------------------------
From 2026-10-11 (commit c4f03d42) until this change the models sampled the
per-state ΔΔG[m, s] ~ Normal(prior_mean, σ_s) and projected it.  In the U0
models (no unfolded state) every constant is a difference of states, so a
shift equal across all states of one mutation changes no observable: P has
the null vector (1, …, 1).  Joint MAP used that direction to zero one
state's residuals and collapse its σ_s toward 0 (an unbounded density): on
a simulated C4_U0 MAP with true σ_s = 0.3 kT, σ_LE2 went to 0.001 and the
ΔΔG error got worse than the old non-centered form.  With the prior on
Δln K there is no per-state latent, so there is no unidentified direction
to exploit, and one σ_s → 0 leaves P · diag(σ²) · Pᵀ positive definite (the
null vector has no zero entry, so P minus any one column keeps full row
rank), so the density stays bounded in that direction.

In the U1 models P is square and invertible (the K_U row anchors the common
shift), so this is a pure change of variables from the per-state model.
Its Jacobian |det P| is constant, so the posterior and the MAP point,
mapped back by P⁻¹, are the same as the per-state model's
(``test_ddG_constant_prior.py`` checks it numerically).

Identifiability of σ_s
----------------------
σ_s enters only through the C × C covariance P · diag(σ²) · Pᵀ.  Its
entries are linear in σ², with coefficient P[i, s]·P[j, s] for state s, so
every σ_s is identified (from enough mutations) when the S rank-one
matrices p_s p_sᵀ (p_s the column of state s) are linearly independent.
They are in all four families: C4_U0 (H alone has a K_op × K_HL term, L
alone a K_HL × K_E term, HD and LE2 then from the K_op and K_E variances:
σ_H² = cov(K_op, K_HL), σ_L² = −2·cov(K_HL, K_E)); C12_U0 (H alone couples
K_h_l, K_h_o and K_h_e, L alone couples K_h_l, K_l_o and K_l_e, and HO, LO,
HE2, LE2 each sit in one variance); and the U1 families, where P is
invertible.  :func:`identifiable_sigma` computes the rank.

Post-hoc per-state ΔΔG
----------------------
The deterministic site ``{name}_ddG`` (M × S) keeps the per-state tables
meaningful:

* U1 (square P): ΔΔG = P⁻¹ · Δln K, exact, with ΔΔG_U = 0.
* U0: the conditional mean under the per-state prior,
  ΔΔG = μ + Σ Pᵀ (P Σ Pᵀ)⁻¹ (Δln K − P μ), Σ = diag(σ_s²), μ the prior
  mean.  The component along the null direction (a common shift of all
  states) is set by the prior, i.e. by the table, not by the data.  Score
  per-state ΔΔG only for U1 models; ``tfs-summarize-fit`` flags the U0
  tables and leaves them out of calibration.

When Δln K equals P · μ (the data agree with the table) both give ΔΔG = μ.

Prior on σ_s
------------
σ_s is a positive scale whose plausible range spans two decades (a
structural prediction good to 0.1 kT, or one off by 10 kT), so a prior
symmetric in log σ_s is the natural weak choice.  The defaults
(``theta_ddG_sigma_loc = 0``, median 1 kT, and ``theta_ddG_sigma_scale =
1.2``) put 95% of the mass on 0.095-10.5 kT.  With hundreds of mutations the
likelihood of the Δln K about their prior means outweighs this prior, so it
mainly matters for states the data barely inform.  The prior is centered:
the density of Δln K carries −M·log|det L| (L the Cholesky factor of the
covariance), which pins σ_s near the scatter of the fitted Δln K about the
prior means.  Until 2026-10-11 σ_s was a ``pyro.param`` with no prior in a
non-centered form, which MAP drove to 4-37 kT whatever the truth.

Known MAP limit.  The density is still unbounded where the covariance turns
singular with every residual inside the remaining subspace, so with weak
data (true σ_s below about √3 times the standard error of the direction
that measures it) a σ_s can drift toward 0 and the fit goes to the prior
means.  A σ_s near 0 from MAP means "not resolved", not "no noise".

Implementation
--------------
The Cholesky factor of P · diag(σ²) · Pᵀ is computed once per evaluation
from the QR decomposition of (P · diag(σ))ᵀ (no squaring of the condition
number), and the per-mutation MultivariateNormal shares that ``scale_tril``
across the M mutations (batch shape M, event shape C).  The cost is one
small QR plus M triangular solves of size C ≤ 6.  No non-centered option is
offered; for NUTS on weakly informed problems a reparameterization of
``{name}_dlnK`` can be added without changing this module.

Recorded sites
--------------
``{name}_ddG_sigma`` (S,) and ``{name}_dlnK`` (M, C) are sample sites;
``{name}_ddG_prior_mean`` (M, S) and ``{name}_ddG`` (M, S, the post-hoc
ΔΔG) are deterministic.  The state axis is in the order of the
``struct_names`` passed in (every model passes its ``STRUCTURE_NAMES``).
:func:`guide_dlnK` is the matching component-guide block,
:func:`dlnK_guesses` the starting point, and :func:`ddG_extract_specs` turns
the sites into ``tfs-extract-params`` tables.  The simulator draws its truth
per state and conditions ``{name}_dlnK`` on its projection
(:func:`per_state_truth`; ``simulate/sample_theta.py``).
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpyro as pyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints

# Default LogNormal prior on σ_s (kT): median exp(0) = 1 kT, 95% of the mass
# on exp(±1.96 · 1.2) = 0.095-10.5 kT.  The models' get_hyperparameters
# expose these as theta_ddG_sigma_loc / theta_ddG_sigma_scale.
DEFAULT_DDG_SIGMA_LOC = 0.0
DEFAULT_DDG_SIGMA_SCALE = 1.2

# Starting SD of the component guide's LogNormal on σ_s (log units).
_GUIDE_SIGMA_SCALE_INIT = 0.1


def ddG_sigma_site(name="theta"):
    """Name of the per-state σ_s sample site."""
    return f"{name}_ddG_sigma"


def dlnK_site(name="theta"):
    """Name of the per-mutation Δln K sample site (M × C)."""
    return f"{name}_dlnK"


def projection(proj, proj_u=None):
    """
    The full projection from per-state ΔΔG to the sampled constants.

    Parameters
    ----------
    proj : array, shape (C0, S)
        The folded-state rows (``mwc.PROJ_C4`` or ``mwc.PROJ_C12``).
    proj_u : array, shape (S,), optional
        The unfolding row of a U1 model (``mwc.PROJ_U_*``), appended last.

    Returns
    -------
    np.ndarray, shape (C, S)
    """
    P = np.asarray(proj, dtype=float)
    if proj_u is not None:
        P = np.vstack([P, np.asarray(proj_u, dtype=float).reshape(1, -1)])
    return P


def ddG_per_state_identified(proj):
    """True when per-state ΔΔG follows from Δln K exactly (P square, invertible)."""
    P = np.asarray(proj, dtype=float)
    return bool(P.shape[0] == P.shape[1]
                and np.linalg.matrix_rank(P) == P.shape[0])


def identifiable_sigma(proj):
    """
    Whether every σ_s is identified by the covariance P · diag(σ²) · Pᵀ.

    The covariance's upper-triangle entries are linear in σ², with
    coefficient P[i, s]·P[j, s]; σ is identified when that (C(C+1)/2 × S)
    matrix has rank S.

    Returns
    -------
    bool
    """
    P = np.asarray(proj, dtype=float)
    C, S = P.shape
    iu = np.triu_indices(C)
    design = np.stack([np.outer(P[:, s], P[:, s])[iu] for s in range(S)],
                      axis=1)
    return bool(np.linalg.matrix_rank(design) == S)


def constant_scale_tril(proj, sigma):
    """
    Lower Cholesky factor of P · diag(σ²) · Pᵀ.

    Computed from the QR decomposition of A = (P · diag(σ))ᵀ (S × C,
    S ≥ C): A = Q R gives Aᵀ A = Rᵀ R, so L = Rᵀ with the signs of R's rows
    made positive.  This avoids forming the covariance and squaring its
    condition number.

    Parameters
    ----------
    proj : array, shape (C, S)
    sigma : array, shape (S,)

    Returns
    -------
    jnp.ndarray, shape (C, C)
    """
    A = (jnp.asarray(proj) * sigma[None, :]).T          # (S, C)
    _, r = jnp.linalg.qr(A)                             # r: (C, C)
    d = jnp.diagonal(r)
    sign = jnp.where(d < 0, -1.0, 1.0)
    return (r * sign[:, None]).T


def posthoc_ddG(dlnK, prior_means, sigma, proj):
    """
    Per-state ΔΔG implied by Δln K.

    U1 (square P): ΔΔG = P⁻¹ · Δln K exactly.  U0: the conditional mean
    under the per-state prior, μ + Σ Pᵀ (P Σ Pᵀ)⁻¹ (Δln K − P μ) with
    Σ = diag(σ²); its common-mode component comes from the prior means.

    Parameters
    ----------
    dlnK : array, shape (M, C)
    prior_means : array, shape (M, S)
    sigma : array, shape (S,)
    proj : np.ndarray, shape (C, S)
        A concrete (non-traced) array.

    Returns
    -------
    jnp.ndarray, shape (M, S)
    """
    P = np.asarray(proj, dtype=float)
    dlnK = jnp.asarray(dlnK)
    if ddG_per_state_identified(P):
        return dlnK @ jnp.asarray(np.linalg.inv(P).T, dtype=dlnK.dtype)
    Pj = jnp.asarray(P, dtype=dlnK.dtype)
    resid = dlnK - prior_means @ Pj.T                   # (M, C)
    L = constant_scale_tril(Pj, sigma)
    x = jsl.cho_solve((L, True), resid.T).T             # (M, C)
    return prior_means + (x @ Pj) * (sigma ** 2)[None, :]


def sample_dlnK(name, struct_names, num_mut, prior_means, proj,
                sigma_loc=DEFAULT_DDG_SIGMA_LOC,
                sigma_scale=DEFAULT_DDG_SIGMA_SCALE):
    """
    Sample σ_s per state and the per-mutation Δln K under the structural prior.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sites.
    struct_names : sequence of str, length S
        State names (only S is used).
    num_mut : int
        Number of mutations M.
    prior_means : jnp.ndarray, shape (M, S)
        Per-state ΔΔG prior means (the ΔΔG table or the MLP output).
    proj : array, shape (C, S)
        Projection from per-state ΔΔG to the constants (:func:`projection`).
    sigma_loc, sigma_scale : float
        LogNormal prior on σ_s, in log kT (the models' priors
        ``theta_ddG_sigma_loc`` and ``theta_ddG_sigma_scale``).

    Returns
    -------
    jnp.ndarray, shape (M, C)
        Sampled Δln K, constants in the row order of ``proj``.
    """
    S = len(struct_names)
    proj = np.asarray(proj, dtype=float)    # concrete, also under jit
    P = jnp.asarray(proj, dtype=jnp.float32)
    C = P.shape[0]

    with pyro.plate(f"{name}_struct_plate", S, dim=-1):
        sigma_s = pyro.sample(ddG_sigma_site(name),
                              dist.LogNormal(sigma_loc, sigma_scale))   # (S,)

    pyro.deterministic(f"{name}_ddG_prior_mean", prior_means)

    # A wt-only library (num_mut == 0, e.g. a stratified binding pool)
    # skips the site, since numpyro.plate needs size > 0; the guides skip
    # it the same way.
    if num_mut == 0:
        pyro.deterministic(f"{name}_ddG", jnp.zeros((0, S)))
        return jnp.zeros((0, C))

    loc = prior_means @ P.T                                   # (M, C)
    scale_tril = constant_scale_tril(P, sigma_s)              # (C, C)
    # validate_args=False lets the simulator condition σ_s to exactly 0
    # (truth = table), where scale_tril is singular; a fit only ever sees
    # σ_s > 0 from the LogNormal.
    with pyro.plate(f"{name}_mut_plate", num_mut, dim=-1):
        dlnK = pyro.sample(dlnK_site(name),
                           dist.MultivariateNormal(loc, scale_tril=scale_tril,
                                                   validate_args=False))  # (M, C)

    pyro.deterministic(f"{name}_ddG",
                       posthoc_ddG(dlnK, prior_means, sigma_s, proj))
    return dlnK


def guide_dlnK(name, num_mut, num_struct, proj, prior_means_init):
    """
    Component-guide block matching :func:`sample_dlnK`.

    Mean field: σ_s ~ LogNormal(``{name}_ddG_sigma_locs``,
    ``{name}_ddG_sigma_scales``) per state, and Δln K ~
    Normal(``{name}_dlnK_locs``, ``{name}_dlnK_scales``) per mutation and
    constant (an event of size C, matching the model's MultivariateNormal).
    The names follow the ``{site}_loc(s)``/``{site}_scale(s)`` convention
    that ``RunInference.component_guide_start`` maps a MAP onto.

    Parameters
    ----------
    name : str
        Site-name prefix.
    num_mut, num_struct : int
        M and S.
    proj : array, shape (C, S)
    prior_means_init : array, shape (M, S)
        Per-state prior means at the start of the fit; the Δln K locations
        start at P · prior_means_init.

    Returns
    -------
    jnp.ndarray, shape (M, C)
        The guide's Δln K draw.
    """
    S = num_struct
    P = jnp.asarray(proj, dtype=jnp.float32)
    C = P.shape[0]
    sigma_locs = pyro.param(f"{name}_ddG_sigma_locs", jnp.zeros(S))
    sigma_scales = pyro.param(f"{name}_ddG_sigma_scales",
                              jnp.full(S, _GUIDE_SIGMA_SCALE_INIT),
                              constraint=constraints.positive)
    with pyro.plate(f"{name}_struct_plate", S, dim=-1):
        pyro.sample(ddG_sigma_site(name),
                    dist.LogNormal(sigma_locs, sigma_scales))
    if num_mut == 0:
        return jnp.zeros((0, C))
    loc_init = jnp.asarray(prior_means_init, dtype=jnp.float32) @ P.T
    dlnK_locs = pyro.param(f"{name}_dlnK_locs", loc_init)
    dlnK_scales = pyro.param(f"{name}_dlnK_scales", jnp.ones((num_mut, C)),
                             constraint=constraints.positive)
    with pyro.plate(f"{name}_mut_plate", num_mut, dim=-1):
        return pyro.sample(dlnK_site(name),
                           dist.Normal(dlnK_locs, dlnK_scales).to_event(1))


def dlnK_guesses(name, prior_means, proj):
    """
    Starting values: Δln K at P · prior means, σ_s at 1 kT.

    Parameters
    ----------
    name : str
        Site-name prefix.
    prior_means : array, shape (M, S)
        The per-state prior means at the start of the fit (the ΔΔG table for
        PddG; zeros for PnnC, whose MLP output starts at 0).
    proj : array, shape (C, S)
    """
    prior_means = jnp.asarray(prior_means, dtype=jnp.float32)
    P = jnp.asarray(proj, dtype=jnp.float32)
    return {dlnK_site(name): prior_means @ P.T,
            ddG_sigma_site(name): jnp.ones(prior_means.shape[1])}


def per_state_truth(prior_means, sigma, proj, rng_key):
    """
    Draw per-state ΔΔG truth and its projection (simulator).

    ΔΔG = prior_mean + σ_s · z, z ~ Normal(0, 1) per mutation and state, and
    Δln K = P · ΔΔG, which is distributed exactly as the model's
    MultivariateNormal on Δln K.  The simulator conditions ``{name}_dlnK``
    on the projection, so the per-state truth it writes is the draw itself
    (not the post-hoc ΔΔG, which in U0 cannot recover the common shift).

    Parameters
    ----------
    prior_means : array, shape (M, S)
    sigma : array, shape (S,)
        σ_s; 0 gives the prior means exactly.
    proj : array, shape (C, S)
    rng_key : jax.random.PRNGKey

    Returns
    -------
    ddG : np.ndarray, shape (M, S)
    dlnK : np.ndarray, shape (M, C)
    """
    prior_means = np.asarray(prior_means, dtype=float)
    z = np.asarray(jax.random.normal(rng_key, prior_means.shape))
    ddG = prior_means + np.asarray(sigma, dtype=float)[None, :] * z
    dlnK = ddG @ np.asarray(proj, dtype=float).T
    return ddG.astype(np.float32), dlnK.astype(np.float32)


def ddG_extract_specs(ctx, struct_names, name="theta"):
    """
    ``get_extract_specs`` entries for the per-state ΔΔG sites.

    The (M, S) deterministic sites ``{name}_ddG`` (post-hoc ΔΔG; see the
    module docstring) and ``{name}_ddG_prior_mean`` become one table per
    state, ``ddG_{structure}`` and ``ddG_prior_mean_{structure}``
    (``tfs-extract-params`` writes ``{out_prefix}_ddG_H.csv`` and so on),
    keyed by ``mutation``.  The names match the simulator's
    ``ddG_{structure}`` truth columns in ``*_sim_mutation_parameters.csv``,
    which ``tfs-summarize-fit`` joins on ``mutation`` (and flags for U0
    models, whose per-state ΔΔG common mode comes from the prior).  The (S,)
    site ``{name}_ddG_sigma`` becomes one table, ``ddG_sigma``, with one row
    per state keyed by ``structure``; it matches the simulator's
    ``*_sim_ddG_sigma.csv`` truth.

    Parameters
    ----------
    ctx : ExtractionContext
        Supplies ``mut_labels``.
    struct_names : sequence of str
        The state axis of the sites, i.e. the model's ``STRUCTURE_NAMES``.
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
        # Separate specs, so a posterior file without one of the sites
        # still gives the other's tables.
        for site in ("ddG", "ddG_prior_mean"):
            specs.append(dict(
                input_df=mut_df,
                params_to_get=[site],
                out_names=[f"{site}_{struct}"],
                map_column="map_ddG",
                get_columns=["mutation"],
                in_run_prefix=f"{name}_",
            ))
    specs.append(dict(
        input_df=pd.DataFrame({"structure": list(struct_names),
                               "map_struct": range(S)}),
        params_to_get=["ddG_sigma"],
        out_names=["ddG_sigma"],
        map_column="map_struct",
        get_columns=["structure"],
        in_run_prefix=f"{name}_",
    ))
    return specs
