import numpy as np
import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Dict, Any, Optional

from tfscreen.tfmodel.data_class import DataClass


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the Hill model priors.

    Attributes
    ----------
    theta_logit_low_hyper_loc_loc : float
    theta_logit_low_hyper_loc_scale : float
    theta_logit_low_hyper_scale : float
    theta_logit_delta_hyper_loc_loc : float
    theta_logit_delta_hyper_loc_scale : float
    theta_logit_delta_hyper_scale : float
    theta_log_hill_K_hyper_loc_loc : float
    theta_log_hill_K_hyper_loc_scale : float
    theta_log_hill_K_hyper_scale : float
    theta_log_hill_n_hyper_loc_loc : float
    theta_log_hill_n_hyper_loc_scale : float
    theta_log_hill_n_hyper_scale : float
    """

    theta_logit_low_hyper_loc_loc: float
    theta_logit_low_hyper_loc_scale: float
    theta_logit_low_hyper_scale: float
    theta_logit_delta_hyper_loc_loc: float
    theta_logit_delta_hyper_loc_scale: float
    theta_logit_delta_hyper_scale: float

    theta_log_hill_K_hyper_loc_loc: float
    theta_log_hill_K_hyper_loc_scale: float
    theta_log_hill_K_hyper_scale: float

    theta_log_hill_n_hyper_loc_loc: float
    theta_log_hill_n_hyper_loc_scale: float
    theta_log_hill_n_hyper_scale: float


@dataclass(frozen=True)
class ThetaParam:
    """
    JAX Pytree holding the sampled Hill equation parameters.

    Per-genotype fields have shape ``(num_titrant_name, num_genotype)``.
    Population moment fields have shape ``(num_titrant_name, num_titrant_conc, 1)``.

    Attributes
    ----------
    theta_low : jnp.ndarray
        The minimum fractional occupancy (baseline), shape (T, G).
    theta_high : jnp.ndarray
        The maximum fractional occupancy (saturation), shape (T, G).
    log_hill_K : jnp.ndarray
        The Hill constant (K_D) in log-space, shape (T, G).
    hill_n : jnp.ndarray
        The Hill coefficient, shape (T, G).
    mu : jnp.ndarray
        Population mean of logit(theta) at each concentration, shape (T, C, 1).
    sigma : jnp.ndarray
        Population std of logit(theta) at each concentration, shape (T, C, 1).
    """

    theta_low: jnp.ndarray
    theta_high: jnp.ndarray
    log_hill_K: jnp.ndarray
    hill_n: jnp.ndarray
    mu: Optional[jnp.ndarray]
    sigma: Optional[jnp.ndarray]


@dataclass(frozen=True)
class SimPriors:
    """
    Parameters controlling perturbation-based theta simulation for hill_geno.

    Genotypes are drawn from one of four phenotype categories:

    * **normal** (probability ``1 - p_stuck_bound - p_never_binds - p_inverted``):
      Hill curve perturbed around the wildtype reference.
    * **stuck-bound** (``p_stuck_bound``): theta ≈ ``wt_theta_low`` throughout.
    * **never-binds** (``p_never_binds``): theta ≈ ``wt_theta_high`` throughout.
    * **inverted** (``p_inverted``): curve runs from ``wt_theta_high`` to ``wt_theta_low``.

    Attributes
    ----------
    wt_theta_low : float
        Wildtype theta at zero ligand concentration.
    wt_theta_high : float
        Wildtype theta at saturating ligand concentration.
    wt_log_K : float
        Wildtype log(K_D); sets the Hill-curve midpoint.
    wt_hill_n : float
        Wildtype Hill coefficient (cooperativity).
    sigma_logit_low : float
        Std dev of additive Normal noise on logit(theta_low) for normal genotypes.
    sigma_logit_delta : float
        Std dev of additive Normal noise on logit_delta (= logit_high - logit_low)
        for normal and inverted genotypes.
    sigma_log_K : float
        Std dev of additive Normal noise on log_K for all genotypes.
    sigma_log_n : float
        Std dev of additive Normal noise on log(hill_n) for all genotypes.
    p_stuck_bound : float
        Fraction of genotypes assigned to the stuck-bound category.
    p_never_binds : float
        Fraction of genotypes assigned to the never-binds category.
    p_inverted : float
        Fraction of genotypes assigned to the inverted category.
    """

    wt_theta_low: float
    wt_theta_high: float
    wt_log_K: float
    wt_hill_n: float

    sigma_logit_low: float
    sigma_logit_delta: float
    sigma_log_K: float
    sigma_log_n: float

    p_stuck_bound: float
    p_never_binds: float
    p_inverted: float


def _population_moments(
    logit_low_hyper_loc,    # (T,)
    logit_low_hyper_scale,  # (T,)
    logit_delta_hyper_loc,    # (T,)
    logit_delta_hyper_scale,  # (T,)
    log_K_hyper_loc,    # (T,)
    log_K_hyper_scale,  # (T,)
    log_n_hyper_loc,    # (T,)
    log_n_hyper_scale,  # (T,)
    log_titrant_conc,   # (C,)
):
    """
    Compute per-concentration population moments of logit(theta) using a ghost
    population sampled from the per-titrant hyperpriors.

    Returns
    -------
    mu : jnp.ndarray, shape (T, C, 1)
    sigma : jnp.ndarray, shape (T, C, 1)
    """
    n_ghost = 100
    eps = 1e-6

    # Ghost population: shape (T, n_ghost)
    ghost_low   = (logit_low_hyper_loc[:, None]
                   + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(0), (n_ghost,))
                   * logit_low_hyper_scale[:, None])
    ghost_delta = (logit_delta_hyper_loc[:, None]
                   + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(1), (n_ghost,))
                   * logit_delta_hyper_scale[:, None])
    ghost_high  = ghost_low + ghost_delta
    ghost_K     = (log_K_hyper_loc[:, None]
                   + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(2), (n_ghost,))
                   * log_K_hyper_scale[:, None])
    ghost_n     = (log_n_hyper_loc[:, None]
                   + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(3), (n_ghost,))
                   * log_n_hyper_scale[:, None])

    # Broadcast to (T, C, n_ghost)
    log_conc = log_titrant_conc[None, :, None]  # (1, C, 1)
    g_low  = ghost_low[:, None, :]   # (T, 1, n_ghost)
    g_high = ghost_high[:, None, :]
    g_K    = ghost_K[:, None, :]
    g_n    = jnp.exp(ghost_n[:, None, :])

    sig = dist.transforms.SigmoidTransform()
    g_occ   = jax.nn.sigmoid(g_n * (log_conc - g_K))
    g_theta = jnp.clip(sig(g_low) + (sig(g_high) - sig(g_low)) * g_occ, eps, 1.0 - eps)
    g_logit = jax.scipy.special.logit(g_theta)  # (T, C, n_ghost)

    mu    = jnp.mean(g_logit, axis=-1, keepdims=True)  # (T, C, 1)
    sigma = jnp.std(g_logit,  axis=-1, keepdims=True)  # (T, C, 1)
    return mu, sigma


def define_model(name: str,
                 data: DataClass,
                 priors: ModelPriors) -> ThetaParam:
    """
    Defines the hierarchical Hill model parameters with per-titrant hyperpriors.

    Hyperpriors are plated over ``num_titrant_name`` so each titrant gets its
    own pooled Hill-curve population.  Per-genotype offsets are sampled for the
    full genotype population (not a mini-batch), producing a ThetaParam whose
    last dimension is ``num_genotype``.  This matches the shape contract of
    ``hill_mut``: ``run_model`` selects genotypes via
    ``data.batch_idx[data.geno_theta_idx]``.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample sites.
    data : DataClass
        Must expose ``num_titrant_name``, ``num_genotype``, and
        ``log_titrant_conc``.
    priors : ModelPriors

    Returns
    -------
    ThetaParam
        Per-genotype fields shape ``(num_titrant_name, num_genotype)``;
        population moment fields shape ``(num_titrant_name, num_titrant_conc, 1)``.
    """
    T = data.num_titrant_name

    # ------------------------------------------------------------------
    # Per-titrant hyperpriors: shape (T,)
    # ------------------------------------------------------------------
    with pyro.plate(f"{name}_hyper_plate", T, dim=-1):
        logit_low_hyper_loc = pyro.sample(
            f"{name}_logit_low_hyper_loc",
            dist.Normal(priors.theta_logit_low_hyper_loc_loc,
                        priors.theta_logit_low_hyper_loc_scale))
        logit_low_hyper_scale = pyro.sample(
            f"{name}_logit_low_hyper_scale",
            dist.HalfNormal(priors.theta_logit_low_hyper_scale))
        logit_delta_hyper_loc = pyro.sample(
            f"{name}_logit_delta_hyper_loc",
            dist.Normal(priors.theta_logit_delta_hyper_loc_loc,
                        priors.theta_logit_delta_hyper_loc_scale))
        logit_delta_hyper_scale = pyro.sample(
            f"{name}_logit_delta_hyper_scale",
            dist.HalfNormal(priors.theta_logit_delta_hyper_scale))
        log_K_hyper_loc = pyro.sample(
            f"{name}_log_hill_K_hyper_loc",
            dist.Normal(priors.theta_log_hill_K_hyper_loc_loc,
                        priors.theta_log_hill_K_hyper_loc_scale))
        log_K_hyper_scale = pyro.sample(
            f"{name}_log_hill_K_hyper_scale",
            dist.HalfNormal(priors.theta_log_hill_K_hyper_scale))
        log_n_hyper_loc = pyro.sample(
            f"{name}_log_hill_n_hyper_loc",
            dist.Normal(priors.theta_log_hill_n_hyper_loc_loc,
                        priors.theta_log_hill_n_hyper_loc_scale))
        log_n_hyper_scale = pyro.sample(
            f"{name}_log_hill_n_hyper_scale",
            dist.HalfNormal(priors.theta_log_hill_n_hyper_scale))

    # ------------------------------------------------------------------
    # Full-population per-genotype offsets: shape (T, G)
    # ------------------------------------------------------------------
    with pyro.plate(f"{name}_titrant_name_plate", T, dim=-2):
        with pyro.plate(f"{name}_genotype_plate", data.num_genotype, dim=-1):
            logit_low_offset   = pyro.sample(f"{name}_logit_low_offset",   dist.Normal(0.0, 1.0))
            logit_delta_offset = pyro.sample(f"{name}_logit_delta_offset", dist.Normal(0.0, 1.0))
            log_K_offset       = pyro.sample(f"{name}_log_hill_K_offset",  dist.Normal(0.0, 1.0))
            log_n_offset       = pyro.sample(f"{name}_log_hill_n_offset",  dist.Normal(0.0, 1.0))

    # ------------------------------------------------------------------
    # Assemble per-genotype parameters: shape (T, G)
    # ------------------------------------------------------------------
    logit_theta_low   = logit_low_hyper_loc[:, None]   + logit_low_offset   * logit_low_hyper_scale[:, None]
    logit_theta_delta = logit_delta_hyper_loc[:, None] + logit_delta_offset * logit_delta_hyper_scale[:, None]
    logit_theta_high  = logit_theta_low + logit_theta_delta
    log_hill_K        = log_K_hyper_loc[:, None] + log_K_offset * log_K_hyper_scale[:, None]
    log_hill_n        = log_n_hyper_loc[:, None] + log_n_offset * log_n_hyper_scale[:, None]

    sig = dist.transforms.SigmoidTransform()
    theta_low  = sig(logit_theta_low)
    theta_high = sig(logit_theta_high)
    hill_n     = jnp.exp(log_hill_n)

    pyro.deterministic(f"{name}_theta_low",  theta_low)
    pyro.deterministic(f"{name}_theta_high", theta_high)
    pyro.deterministic(f"{name}_log_hill_K", log_hill_K)
    pyro.deterministic(f"{name}_hill_n",     hill_n)

    mu, sigma = _population_moments(
        logit_low_hyper_loc, logit_low_hyper_scale,
        logit_delta_hyper_loc, logit_delta_hyper_scale,
        log_K_hyper_loc, log_K_hyper_scale,
        log_n_hyper_loc, log_n_hyper_scale,
        data.log_titrant_conc,
    )

    return ThetaParam(theta_low=theta_low, theta_high=theta_high,
                      log_hill_K=log_hill_K, hill_n=hill_n,
                      mu=mu, sigma=sigma)


def guide(name: str,
          data: DataClass,
          priors: ModelPriors) -> ThetaParam:
    """
    Guide corresponding to the hierarchical Hill model.

    Mirrors the structure of ``define_model``: per-titrant variational
    parameters for hyperpriors, full-population variational parameters for
    per-genotype offsets.
    """
    T = data.num_titrant_name
    G = data.num_genotype

    # ------------------------------------------------------------------
    # Per-titrant hyperprior variational parameters: shape (T,)
    # ------------------------------------------------------------------
    h_low_loc_loc   = pyro.param(f"{name}_logit_low_hyper_loc_loc",
                                 jnp.full(T, priors.theta_logit_low_hyper_loc_loc))
    h_low_loc_scale = pyro.param(f"{name}_logit_low_hyper_loc_scale",
                                 jnp.full(T, priors.theta_logit_low_hyper_loc_scale),
                                 constraint=dist.constraints.greater_than(1e-4))
    h_low_scale_loc   = pyro.param(f"{name}_logit_low_hyper_scale_loc",   jnp.full(T, -1.0))
    h_low_scale_scale = pyro.param(f"{name}_logit_low_hyper_scale_scale", jnp.full(T, 0.1),
                                   constraint=dist.constraints.greater_than(1e-4))

    h_delta_loc_loc   = pyro.param(f"{name}_logit_delta_hyper_loc_loc",
                                   jnp.full(T, priors.theta_logit_delta_hyper_loc_loc))
    h_delta_loc_scale = pyro.param(f"{name}_logit_delta_hyper_loc_scale",
                                   jnp.full(T, priors.theta_logit_delta_hyper_loc_scale),
                                   constraint=dist.constraints.greater_than(1e-4))
    h_delta_scale_loc   = pyro.param(f"{name}_logit_delta_hyper_scale_loc",   jnp.full(T, -1.0))
    h_delta_scale_scale = pyro.param(f"{name}_logit_delta_hyper_scale_scale", jnp.full(T, 0.1),
                                     constraint=dist.constraints.greater_than(1e-4))

    h_K_loc_loc   = pyro.param(f"{name}_log_hill_K_hyper_loc_loc",
                               jnp.full(T, priors.theta_log_hill_K_hyper_loc_loc))
    h_K_loc_scale = pyro.param(f"{name}_log_hill_K_hyper_loc_scale",
                               jnp.full(T, priors.theta_log_hill_K_hyper_loc_scale),
                               constraint=dist.constraints.greater_than(1e-4))
    h_K_scale_loc   = pyro.param(f"{name}_log_hill_K_hyper_scale_loc",   jnp.full(T, -1.0))
    h_K_scale_scale = pyro.param(f"{name}_log_hill_K_hyper_scale_scale", jnp.full(T, 0.1),
                                 constraint=dist.constraints.greater_than(1e-4))

    h_n_loc_loc   = pyro.param(f"{name}_log_hill_n_hyper_loc_loc",
                               jnp.full(T, priors.theta_log_hill_n_hyper_loc_loc))
    h_n_loc_scale = pyro.param(f"{name}_log_hill_n_hyper_loc_scale",
                               jnp.full(T, priors.theta_log_hill_n_hyper_loc_scale),
                               constraint=dist.constraints.greater_than(1e-4))
    h_n_scale_loc   = pyro.param(f"{name}_log_hill_n_hyper_scale_loc",   jnp.full(T, -1.0))
    h_n_scale_scale = pyro.param(f"{name}_log_hill_n_hyper_scale_scale", jnp.full(T, 0.1),
                                 constraint=dist.constraints.greater_than(1e-4))

    with pyro.plate(f"{name}_hyper_plate", T, dim=-1):
        logit_low_hyper_loc   = pyro.sample(f"{name}_logit_low_hyper_loc",
                                            dist.Normal(h_low_loc_loc, h_low_loc_scale))
        logit_low_hyper_scale = pyro.sample(f"{name}_logit_low_hyper_scale",
                                            dist.LogNormal(h_low_scale_loc, h_low_scale_scale))
        logit_delta_hyper_loc   = pyro.sample(f"{name}_logit_delta_hyper_loc",
                                              dist.Normal(h_delta_loc_loc, h_delta_loc_scale))
        logit_delta_hyper_scale = pyro.sample(f"{name}_logit_delta_hyper_scale",
                                              dist.LogNormal(h_delta_scale_loc, h_delta_scale_scale))
        log_K_hyper_loc   = pyro.sample(f"{name}_log_hill_K_hyper_loc",
                                        dist.Normal(h_K_loc_loc, h_K_loc_scale))
        log_K_hyper_scale = pyro.sample(f"{name}_log_hill_K_hyper_scale",
                                        dist.LogNormal(h_K_scale_loc, h_K_scale_scale))
        log_n_hyper_loc   = pyro.sample(f"{name}_log_hill_n_hyper_loc",
                                        dist.Normal(h_n_loc_loc, h_n_loc_scale))
        log_n_hyper_scale = pyro.sample(f"{name}_log_hill_n_hyper_scale",
                                        dist.LogNormal(h_n_scale_loc, h_n_scale_scale))

    # ------------------------------------------------------------------
    # Full-population per-genotype offset variational parameters: shape (T, G)
    # ------------------------------------------------------------------
    low_offset_locs   = pyro.param(f"{name}_logit_low_offset_locs",   jnp.zeros((T, G), dtype=float))
    low_offset_scales = pyro.param(f"{name}_logit_low_offset_scales", jnp.ones((T, G),  dtype=float),
                                   constraint=dist.constraints.positive)
    delta_offset_locs   = pyro.param(f"{name}_logit_delta_offset_locs",   jnp.zeros((T, G), dtype=float))
    delta_offset_scales = pyro.param(f"{name}_logit_delta_offset_scales", jnp.ones((T, G),  dtype=float),
                                     constraint=dist.constraints.positive)
    K_offset_locs   = pyro.param(f"{name}_log_hill_K_offset_locs",   jnp.zeros((T, G), dtype=float))
    K_offset_scales = pyro.param(f"{name}_log_hill_K_offset_scales", jnp.ones((T, G),  dtype=float),
                                 constraint=dist.constraints.positive)
    n_offset_locs   = pyro.param(f"{name}_log_hill_n_offset_locs",   jnp.zeros((T, G), dtype=float))
    n_offset_scales = pyro.param(f"{name}_log_hill_n_offset_scales", jnp.ones((T, G),  dtype=float),
                                 constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_titrant_name_plate", T, dim=-2):
        with pyro.plate(f"{name}_genotype_plate", G, dim=-1):
            logit_low_offset   = pyro.sample(f"{name}_logit_low_offset",
                                             dist.Normal(low_offset_locs,   low_offset_scales))
            logit_delta_offset = pyro.sample(f"{name}_logit_delta_offset",
                                             dist.Normal(delta_offset_locs, delta_offset_scales))
            log_K_offset       = pyro.sample(f"{name}_log_hill_K_offset",
                                             dist.Normal(K_offset_locs,     K_offset_scales))
            log_n_offset       = pyro.sample(f"{name}_log_hill_n_offset",
                                             dist.Normal(n_offset_locs,     n_offset_scales))

    # ------------------------------------------------------------------
    # Reconstruct full-population ThetaParam
    # ------------------------------------------------------------------
    sig = dist.transforms.SigmoidTransform()
    logit_theta_low   = logit_low_hyper_loc[:, None]   + logit_low_offset   * logit_low_hyper_scale[:, None]
    logit_theta_delta = logit_delta_hyper_loc[:, None] + logit_delta_offset * logit_delta_hyper_scale[:, None]
    logit_theta_high  = logit_theta_low + logit_theta_delta
    log_hill_K        = log_K_hyper_loc[:, None] + log_K_offset * log_K_hyper_scale[:, None]
    log_hill_n        = log_n_hyper_loc[:, None] + log_n_offset * log_n_hyper_scale[:, None]

    theta_low  = sig(logit_theta_low)
    theta_high = sig(logit_theta_high)
    hill_n     = jnp.exp(log_hill_n)

    mu, sigma = _population_moments(
        logit_low_hyper_loc, logit_low_hyper_scale,
        logit_delta_hyper_loc, logit_delta_hyper_scale,
        log_K_hyper_loc, log_K_hyper_scale,
        log_n_hyper_loc, log_n_hyper_scale,
        data.log_titrant_conc,
    )

    return ThetaParam(theta_low=theta_low, theta_high=theta_high,
                      log_hill_K=log_hill_K, hill_n=hill_n,
                      mu=mu, sigma=sigma)


def simulate(name: str,
             data: DataClass,
             sim_priors: SimPriors,
             rng_key) -> tuple:
    """
    Perturbation-based theta simulation for hill_geno.

    Generates per-genotype Hill curves by perturbing a wildtype reference.
    Genotypes are assigned to one of four phenotype categories (normal,
    stuck-bound, never-binds, inverted) and their parameters drawn from
    Normal distributions centered on the wildtype reference.

    The returned ``ThetaParam`` has ``mu=None`` and ``sigma=None`` because
    population-moment hyperparameters are not sampled in this path.  Downstream
    ``run_model`` and ``_theta_param_to_df`` only read the per-genotype fields
    (``theta_low``, ``theta_high``, ``log_hill_K``, ``hill_n``), so this is safe.

    Parameters
    ----------
    name : str
        Unused; present for interface consistency with ``define_model``.
    data : DataClass
        Must expose ``num_genotype`` and ``log_titrant_conc``.
    sim_priors : SimPriors
        Wildtype reference and perturbation distributions.
    rng_key : jax.random.PRNGKey
        Seed for NumPy RNG (converted internally).

    Returns
    -------
    theta_gc : np.ndarray, shape (num_genotype, num_titrant_conc)
        Fractional occupancy for each genotype at each unique concentration.
    theta_param : ThetaParam
        Per-genotype fields shape ``(1, num_genotype)``; ``mu`` and ``sigma``
        are ``None``.
    """
    _eps = 1e-6
    seed = int(jax.random.randint(rng_key, shape=(), minval=0, maxval=2**30))
    rng = np.random.default_rng(seed)

    G = data.num_genotype
    log_conc = np.array(data.log_titrant_conc)  # (C,)

    # Wildtype reference in logit space
    wt_logit_low   = float(np.log(np.clip(sim_priors.wt_theta_low,  _eps, 1 - _eps)
                                  / (1 - np.clip(sim_priors.wt_theta_low,  _eps, 1 - _eps))))
    wt_logit_high  = float(np.log(np.clip(sim_priors.wt_theta_high, _eps, 1 - _eps)
                                  / (1 - np.clip(sim_priors.wt_theta_high, _eps, 1 - _eps))))
    wt_logit_delta = wt_logit_high - wt_logit_low
    wt_log_n       = float(np.log(sim_priors.wt_hill_n))

    # Assign phenotype categories
    p_normal = 1.0 - sim_priors.p_stuck_bound - sim_priors.p_never_binds - sim_priors.p_inverted
    if p_normal < 0:
        raise ValueError(
            "SimPriors mixture probabilities (p_stuck_bound + p_never_binds + p_inverted) "
            "sum to more than 1.0"
        )
    probs = [p_normal, sim_priors.p_stuck_bound, sim_priors.p_never_binds, sim_priors.p_inverted]
    categories = rng.choice(4, size=G, p=probs)

    # Per-genotype parameters, initialised to wildtype
    logit_low   = np.full(G, wt_logit_low)
    logit_delta = np.full(G, wt_logit_delta)
    log_K       = np.full(G, sim_priors.wt_log_K)
    log_n       = np.full(G, wt_log_n)

    # Category 0: normal (wildtype-like, full perturbation on all parameters)
    m0 = categories == 0
    n0 = int(m0.sum())
    logit_low[m0]   += rng.normal(0.0, sim_priors.sigma_logit_low,   n0)
    logit_delta[m0] += rng.normal(0.0, sim_priors.sigma_logit_delta, n0)
    log_K[m0]       += rng.normal(0.0, sim_priors.sigma_log_K,       n0)
    log_n[m0]       += rng.normal(0.0, sim_priors.sigma_log_n,       n0)

    # Category 1: stuck-bound (theta ≈ wt_theta_low, near-zero transition)
    m1 = categories == 1
    n1 = int(m1.sum())
    logit_low[m1]   = wt_logit_low + rng.normal(0.0, sim_priors.sigma_logit_low, n1)
    logit_delta[m1] = rng.normal(0.0, 0.1, n1)
    log_K[m1]       += rng.normal(0.0, sim_priors.sigma_log_K, n1)
    log_n[m1]       += rng.normal(0.0, sim_priors.sigma_log_n, n1)

    # Category 2: never-binds (theta ≈ wt_theta_high, near-zero transition)
    m2 = categories == 2
    n2 = int(m2.sum())
    logit_low[m2]   = wt_logit_high + rng.normal(0.0, sim_priors.sigma_logit_low, n2)
    logit_delta[m2] = rng.normal(0.0, 0.1, n2)
    log_K[m2]       += rng.normal(0.0, sim_priors.sigma_log_K, n2)
    log_n[m2]       += rng.normal(0.0, sim_priors.sigma_log_n, n2)

    # Category 3: inverted (theta goes from wt_theta_high to wt_theta_low)
    m3 = categories == 3
    n3 = int(m3.sum())
    logit_low[m3]   = wt_logit_high + rng.normal(0.0, sim_priors.sigma_logit_low,   n3)
    logit_delta[m3] = -wt_logit_delta + rng.normal(0.0, sim_priors.sigma_logit_delta, n3)
    log_K[m3]       += rng.normal(0.0, sim_priors.sigma_log_K, n3)
    log_n[m3]       += rng.normal(0.0, sim_priors.sigma_log_n, n3)

    # Convert to probability space
    theta_low_arr  = 1.0 / (1.0 + np.exp(-logit_low))               # (G,)
    theta_high_arr = 1.0 / (1.0 + np.exp(-(logit_low + logit_delta))) # (G,)
    hill_n_arr     = np.exp(log_n)                                     # (G,)

    # Compute theta: (G, C)
    occupancy = 1.0 / (1.0 + np.exp(-hill_n_arr[:, None] * (log_conc[None, :] - log_K[:, None])))
    theta_gc  = theta_low_arr[:, None] + (theta_high_arr - theta_low_arr)[:, None] * occupancy

    theta_param = ThetaParam(
        theta_low  = jnp.array(theta_low_arr)[None, :],    # (1, G)
        theta_high = jnp.array(theta_high_arr)[None, :],   # (1, G)
        log_hill_K = jnp.array(log_K)[None, :],            # (1, G)
        hill_n     = jnp.array(hill_n_arr)[None, :],       # (1, G)
        mu    = None,
        sigma = None,
    )

    return theta_gc, theta_param


def run_model(theta_param: ThetaParam, data: DataClass) -> jnp.ndarray:
    """
    Calculates fractional occupancy (theta) using the Hill equation.

    ``theta_param`` has per-genotype fields of shape ``(T, num_genotype)``.
    ``data.geno_theta_idx`` contains batch-relative indices; translating
    through ``data.batch_idx`` yields full-population genotype indices,
    consistent with the ``hill_mut`` convention.

    Parameters
    ----------
    theta_param : ThetaParam
        Output of ``define_model`` / ``guide``.  Per-genotype fields shape
        ``(num_titrant_name, num_genotype)``.
    data : DataClass
        Must expose ``batch_idx``, ``geno_theta_idx``, ``log_titrant_conc``,
        and ``scatter_theta``.

    Returns
    -------
    jnp.ndarray
        - ``scatter_theta == 0`` → shape ``(T, C, G_subset)``
        - ``scatter_theta == 1`` → shape ``(1, 1, 1, 1, T, C, G_subset)``
    """
    geno_idx = data.batch_idx[data.geno_theta_idx]
    theta_low  = theta_param.theta_low[:, None, geno_idx]
    theta_high = theta_param.theta_high[:, None, geno_idx]
    log_hill_K = theta_param.log_hill_K[:, None, geno_idx]
    hill_n     = theta_param.hill_n[:, None, geno_idx]

    log_titrant = data.log_titrant_conc[None, :, None]
    occupancy   = jax.nn.sigmoid(hill_n * (log_titrant - log_hill_K))
    theta_calc  = theta_low + (theta_high - theta_low) * occupancy

    if data.scatter_theta == 1:
        theta_calc = theta_calc[None, None, None, None, :, :, :]

    return theta_calc


def get_population_moments(theta_param: ThetaParam, data: DataClass) -> tuple:
    """
    Returns the expected population moments (mu, sigma) in logit-space.
    """
    return theta_param.mu, theta_param.sigma


def get_hyperparameters() -> Dict[str, Any]:
    """
    Gets default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names to their default values.
    """
    parameters = {}

    parameters["theta_logit_low_hyper_loc_loc"]   = 2.0
    parameters["theta_logit_low_hyper_loc_scale"]  = 2.0
    parameters["theta_logit_low_hyper_scale"]      = 1.0

    parameters["theta_logit_delta_hyper_loc_loc"]   = -4.0
    parameters["theta_logit_delta_hyper_loc_scale"]  = 2.0
    parameters["theta_logit_delta_hyper_scale"]      = 1.0

    parameters["theta_log_hill_K_hyper_loc_loc"]   = -4.1
    parameters["theta_log_hill_K_hyper_loc_scale"]  = 1.0
    parameters["theta_log_hill_K_hyper_scale"]      = 0.1

    parameters["theta_log_hill_n_hyper_loc_loc"]   = 0.7
    parameters["theta_log_hill_n_hyper_loc_scale"]  = 0.5
    parameters["theta_log_hill_n_hyper_scale"]      = 1.0

    return parameters


def get_sim_hyperparameters() -> Dict[str, Any]:
    """
    Default hyperparameters for perturbation-based simulation (``SimPriors``).

    Wildtype reference is a full-range decreasing Hill curve (theta: ~1→~0)
    matching the lac/IPTG system.  Perturbation sigmas allow realistic spread
    across the genotype library.

    Returns
    -------
    dict[str, Any]
    """
    return {
        "wt_theta_low":  0.99,
        "wt_theta_high": 0.01,
        "wt_log_K":      -4.1,   # ln(0.017 mM) — lac/IPTG default
        "wt_hill_n":     2.0,
        "sigma_logit_low":   0.5,
        "sigma_logit_delta": 0.5,
        "sigma_log_K":   0.5,
        "sigma_log_n":   0.3,
        "p_stuck_bound": 0.05,
        "p_never_binds": 0.05,
        "p_inverted":    0.02,
    }


def get_guesses(name: str, data: DataClass) -> Dict[str, Any]:
    """
    Gets initial guess values for model parameters.

    ``log_hill_K_hyper_loc`` is estimated per-titrant from the median of all
    finite values in ``data.log_titrant_conc``.  All other hyperprior guesses
    are broadcast to shape ``(num_titrant_name,)`` so they match the per-titrant
    plate structure.

    Parameters
    ----------
    name : str
    data : DataClass
        Must expose ``num_titrant_name``, ``num_genotype``, and
        ``log_titrant_conc``.

    Returns
    -------
    dict[str, Any]
    """
    _DEFAULT_LOG_K = -4.1  # ln(0.017 mM) — lac/IPTG system default

    log_conc    = np.array(data.log_titrant_conc)
    finite_conc = log_conc[np.isfinite(log_conc)]
    log_K_guess = float(np.median(finite_conc)) if len(finite_conc) > 0 else _DEFAULT_LOG_K

    T = data.num_titrant_name
    G = data.num_genotype

    guesses = {}
    guesses[f"{name}_logit_low_hyper_loc"]    = jnp.full(T, 2.0)
    guesses[f"{name}_logit_low_hyper_scale"]  = jnp.full(T, 2.0)
    guesses[f"{name}_logit_delta_hyper_loc"]  = jnp.full(T, -4.0)
    guesses[f"{name}_logit_delta_hyper_scale"] = jnp.full(T, 2.0)
    guesses[f"{name}_log_hill_K_hyper_loc"]   = jnp.full(T, log_K_guess)
    guesses[f"{name}_log_hill_K_hyper_scale"] = jnp.full(T, 1.0)
    guesses[f"{name}_log_hill_n_hyper_loc"]   = jnp.full(T, 0.7)
    guesses[f"{name}_log_hill_n_hyper_scale"] = jnp.full(T, 0.3)

    guesses[f"{name}_logit_low_offset"]   = jnp.zeros((T, G), dtype=float)
    guesses[f"{name}_logit_delta_offset"] = jnp.zeros((T, G), dtype=float)
    guesses[f"{name}_log_hill_K_offset"]  = jnp.zeros((T, G), dtype=float)
    guesses[f"{name}_log_hill_n_offset"]  = jnp.zeros((T, G), dtype=float)

    return guesses


def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Returns
    -------
    ModelPriors
    """
    return ModelPriors(**get_hyperparameters())


def get_extract_specs(ctx):
    return [dict(
        input_df=ctx.growth_tm.df,
        params_to_get=["hill_n", "log_hill_K", "theta_high", "theta_low"],
        map_column="map_theta_group",
        get_columns=["genotype", "titrant_name"],
        in_run_prefix="theta_",
    )]


_ZERO_CONC_VALUE = 1e-20


def predict_unmeasured(
    target_genotypes,
    titrant_names,
    manual_titrant_df,
    mut_labels,
    pair_labels,
    param_posteriors,
    q_to_get,
):
    """
    Predict theta for unmeasured genotypes using the per-titrant hyperprior means.

    ``hill`` has no per-mutation decomposition, so any unmeasured genotype
    receives the population-average prediction for its titrant.  All target
    genotypes receive identical predictions per titrant; mutations not seen
    during training are not NaN-masked because the model has no concept of
    individual mutation effects.

    Parameters
    ----------
    target_genotypes : list[str]
    titrant_names : list[str]
        Ordered titrant names matching the T dimension in the posterior.
    manual_titrant_df : pd.DataFrame
        Must have 'titrant_name' and 'titrant_conc' columns.
    mut_labels : list[str]  — unused (no per-mutation structure)
    pair_labels : list[str] — unused
    param_posteriors : dict-like
        Posterior samples; hyperprior locs have shape ``(S, T)``.
    q_to_get : dict[str, float]

    Returns
    -------
    pd.DataFrame
        Columns: 'genotype', 'titrant_name', 'titrant_conc', <quantile names>.
        All genotypes for a given titrant receive the same population-mean
        prediction.
    """
    import numpy as np
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples
    from tfscreen.tfmodel.analysis.predict_unmeasured import (
        _build_prediction_grid,
    )

    target_genotypes = list(target_genotypes)
    calc_df, geno_idx, titrant_idx = _build_prediction_grid(
        target_genotypes, titrant_names, manual_titrant_df
    )

    def _load(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = v[:]
        return np.array(v)

    # Shape (S, T) — one set of hyperpriors per titrant
    logit_low_hyper_loc   = _load("theta_logit_low_hyper_loc")    # (S, T)
    logit_delta_hyper_loc = _load("theta_logit_delta_hyper_loc")  # (S, T)
    log_K_hyper_loc       = _load("theta_log_hill_K_hyper_loc")   # (S, T)
    log_n_hyper_loc       = _load("theta_log_hill_n_hyper_loc")   # (S, T)

    logit_high_hyper_loc = logit_low_hyper_loc + logit_delta_hyper_loc
    theta_low  = 1.0 / (1.0 + np.exp(-logit_low_hyper_loc))    # (S, T)
    theta_high = 1.0 / (1.0 + np.exp(-logit_high_hyper_loc))   # (S, T)
    hill_K     = log_K_hyper_loc                                 # (S, T) log scale
    hill_n     = np.exp(log_n_hyper_loc)                         # (S, T)

    conc_vals = calc_df["titrant_conc"].values.astype(float).copy()
    conc_vals[conc_vals == 0] = _ZERO_CONC_VALUE
    log_conc = np.log(conc_vals)  # (N_rows,)

    # Select per-row parameters using the titrant index for each row
    t_l = theta_low[:, titrant_idx]    # (S, N_rows)
    t_h = theta_high[:, titrant_idx]
    l_K = hill_K[:, titrant_idx]
    h_n = hill_n[:, titrant_idx]

    occupancy     = 1.0 / (1.0 + np.exp(-h_n * (log_conc[np.newaxis, :] - l_K)))
    theta_samples = t_l + (t_h - t_l) * occupancy   # (S, N_rows)

    result_df = calc_df[["genotype", "titrant_name", "titrant_conc"]].copy()
    for q_name, q_val in q_to_get.items():
        result_df[q_name] = np.quantile(theta_samples, q_val, axis=0)
    return result_df


def build_calc_df(model, manual_titrant_df):
    """
    Build the concentration grid DataFrame for theta curve extraction.

    Returns
    -------
    calc_df : pd.DataFrame
        Rows for each (genotype, titrant_name, titrant_conc) combination,
        including the internal ``map_theta_group`` index column.
    internal_cols : list of str
        Columns to strip before returning results to the caller.
    extra_kwargs : dict
        Passed as ``**kwargs`` to ``compute_theta_samples``; empty for this model.
    """
    import pandas as pd
    import tfscreen.util.dataframe

    if manual_titrant_df is None:
        calc_df = (model.training_tm.df[["genotype", "titrant_name", "titrant_conc",
                                       "map_theta_group"]]
                   .drop_duplicates()
                   .reset_index(drop=True))
    else:
        tfscreen.util.dataframe.check_columns(manual_titrant_df,
                                              required_columns=["titrant_name", "titrant_conc"])
        if "genotype" not in manual_titrant_df.columns:
            genotypes = model.training_tm.df["genotype"].unique()
            dfs = [manual_titrant_df.assign(genotype=g) for g in genotypes]
            calc_df = pd.concat(dfs).reset_index(drop=True)
        else:
            calc_df = manual_titrant_df.copy()

        mapping = (model.training_tm.df[["genotype", "titrant_name", "map_theta_group"]]
                   .drop_duplicates()
                   .set_index(["genotype", "titrant_name"])["map_theta_group"]
                   .to_dict())
        calc_df["map_theta_group"] = (calc_df
                                      .set_index(["genotype", "titrant_name"])
                                      .index.map(mapping))
        if calc_df["map_theta_group"].isna().any():
            missing = calc_df[calc_df["map_theta_group"].isna()]
            raise ValueError(
                "Some (genotype, titrant_name) pairs in manual_titrant_df "
                "were not found in the model data: "
                f"{missing[['genotype', 'titrant_name']].drop_duplicates().values}"
            )

    return calc_df, ["map_theta_group"], {}


def compute_theta_samples(calc_df, param_posteriors):
    """
    Compute posterior theta samples for the Hill model.

    Parameters
    ----------
    calc_df : pd.DataFrame
        Output of ``build_calc_df``; must contain ``titrant_conc`` and
        ``map_theta_group`` columns.
    param_posteriors : dict-like
        Posterior samples keyed by parameter name (with ``theta_`` prefix).

    Returns
    -------
    theta_samples : np.ndarray, shape (S, N)
        Posterior theta at each row of ``calc_df``.
    """
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples

    indices = calc_df["map_theta_group"].values.astype(int)

    log_titrant = calc_df["titrant_conc"].values.copy().astype(float)
    log_titrant[log_titrant == 0] = _ZERO_CONC_VALUE
    log_titrant = np.log(log_titrant)[np.newaxis, :]   # (1, N)

    def _load_flat(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = v[:]
        return v.reshape(v.shape[0], -1)   # (S, num_groups)

    hill_n     = _load_flat("theta_hill_n")
    log_hill_K = _load_flat("theta_log_hill_K")
    theta_high = _load_flat("theta_theta_high")
    theta_low  = _load_flat("theta_theta_low")

    h_n = hill_n[:, indices]
    l_K = log_hill_K[:, indices]
    t_h = theta_high[:, indices]
    t_l = theta_low[:, indices]

    occupancy = 1.0 / (1.0 + np.exp(-h_n * (log_titrant - l_K)))
    return t_l + (t_h - t_l) * occupancy   # (S, N)
