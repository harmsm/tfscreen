"""
Mutation-decomposed Hill theta model.

Each Hill parameter is expressed as a wild-type value plus additive
per-mutation deltas (in the appropriate transformed space), optionally
plus pairwise epistasis terms:

    param[t, g] = param_wt[t]
                  + (d_param[t, :] @ M)[:, g]          # additive mut effects
                  + (epi_param[t, :] @ P)[:, g]         # pairwise epistasis

where M[m, g] = 1 if mutation m is in genotype g (shape: num_mutation x G)
and   P[p, g] = 1 if pair   p is in genotype g (shape: num_pair    x G).

The mutation deltas are hierarchical: each delta is drawn from
Normal(0, sigma_d) where sigma_d is inferred from the data (HalfNormal prior).
Epistasis terms are drawn from Normal(0, sigma_epi) likewise.

The four Hill parameters and their spaces:
  logit_theta_low   – logit space
  logit_theta_delta – logit space (theta_high = sigmoid(logit_low + logit_delta))
  log_hill_K        – log space
  log_hill_n        – log space

Epistasis is only sampled when data.num_pair > 0.
"""

import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
import pandas as pd
from flax.struct import dataclass, field
from typing import Dict, Any

from functools import partial
from typing import Union
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData, BindingData
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix


# ---------------------------------------------------------------------------
# Pytrees
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters for the mutation-decomposed Hill model priors."""

    # WT priors (Normal, in transformed space)
    theta_logit_low_wt_loc: float
    theta_logit_low_wt_scale: float
    theta_logit_delta_wt_loc: float
    theta_logit_delta_wt_scale: float
    theta_log_hill_K_wt_loc: float
    theta_log_hill_K_wt_scale: float
    theta_log_hill_n_wt_loc: float
    theta_log_hill_n_wt_scale: float

    # HalfNormal scales for mutation delta distributions
    theta_sigma_d_logit_low_scale: float
    theta_sigma_d_logit_delta_scale: float
    theta_sigma_d_log_hill_K_scale: float
    theta_sigma_d_log_hill_n_scale: float

    # Shared regularised horseshoe hyperparameters for pairwise epistasis
    theta_epi_tau_scale: float     # HalfCauchy scale for global τ (shared across all params)
    theta_epi_slab_scale: float    # typical size of a large epistasis effect
    theta_epi_slab_df: float       # InvGamma shape ν (usually 4)


@dataclass(frozen=True)
class ThetaParam:
    """Assembled Hill equation parameters, shape (num_titrant_name, num_genotype)."""

    theta_low: jnp.ndarray
    theta_high: jnp.ndarray
    log_hill_K: jnp.ndarray
    hill_n: jnp.ndarray
    mu: jnp.ndarray
    sigma: jnp.ndarray


# ---------------------------------------------------------------------------
# Internal helpers shared by model and guide
# ---------------------------------------------------------------------------

def _assemble(wt, d_offsets, sigma_d, M,
              epi_offsets=None, sigma_epi=None, pair_scatter=None):
    """
    Assemble per-genotype parameter array from WT + mutation deltas + epistasis.

    Parameters
    ----------
    wt : array, shape (T,)
    d_offsets : array, shape (T, num_mutation)
    sigma_d : array, shape (T,)
    M : array, shape (num_mutation, G)
    epi_offsets : array, shape (T, num_pair) or None
    sigma_epi : array, shape (T,) or None
    pair_scatter : callable or None
        Callable with signature ``pair_scatter(epi) -> (T, G)`` that scatters
        epistasis values from pair-space to genotype-space.  Typically a
        ``partial`` of ``apply_pair_matrix`` with the COO indices pre-bound.

    Returns
    -------
    array, shape (T, G)
    """
    d = d_offsets * sigma_d[:, None]           # [T, M]
    result = wt[:, None] + d @ M               # [T, G]
    if epi_offsets is not None:
        epi = epi_offsets * sigma_epi[:, None] # [T, P]
        result = result + pair_scatter(epi)    # [T, G]
    return result


def _population_moments(logit_theta_low, logit_theta_high,
                        log_hill_K, log_hill_n, data):
    """
    Compute per-concentration population moments (mu, sigma) of logit(theta)
    over the assembled genotype population.

    Returns arrays of shape (T, num_conc, 1).
    """
    eps = 1e-6
    log_conc = data.log_titrant_conc[None, :, None]       # [1, C, 1]
    K = jnp.exp(log_hill_K[:, None, :])                   # [T, 1, G]
    n = jnp.exp(log_hill_n[:, None, :])                   # [T, 1, G]
    low = dist.transforms.SigmoidTransform()(logit_theta_low[:, None, :])   # [T, 1, G]
    high = dist.transforms.SigmoidTransform()(logit_theta_high[:, None, :]) # [T, 1, G]

    occ = jax.nn.sigmoid(n * (log_conc - K))              # [T, C, G]
    theta_all = jnp.clip(low + (high - low) * occ, eps, 1.0 - eps)
    logit_theta_all = jax.scipy.special.logit(theta_all)  # [T, C, G]

    mu = jnp.mean(logit_theta_all, axis=-1, keepdims=True)    # [T, C, 1]
    sigma = jnp.std(logit_theta_all, axis=-1, keepdims=True)  # [T, C, 1]
    return mu, sigma


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def define_model(name: str,
                 data: Union[GrowthData, BindingData],
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the mutation-decomposed hierarchical Hill model.

    Samples WT Hill parameters and per-mutation delta parameters, then
    assembles per-genotype Hill curves via matrix multiplication.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample sites.
    data : GrowthData
        Must have ``mut_geno_matrix`` (num_mutation x G) and ``num_mutation``.
        If ``num_pair > 0``, must also have ``pair_geno_matrix`` (num_pair x G).
    priors : ModelPriors

    Returns
    -------
    ThetaParam
        Assembled parameters with shape ``(num_titrant_name, num_genotype)``.
    """
    T = data.num_titrant_name
    M_mat = jnp.array(data.mut_geno_matrix)   # [num_mutation, G]
    num_mut = data.num_mutation
    has_epi = data.num_pair > 0

    # ------------------------------------------------------------------
    # WT parameters and mutation-delta scale hyperpriors: shape (T,)
    # ------------------------------------------------------------------
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):

        logit_low_wt = pyro.sample(
            f"{name}_logit_low_wt",
            dist.Normal(priors.theta_logit_low_wt_loc,
                        priors.theta_logit_low_wt_scale))
        logit_delta_wt = pyro.sample(
            f"{name}_logit_delta_wt",
            dist.Normal(priors.theta_logit_delta_wt_loc,
                        priors.theta_logit_delta_wt_scale))
        log_K_wt = pyro.sample(
            f"{name}_log_hill_K_wt",
            dist.Normal(priors.theta_log_hill_K_wt_loc,
                        priors.theta_log_hill_K_wt_scale))
        log_n_wt = pyro.sample(
            f"{name}_log_hill_n_wt",
            dist.Normal(priors.theta_log_hill_n_wt_loc,
                        priors.theta_log_hill_n_wt_scale))

        sigma_d_low = pyro.sample(
            f"{name}_sigma_d_logit_low",
            dist.HalfNormal(priors.theta_sigma_d_logit_low_scale))
        sigma_d_delta = pyro.sample(
            f"{name}_sigma_d_logit_delta",
            dist.HalfNormal(priors.theta_sigma_d_logit_delta_scale))
        sigma_d_K = pyro.sample(
            f"{name}_sigma_d_log_hill_K",
            dist.HalfNormal(priors.theta_sigma_d_log_hill_K_scale))
        sigma_d_n = pyro.sample(
            f"{name}_sigma_d_log_hill_n",
            dist.HalfNormal(priors.theta_sigma_d_log_hill_n_scale))

    # ------------------------------------------------------------------
    # Mutation delta offsets: shape (T, num_mutation)
    # ------------------------------------------------------------------
    with pyro.plate(f"{name}_titrant_mut_outer_plate", T, dim=-2):
        with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
            d_low_off = pyro.sample(
                f"{name}_d_logit_low_offset", dist.Normal(0.0, 1.0))
            d_delta_off = pyro.sample(
                f"{name}_d_logit_delta_offset", dist.Normal(0.0, 1.0))
            d_K_off = pyro.sample(
                f"{name}_d_log_hill_K_offset", dist.Normal(0.0, 1.0))
            d_n_off = pyro.sample(
                f"{name}_d_log_hill_n_offset", dist.Normal(0.0, 1.0))

    # ------------------------------------------------------------------
    # Optional epistasis: shape (T, num_pair)
    # ------------------------------------------------------------------
    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                              pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                              pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                              num_genotype=data.num_genotype)
        num_pair = data.num_pair

        # Shared global scale and slab variance across all Hill parameters
        tau_epi = pyro.sample(
            f"{name}_epi_tau",
            dist.HalfCauchy(priors.theta_epi_tau_scale))
        c2_epi = pyro.sample(
            f"{name}_epi_c2",
            dist.InverseGamma(priors.theta_epi_slab_df / 2.0,
                              priors.theta_epi_slab_df * priors.theta_epi_slab_scale ** 2 / 2.0))

        # Per-parameter-type local scales and offsets: shape (T, num_pair)
        with pyro.plate(f"{name}_titrant_pair_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                epi_low_lam = pyro.sample(
                    f"{name}_epi_logit_low_lambda", dist.HalfCauchy(1.0))
                epi_low_off = pyro.sample(
                    f"{name}_epi_logit_low_offset", dist.Normal(0.0, 1.0))
                epi_delta_lam = pyro.sample(
                    f"{name}_epi_logit_delta_lambda", dist.HalfCauchy(1.0))
                epi_delta_off = pyro.sample(
                    f"{name}_epi_logit_delta_offset", dist.Normal(0.0, 1.0))
                epi_K_lam = pyro.sample(
                    f"{name}_epi_log_hill_K_lambda", dist.HalfCauchy(1.0))
                epi_K_off = pyro.sample(
                    f"{name}_epi_log_hill_K_offset", dist.Normal(0.0, 1.0))
                epi_n_lam = pyro.sample(
                    f"{name}_epi_log_hill_n_lambda", dist.HalfCauchy(1.0))
                epi_n_off = pyro.sample(
                    f"{name}_epi_log_hill_n_offset", dist.Normal(0.0, 1.0))
    else:
        pair_scatter = None
        epi_low_off = epi_delta_off = epi_K_off = epi_n_off = None

    # ------------------------------------------------------------------
    # Per-mutation deltas in transformed space: shape (T, M)
    # ------------------------------------------------------------------
    d_logit_low   = d_low_off   * sigma_d_low[:, None]    # [T, M]
    d_logit_delta = d_delta_off * sigma_d_delta[:, None]
    d_log_hill_K  = d_K_off     * sigma_d_K[:, None]
    d_log_hill_n  = d_n_off     * sigma_d_n[:, None]

    pyro.deterministic(f"{name}_d_logit_low",   d_logit_low)
    pyro.deterministic(f"{name}_d_logit_delta", d_logit_delta)
    pyro.deterministic(f"{name}_d_log_hill_K",  d_log_hill_K)
    pyro.deterministic(f"{name}_d_log_hill_n",  d_log_hill_n)

    if has_epi:
        def _lam_tilde(lam):
            return jnp.sqrt(c2_epi * lam ** 2 / (c2_epi + tau_epi ** 2 * lam ** 2))

        epi_logit_low   = epi_low_off   * tau_epi * _lam_tilde(epi_low_lam)    # [T, P]
        epi_logit_delta = epi_delta_off * tau_epi * _lam_tilde(epi_delta_lam)
        epi_log_hill_K  = epi_K_off     * tau_epi * _lam_tilde(epi_K_lam)
        epi_log_hill_n  = epi_n_off     * tau_epi * _lam_tilde(epi_n_lam)

        pyro.deterministic(f"{name}_epi_logit_low",   epi_logit_low)
        pyro.deterministic(f"{name}_epi_logit_delta", epi_logit_delta)
        pyro.deterministic(f"{name}_epi_log_hill_K",  epi_log_hill_K)
        pyro.deterministic(f"{name}_epi_log_hill_n",  epi_log_hill_n)

    # ------------------------------------------------------------------
    # Assemble per-genotype parameters: shape (T, G)
    # ------------------------------------------------------------------
    logit_theta_low   = _assemble(logit_low_wt,   d_low_off,   sigma_d_low,   M_mat)
    logit_theta_delta = _assemble(logit_delta_wt, d_delta_off, sigma_d_delta, M_mat)
    log_hill_K        = _assemble(log_K_wt,       d_K_off,     sigma_d_K,     M_mat)
    log_hill_n        = _assemble(log_n_wt,       d_n_off,     sigma_d_n,     M_mat)

    if has_epi:
        logit_theta_low   = logit_theta_low   + pair_scatter(epi_logit_low)
        logit_theta_delta = logit_theta_delta + pair_scatter(epi_logit_delta)
        log_hill_K        = log_hill_K        + pair_scatter(epi_log_hill_K)
        log_hill_n        = log_hill_n        + pair_scatter(epi_log_hill_n)

    logit_theta_high = logit_theta_low + logit_theta_delta

    # Transform to natural scale and register
    theta_low = dist.transforms.SigmoidTransform()(logit_theta_low)
    theta_high = dist.transforms.SigmoidTransform()(logit_theta_high)
    hill_n = jnp.exp(log_hill_n)

    pyro.deterministic(f"{name}_theta_low", theta_low)
    pyro.deterministic(f"{name}_theta_high", theta_high)
    pyro.deterministic(f"{name}_log_hill_K", log_hill_K)
    pyro.deterministic(f"{name}_hill_n", hill_n)

    mu, sigma = _population_moments(logit_theta_low, logit_theta_high,
                                    log_hill_K, log_hill_n, data)

    return ThetaParam(theta_low=theta_low,
                      theta_high=theta_high,
                      log_hill_K=log_hill_K,
                      hill_n=hill_n,
                      mu=mu,
                      sigma=sigma)


# ---------------------------------------------------------------------------
# Guide
# ---------------------------------------------------------------------------

def guide(name: str,
          data: Union[GrowthData, BindingData],
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the mutation-decomposed Hill model."""

    T = data.num_titrant_name
    num_mut = data.num_mutation
    M_mat = jnp.array(data.mut_geno_matrix)
    has_epi = data.num_pair > 0

    # ------------------------------------------------------------------
    # Variational parameters for WT and sigma hyperpriors
    # ------------------------------------------------------------------
    # WT locs/scales
    wt_init = {
        "logit_low_wt":   (priors.theta_logit_low_wt_loc,   1.0),
        "logit_delta_wt": (priors.theta_logit_delta_wt_loc, 1.0),
        "log_hill_K_wt":  (priors.theta_log_hill_K_wt_loc,  1.0),
        "log_hill_n_wt":  (priors.theta_log_hill_n_wt_loc,  0.3),
    }
    wt_loc_params = {}
    wt_scale_params = {}
    for k, (loc0, scale0) in wt_init.items():
        wt_loc_params[k] = pyro.param(
            f"{name}_{k}_loc", jnp.full(T, loc0))
        wt_scale_params[k] = pyro.param(
            f"{name}_{k}_scale", jnp.full(T, scale0),
            constraint=dist.constraints.positive)

    # sigma hyperpriors (LogNormal guide for HalfNormal prior)
    sigma_names = ["sigma_d_logit_low", "sigma_d_logit_delta",
                   "sigma_d_log_hill_K", "sigma_d_log_hill_n"]
    sigma_loc_params = {}
    sigma_scale_params = {}
    for k in sigma_names:
        sigma_loc_params[k] = pyro.param(
            f"{name}_{k}_loc", jnp.full(T, -1.0))
        sigma_scale_params[k] = pyro.param(
            f"{name}_{k}_scale", jnp.full(T, 0.1),
            constraint=dist.constraints.positive)

    # ------------------------------------------------------------------
    # Variational parameters for mutation delta offsets: shape (T, M)
    # ------------------------------------------------------------------
    delta_names = ["d_logit_low_offset", "d_logit_delta_offset",
                   "d_log_hill_K_offset", "d_log_hill_n_offset"]
    d_loc_params = {}
    d_scale_params = {}
    for k in delta_names:
        d_loc_params[k] = pyro.param(
            f"{name}_{k}_locs", jnp.zeros((T, num_mut)))
        d_scale_params[k] = pyro.param(
            f"{name}_{k}_scales", jnp.ones((T, num_mut)),
            constraint=dist.constraints.positive)

    # ------------------------------------------------------------------
    # Optional epistasis variational parameters
    # ------------------------------------------------------------------
    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)
        num_pair = data.num_pair

        # Shared scalar τ and c²
        tau_epi_loc = pyro.param(f"{name}_epi_tau_loc", jnp.array(-1.0))
        tau_epi_scale_p = pyro.param(f"{name}_epi_tau_scale", jnp.array(0.1),
                                     constraint=dist.constraints.positive)
        c2_epi_loc = pyro.param(f"{name}_epi_c2_loc", jnp.array(1.4))
        c2_epi_scale_p = pyro.param(f"{name}_epi_c2_scale", jnp.array(0.5),
                                    constraint=dist.constraints.positive)

        # Per-type local scales and offsets: shape (T, num_pair)
        epi_lam_names = ["epi_logit_low_lambda", "epi_logit_delta_lambda",
                         "epi_log_hill_K_lambda", "epi_log_hill_n_lambda"]
        epi_names = ["epi_logit_low_offset", "epi_logit_delta_offset",
                     "epi_log_hill_K_offset", "epi_log_hill_n_offset"]
        epi_lam_loc_params = {}
        epi_lam_scale_params = {}
        epi_loc_params = {}
        epi_scale_params = {}
        for k in epi_lam_names:
            epi_lam_loc_params[k] = pyro.param(
                f"{name}_{k}_locs", jnp.zeros((T, num_pair)))
            epi_lam_scale_params[k] = pyro.param(
                f"{name}_{k}_scales", jnp.ones((T, num_pair)),
                constraint=dist.constraints.positive)
        for k in epi_names:
            epi_loc_params[k] = pyro.param(
                f"{name}_{k}_locs", jnp.zeros((T, num_pair)))
            epi_scale_params[k] = pyro.param(
                f"{name}_{k}_scales", jnp.ones((T, num_pair)),
                constraint=dist.constraints.positive)
    else:
        pair_scatter = None

    # ------------------------------------------------------------------
    # Sampling within plates
    # ------------------------------------------------------------------
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        logit_low_wt = pyro.sample(
            f"{name}_logit_low_wt",
            dist.Normal(wt_loc_params["logit_low_wt"],
                        wt_scale_params["logit_low_wt"]))
        logit_delta_wt = pyro.sample(
            f"{name}_logit_delta_wt",
            dist.Normal(wt_loc_params["logit_delta_wt"],
                        wt_scale_params["logit_delta_wt"]))
        log_K_wt = pyro.sample(
            f"{name}_log_hill_K_wt",
            dist.Normal(wt_loc_params["log_hill_K_wt"],
                        wt_scale_params["log_hill_K_wt"]))
        log_n_wt = pyro.sample(
            f"{name}_log_hill_n_wt",
            dist.Normal(wt_loc_params["log_hill_n_wt"],
                        wt_scale_params["log_hill_n_wt"]))

        sigma_d_low = pyro.sample(
            f"{name}_sigma_d_logit_low",
            dist.LogNormal(sigma_loc_params["sigma_d_logit_low"],
                           sigma_scale_params["sigma_d_logit_low"]))
        sigma_d_delta = pyro.sample(
            f"{name}_sigma_d_logit_delta",
            dist.LogNormal(sigma_loc_params["sigma_d_logit_delta"],
                           sigma_scale_params["sigma_d_logit_delta"]))
        sigma_d_K = pyro.sample(
            f"{name}_sigma_d_log_hill_K",
            dist.LogNormal(sigma_loc_params["sigma_d_log_hill_K"],
                           sigma_scale_params["sigma_d_log_hill_K"]))
        sigma_d_n = pyro.sample(
            f"{name}_sigma_d_log_hill_n",
            dist.LogNormal(sigma_loc_params["sigma_d_log_hill_n"],
                           sigma_scale_params["sigma_d_log_hill_n"]))

    with pyro.plate(f"{name}_titrant_mut_outer_plate", T, dim=-2):
        with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
            d_low_off = pyro.sample(
                f"{name}_d_logit_low_offset",
                dist.Normal(d_loc_params["d_logit_low_offset"],
                            d_scale_params["d_logit_low_offset"]))
            d_delta_off = pyro.sample(
                f"{name}_d_logit_delta_offset",
                dist.Normal(d_loc_params["d_logit_delta_offset"],
                            d_scale_params["d_logit_delta_offset"]))
            d_K_off = pyro.sample(
                f"{name}_d_log_hill_K_offset",
                dist.Normal(d_loc_params["d_log_hill_K_offset"],
                            d_scale_params["d_log_hill_K_offset"]))
            d_n_off = pyro.sample(
                f"{name}_d_log_hill_n_offset",
                dist.Normal(d_loc_params["d_log_hill_n_offset"],
                            d_scale_params["d_log_hill_n_offset"]))

    if has_epi:
        tau_epi = pyro.sample(f"{name}_epi_tau",
                              dist.LogNormal(tau_epi_loc, tau_epi_scale_p))
        c2_epi = pyro.sample(f"{name}_epi_c2",
                             dist.LogNormal(c2_epi_loc, c2_epi_scale_p))

        with pyro.plate(f"{name}_titrant_pair_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                epi_low_lam = pyro.sample(
                    f"{name}_epi_logit_low_lambda",
                    dist.LogNormal(epi_lam_loc_params["epi_logit_low_lambda"],
                                   epi_lam_scale_params["epi_logit_low_lambda"]))
                epi_low_off = pyro.sample(
                    f"{name}_epi_logit_low_offset",
                    dist.Normal(epi_loc_params["epi_logit_low_offset"],
                                epi_scale_params["epi_logit_low_offset"]))
                epi_delta_lam = pyro.sample(
                    f"{name}_epi_logit_delta_lambda",
                    dist.LogNormal(epi_lam_loc_params["epi_logit_delta_lambda"],
                                   epi_lam_scale_params["epi_logit_delta_lambda"]))
                epi_delta_off = pyro.sample(
                    f"{name}_epi_logit_delta_offset",
                    dist.Normal(epi_loc_params["epi_logit_delta_offset"],
                                epi_scale_params["epi_logit_delta_offset"]))
                epi_K_lam = pyro.sample(
                    f"{name}_epi_log_hill_K_lambda",
                    dist.LogNormal(epi_lam_loc_params["epi_log_hill_K_lambda"],
                                   epi_lam_scale_params["epi_log_hill_K_lambda"]))
                epi_K_off = pyro.sample(
                    f"{name}_epi_log_hill_K_offset",
                    dist.Normal(epi_loc_params["epi_log_hill_K_offset"],
                                epi_scale_params["epi_log_hill_K_offset"]))
                epi_n_lam = pyro.sample(
                    f"{name}_epi_log_hill_n_lambda",
                    dist.LogNormal(epi_lam_loc_params["epi_log_hill_n_lambda"],
                                   epi_lam_scale_params["epi_log_hill_n_lambda"]))
                epi_n_off = pyro.sample(
                    f"{name}_epi_log_hill_n_offset",
                    dist.Normal(epi_loc_params["epi_log_hill_n_offset"],
                                epi_scale_params["epi_log_hill_n_offset"]))
    else:
        epi_low_off = epi_delta_off = epi_K_off = epi_n_off = None

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    logit_theta_low   = _assemble(logit_low_wt,   d_low_off,   sigma_d_low,   M_mat)
    logit_theta_delta = _assemble(logit_delta_wt, d_delta_off, sigma_d_delta, M_mat)
    log_hill_K        = _assemble(log_K_wt,       d_K_off,     sigma_d_K,     M_mat)
    log_hill_n        = _assemble(log_n_wt,       d_n_off,     sigma_d_n,     M_mat)

    if has_epi:
        def _lam_tilde(lam):
            return jnp.sqrt(c2_epi * lam ** 2 / (c2_epi + tau_epi ** 2 * lam ** 2))

        logit_theta_low   = logit_theta_low   + pair_scatter(epi_low_off   * tau_epi * _lam_tilde(epi_low_lam))
        logit_theta_delta = logit_theta_delta + pair_scatter(epi_delta_off * tau_epi * _lam_tilde(epi_delta_lam))
        log_hill_K        = log_hill_K        + pair_scatter(epi_K_off     * tau_epi * _lam_tilde(epi_K_lam))
        log_hill_n        = log_hill_n        + pair_scatter(epi_n_off     * tau_epi * _lam_tilde(epi_n_lam))

    logit_theta_high = logit_theta_low + logit_theta_delta

    theta_low = dist.transforms.SigmoidTransform()(logit_theta_low)
    theta_high = dist.transforms.SigmoidTransform()(logit_theta_high)
    hill_n = jnp.exp(log_hill_n)

    mu, sigma = _population_moments(logit_theta_low, logit_theta_high,
                                    log_hill_K, log_hill_n, data)

    return ThetaParam(theta_low=theta_low,
                      theta_high=theta_high,
                      log_hill_K=log_hill_K,
                      hill_n=hill_n,
                      mu=mu,
                      sigma=sigma)


# ---------------------------------------------------------------------------
# run_model – identical to hill.py
# ---------------------------------------------------------------------------

def run_model(theta_param: ThetaParam, data) -> jnp.ndarray:
    """
    Calculate theta via the Hill equation using the assembled parameters.

    Identical to ``hill.run_model``; the assembled ``ThetaParam`` already
    has the correct per-genotype shape so no additional transformation is needed.
    """
    geno_idx = data.batch_idx[data.geno_theta_idx]
    theta_low = theta_param.theta_low[:, None, geno_idx]
    theta_high = theta_param.theta_high[:, None, geno_idx]
    log_hill_K = theta_param.log_hill_K[:, None, geno_idx]
    hill_n = theta_param.hill_n[:, None, geno_idx]

    log_titrant = data.log_titrant_conc[None, :, None]
    occupancy = jax.nn.sigmoid(hill_n * (log_titrant - log_hill_K))
    theta_calc = theta_low + (theta_high - theta_low) * occupancy

    if data.scatter_theta == 1:
        theta_calc = theta_calc[None, None, None, None, :, :, :]

    return theta_calc


def get_population_moments(theta_param: ThetaParam, data) -> tuple:
    return theta_param.mu, theta_param.sigma


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_hyperparameters() -> Dict[str, Any]:
    p = {}
    p["theta_logit_low_wt_loc"] = 2.0
    p["theta_logit_low_wt_scale"] = 2.0
    p["theta_logit_delta_wt_loc"] = -4.0
    p["theta_logit_delta_wt_scale"] = 2.0
    p["theta_log_hill_K_wt_loc"] = -4.1
    p["theta_log_hill_K_wt_scale"] = 2.0
    p["theta_log_hill_n_wt_loc"] = 0.7
    p["theta_log_hill_n_wt_scale"] = 0.3
    p["theta_sigma_d_logit_low_scale"] = 1.0
    p["theta_sigma_d_logit_delta_scale"] = 1.0
    p["theta_sigma_d_log_hill_K_scale"] = 1.0
    p["theta_sigma_d_log_hill_n_scale"] = 0.5
    p["theta_epi_tau_scale"] = 0.1
    p["theta_epi_slab_scale"] = 2.0
    p["theta_epi_slab_df"] = 4.0
    return p


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    T = data.num_titrant_name
    M = data.num_mutation
    guesses = {}
    guesses[f"{name}_logit_low_wt"] = jnp.full(T, 2.0)
    guesses[f"{name}_logit_delta_wt"] = jnp.full(T, -4.0)
    guesses[f"{name}_log_hill_K_wt"] = jnp.full(T, -4.1)
    guesses[f"{name}_log_hill_n_wt"] = jnp.full(T, 0.7)
    guesses[f"{name}_sigma_d_logit_low"] = jnp.full(T, 0.5)
    guesses[f"{name}_sigma_d_logit_delta"] = jnp.full(T, 0.5)
    guesses[f"{name}_sigma_d_log_hill_K"] = jnp.full(T, 0.5)
    guesses[f"{name}_sigma_d_log_hill_n"] = jnp.full(T, 0.3)
    guesses[f"{name}_d_logit_low_offset"] = jnp.zeros((T, M))
    guesses[f"{name}_d_logit_delta_offset"] = jnp.zeros((T, M))
    guesses[f"{name}_d_log_hill_K_offset"] = jnp.zeros((T, M))
    guesses[f"{name}_d_log_hill_n_offset"] = jnp.zeros((T, M))
    if data.num_pair > 0:
        P = data.num_pair
        guesses[f"{name}_epi_tau"] = 0.05
        guesses[f"{name}_epi_c2"] = 4.0
        guesses[f"{name}_epi_logit_low_lambda"] = jnp.ones((T, P)) * 0.5
        guesses[f"{name}_epi_logit_delta_lambda"] = jnp.ones((T, P)) * 0.5
        guesses[f"{name}_epi_log_hill_K_lambda"] = jnp.ones((T, P)) * 0.5
        guesses[f"{name}_epi_log_hill_n_lambda"] = jnp.ones((T, P)) * 0.5
        guesses[f"{name}_epi_logit_low_offset"] = jnp.zeros((T, P))
        guesses[f"{name}_epi_logit_delta_offset"] = jnp.zeros((T, P))
        guesses[f"{name}_epi_log_hill_K_offset"] = jnp.zeros((T, P))
        guesses[f"{name}_epi_log_hill_n_offset"] = jnp.zeros((T, P))
    return guesses


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())


def get_extract_specs(ctx):
    geno_dim = ctx.growth_tm.tensor_dim_names.index("genotype")
    num_genotype = len(ctx.growth_tm.tensor_dim_labels[geno_dim])
    titrant_dim = ctx.growth_tm.tensor_dim_names.index("titrant_name")
    titrant_names = list(ctx.growth_tm.tensor_dim_labels[titrant_dim])
    num_mut = len(ctx.mut_labels)

    theta_mut_df = (ctx.growth_tm.df[["genotype", "titrant_name",
                                      "genotype_idx", "titrant_name_idx"]]
                    .drop_duplicates()
                    .copy())
    theta_mut_df["map_theta_mut"] = (theta_mut_df["titrant_name_idx"] * num_genotype
                                     + theta_mut_df["genotype_idx"])
    specs = [dict(
        input_df=theta_mut_df,
        params_to_get=["theta_low", "theta_high", "log_hill_K", "hill_n"],
        map_column="map_theta_mut",
        get_columns=["genotype", "titrant_name"],
        in_run_prefix="theta_",
    )]

    theta_d_rows = [
        {"titrant_name": t, "mutation": m,
         "map_theta_d_mut": ti * num_mut + mi}
        for ti, t in enumerate(titrant_names)
        for mi, m in enumerate(ctx.mut_labels)
    ]
    specs.append(dict(
        input_df=pd.DataFrame(theta_d_rows),
        params_to_get=["d_logit_low", "d_logit_delta", "d_log_hill_K", "d_log_hill_n"],
        map_column="map_theta_d_mut",
        get_columns=["titrant_name", "mutation"],
        in_run_prefix="theta_",
    ))

    if ctx.pair_labels:
        num_pair = len(ctx.pair_labels)
        theta_epi_rows = [
            {"titrant_name": t, "pair": p,
             "map_theta_epi": ti * num_pair + pi}
            for ti, t in enumerate(titrant_names)
            for pi, p in enumerate(ctx.pair_labels)
        ]
        specs.append(dict(
            input_df=pd.DataFrame(theta_epi_rows),
            params_to_get=["epi_logit_low", "epi_logit_delta",
                           "epi_log_hill_K", "epi_log_hill_n"],
            map_column="map_theta_epi",
            get_columns=["titrant_name", "pair"],
            in_run_prefix="theta_",
        ))

    return specs
