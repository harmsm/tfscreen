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

import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
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

    # HalfNormal scales for pairwise epistasis distributions
    theta_sigma_epi_logit_low_scale: float
    theta_sigma_epi_logit_delta_scale: float
    theta_sigma_epi_log_hill_K_scale: float
    theta_sigma_epi_log_hill_n_scale: float


@dataclass
class ThetaParam:
    """Assembled Hill equation parameters, shape (num_titrant_name, num_genotype)."""

    theta_low: torch.Tensor
    theta_high: torch.Tensor
    log_hill_K: torch.Tensor
    hill_n: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor


# ---------------------------------------------------------------------------
# Internal helpers shared by model and guide
# ---------------------------------------------------------------------------

def _assemble(wt, d_offsets, sigma_d, M,
              epi_offsets=None, sigma_epi=None, P=None):
    """
    Assemble per-genotype parameter array from WT + mutation deltas + epistasis.

    Parameters
    ----------
    wt : tensor, shape (T,)
    d_offsets : tensor, shape (T, num_mutation)
    sigma_d : tensor, shape (T,)
    M : tensor, shape (num_mutation, G)
    epi_offsets : tensor, shape (T, num_pair) or None
    sigma_epi : tensor, shape (T,) or None
    P : tensor, shape (num_pair, G) or None

    Returns
    -------
    tensor, shape (T, G)
    """
    d = d_offsets * sigma_d[:, None]           # [T, M]
    result = wt[:, None] + d @ M               # [T, G]
    if epi_offsets is not None:
        epi = epi_offsets * sigma_epi[:, None] # [T, P]
        result = result + epi @ P              # [T, G]
    return result


def _population_moments(logit_theta_low, logit_theta_high,
                        log_hill_K, log_hill_n, data):
    """
    Compute per-concentration population moments (mu, sigma) of logit(theta)
    over the assembled genotype population.

    Returns tensors of shape (T, num_conc, 1).
    """
    eps = 1e-6
    log_conc = torch.as_tensor(data.log_titrant_conc).float()[None, :, None]  # [1, C, 1]
    K = torch.exp(log_hill_K[:, None, :])                   # [T, 1, G]
    n = torch.exp(log_hill_n[:, None, :])                   # [T, 1, G]
    low = torch.sigmoid(logit_theta_low[:, None, :])        # [T, 1, G]
    high = torch.sigmoid(logit_theta_high[:, None, :])      # [T, 1, G]

    occ = torch.sigmoid(n * (log_conc - K))                 # [T, C, G]
    theta_all = torch.clamp(low + (high - low) * occ, eps, 1.0 - eps)
    logit_theta_all = torch.logit(theta_all)                # [T, C, G]

    mu = logit_theta_all.mean(dim=-1, keepdim=True)    # [T, C, 1]
    sigma = logit_theta_all.std(dim=-1, keepdim=True)  # [T, C, 1]
    return mu, sigma


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the mutation-decomposed hierarchical Hill model.

    Samples WT Hill parameters and per-mutation delta parameters, then
    assembles per-genotype Hill curves via matrix multiplication.

    Parameters
    ----------
    name : str
        Prefix for all Pyro sample sites.
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
    M_mat = torch.as_tensor(data.mut_geno_matrix)   # [num_mutation, G]
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
        P_mat = torch.as_tensor(data.pair_geno_matrix)  # [num_pair, G]
        num_pair = data.num_pair

        with pyro.plate(f"{name}_wt_epi_plate", T, dim=-1):
            sigma_epi_low = pyro.sample(
                f"{name}_sigma_epi_logit_low",
                dist.HalfNormal(priors.theta_sigma_epi_logit_low_scale))
            sigma_epi_delta = pyro.sample(
                f"{name}_sigma_epi_logit_delta",
                dist.HalfNormal(priors.theta_sigma_epi_logit_delta_scale))
            sigma_epi_K = pyro.sample(
                f"{name}_sigma_epi_log_hill_K",
                dist.HalfNormal(priors.theta_sigma_epi_log_hill_K_scale))
            sigma_epi_n = pyro.sample(
                f"{name}_sigma_epi_log_hill_n",
                dist.HalfNormal(priors.theta_sigma_epi_log_hill_n_scale))

        with pyro.plate(f"{name}_titrant_pair_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                epi_low_off = pyro.sample(
                    f"{name}_epi_logit_low_offset", dist.Normal(0.0, 1.0))
                epi_delta_off = pyro.sample(
                    f"{name}_epi_logit_delta_offset", dist.Normal(0.0, 1.0))
                epi_K_off = pyro.sample(
                    f"{name}_epi_log_hill_K_offset", dist.Normal(0.0, 1.0))
                epi_n_off = pyro.sample(
                    f"{name}_epi_log_hill_n_offset", dist.Normal(0.0, 1.0))
    else:
        P_mat = None
        sigma_epi_low = sigma_epi_delta = sigma_epi_K = sigma_epi_n = None
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
        epi_logit_low   = epi_low_off   * sigma_epi_low[:, None]    # [T, P]
        epi_logit_delta = epi_delta_off * sigma_epi_delta[:, None]
        epi_log_hill_K  = epi_K_off     * sigma_epi_K[:, None]
        epi_log_hill_n  = epi_n_off     * sigma_epi_n[:, None]

        pyro.deterministic(f"{name}_epi_logit_low",   epi_logit_low)
        pyro.deterministic(f"{name}_epi_logit_delta", epi_logit_delta)
        pyro.deterministic(f"{name}_epi_log_hill_K",  epi_log_hill_K)
        pyro.deterministic(f"{name}_epi_log_hill_n",  epi_log_hill_n)

    # ------------------------------------------------------------------
    # Assemble per-genotype parameters: shape (T, G)
    # ------------------------------------------------------------------
    logit_theta_low = _assemble(logit_low_wt, d_low_off, sigma_d_low, M_mat,
                                epi_low_off, sigma_epi_low, P_mat)
    logit_theta_delta = _assemble(logit_delta_wt, d_delta_off, sigma_d_delta, M_mat,
                                  epi_delta_off, sigma_epi_delta, P_mat)
    logit_theta_high = logit_theta_low + logit_theta_delta
    log_hill_K = _assemble(log_K_wt, d_K_off, sigma_d_K, M_mat,
                           epi_K_off, sigma_epi_K, P_mat)
    log_hill_n = _assemble(log_n_wt, d_n_off, sigma_d_n, M_mat,
                           epi_n_off, sigma_epi_n, P_mat)

    # Transform to natural scale and register
    theta_low = torch.sigmoid(logit_theta_low)
    theta_high = torch.sigmoid(logit_theta_high)
    hill_n = torch.exp(log_hill_n)

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
          data: GrowthData,
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the mutation-decomposed Hill model."""

    T = data.num_titrant_name
    num_mut = data.num_mutation
    M_mat = torch.as_tensor(data.mut_geno_matrix)
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
            f"{name}_{k}_loc", torch.full((T,), float(loc0)))
        wt_scale_params[k] = pyro.param(
            f"{name}_{k}_scale", torch.full((T,), float(scale0)),
            constraint=torch.distributions.constraints.positive)

    # sigma hyperpriors (LogNormal guide for HalfNormal prior)
    sigma_names = ["sigma_d_logit_low", "sigma_d_logit_delta",
                   "sigma_d_log_hill_K", "sigma_d_log_hill_n"]
    sigma_loc_params = {}
    sigma_scale_params = {}
    for k in sigma_names:
        sigma_loc_params[k] = pyro.param(
            f"{name}_{k}_loc", torch.full((T,), -1.0))
        sigma_scale_params[k] = pyro.param(
            f"{name}_{k}_scale", torch.full((T,), 0.1),
            constraint=torch.distributions.constraints.positive)

    # ------------------------------------------------------------------
    # Variational parameters for mutation delta offsets: shape (T, M)
    # ------------------------------------------------------------------
    delta_names = ["d_logit_low_offset", "d_logit_delta_offset",
                   "d_log_hill_K_offset", "d_log_hill_n_offset"]
    d_loc_params = {}
    d_scale_params = {}
    for k in delta_names:
        d_loc_params[k] = pyro.param(
            f"{name}_{k}_locs", torch.zeros(T, num_mut))
        d_scale_params[k] = pyro.param(
            f"{name}_{k}_scales", torch.ones(T, num_mut),
            constraint=torch.distributions.constraints.positive)

    # ------------------------------------------------------------------
    # Optional epistasis variational parameters
    # ------------------------------------------------------------------
    if has_epi:
        P_mat = torch.as_tensor(data.pair_geno_matrix)
        num_pair = data.num_pair
        epi_sigma_names = ["sigma_epi_logit_low", "sigma_epi_logit_delta",
                           "sigma_epi_log_hill_K", "sigma_epi_log_hill_n"]
        epi_sigma_loc_params = {}
        epi_sigma_scale_params = {}
        for k in epi_sigma_names:
            epi_sigma_loc_params[k] = pyro.param(
                f"{name}_{k}_loc", torch.full((T,), -1.0))
            epi_sigma_scale_params[k] = pyro.param(
                f"{name}_{k}_scale", torch.full((T,), 0.1),
                constraint=torch.distributions.constraints.positive)

        epi_names = ["epi_logit_low_offset", "epi_logit_delta_offset",
                     "epi_log_hill_K_offset", "epi_log_hill_n_offset"]
        epi_loc_params = {}
        epi_scale_params = {}
        for k in epi_names:
            epi_loc_params[k] = pyro.param(
                f"{name}_{k}_locs", torch.zeros(T, num_pair))
            epi_scale_params[k] = pyro.param(
                f"{name}_{k}_scales", torch.ones(T, num_pair),
                constraint=torch.distributions.constraints.positive)
    else:
        P_mat = None

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
        with pyro.plate(f"{name}_wt_epi_plate", T, dim=-1):
            sigma_epi_low = pyro.sample(
                f"{name}_sigma_epi_logit_low",
                dist.LogNormal(epi_sigma_loc_params["sigma_epi_logit_low"],
                               epi_sigma_scale_params["sigma_epi_logit_low"]))
            sigma_epi_delta = pyro.sample(
                f"{name}_sigma_epi_logit_delta",
                dist.LogNormal(epi_sigma_loc_params["sigma_epi_logit_delta"],
                               epi_sigma_scale_params["sigma_epi_logit_delta"]))
            sigma_epi_K = pyro.sample(
                f"{name}_sigma_epi_log_hill_K",
                dist.LogNormal(epi_sigma_loc_params["sigma_epi_log_hill_K"],
                               epi_sigma_scale_params["sigma_epi_log_hill_K"]))
            sigma_epi_n = pyro.sample(
                f"{name}_sigma_epi_log_hill_n",
                dist.LogNormal(epi_sigma_loc_params["sigma_epi_log_hill_n"],
                               epi_sigma_scale_params["sigma_epi_log_hill_n"]))

        with pyro.plate(f"{name}_titrant_pair_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                epi_low_off = pyro.sample(
                    f"{name}_epi_logit_low_offset",
                    dist.Normal(epi_loc_params["epi_logit_low_offset"],
                                epi_scale_params["epi_logit_low_offset"]))
                epi_delta_off = pyro.sample(
                    f"{name}_epi_logit_delta_offset",
                    dist.Normal(epi_loc_params["epi_logit_delta_offset"],
                                epi_scale_params["epi_logit_delta_offset"]))
                epi_K_off = pyro.sample(
                    f"{name}_epi_log_hill_K_offset",
                    dist.Normal(epi_loc_params["epi_log_hill_K_offset"],
                                epi_scale_params["epi_log_hill_K_offset"]))
                epi_n_off = pyro.sample(
                    f"{name}_epi_log_hill_n_offset",
                    dist.Normal(epi_loc_params["epi_log_hill_n_offset"],
                                epi_scale_params["epi_log_hill_n_offset"]))
    else:
        sigma_epi_low = sigma_epi_delta = sigma_epi_K = sigma_epi_n = None
        epi_low_off = epi_delta_off = epi_K_off = epi_n_off = None

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    logit_theta_low = _assemble(logit_low_wt, d_low_off, sigma_d_low, M_mat,
                                epi_low_off, sigma_epi_low, P_mat)
    logit_theta_delta = _assemble(logit_delta_wt, d_delta_off, sigma_d_delta, M_mat,
                                  epi_delta_off, sigma_epi_delta, P_mat)
    logit_theta_high = logit_theta_low + logit_theta_delta
    log_hill_K = _assemble(log_K_wt, d_K_off, sigma_d_K, M_mat,
                           epi_K_off, sigma_epi_K, P_mat)
    log_hill_n = _assemble(log_n_wt, d_n_off, sigma_d_n, M_mat,
                           epi_n_off, sigma_epi_n, P_mat)

    theta_low = torch.sigmoid(logit_theta_low)
    theta_high = torch.sigmoid(logit_theta_high)
    hill_n = torch.exp(log_hill_n)

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

def run_model(theta_param: ThetaParam, data) -> torch.Tensor:
    """
    Calculate theta via the Hill equation using the assembled parameters.

    Identical to ``hill.run_model``; the assembled ``ThetaParam`` already
    has the correct per-genotype shape so no additional transformation is needed.
    """
    # Select genotypes from the last dim; handles 2D [T,G] and 3D [S,T,G] shapes.
    geno_idx = torch.as_tensor(data.geno_theta_idx, dtype=torch.long)
    theta_low = theta_param.theta_low[..., geno_idx].unsqueeze(-2)
    theta_high = theta_param.theta_high[..., geno_idx].unsqueeze(-2)
    log_hill_K = theta_param.log_hill_K[..., geno_idx].unsqueeze(-2)
    hill_n = theta_param.hill_n[..., geno_idx].unsqueeze(-2)

    log_titrant = torch.as_tensor(data.log_titrant_conc).float()[None, :, None]
    occupancy = torch.sigmoid(hill_n * (log_titrant - log_hill_K))
    theta_calc = theta_low + (theta_high - theta_low) * occupancy

    if data.scatter_theta == 1:
        new_shape = theta_calc.shape[:-3] + (1, 1, 1, 1) + theta_calc.shape[-3:]
        theta_calc = theta_calc.reshape(new_shape)

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
    p["theta_sigma_epi_logit_low_scale"] = 0.5
    p["theta_sigma_epi_logit_delta_scale"] = 0.5
    p["theta_sigma_epi_log_hill_K_scale"] = 0.5
    p["theta_sigma_epi_log_hill_n_scale"] = 0.25
    return p


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    T = data.num_titrant_name
    M = data.num_mutation
    guesses = {}
    guesses[f"{name}_logit_low_wt"] = torch.full((T,), 2.0)
    guesses[f"{name}_logit_delta_wt"] = torch.full((T,), -4.0)
    guesses[f"{name}_log_hill_K_wt"] = torch.full((T,), -4.1)
    guesses[f"{name}_log_hill_n_wt"] = torch.full((T,), 0.7)
    guesses[f"{name}_sigma_d_logit_low"] = torch.full((T,), 0.5)
    guesses[f"{name}_sigma_d_logit_delta"] = torch.full((T,), 0.5)
    guesses[f"{name}_sigma_d_log_hill_K"] = torch.full((T,), 0.5)
    guesses[f"{name}_sigma_d_log_hill_n"] = torch.full((T,), 0.3)
    guesses[f"{name}_d_logit_low_offset"] = torch.zeros(T, M)
    guesses[f"{name}_d_logit_delta_offset"] = torch.zeros(T, M)
    guesses[f"{name}_d_log_hill_K_offset"] = torch.zeros(T, M)
    guesses[f"{name}_d_log_hill_n_offset"] = torch.zeros(T, M)
    if data.num_pair > 0:
        P = data.num_pair
        guesses[f"{name}_sigma_epi_logit_low"] = torch.full((T,), 0.3)
        guesses[f"{name}_sigma_epi_logit_delta"] = torch.full((T,), 0.3)
        guesses[f"{name}_sigma_epi_log_hill_K"] = torch.full((T,), 0.3)
        guesses[f"{name}_sigma_epi_log_hill_n"] = torch.full((T,), 0.2)
        guesses[f"{name}_epi_logit_low_offset"] = torch.zeros(T, P)
        guesses[f"{name}_epi_logit_delta_offset"] = torch.zeros(T, P)
        guesses[f"{name}_epi_log_hill_K_offset"] = torch.zeros(T, P)
        guesses[f"{name}_epi_log_hill_n_offset"] = torch.zeros(T, P)
    return guesses


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())
