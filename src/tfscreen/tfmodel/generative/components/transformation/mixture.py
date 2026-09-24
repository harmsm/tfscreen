"""
Congression as an observable-level mixture of clean and congressed cells.

A genotype's cells are a mixture: most carry only its plasmid ("clean"),
and a fraction ``w_cong = f_g (1 - exp(-lambda))`` also carry co-resident
plasmids (Poisson(lambda) co-residents, drawn from the bulk library).
Each class grows at its own rate, so the classes are mixed at the level of
``exp(ln_cfu)``, not rates or theta:

    ln_cfu_g(t) = ln_cfu0_g + log sum_c w_c exp(G_c(t))

The congressed term ``E_S exp(G_{g,S}(t))`` over random co-resident sets S is
evaluated by a fixed quadrature: K co-resident sets per genotype, drawn once
by ``ModelOrchestrator._draw_coresident_sets`` (``data.coresident_idx``,
``data.coresident_n``) and stratified by co-resident count. A congressed cell
takes its theta and TF activity from the plasmid with the highest theta (at
each titrant concentration) and the share-weighted mean dk_geno of its
plasmids (dilution), matching ``simulate/cell_rules.py``.

See ``planning/congression-physics-plan.md`` ("Fit design (step 3)") and
``planning/studies/congression-estimator/`` for why this estimator.
"""

import math

import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
import pandas as pd
from flax.struct import dataclass

from tfscreen.tfmodel.data_class import GrowthData
from tfscreen.tfmodel.generative.components.transformation._classes import (
    CellClasses,
)


# The mixture's congressed cells draw co-residents' theta, activity and
# dk_geno from the whole library, so jax_model must supply library-ordered
# populations of all three (computed locally in training, or passed in as
# data.external_*_population by prediction code).
NEEDS_POPULATION = True


@dataclass(frozen=True)
class ModelPriors:
    """
    LogNormal prior on the congression Poisson rate lambda (the log-space
    location and scale).
    """

    lam_loc: float
    lam_scale: float


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Sample the global congression rate lambda.

    Returns
    -------
    jnp.ndarray
        lambda (scalar).
    """
    lam = pyro.sample(f"{name}_lam",
                      dist.LogNormal(priors.lam_loc, priors.lam_scale))
    pyro.deterministic(f"{name}_lam_value", lam)
    return lam


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """
    LogNormal variational posterior for lambda.
    """
    lam_loc = pyro.param(f"{name}_lam_loc", jnp.array(priors.lam_loc))
    lam_scale = pyro.param(f"{name}_lam_scale", jnp.array(priors.lam_scale),
                           constraint=dist.constraints.positive)
    return pyro.sample(f"{name}_lam", dist.LogNormal(lam_loc, lam_scale))


def _log_class_weights(lam, bulk_fraction, coresident_n):
    """
    Log mixture weights, shape (1 + K, B): clean first, then the K sets.

    ``w_cong = f (1 - exp(-lambda))``; set k (with n_k co-residents) gets
    ``w_cong * P(N = n_k | N in strata; lambda) / K_{n_k}``, where the
    stratum probabilities are Poisson(lambda) renormalized over the counts
    that have sets. Written in log space so the lambda gradient stays finite
    when ``f = 0`` (those congressed weights are -inf and drop out of the
    logsumexp).
    """
    n = coresident_n.astype(float)                                  # (K,)
    num_in_stratum = jnp.sum(coresident_n[None, :] == coresident_n[:, None],
                             axis=1).astype(float)                  # (K,)

    # Unnormalized log Poisson pmf; exp(-lambda) cancels in the normalization.
    log_pmf = n * jnp.log(lam) - jax.scipy.special.gammaln(n + 1.0)
    # Each stratum appears K_n times; divide by K_n so it counts once.
    log_norm = jax.scipy.special.logsumexp(log_pmf - jnp.log(num_in_stratum))
    log_stratum = log_pmf - log_norm - jnp.log(num_in_stratum)     # (K,)

    log_p_cong = jnp.log(-jnp.expm1(-lam))
    log_f = jnp.log(bulk_fraction)                                  # (B,)

    log_w_sets = log_f[None, :] + log_p_cong + log_stratum[:, None]  # (K, B)
    log_w_clean = jnp.log1p(-bulk_fraction * (-jnp.expm1(-lam)))     # (B,)

    return jnp.concatenate([log_w_clean[None, :], log_w_sets], axis=0)


def cell_classes(focal, population, lam, data: GrowthData) -> CellClasses:
    """
    Build the clean and congressed cell classes for the batch's genotypes.

    Parameters
    ----------
    focal : tuple of jnp.ndarray
        ``(theta, activity, dk_geno)`` for the batch. ``theta`` has the
        growth layout ``(1, 1, 1, 1, num_titrant_name, num_titrant_conc,
        batch)``; ``activity`` and ``dk_geno`` are ``(1, ..., 1, batch)``.
    population : tuple of jnp.ndarray
        Library-ordered ``(theta, activity, dk_geno)`` for every genotype:
        ``theta`` ``(1, 1, 1, 1, tn, tc, num_genotype)``; ``activity`` and
        ``dk_geno`` ``(num_genotype,)``.
    lam : jnp.ndarray
        Congression Poisson rate.
    data : GrowthData
        Uses ``batch_idx``, ``bulk_fraction``, ``coresident_idx``,
        ``coresident_n``.

    Returns
    -------
    CellClasses
        Leading class axis of size 1 + K (clean first).
    """
    theta, activity, dk_geno = focal
    theta_pop, activity_pop, dk_pop = population

    idx = data.coresident_idx[data.batch_idx]                  # (B, K, N)
    valid = idx >= 0
    safe = jnp.where(valid, idx, 0)

    # --- theta and activity: the highest-theta plasmid sets both ---------
    # Slots: the focal plasmid, then the set's co-residents.
    co_theta = theta_pop[..., safe]                             # (..., B, K, N)
    focal_theta = jnp.broadcast_to(theta[..., None, None],
                                   co_theta.shape[:-1] + (1,))
    slot_theta = jnp.concatenate([focal_theta, co_theta], axis=-1)
    slot_valid = jnp.concatenate(
        [jnp.ones(valid.shape[:-1] + (1,), dtype=bool), valid], axis=-1)
    slot_theta = jnp.where(slot_valid, slot_theta, -jnp.inf)
    best = jnp.argmax(slot_theta, axis=-1)[..., None]           # (..., B, K, 1)
    cell_theta = jnp.take_along_axis(slot_theta, best, axis=-1)[..., 0]

    act_focal = jnp.broadcast_to(activity, theta.shape)
    slot_act = jnp.concatenate(
        [jnp.broadcast_to(act_focal[..., None, None], focal_theta.shape),
         jnp.broadcast_to(activity_pop[safe], co_theta.shape)], axis=-1)
    cell_act = jnp.take_along_axis(slot_act, best, axis=-1)[..., 0]

    # --- dk_geno: share-weighted mean (dilution) --------------------------
    co_dk = jnp.where(valid, dk_pop[safe], 0.0)                 # (B, K, N)
    num_plasmids = 1.0 + jnp.sum(valid, axis=-1)                # (B, K)
    cell_dk = (dk_geno[..., None] + jnp.sum(co_dk, axis=-1)) / num_plasmids

    # --- stack classes on a leading axis (clean first) --------------------
    def stack(clean, congressed):
        return jnp.concatenate([clean[None],
                                jnp.moveaxis(congressed, -1, 0)], axis=0)

    classes_theta = stack(theta, cell_theta)
    classes_act = stack(act_focal, cell_act)
    classes_dk = stack(dk_geno, cell_dk)

    bulk_fraction = data.bulk_fraction[data.batch_idx]
    log_w = _log_class_weights(lam, bulk_fraction, data.coresident_n)
    log_w = log_w.reshape((log_w.shape[0],) + (1,) * (theta.ndim - 1)
                          + (log_w.shape[-1],))

    return CellClasses(theta=classes_theta, activity=classes_act,
                       dk_geno=classes_dk, log_weight=log_w)


def get_hyperparameters(lam_mean=None, lam_std=None):
    """
    Default hyperparameters: the LogNormal prior on lambda.

    Parameters
    ----------
    lam_mean, lam_std : float, optional
        Experimentally measured mean and standard deviation of lambda, in
        linear space, moment-matched onto the LogNormal. Provide both or
        neither; without them a placeholder prior is used (tfs-configure-model
        requires a measured value for this component).

    Raises
    ------
    ValueError
        If exactly one of ``lam_mean``/``lam_std`` is given, or either is
        not strictly positive.
    """
    if (lam_mean is None) != (lam_std is None):
        raise ValueError(
            "lam_mean and lam_std must be provided together (or neither)."
        )

    parameters = {}
    if lam_mean is not None:
        if lam_mean <= 0:
            raise ValueError(f"lam_mean must be > 0; got {lam_mean}")
        if lam_std <= 0:
            raise ValueError(f"lam_std must be > 0; got {lam_std}")
        sigma2 = math.log(1.0 + (lam_std / lam_mean) ** 2)
        parameters["lam_loc"] = math.log(lam_mean) - sigma2 / 2.0
        parameters["lam_scale"] = math.sqrt(sigma2)
    else:
        parameters["lam_loc"] = 0.0
        parameters["lam_scale"] = 1.0

    return parameters


def get_guesses(name, data, lam_mean=None):
    """Initial guess for lambda: the measured mean if given, else 1.0."""
    return {f"{name}_lam": lam_mean if lam_mean is not None else 1.0}


def get_priors(lam_mean=None, lam_std=None):
    """Build the ModelPriors from ``get_hyperparameters``."""
    return ModelPriors(**get_hyperparameters(lam_mean=lam_mean,
                                             lam_std=lam_std))


def get_extract_specs(ctx):
    lam_df = pd.DataFrame({"parameter": ["lam"], "map_all": [0]})
    return [dict(
        input_df=lam_df,
        params_to_get=["lam"],
        map_column="map_all",
        get_columns=["parameter"],
        in_run_prefix="transformation_",
    )]
