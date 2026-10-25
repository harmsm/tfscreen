"""
Per-tube level offset: one ln_cfu shift per tube, shared by every genotype
in it, with a constant prior.

A tube is one sequenced sample (replicate x time x condition x titrant; see
``docs/source/process-raw.rst``, "One tube per time-point"). Anything that
moves every genotype's reads in a tube together lands here: the tube's
composition offset (PCR and genotype-calling efficiency, the ``__unknown__``
share; study 0b measured about 0.02 ln units) and any error in the tube's
supplied total cells (per-tube OD600 scattered about 0.17 ln units around a
smooth curve in study 0c). Roadmap D2 (planning/analysis-roadmap.md): unlike
``normal``, whose offset is a growth-rate shift scaled by elapsed time, this
one is a level with no time dependence.

Prior
-----
    sigma          ~ HalfNormal(sigma_prior_scale)
    offset[tube]   ~ Normal(0, sigma)

Guide
-----
    sigma          ~ LogNormal(sigma_loc, sigma_scale)
    offset[tube]   ~ Normal(offset_loc, offset_scale)

The offset is a per-tube global site (not per genotype), so it is
mini-batch safe. Its contribution has shape ``(*tube_shape, 1)`` and
broadcasts over the genotype axis of ``ln_cfu_pred``.
"""

import math
from typing import Any, Dict

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass, field

from tfscreen.tfmodel.data_class import GrowthData

from ._tubes import tube_extract_spec


@dataclass(frozen=True)
class ModelPriors:
    """
    sigma_prior_scale : float
        Scale of the HalfNormal prior on sigma, the SD of the per-tube
        offsets (ln units).
    sigma_fixed : float
        If > 0, sigma is held at this value (a deterministic site, no guide
        parameters) instead of learned. Default 0 (learned). A learned sigma
        lets the offsets grow into a free level per tube: on the first
        real-data fit (2026-09-29) it went from its 0.2 prior scale to 0.53,
        the offsets ran from -2.7 to +3.0 in a smooth pattern over IPTG and
        time, and they carried the population's growth in place of k and m,
        which went unphysical. Holding sigma near the per-tube OD600 scatter
        (about 0.17, study 0c) keeps the tube totals binding. Static.
    """
    sigma_prior_scale: float
    sigma_fixed: float = field(pytree_node=False, default=0.0)


def _tube_shape(data: GrowthData):
    return (data.num_replicate, data.num_time, data.num_condition_pre,
            data.num_condition_sel, data.num_titrant_name, data.num_titrant_conc)


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Sample the per-tube offsets and return them shaped
    ``(*tube_shape, 1)`` for adding to ``ln_cfu_pred``.
    """
    if float(priors.sigma_fixed) > 0:
        sigma = pyro.deterministic(f"{name}_sigma",
                                   jnp.array(float(priors.sigma_fixed)))
    else:
        sigma = pyro.sample(f"{name}_sigma",
                            dist.HalfNormal(priors.sigma_prior_scale))
    shape = _tube_shape(data)
    num_tubes = math.prod(shape)
    with pyro.plate(f"{name}_tubes", num_tubes, dim=-1):
        offset = pyro.sample(f"{name}_offset", dist.Normal(0.0, sigma))
    return offset.reshape(*shape, 1)


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """LogNormal guide for sigma, per-tube Normal guides for the offsets."""
    shape = _tube_shape(data)
    num_tubes = math.prod(shape)

    if not float(priors.sigma_fixed) > 0:
        sigma_loc = pyro.param(f"{name}_sigma_loc",
                               jnp.log(jnp.array(priors.sigma_prior_scale) / 4.0))
        sigma_scale = pyro.param(f"{name}_sigma_scale", jnp.array(0.5),
                                 constraint=dist.constraints.greater_than(1e-4))
        pyro.sample(f"{name}_sigma", dist.LogNormal(sigma_loc, sigma_scale))

    offset_loc = pyro.param(f"{name}_offset_loc", jnp.zeros(num_tubes))
    offset_scale = pyro.param(f"{name}_offset_scale", jnp.full(num_tubes, 1e-2),
                              constraint=dist.constraints.greater_than(1e-4))
    with pyro.plate(f"{name}_tubes", num_tubes, dim=-1):
        offset = pyro.sample(f"{name}_offset",
                             dist.Normal(offset_loc, offset_scale))
    return offset.reshape(*shape, 1)


def get_hyperparameters() -> Dict[str, Any]:
    """
    sigma_prior_scale = 0.2: wide enough for the per-tube OD600 scatter of
    study 0c (about 0.17 ln units) when each tube's total comes from its own
    OD600, and for the small composition offset (about 0.02, study 0b).
    """
    return {"sigma_prior_scale": 0.2, "sigma_fixed": 0.0}


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    # a guess for a held sigma is unused (the site is deterministic)
    return {f"{name}_sigma": jnp.array(0.05)}


def get_extract_specs(ctx) -> list:
    """Per-tube offsets labeled by tube design, and their SD."""
    return tube_extract_spec(ctx, "sample_offset", "offset", "sigma")
