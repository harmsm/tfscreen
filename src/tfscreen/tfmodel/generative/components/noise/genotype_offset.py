"""
Per-genotype binding offset: binding sees each binding genotype's theta
shifted by one offset of its own.

    theta_binding = theta_calc + delta_g,    delta_g = tau * z_g,
    z_g ~ Normal(0, 1),    tau ~ HalfNormal(offset_scale_scale)

One offset per binding genotype, the same at every titrant concentration,
added in theta units (the binding likelihood is an unclipped Normal, so the
sum need not stay in [0, 1]). For ``theta_binding_noise`` only: the growth
side never sees it, so the curve shared with growth is the in vivo one and
binding measures it up to the offset.

Why (planning/joint-fit-return.md step 4b): a genotype's in vitro binding
can differ from its in vivo occupancy (M42I grows 0.3-0.4 below its 2 uM
binding at low IPTG), and each binding curve carries its day's
normalization error, which is shared by the whole curve (wt's day-to-day
scatter at 2 uM is about three times the point-to-point scatter within a
curve). Without an offset, one such genotype pulls the shared growth
parameters; with it, the genotype disagrees alone, and tau says how far
binding and growth disagree across genotypes. tau is learned unless
``offset_scale_fixed`` > 0 holds it.

Non-centered (z, not delta, is the latent) so a MAP has no funnel at
tau -> 0. The offsets live in a plate over the binding genotypes only
(``{name}_binding_plate``, sized ``data.num_genotype`` of the binding data)
and are indexed by ``data.binding_ordinal``, so a library of 200,000
genotypes does not carry 200,000 prior-only offsets. The plate name holds no
growth tensor dimension name, so prediction remaps it by binding genotype
label (``analysis/prediction.py``).
"""

from typing import Any, Dict

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass, field

from tfscreen.tfmodel.data_class import BindingData


@dataclass(frozen=True)
class ModelPriors:
    """
    Attributes
    ----------
    offset_scale_scale : float
        Scale of the HalfNormal prior on tau, in theta units.
    offset_scale_fixed : float
        If > 0, tau is held at this value (a deterministic site, no guide
        parameters) instead of learned. Default 0 keeps it learned.
    """
    offset_scale_scale: float
    offset_scale_fixed: float = field(pytree_node=False, default=0.0)


def _tau_fixed(priors):
    return float(getattr(priors, "offset_scale_fixed", 0.0))


def _ordinal(fx_calc, data):
    if data is None or getattr(data, "binding_ordinal", None) is None:
        return jnp.arange(fx_calc.shape[-1])
    return data.binding_ordinal


def _num_binding(fx_calc, data):
    return fx_calc.shape[-1] if data is None else data.num_genotype


def _apply(name, fx_calc, tau, z, data):
    offset = tau * z
    pyro.deterministic(f"{name}_offset", offset)
    if offset.shape[-1] == _num_binding(fx_calc, data):
        offset = offset[..., _ordinal(fx_calc, data)]
    return fx_calc + offset


def define_model(name: str,
                 fx_calc: jnp.ndarray,
                 priors: ModelPriors,
                 data: BindingData = None) -> jnp.ndarray:
    """
    Add each binding genotype's offset to its theta.

    Parameters
    ----------
    name : str
        Prefix for the Numpyro sites.
    fx_calc : jnp.ndarray
        Theta at the binding concentrations; the last axis is the batch's
        binding genotypes.
    priors : ModelPriors
    data : BindingData, optional
        Supplies ``num_genotype`` (the plate size) and ``binding_ordinal``.
        None: one offset per position of ``fx_calc``'s last axis.

    Returns
    -------
    jnp.ndarray
        ``fx_calc`` plus the offsets, same shape.
    """
    fixed = _tau_fixed(priors)
    if fixed > 0:
        tau = pyro.deterministic(f"{name}_offset_scale", jnp.asarray(fixed))
    else:
        tau = pyro.sample(f"{name}_offset_scale",
                          dist.HalfNormal(priors.offset_scale_scale))
    with pyro.plate(f"{name}_binding_plate", _num_binding(fx_calc, data), dim=-1):
        z = pyro.sample(f"{name}_offset_z", dist.Normal(0.0, 1.0))
    return _apply(name, fx_calc, tau, z, data)


def guide(name: str,
          fx_calc: jnp.ndarray,
          priors: ModelPriors,
          data: BindingData = None) -> jnp.ndarray:
    """Mean-field guide: LogNormal tau, Normal z per binding genotype."""
    fixed = _tau_fixed(priors)
    if fixed > 0:
        tau = jnp.asarray(fixed)
    else:
        loc = pyro.param(f"{name}_offset_scale_loc",
                         jnp.log(jnp.asarray(priors.offset_scale_scale) / 2.0))
        scale = pyro.param(f"{name}_offset_scale_scale", jnp.asarray(0.1),
                           constraint=dist.constraints.greater_than(1e-4))
        tau = pyro.sample(f"{name}_offset_scale", dist.LogNormal(loc, scale))
    n = _num_binding(fx_calc, data)
    z_locs = pyro.param(f"{name}_offset_z_locs", jnp.zeros(n))
    z_scales = pyro.param(f"{name}_offset_z_scales", jnp.ones(n),
                          constraint=dist.constraints.greater_than(1e-4))
    with pyro.plate(f"{name}_binding_plate", n, dim=-1):
        z = pyro.sample(f"{name}_offset_z", dist.Normal(z_locs, z_scales))
    offset = tau * z
    if offset.shape[-1] == n:
        offset = offset[..., _ordinal(fx_calc, data)]
    return fx_calc + offset


def get_hyperparameters() -> Dict[str, Any]:
    """
    ``offset_scale_scale`` 0.2: tau mostly below 0.4 in theta, about the
    largest disagreement seen (M42I) and the day-to-day scatter (0.13).
    """
    return {"offset_scale_scale": 0.2, "offset_scale_fixed": 0.0}


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())


def get_guesses(name: str, data: BindingData) -> Dict[str, Any]:
    """Start at no offsets and a small tau."""
    return {f"{name}_offset_scale": jnp.asarray(0.05),
            f"{name}_offset_z": jnp.zeros(data.num_genotype)}


def get_extract_specs(ctx) -> list:
    """tau, and one offset per binding genotype."""
    import pandas as pd
    specs = [dict(
        input_df=pd.DataFrame({"parameter": ["offset_scale"], "map_all": [0]}),
        params_to_get=["offset_scale"],
        map_column="map_all",
        get_columns=["parameter"],
        in_run_prefix="theta_binding_noise_",
    )]
    labels = getattr(ctx, "binding_genotypes", None)
    if labels is not None:
        specs.append(dict(
            input_df=pd.DataFrame({"genotype": list(labels),
                                   "map_binding": range(len(labels))}),
            params_to_get=["offset"],
            map_column="map_binding",
            get_columns=["genotype"],
            in_run_prefix="theta_binding_noise_",
        ))
    return specs
