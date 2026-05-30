"""
Per-tube growth-rate offset component.

Each physical tube (unique combination of replicate × time × condition_pre ×
condition_sel × titrant_name × titrant_conc) may experience slightly different
environmental conditions, producing a shared growth-rate offset delta_k for
every genotype in that tube.  Its contribution to ln_cfu is:

    delta_sample = delta_k * (t_pre + t_sel)

where t_pre and t_sel are the per-tube growth durations (hr).  Because the
offset is in growth-rate units (hr⁻¹), longer experiments naturally amplify
the noise, which is the correct physical behaviour.

Prior
-----
    sigma_env  ~ HalfNormal(sigma_env_scale)   [global scale, hr⁻¹]
    delta_k[s] ~ Normal(0, sigma_env)           [per tube, hr⁻¹]

sigma_env_scale should be set to roughly the expected tube-to-tube growth-rate
standard deviation.  A value of ~0.001–0.002 hr⁻¹ is typical for well-controlled
liquid cultures (≈3–5 % of a representative k ~ 0.04 hr⁻¹).

Guide
-----
    sigma_env  ~ LogNormal(loc, scale)
    delta_k[s] ~ Normal(loc_s, scale_s)   per-tube variational params

Implementation note on plate dimensions
----------------------------------------
All R×T×CP×CS×TN×TC tubes are flattened into a single plate at dim=-1.
The sample site is a scalar Normal (the plate handles independence), giving
delta_k_flat shape (num_tubes,).  After sampling the flat array is reshaped
to (*shape, 1) so the trailing singleton broadcasts over the genotype dim
when added to ln_cfu_pred.  The tubes plate is closed before the genotype
plate opens, so sharing dim=-1 causes no conflict.
"""

import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from tfscreen.tfmodel.data_class import GrowthData
from typing import Dict, Any


@dataclass(frozen=True)
class ModelPriors:
    """
    Hyperparameters for the per-tube sample-offset model.

    Attributes
    ----------
    sigma_env_scale : float
        Scale of the HalfNormal prior on sigma_env (hr⁻¹).
    """
    sigma_env_scale: float


def _num_tubes(data: GrowthData) -> int:
    return (data.num_replicate * data.num_time * data.num_condition_pre *
            data.num_condition_sel * data.num_titrant_name * data.num_titrant_conc)


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Sample per-tube growth-rate offsets and return their ln_cfu contribution.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample / deterministic sites.
    data : GrowthData
        Observed growth data; used for tensor-shape metadata and t_pre / t_sel.
    priors : ModelPriors
        Pytree containing ``sigma_env_scale``.

    Returns
    -------
    jnp.ndarray
        Shape (num_replicate, num_time, num_condition_pre, num_condition_sel,
        num_titrant_name, num_titrant_conc, 1).  The trailing singleton
        broadcasts over the genotype dimension when added to ln_cfu_pred.
    """
    sigma_env = pyro.sample(
        f"{name}_sigma_env",
        dist.HalfNormal(priors.sigma_env_scale)
    )

    num_tubes = _num_tubes(data)
    shape = (data.num_replicate, data.num_time, data.num_condition_pre,
             data.num_condition_sel, data.num_titrant_name, data.num_titrant_conc)

    with pyro.plate(f"{name}_tubes", num_tubes, dim=-1):
        delta_k_flat = pyro.sample(
            f"{name}_delta_k",
            dist.Normal(0.0, sigma_env)
        )

    # Reshape to (*shape, 1): trailing 1 broadcasts over the genotype plate.
    delta_k = delta_k_flat.reshape(*shape, 1)

    # t_pre and t_sel are experimentally set — same for every genotype in a
    # given tube.  Keep genotype axis as a singleton (0:1) to match delta_k.
    t_total = data.t_pre[..., 0:1] + data.t_sel[..., 0:1]

    return delta_k * t_total


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """
    Variational guide for the per-tube sample offset.

    Uses a LogNormal guide for sigma_env and per-tube Normal guides for delta_k.
    """
    num_tubes = _num_tubes(data)
    shape = (data.num_replicate, data.num_time, data.num_condition_pre,
             data.num_condition_sel, data.num_titrant_name, data.num_titrant_conc)

    # --- sigma_env ---
    sigma_env_loc = pyro.param(
        f"{name}_sigma_env_loc",
        jnp.log(jnp.array(priors.sigma_env_scale) / 4.0),
    )
    sigma_env_scale_param = pyro.param(
        f"{name}_sigma_env_scale",
        jnp.array(0.5),
        constraint=dist.constraints.greater_than(1e-4),
    )
    pyro.sample(
        f"{name}_sigma_env",
        dist.LogNormal(sigma_env_loc, sigma_env_scale_param)
    )

    # --- delta_k[num_tubes] ---
    delta_k_loc = pyro.param(
        f"{name}_delta_k_loc",
        jnp.zeros(num_tubes),
    )
    delta_k_scale = pyro.param(
        f"{name}_delta_k_scale",
        jnp.full(num_tubes, 1e-2),
        constraint=dist.constraints.greater_than(1e-4),
    )

    with pyro.plate(f"{name}_tubes", num_tubes, dim=-1):
        delta_k_flat = pyro.sample(
            f"{name}_delta_k",
            dist.Normal(delta_k_loc, delta_k_scale)
        )

    delta_k = delta_k_flat.reshape(*shape, 1)
    t_total = data.t_pre[..., 0:1] + data.t_sel[..., 0:1]
    return delta_k * t_total


def get_hyperparameters() -> Dict[str, Any]:
    """
    Default hyperparameters.

    sigma_env_scale = 0.002 hr⁻¹ is roughly 5 % of a representative
    growth rate of 0.04 hr⁻¹.  Adjust to match the expected magnitude of
    tube-to-tube environmental variation in your experiment.
    """
    return {"sigma_env_scale": 0.002}


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    return {f"{name}_sigma_env": jnp.array(1e-4)}


def get_extract_specs(ctx) -> list:
    return []
