import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Tuple

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    growth_k_hyper_loc_loc: float
    growth_k_hyper_loc_scale: float
    growth_k_hyper_scale: float
    
    growth_m_hyper_loc_loc: float
    growth_m_hyper_loc_scale: float
    growth_m_hyper_scale: float


def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> Tuple[jnp.ndarray, ...]:
    """
    Growth parameters k_xx and m_xx versus condition, where xx are things like
    pheS+4CP, kanR-kan, etc. These go into the model as k + m*theta. Assigns
    each condition/replicate a normal prior. Returns full k_pre, m_pre, k_sel
    and m_sel tensors.

    Priors
    ------
    priors.growth_k_hyper_loc_loc
    priors.growth_k_hyper_loc_scale
    priors.growth_k_hyper_scale
    priors.growth_m_hyper_loc_loc
    priors.growth_m_hyper_loc_scale
    priors.growth_m_hyper_scale

    Data
    ----
    data.num_condition
    data.num_replicate
    data.map_condition_pre
    data.map_condition_sel
    """

    growth_k_hyper_loc = pyro.sample(
        f"{name}_k_hyper_loc",
        dist.Normal(priors.growth_k_hyper_loc_loc,
                    priors.growth_k_hyper_loc_scale)
    )
    growth_k_hyper_scale = pyro.sample(
        f"{name}_k_hyper_scale",
        dist.HalfNormal(priors.growth_k_hyper_scale)
    )

    growth_m_hyper_loc = pyro.sample(
        f"{name}_m_hyper_loc",
        dist.Normal(priors.growth_m_hyper_loc_loc,
                    priors.growth_m_hyper_loc_scale)
    )
    growth_m_hyper_scale = pyro.sample(
        f"{name}_m_hyper_scale",
        dist.HalfNormal(priors.growth_m_hyper_scale)
    )
    
    # Loop over conditions and replicates
    with pyro.plate(f"{name}_condition_parameters",data.num_condition):
        growth_k_offset = pyro.sample(f"{name}_k_offset", dist.Normal(0, 1))
        growth_m_offset = pyro.sample(f"{name}_m_offset", dist.Normal(0, 1))
    
    growth_k_per_condition = growth_k_hyper_loc + growth_k_offset * growth_k_hyper_scale
    growth_m_per_condition = growth_m_hyper_loc + growth_m_offset * growth_m_hyper_scale

    # Register dists
    pyro.deterministic(f"{name}_k", growth_k_per_condition)
    pyro.deterministic(f"{name}_m", growth_m_per_condition)

    # Expand to full-sized tensors
    k_pre = growth_k_per_condition[data.map_condition_pre]
    m_pre = growth_m_per_condition[data.map_condition_pre]
    k_sel = growth_k_per_condition[data.map_condition_sel]
    m_sel = growth_m_per_condition[data.map_condition_sel]

    return k_pre, m_pre, k_sel, m_sel

def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors) -> Tuple[jnp.ndarray, ...]:
    """
    Guide
    """

    k_loc_loc = pyro.param(f"{name}_k_hyper_loc_loc", jnp.array(priors.growth_k_hyper_loc_loc))
    k_loc_scale = pyro.param(f"{name}_k_hyper_loc_scale", jnp.array(priors.growth_k_hyper_loc_scale),
                             constraint=dist.constraints.positive)
    growth_k_hyper_loc = pyro.sample(
        f"{name}_k_hyper_loc",
        dist.Normal(k_loc_loc,k_loc_scale)
    )

    k_scale_loc = pyro.param(f"{name}_k_hyper_scale_loc", jnp.array(-1.0))
    k_scale_scale = pyro.param(f"{name}_k_hyper_scale_scale",jnp.array(0.1),
                               constraint=dist.constraints.positive)
    growth_k_hyper_scale = pyro.sample(
        f"{name}_k_hyper_scale",
        dist.LogNormal(k_scale_loc, k_scale_scale)
    )

    m_loc_loc = pyro.param(f"{name}_m_hyper_loc_loc", jnp.array(priors.growth_m_hyper_loc_loc))
    m_loc_scale = pyro.param(f"{name}_m_hyper_loc_scale", jnp.array(priors.growth_m_hyper_loc_scale),
                             constraint=dist.constraints.positive)
    growth_m_hyper_loc = pyro.sample(
        f"{name}_m_hyper_loc",
        dist.Normal(m_loc_loc,m_loc_scale)
    )

    m_scale_loc = pyro.param(f"{name}_m_hyper_scale_loc", jnp.array(-1.0))
    m_scale_scale = pyro.param(f"{name}_m_hyper_scale_scale",jnp.array(0.1),
                               constraint=dist.constraints.positive)
    growth_m_hyper_scale = pyro.sample(
        f"{name}_m_hyper_scale",
        dist.LogNormal(m_scale_loc, m_scale_scale)
    )
    
    k_offset_locs = pyro.param(f"{name}_k_offset_locs",
                               jnp.zeros(data.num_condition))
    k_offset_scales = pyro.param(f"{name}_k_offset_scales",
                                 jnp.ones(data.num_condition),
                                 constraint=dist.constraints.positive)


    m_offset_locs = pyro.param(f"{name}_m_offset_locs",
                               jnp.zeros(data.num_condition))
    m_offset_scales = pyro.param(f"{name}_m_offset_scales",
                                 jnp.ones(data.num_condition),
                                 constraint=dist.constraints.positive)


    # Loop over conditions and replicates
    with pyro.plate(f"{name}_condition_parameters",data.num_condition) as idx:

        k_batch_locs = k_offset_locs[...,idx]
        k_batch_scales = k_offset_scales[...,idx]
        m_batch_locs = m_offset_locs[...,idx]
        m_batch_scales = m_offset_scales[...,idx]

        growth_k_offset = pyro.sample(f"{name}_k_offset", dist.Normal(k_batch_locs,k_batch_scales))
        growth_m_offset = pyro.sample(f"{name}_m_offset", dist.Normal(m_batch_locs,m_batch_scales))
    
    growth_k_per_condition = growth_k_hyper_loc + growth_k_offset * growth_k_hyper_scale
    growth_m_per_condition = growth_m_hyper_loc + growth_m_offset * growth_m_hyper_scale

    # Expand to full-sized tensors
    k_pre = growth_k_per_condition[data.map_condition_pre]
    m_pre = growth_m_per_condition[data.map_condition_pre]
    k_sel = growth_k_per_condition[data.map_condition_sel]
    m_sel = growth_m_per_condition[data.map_condition_sel]

    return k_pre, m_pre, k_sel, m_sel


def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}
    parameters["growth_k_hyper_loc_loc"] = 0.025
    parameters["growth_k_hyper_loc_scale"] = 0.1
    parameters["growth_k_hyper_scale"] = 0.1

    parameters["growth_m_hyper_loc_loc"] = 0.0
    parameters["growth_m_hyper_loc_scale"] = 0.01
    parameters["growth_m_hyper_scale"] = 0.1

    return parameters

def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """

    shape = data.num_condition

    guesses = {}
    guesses[f"{name}_k_hyper_loc"] = 0.025
    guesses[f"{name}_k_hyper_scale"] = 0.1
    guesses[f"{name}_m_hyper_loc"] = 0.0
    guesses[f"{name}_m_hyper_scale"] = 0.01
    guesses[f"{name}_k_offset"] = jnp.zeros(shape)
    guesses[f"{name}_m_offset"] = jnp.zeros(shape)

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())