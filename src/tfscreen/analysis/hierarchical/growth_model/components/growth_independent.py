import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    # dims are num_conditions long
    growth_k_hyper_loc_loc: jnp.ndarray
    growth_k_hyper_loc_scale: jnp.ndarray
    growth_k_hyper_scale: jnp.ndarray

    growth_m_hyper_loc_loc: jnp.ndarray
    growth_m_hyper_loc_scale: jnp.ndarray
    growth_m_hyper_scale: jnp.ndarray

def define_model(name,data,priors):
    """
    Growth parameters k_xx and m_xx versus condition, where xx are things like
    pheS+4CP, kanR-kan, etc. These go into the model as k + m*theta. Assigns
    each condition a normal hyper prior. This prior is shared by each replicate. 
    Returns full k_pre, m_pre, k_sel and m_sel tensors.

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

    # Loop over conditions
    with pyro.plate(f"{name}_condition_parameters",data.num_condition):

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

        # Loop over replicates
        with pyro.plate(f"{name}_replicate_parameters", data.num_replicate):
            k_offset = pyro.sample(f"{name}_k_offset", dist.Normal(0, 1))
            m_offset = pyro.sample(f"{name}_m_offset", dist.Normal(0, 1))
    
        growth_k_dist = growth_k_hyper_loc + k_offset * growth_k_hyper_scale
        growth_m_dist = growth_m_hyper_loc + m_offset * growth_m_hyper_scale
    
    # Flatten array
    growth_k_dist_1d = growth_k_dist.ravel()
    growth_m_dist_1d = growth_m_dist.ravel()

    # Register dists
    pyro.deterministic(f"{name}_k", growth_k_dist_1d)
    pyro.deterministic(f"{name}_m", growth_m_dist_1d)

    # Expand to full-sized tensors
    k_pre = growth_k_dist_1d[data.map_condition_pre]
    m_pre = growth_m_dist_1d[data.map_condition_pre]
    k_sel = growth_k_dist_1d[data.map_condition_sel]
    m_sel = growth_m_dist_1d[data.map_condition_sel]

    return k_pre, m_pre, k_sel, m_sel

def get_hyperparameters(num_condition):
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}
    parameters["growth_k_hyper_loc_loc"] = jnp.ones(num_condition)*0.025
    parameters["growth_k_hyper_loc_scale"] = jnp.ones(num_condition)*0.1
    parameters["growth_k_hyper_scale"] = jnp.ones(num_condition)*1.0
    parameters["growth_m_hyper_loc_loc"] = jnp.zeros(num_condition)
    parameters["growth_m_hyper_loc_scale"] = jnp.ones(num_condition)*0.01
    parameters["growth_m_hyper_scale"] = jnp.ones(num_condition)*1.0

    return parameters


def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """

    shape = (data.num_condition,data.num_replicate)

    guesses = {}
    guesses[f"{name}_k_hyper_loc"] = 1.0
    guesses[f"{name}_k_hyper_scale"] = 0.1
    guesses[f"{name}_m_hyper_loc"] = 1.0
    guesses[f"{name}_m_hyper_scale"] = 0.1
    guesses[f"{name}_k_offset"] = jnp.ones(shape)*0.1
    guesses[f"{name}_m_offset"] = jnp.ones(shape)*0.1

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())

    