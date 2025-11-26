import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    growth_k_per_cond_rep: jnp.ndarray 
    growth_m_per_cond_rep: jnp.ndarray 

def define_model(name,data,priors):
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

    # Expand to full-sized tensors
    k_pre = priors.growth_k_per_cond_rep[data.map_condition_pre]
    m_pre = priors.growth_m_per_cond_rep[data.map_condition_pre]
    k_sel = priors.growth_k_per_cond_rep[data.map_condition_sel]
    m_sel = priors.growth_m_per_cond_rep[data.map_condition_sel]

    return k_pre, m_pre, k_sel, m_sel

def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}
    parameters["growth_k_per_cond_rep"] = jnp.array([ 0.010696,0.015366,0.021437, 0.028558])
    parameters["growth_m_per_cond_rep"] = jnp.array([-0.009933,0.000808,0.006226,-0.000344])

    return parameters

def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """

    return {}

def get_priors():
    return ModelPriors(**get_hyperparameters())