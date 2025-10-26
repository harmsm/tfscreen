import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

@dataclass 
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
    data.map_cond_pre
    data.map_cond_sel
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
        with pyro.plate(f"{name}_replicate_parameters", data.num_replicate):
            growth_k_offset = pyro.sample(f"{name}_k_offset", dist.Normal(0, 1))
            growth_m_offset = pyro.sample(f"{name}_m_offset", dist.Normal(0, 1))
    
    growth_k_dist = growth_k_hyper_loc + growth_k_offset * growth_k_hyper_scale
    growth_m_dist = growth_m_hyper_loc + growth_m_offset * growth_m_hyper_scale

    growth_k_dist_1d = growth_k_dist.ravel()
    growth_m_dist_1d = growth_m_dist.ravel()

    # Register dists
    pyro.deterministic(f"{name}_k", growth_k_dist_1d)
    pyro.deterministic(f"{name}_m", growth_m_dist_1d)

    # Expand to full-sized tensors
    k_pre = growth_k_dist_1d[data.map_cond_pre]
    m_pre = growth_m_dist_1d[data.map_cond_pre]
    k_sel = growth_k_dist_1d[data.map_cond_sel]
    m_sel = growth_m_dist_1d[data.map_cond_sel]

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

    shape = (data.num_condition,data.num_replicate)

    guesses = {}
    guesses[f"{name}_k_hyper_loc"] = 0.025
    guesses[f"{name}_k_hyper_scale"] = 0.1
    guesses[f"{name}_m_hyper_loc"] = 0.0
    guesses[f"{name}_m_hyper_scale"] = 0.01
    guesses[f"{name}_k_offset"] = jnp.zeros(shape)
    guesses[f"{name}_m_offset"] = jnp.zeros(shape)

    return guesses
