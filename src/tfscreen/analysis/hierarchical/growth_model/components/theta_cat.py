import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    theta_log_alpha_hyper_loc_loc: float
    theta_log_alpha_hyper_loc_scale: float
    theta_log_beta_hyper_loc_loc: float
    theta_log_beta_hyper_loc_scale: float


@dataclass(frozen=True)
class ThetaParam:
    """
    """

    theta: jnp.ndarray


def define_model(name,data,priors):
    """
    priors.theta_log_alpha_hyper_loc_loc
    priors.theta_log_alpha_hyper_loc_scale
    priors.theta_log_beta_hyper_loc_loc
    priors.theta_log_beta_hyper_loc_scale

    Data
    ----
    data.num_titrant
    data.num_theta
    data.map_theta
    """

    # Go over all titrants
    with pyro.plate(f"{name}_all_titrants",data.num_titrant):

        # Sample to create a beta distribution with unique shape parameters 
        # alpha/beta for each titrant
        log_alpha_hyper_loc = pyro.sample(
            f"{name}_log_alpha_hyper_loc",
            dist.Normal(priors.theta_log_alpha_hyper_loc_loc,
                        priors.theta_log_alpha_hyper_loc_scale)
        )
        log_alpha_hyper_offset = pyro.sample(f"{name}_log_alpha_hyper_offset",
                                             dist.Normal(0, 1))
        theta_alpha = jnp.clip(jnp.exp(log_alpha_hyper_loc + log_alpha_hyper_offset),a_max=1e30)
        pyro.deterministic(f"{name}_alpha",theta_alpha)

        log_beta_hyper_loc = pyro.sample(
            f"{name}_log_beta_hyper_loc",
            dist.Normal(priors.theta_log_beta_hyper_loc_loc,
                        priors.theta_log_beta_hyper_loc_scale)
        )
        log_beta_hyper_offset = pyro.sample(f"{name}_log_beta_hyper_offset",dist.Normal(0, 1))
        theta_beta = jnp.clip(jnp.exp(log_beta_hyper_loc + log_beta_hyper_offset),a_max=1e30)
        pyro.deterministic(f"{name}_beta",theta_beta)

        # We're going to sample theta from this distribution for this titrant
        theta_at_titrant_dist = dist.Beta(theta_alpha, theta_beta)
        with pyro.plate(f"{name}_parameters",data.num_theta):
            new_theta_dists = pyro.sample(f"{name}_value", theta_at_titrant_dist)

    theta_dists = new_theta_dists.ravel()

    # Register dists
    pyro.deterministic(name, theta_dists)

    # Expand to full-sized tensor
    theta = theta_dists[data.map_theta]
    
    return theta

def run_model(theta_param,data):
    
    raise NotImplementedError(
        "the categorical model is not fully implemented."
    )

def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}

    parameters["theta_log_alpha_hyper_loc_loc"] = 0.9
    parameters["theta_log_alpha_hyper_loc_scale"] = 0.1
    parameters["theta_log_beta_hyper_loc_loc"] = 0.9
    parameters["theta_log_beta_hyper_loc_scale"] = 0.1
    
    return parameters


def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """

    guesses = {}

    guesses[f"{name}_log_alpha_hyper_loc"] = jnp.ones(data.num_titrant)*0.9
    guesses[f"{name}_log_alpha_hyper_offset"] = jnp.ones(data.num_titrant)
    guesses[f"{name}_log_beta_hyper_loc"] = jnp.ones(data.num_titrant)
    guesses[f"{name}_log_beta_hyper_offset"] = jnp.ones(data.num_titrant)

    return {}

def get_priors():
    return ModelPriors(**get_hyperparameters())