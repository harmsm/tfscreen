import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass


@dataclass 
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    theta_log_alpha_hyper_loc_loc: float
    theta_log_alpha_hyper_loc_scale: float
    theta_log_beta_hyper_loc_loc: float
    theta_log_beta_hyper_loc_scale: float

    theta_meas_loc: jnp.ndarray
    theta_meas_scale: jnp.ndarray
    theta_num_measured: int
    theta_measured_mask: jnp.ndarray


def define_model(name,data,priors):
    """
    priors.theta_meas_loc,
    priors.theta_meas_scale
    priors.theta_num_measured
    priors.theta_measured_mask

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

    # Load independently measured theta values with their own priors. This is 
    # is 1d
    measured_theta_dists = pyro.sample(
        f"{name}_previously_measured",
        dist.TruncatedNormal(loc=priors.theta_meas_loc,
                             scale=priors.theta_meas_scale,
                             low=0.0,
                             high=1.0)
        )

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
        theta_alpha = jnp.exp(log_alpha_hyper_loc + log_alpha_hyper_offset)
        pyro.deterministic(f"{name}_alpha",theta_alpha)

        log_beta_hyper_loc = pyro.sample(
            f"{name}_log_beta_hyper_loc",
            dist.Normal(priors.theta_log_beta_hyper_loc_loc,
                        priors.theta_log_beta_hyper_loc_scale)
        )
        log_beta_hyper_offset = pyro.sample(f"{name}_log_beta_hyper_offset",dist.Normal(0, 1))
        theta_beta = jnp.exp(log_beta_hyper_loc + log_beta_hyper_offset)
        pyro.deterministic(f"{name}_beta",theta_beta)

        # We're going to sample theta from this distribution for this titrant
        theta_at_titrant_dist = dist.Beta(theta_alpha, theta_beta)
        with pyro.plate(f"{name}_parameters",data.num_theta - priors.theta_num_measured):
            new_theta_dists = pyro.sample(f"{name}_new_measured", theta_at_titrant_dist)

    new_theta_dists_1d = new_theta_dists.ravel()

    # Record thetas         
    theta_dists = jnp.empty(data.num_theta)
    theta_dists = theta_dists.at[priors.theta_measured_mask].set(measured_theta_dists)
    theta_dists = theta_dists.at[~priors.theta_measured_mask].set(new_theta_dists_1d)

    # Register dists
    pyro.deterministic(name, theta_dists)

    # Expand to full-sized tensor
    theta = theta_dists[data.map_theta]
    
    return theta

def get_hyperparameters(data,num_theta_measured):
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}

    parameters["theta_log_alpha_hyper_loc_loc"] = 0.9
    parameters["theta_log_alpha_hyper_loc_scale"] = 0.1
    parameters["theta_log_beta_hyper_loc_loc"] = 0.9
    parameters["theta_log_beta_hyper_loc_scale"] = 0.1

    parameters["theta_meas_loc"] = jnp.ones(num_theta_measured)*0.5
    parameters["theta_meas_scale"] = jnp.ones(num_theta_measured)*1.0
    parameters["theta_num_measured"] = num_theta_measured
    parameters["theta_measured_mask"] = jnp.zeros(data.num_theta,dtype=bool)
    
    return parameters


def get_guesses(name,data,num_theta_measured):
    """
    Get guesses for the model parameters. 
    """

    guesses = {}

    guesses[f"{name}_previously_measured"] = jnp.ones(num_theta_measured)*0.5
    guesses[f"{name}_log_alpha_hyper_loc"] = jnp.ones(data.num_titrant)*0.9
    guesses[f"{name}_log_alpha_hyper_offset"] = jnp.ones(data.num_titrant)
    guesses[f"{name}_log_beta_hyper_loc"] = jnp.ones(data.num_titrant)
    guesses[f"{name}_log_beta_hyper_offset"] = jnp.ones(data.num_titrant)

    shape = (data.num_titrant,data.num_theta - num_theta_measured)
    guesses[f"{name}_new_measured"] = jnp.ones(shape)

    return {}