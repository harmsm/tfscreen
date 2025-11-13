import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import (
    dataclass,
    field
)

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    theta_logit_min_hyper_loc_loc: float
    theta_logit_min_hyper_loc_scale: float
    theta_logit_min_hyper_scale: float

    theta_logit_max_hyper_loc_loc: float
    theta_logit_max_hyper_loc_scale: float
    theta_logit_max_hyper_scale: float

    theta_log_hill_K_hyper_loc_loc: float
    theta_log_hill_K_hyper_loc_scale: float
    theta_log_hill_K_hyper_scale: float

    theta_log_hill_n_hyper_loc_loc: float
    theta_log_hill_n_hyper_loc_scale: float
    theta_log_hill_n_hyper_scale: float


@dataclass(frozen=True)
class ThetaParam:
    """
    """

    theta_min: jnp.ndarray
    theta_max: jnp.ndarray
    hill_K: jnp.ndarray
    hill_n: jnp.ndarray


def define_model(name,data,priors):
    """
    Hill model parameters (theta_min, theta_max, K, and n). theta_min and 
    theta_max each use pooled logit-scaled hyperpriors. hill_K and hill_n each
    use pooled log-scaled hyperpriors. 
    
    Priors
    ------
    priors.theta_logit_min_hyper_loc_loc
    priors.theta_logit_min_hyper_loc_scale
    priors.theta_logit_min_hyper_scale

    priors.theta_logit_max_hyper_loc_loc
    priors.theta_logit_max_hyper_loc_scale
    priors.theta_logit_max_hyper_scale

    priors.theta_log_hill_K_hyper_loc_loc
    priors.theta_log_hill_K_hyper_loc_scale
    priors.theta_log_hill_K_hyper_scale

    priors.theta_log_hill_n_hyper_loc_loc
    priors.theta_log_hill_n_hyper_loc_scale
    priors.theta_log_hill_n_hyper_scale

    Data
    ----
    data.num_titrant_name 
    data.num_genotype
    """

    # --------------------------------------------------------------------------
    # Hyperpriors for the Hill model parameters to be inferred 
    
    # hyperpriors for the min theta (logit scale)
    logit_theta_min_hyper_loc = pyro.sample(
        f"{name}_theta_logit_min_hyper_loc",
        dist.Normal(priors.theta_logit_min_hyper_loc_loc,
                    priors.theta_logit_min_hyper_loc_scale)
    )
    logit_theta_min_hyper_scale = pyro.sample(
        f"{name}_theta_logit_min_hyper_scale",
        dist.HalfNormal(priors.theta_logit_min_hyper_scale)
    )

    # hyperpriors for max theta (logit scale)
    logit_theta_max_hyper_loc = pyro.sample(
        f"{name}_theta_logit_max_hyper_loc",
        dist.Normal(priors.theta_logit_max_hyper_loc_loc,
                    priors.theta_logit_max_hyper_loc_scale)
    )
    logit_theta_max_hyper_scale = pyro.sample(
        f"{name}_theta_logit_max_hyper_scale",
        dist.HalfNormal(priors.theta_logit_max_hyper_scale)
    )
    
     # hyperpriors for hill K (log scale)
    log_hill_K_hyper_loc = pyro.sample(
        f"{name}_theta_log_hill_K_hyper_loc",
        dist.Normal(priors.theta_log_hill_K_hyper_loc_loc,
                    priors.theta_log_hill_K_hyper_loc_scale)
    )
    log_hill_K_hyper_scale = pyro.sample(
        f"{name}_theta_log_hill_K_hyper_scale",
        dist.HalfNormal(priors.theta_log_hill_K_hyper_scale)
    )

    # hyperpriors for hill n (log scale)
    log_hill_n_hyper_loc = pyro.sample(
        f"{name}_theta_log_hill_n_hyper_loc",
        dist.Normal(priors.theta_log_hill_n_hyper_loc_loc,
                    priors.theta_log_hill_n_hyper_loc_scale)
    )
    log_hill_n_hyper_scale = pyro.sample(
        f"{name}_theta_log_hill_n_hyper_scale",
        dist.HalfNormal(priors.theta_log_hill_n_hyper_scale)
    )

    # --------------------------------------------------------------------------
    # Sample curve parameters for each (genotype, titrant_name) group 

    with pyro.plate(f"{name}_titrant_name_plate",data.num_titrant_name):
        with pyro.plate(f"{name}_genotype_plate",data.num_genotype):
            logit_theta_min_offset = pyro.sample(f"{name}_logit_min_offset", dist.Normal(0, 1))
            logit_theta_max_offset = pyro.sample(f"{name}_logit_max_offset", dist.Normal(0, 1))
            log_hill_K_offset = pyro.sample(f"{name}_log_hill_K_offset", dist.Normal(0, 1))
            log_hill_n_offset = pyro.sample(f"{name}_log_hill_n_offset", dist.Normal(0, 1))

    logit_theta_min = logit_theta_min_hyper_loc + logit_theta_min_offset * logit_theta_min_hyper_scale
    logit_theta_max = logit_theta_max_hyper_loc + logit_theta_max_offset * logit_theta_max_hyper_scale
    log_hill_K = log_hill_K_hyper_loc + log_hill_K_offset * log_hill_K_hyper_scale
    log_hill_n = log_hill_n_hyper_loc + log_hill_n_offset * log_hill_n_hyper_scale

    # --------------------------------------------------------------------------
    # Expand parameters 

    # Transform parameters to their natural scale
    theta_min = dist.transforms.SigmoidTransform()(logit_theta_min)
    theta_max = dist.transforms.SigmoidTransform()(logit_theta_max)
    hill_K = jnp.clip(jnp.exp(log_hill_K),a_max=1e30)
    hill_n = jnp.clip(jnp.exp(log_hill_n),a_max=1e30)

    # Register parameter values
    pyro.deterministic(f"{name}_theta_min",theta_min)
    pyro.deterministic(f"{name}_theta_max",theta_max)
    pyro.deterministic(f"{name}_hill_K",hill_K)
    pyro.deterministic(f"{name}_hill_n",hill_n)

    theta_param = ThetaParam(theta_min=theta_min,
                             theta_max=theta_max,
                             hill_K=hill_K,
                             hill_n=hill_n)
    
    return theta_param

def run_model(theta_param,data):
    """
    Calculate the values of the hill model using the parameters in theta_param. 

    theta_param is generated by `define_model`. 

    theta_param.hill_K et al have dimensions: 
        [titrant_name,genotype]

    data.map_theta_group and data.titrant_conc have dimensions
        [titrant_name,titrant_conc,genotype]

    data.map_theta has dimensions
        [replicate,time,treatment,genotype]

    This function returns a tensor of calculated theta values. This either has
    dimensions [titrant_name,titrant_conc,genotype] (scatter_theta == 0) or
    dimensions [replicate,time,treatment,genotype] (scatter_theta == 1). 
    """
    
    # Create [titrant_name,titrant_conc,genotype]-sized tensors of all 
    # parameters.
    theta_min = theta_param.theta_min.ravel()[data.map_theta_group]
    theta_max = theta_param.theta_max.ravel()[data.map_theta_group]
    hill_K = theta_param.hill_K.ravel()[data.map_theta_group]
    hill_n = theta_param.hill_n.ravel()[data.map_theta_group]

    # Calculate theta 
    c_pow_n = jnp.clip(jnp.power(data.titrant_conc, hill_n),a_max=1e30) 
    Kd_pow_n = jnp.power(hill_K, hill_n)
    epsilon = 1e-20 # prevent x/0
    theta_calc = theta_min + (theta_max - theta_min) * (c_pow_n / (Kd_pow_n + c_pow_n + epsilon))

    # Scatter to the full-sized tensor
    if data.scatter_theta == 1:
        theta_calc = theta_calc.ravel()[data.map_theta]
    
    return theta_calc


def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}

    parameters["theta_logit_min_hyper_loc_loc"] = -1
    parameters["theta_logit_min_hyper_loc_scale"] = 1.5
    parameters["theta_logit_min_hyper_scale"] = 1.0

    parameters["theta_logit_max_hyper_loc_loc"] = 1
    parameters["theta_logit_max_hyper_loc_scale"] = 1.5
    parameters["theta_logit_max_hyper_scale"] = 1.0

    parameters["theta_log_hill_K_hyper_loc_loc"] = -4.1
    parameters["theta_log_hill_K_hyper_loc_scale"] = 2
    parameters["theta_log_hill_K_hyper_scale"] = 1.0

    parameters["theta_log_hill_n_hyper_loc_loc"] = 0.693
    parameters["theta_log_hill_n_hyper_loc_scale"] = 0.5
    parameters["theta_log_hill_n_hyper_scale"] = 1.0

    return parameters


def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """
    
    guesses = {}

    guesses[f"{name}_logit_min_hyper_loc"] = -1
    guesses[f"{name}_logit_min_hyper_scale"] = 1.5
    guesses[f"{name}_logit_max_hyper_loc"] = 1
    guesses[f"{name}_logit_max_hyper_scale"] = 1.5
    guesses[f"{name}_theta_log_hill_K_hyper_loc"] = -4.14433344452323 # ln(0.017 mM)
    guesses[f"{name}_theta_log_hill_K_hyper_scale"] = 1
    guesses[f"{name}_theta_log_hill_n_hyper_loc"] = 0.693 # ln(2)
    guesses[f"{name}_theta_log_hill_n_hyper_scale"] = 0.3

    guesses[f"{name}_logit_min_offset"] = jnp.ones((data.num_titrant_name,data.num_genotype))*0.076
    guesses[f"{name}_logit_max_offset"] = jnp.ones((data.num_titrant_name,data.num_genotype))*0.924
    guesses[f"{name}_log_hill_K_offset"] = jnp.ones((data.num_titrant_name,data.num_genotype))*(-4.144) #ln(0.017 mM)
    guesses[f"{name}_log_hill_n_offset"] = jnp.ones((data.num_titrant_name,data.num_genotype))*0.693 #ln(2)  

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())