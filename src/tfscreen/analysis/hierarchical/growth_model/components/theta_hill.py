import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import (
    dataclass,
    field
)

@dataclass 
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

    theta_kappa_loc: float
    theta_kappa_scale: float

    measured_logit_theta_min_loc: jnp.ndarray
    measured_logit_theta_min_scale: jnp.ndarray
    measured_logit_theta_max_loc: jnp.ndarray
    measured_logit_theta_max_scale: jnp.ndarray
    measured_log_theta_hill_K_loc: jnp.ndarray
    measured_log_theta_hill_K_scale: jnp.ndarray
    measured_log_theta_hill_n_loc: jnp.ndarray
    measured_log_theta_hill_n_scale: jnp.ndarray

    measured_theta_hill_indices: jnp.ndarray
    unmeasured_theta_hill_indices: jnp.ndarray
    num_measured_theta_hill: int = field(pytree_node=False)


def define_model(name,data,priors):
    """
    Hill model parameters (theta_min, theta_max, K, and n). theta_min and 
    theta_max each use pooled logit-scaled hyperpriors. hill_K and hill_n each
    use pooled log-scaled hyperpriors. A final parameter, kappa, captures the 
    spread between the output of the deterministic Hill model and the  
    biological outputs. Also allows setting independent priors (logit and log 
    scaled) on independently measured genotypes. 
    

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

    priors.theta_kappa_loc
    priors.theta_kappa_scale

    priors.measured_logit_theta_min_loc
    priors.measured_logit_theta_min_scale
    priors.measured_logit_theta_max_loc
    priors.measured_logit_theta_max_scale
    priors.measured_log_theta_hill_K_loc
    priors.measured_log_theta_hill_K_scale
    priors.measured_log_theta_hill_n_loc
    priors.measured_log_theta_hill_n_scale

    priors.measured_theta_hill_indices
    priors.unmeasured_theta_hill_indices
    priors.num_measured_theta_hill

    Data
    ----
    data.num_theta_group (num_genotype*num_titrant_names)
    data.map_theta_group (full tensor dimensions)
    data.titrant_conc (full tensor dimensions)
    """

    # --------------------------------------------------------------------------
    # Hyperpriors for the Hill model parameters to be inferred for genotypes 
    # without previous measurements.
    
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

    # Parameter for the beta distribution modeling the stochastic nature of the
    # experiment (shared by un-measured and measured). 
    kappa = pyro.sample(
        f"{name}_theta_kappa",
        dist.Gamma(priors.theta_kappa_loc,
                   priors.theta_kappa_scale)
    ) 
    
    # --------------------------------------------------------------------------
    # Sample curve parameters for each (genotype, titrant_name) group that was
    # not previously measured

    num_unmeas = data.num_theta_group - priors.num_measured_theta_hill
    
    if num_unmeas > 0:

        with pyro.plate(f"{name}_unmeasured_parameters", num_unmeas):
            logit_theta_min_offset = pyro.sample(f"{name}_unmeasured_logit_min_offset", dist.Normal(0, 1))
            logit_theta_max_offset = pyro.sample(f"{name}_unmeasured_logit_max_offset", dist.Normal(0, 1))
            log_hill_K_offset = pyro.sample(f"{name}_unmeasured_log_hill_K_offset", dist.Normal(0, 1))
            log_hill_n_offset = pyro.sample(f"{name}_unmeasured_log_hill_n_offset", dist.Normal(0, 1))

        unmeasured_logit_theta_min = logit_theta_min_hyper_loc + logit_theta_min_offset * logit_theta_min_hyper_scale
        unmeasured_logit_theta_max = logit_theta_max_hyper_loc + logit_theta_max_offset * logit_theta_max_hyper_scale
        unmeasured_log_hill_K = log_hill_K_hyper_loc + log_hill_K_offset * log_hill_K_hyper_scale
        unmeasured_log_hill_n = log_hill_n_hyper_loc + log_hill_n_offset * log_hill_n_hyper_scale
    else:
        unmeasured_logit_theta_min = jnp.array([])
        unmeasured_logit_theta_max = jnp.array([])
        unmeasured_log_hill_K = jnp.array([])
        unmeasured_log_hill_n = jnp.array([])

    # --------------------------------------------------------------------------
    # Sample curve parameters for each (genotype, titrant_name) group that was
    # previously measured

    with pyro.plate(f"{name}_measured_parameters", priors.num_measured_theta_hill):
    
        measured_logit_theta_min = pyro.sample(
            f"{name}_measured_logit_theta_min",
            dist.Normal(priors.measured_logit_theta_min_loc,
                        priors.measured_logit_theta_min_scale)
        )
        measured_logit_theta_max = pyro.sample(
            f"{name}_measured_logit_theta_max",
            dist.Normal(priors.measured_logit_theta_max_loc,
                        priors.measured_logit_theta_max_scale)
        )
        measured_log_hill_K = pyro.sample(
            f"{name}_measured_log_hill_K",
            dist.Normal(priors.measured_log_theta_hill_K_loc,
                        priors.measured_log_theta_hill_K_scale)
        )
        measured_log_hill_n = pyro.sample(
            f"{name}_measured_log_hill_n",
            dist.Normal(priors.measured_log_theta_hill_n_loc,
                        priors.measured_log_theta_hill_n_scale)
        )

    # --------------------------------------------------------------------------
    # Assemble measured and unmeasured Hill parameters

    logit_theta_min = jnp.empty(data.num_theta_group)
    logit_theta_min = logit_theta_min.at[priors.measured_theta_hill_indices].set(measured_logit_theta_min)
    logit_theta_min = logit_theta_min.at[priors.unmeasured_theta_hill_indices].set(unmeasured_logit_theta_min)
    
    logit_theta_max = jnp.empty(data.num_theta_group)
    logit_theta_max = logit_theta_max.at[priors.measured_theta_hill_indices].set(measured_logit_theta_max)
    logit_theta_max = logit_theta_max.at[priors.unmeasured_theta_hill_indices].set(unmeasured_logit_theta_max)

    log_hill_K = jnp.empty(data.num_theta_group)
    log_hill_K = log_hill_K.at[priors.measured_theta_hill_indices].set(measured_log_hill_K)
    log_hill_K = log_hill_K.at[priors.unmeasured_theta_hill_indices].set(unmeasured_log_hill_K)

    log_hill_n = jnp.empty(data.num_theta_group)
    log_hill_n = log_hill_n.at[priors.measured_theta_hill_indices].set(measured_log_hill_n)
    log_hill_n = log_hill_n.at[priors.unmeasured_theta_hill_indices].set(unmeasured_log_hill_n)

    # --------------------------------------------------------------------------
    # Expand parameters and calculate Hill model

    # Transform parameters to their natural scale
    theta_min_1d = dist.transforms.SigmoidTransform()(logit_theta_min)
    theta_max_1d = dist.transforms.SigmoidTransform()(logit_theta_max)
    hill_K_1d = jnp.exp(log_hill_K) # no clips here because we take care of inf cases below
    hill_n_1d = jnp.exp(log_hill_n)

    # Register parameter values
    pyro.deterministic(f"{name}_theta_min",theta_min_1d)
    pyro.deterministic(f"{name}_theta_max",theta_max_1d)
    pyro.deterministic(f"{name}_hill_K",hill_K_1d)
    pyro.deterministic(f"{name}_hill_n",hill_n_1d)

    # Expand to full tensor
    theta_min = theta_min_1d[data.map_theta_group]
    theta_max = theta_max_1d[data.map_theta_group]
    hill_K = hill_K_1d[data.map_theta_group]
    hill_n = hill_n_1d[data.map_theta_group]

    # Calculate theta (mean) using Hill model applied to a full titrant tensor
    c_pow_n = jnp.clip(jnp.power(data.titrant_conc, hill_n),a_max=1e30) # high clip to prevent inf/inf
    Kd_pow_n = jnp.power(hill_K, hill_n)
    epsilon = 1e-20 # prevent x/0
    mean_theta = theta_min + (theta_max - theta_min) * (c_pow_n / (Kd_pow_n + c_pow_n + epsilon))

    # --------------------------------------------------------------------------
    # Convert to alpha, beta and sample the final thetas
    alpha = mean_theta * kappa
    beta = (1.0 - mean_theta) * kappa

    # Clip alpha and beta for stability
    alpha = jnp.clip(alpha, a_min=1e-10, a_max=1e10)
    beta = jnp.clip(beta, a_min=1e-10, a_max=1e10)

    # Sample from beta distribution centered on theta_mean with spread dictated
    # by kappa
    theta = pyro.sample(f"{name}_dist", dist.Beta(alpha, beta))
    
    # Register final tensors
    pyro.deterministic(name, theta)
    
    return theta

def get_hyperparameters(data,
                        num_measured_theta_hill):
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

    # Center gamma on 300
    parameters["theta_kappa_loc"] = 25.0
    parameters["theta_kappa_scale"] = 0.05

    parameters["measured_logit_theta_min_loc"] = jnp.ones(num_measured_theta_hill)*(-1)
    parameters["measured_logit_theta_min_scale"] = jnp.ones(num_measured_theta_hill)*1
    parameters["measured_logit_theta_max_loc"] = jnp.ones(num_measured_theta_hill)*(1)
    parameters["measured_logit_theta_max_scale"] = jnp.ones(num_measured_theta_hill)*1
    parameters["measured_log_theta_hill_K_loc"] = jnp.ones(num_measured_theta_hill)*(-4.1)
    parameters["measured_log_theta_hill_K_scale"] = jnp.ones(num_measured_theta_hill)*0.05
    parameters["measured_log_theta_hill_n_loc"] = jnp.ones(num_measured_theta_hill)*(0.693)
    parameters["measured_log_theta_hill_n_scale"] = jnp.ones(num_measured_theta_hill)*0.05

    parameters["measured_theta_hill_indices"] = jnp.zeros(data.num_theta_group)
    parameters["unmeasured_theta_hill_indices"] = jnp.zeros(data.num_theta_group)
    parameters["num_measured_theta_hill"] = num_measured_theta_hill

    return parameters


def get_guesses(name,
                data,
                num_measured_theta_hill):
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
    guesses[f"{name}_theta_kappa"] = 500

    shape = (data.num_theta_group - num_measured_theta_hill,)
    guesses[f"{name}_unmeasured_logit_min_offset"] = jnp.zeros(shape) # <- 0.076
    guesses[f"{name}_unmeasured_logit_max_offset"] = jnp.zeros(shape) # <- 0.924
    guesses[f"{name}_unmeasured_log_hill_K_offset"] = jnp.zeros(shape)  # <- -4.1
    guesses[f"{name}_unmeasured_log_hill_n_offset"] = jnp.zeros(shape)  # <- ln(2)  

    shape = (num_measured_theta_hill,)
    guesses[f"{name}_measured_logit_theta_min"] = -1*jnp.ones(shape)
    guesses[f"{name}_measured_logit_theta_max"] =  1*jnp.ones(shape)
    guesses[f"{name}_measured_log_hill_K"] = -4.14433344452323*jnp.ones(shape)
    guesses[f"{name}_measured_log_hill_n"] = 0.693*jnp.ones(shape)

    return guesses