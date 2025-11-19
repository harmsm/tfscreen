import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import (
    dataclass,
    field
)
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the Hill model priors.
    
    Attributes
    ----------
    theta_logit_min_hyper_loc_loc : float
    theta_logit_min_hyper_loc_scale : float
    theta_logit_min_hyper_scale : float
    theta_logit_max_hyper_loc_loc : float
    theta_logit_max_hyper_loc_scale : float
    theta_logit_max_hyper_scale : float
    theta_log_hill_K_hyper_loc_loc : float
    theta_log_hill_K_hyper_loc_scale : float
    theta_log_hill_K_hyper_scale : float
    theta_hill_n_hyper_loc_loc : float
    theta_hill_n_hyper_loc_scale : float
    theta_hill_n_hyper_scale : float
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

    theta_hill_n_hyper_loc_loc: float
    theta_hill_n_hyper_loc_scale: float
    theta_hill_n_hyper_scale: float


@dataclass(frozen=True)
class ThetaParam:
    """
    JAX Pytree holding the sampled Hill equation parameters.
    
    These are the parameters sampled in their natural scale.

    Attributes
    ----------
    theta_min : jnp.ndarray
        The minimum fractional occupancy (baseline).
    theta_max : jnp.ndarray
        The maximum fractional occupancy (saturation).
    hill_K : jnp.ndarray
        The Hill constant (K_D).
    hill_n : jnp.ndarray
        The Hill coefficient.
    """

    theta_min: jnp.ndarray
    theta_max: jnp.ndarray
    hill_K: jnp.ndarray
    hill_n: jnp.ndarray


def define_model(name: str, data: DataClass, priors: ModelPriors) -> ThetaParam:
    """
    Defines the hierarchical Hill model parameters.
    
    This function defines the Numpyro ``sample`` sites for a non-centered
    hierarchical model of Hill parameters (theta_min, theta_max, K, and n).
    
    - ``theta_min`` and ``theta_max`` use pooled logit-scaled hyperpriors.
    - ``hill_K`` and ``hill_n`` use pooled log-scaled hyperpriors.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample sites (e.g., "theta").
    data : DataClass
        A data object containing metadata, primarily:
        - ``data.num_titrant_name`` : (int) Number of titrants.
        - ``data.num_genotype`` : (int) Number of genotypes.
    priors : ModelPriors
        A Pytree containing all hyperparameters for the model, including:
        - ``priors.theta_logit_min_hyper_loc_loc``
        - ``priors.theta_logit_min_hyper_loc_scale``
        - ``priors.theta_logit_min_hyper_scale``
        - ``priors.theta_logit_max_hyper_loc_loc``
        - ``priors.theta_logit_max_hyper_loc_scale``
        - ``priors.theta_logit_max_hyper_scale``
        - ``priors.theta_log_hill_K_hyper_loc_loc``
        - ``priors.theta_log_hill_K_hyper_loc_scale``
        - ``priors.theta_log_hill_K_hyper_scale``
        - ``priors.theta_hill_n_hyper_loc_loc``
        - ``priors.theta_hill_n_hyper_loc_scale``
        - ``priors.theta_hill_n_hyper_scale``

    Returns
    -------
    ThetaParam
        A Pytree containing the sampled Hill parameters (theta_min,
        theta_max, hill_K, hill_n), each with shape
        ``[num_titrant_name, num_genotype]``.
    """

    # --------------------------------------------------------------------------
    # Hyperpriors for the Hill model parameters to be inferred 
    
    # hyperpriors for the min theta (logit scale)
    logit_theta_min_hyper_loc = pyro.sample(
        f"{name}_logit_min_hyper_loc",
        dist.Normal(priors.theta_logit_min_hyper_loc_loc,
                    priors.theta_logit_min_hyper_loc_scale)
    )
    logit_theta_min_hyper_scale = pyro.sample(
        f"{name}_logit_min_hyper_scale",
        dist.HalfNormal(priors.theta_logit_min_hyper_scale)
    )

    # hyperpriors for max theta (logit scale)
    logit_theta_max_hyper_loc = pyro.sample(
        f"{name}_logit_max_hyper_loc",
        dist.Normal(priors.theta_logit_max_hyper_loc_loc,
                    priors.theta_logit_max_hyper_loc_scale)
    )
    logit_theta_max_hyper_scale = pyro.sample(
        f"{name}_logit_max_hyper_scale",
        dist.HalfNormal(priors.theta_logit_max_hyper_scale)
    )
    
     # hyperpriors for hill K (log scale)
    log_hill_K_hyper_loc = pyro.sample(
        f"{name}_log_hill_K_hyper_loc",
        dist.Normal(priors.theta_log_hill_K_hyper_loc_loc,
                    priors.theta_log_hill_K_hyper_loc_scale)
    )
    log_hill_K_hyper_scale = pyro.sample(
        f"{name}_log_hill_K_hyper_scale",
        dist.HalfNormal(priors.theta_log_hill_K_hyper_scale)
    )

    # hyperpriors for hill n (log scale)
    hill_n_hyper_loc = pyro.sample(
        f"{name}_hill_n_hyper_loc",
        dist.Normal(priors.theta_hill_n_hyper_loc_loc,
                    priors.theta_hill_n_hyper_loc_scale)
    )
    hill_n_hyper_scale = pyro.sample(
        f"{name}_hill_n_hyper_scale",
        dist.HalfNormal(priors.theta_hill_n_hyper_scale)
    )

    # --------------------------------------------------------------------------
    # Sample curve parameters for each (genotype, titrant_name) group 

    with pyro.plate(f"{name}_titrant_name_plate",data.num_titrant_name,dim=-2):
        with pyro.plate(f"{name}_genotype_plate",data.num_genotype,dim=-1):
            logit_theta_min_offset = pyro.sample(f"{name}_logit_min_offset", dist.Normal(0, 1))
            logit_theta_max_offset = pyro.sample(f"{name}_logit_max_offset", dist.Normal(0, 1))
            log_hill_K_offset = pyro.sample(f"{name}_log_hill_K_offset", dist.Normal(0, 1))
            hill_n_offset = pyro.sample(f"{name}_hill_n_offset", dist.Normal(0, 1))

    logit_theta_min = logit_theta_min_hyper_loc + logit_theta_min_offset * logit_theta_min_hyper_scale
    logit_theta_max = logit_theta_max_hyper_loc + logit_theta_max_offset * logit_theta_max_hyper_scale
    log_hill_K = log_hill_K_hyper_loc + log_hill_K_offset * log_hill_K_hyper_scale
    hill_n = hill_n_hyper_loc + hill_n_offset * hill_n_hyper_scale

    # --------------------------------------------------------------------------
    # Expand parameters 

    # Transform parameters to their natural scale
    theta_min = dist.transforms.SigmoidTransform()(logit_theta_min)
    theta_max = dist.transforms.SigmoidTransform()(logit_theta_max)
    hill_K = jnp.clip(jnp.exp(log_hill_K),max=1e30)
    # hill_n already on its natural scale

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

def run_model(theta_param: ThetaParam, data: DataClass) -> jnp.ndarray:
    """
    Calculates fractional occupancy (theta) using the Hill equation.

    This is a pure JAX function that deterministically calculates theta
    values using the sampled parameters from ``define_model``.

    Parameters
    ----------
    theta_param : ThetaParam
        A Pytree generated by ``define_model`` containing the sampled
        Hill parameters. Tensors within (e.g., ``theta_param.hill_K``)
        are expected to have dimensions ``[titrant_name, genotype]``.
    data : DataClass
        A data object (e.g., ``GrowthData`` or ``BindingData``) containing:
        - ``data.map_theta_group``: (jnp.ndarray) Mapper with dimensions
          ``[titrant_name, titrant_conc, genotype]``.
        - ``data.titrant_conc``: (jnp.ndarray) Titrant concentrations,
          with dimensions ``[titrant_name, titrant_conc, genotype]``.
        - ``data.map_theta``: (jnp.ndarray) Mapper with dimensions
          ``[replicate, time, treatment, genotype]``.
        - ``data.scatter_theta``: (int) A flag (0 or 1) indicating
          whether to scatter the final tensor.

    Returns
    -------
    jnp.ndarray
        A tensor of calculated theta values.
        - If ``data.scatter_theta == 0``, shape is
          ``[titrant_name, titrant_conc, genotype]``.
        - If ``data.scatter_theta == 1``, shape is
          ``[replicate, time, treatment, genotype]``.
    """
    
    # Create [titrant_name,titrant_conc,genotype]-sized tensors of all 
    # parameters.
    theta_min = theta_param.theta_min.ravel()[data.map_theta_group]
    theta_max = theta_param.theta_max.ravel()[data.map_theta_group]
    hill_K = theta_param.hill_K.ravel()[data.map_theta_group]
    hill_n = theta_param.hill_n.ravel()[data.map_theta_group]

    # Calculate theta 
    c_pow_n = jnp.clip(jnp.power(data.titrant_conc, hill_n),max=1e30) 
    Kd_pow_n = jnp.power(hill_K, hill_n)
    epsilon = 1e-20 # prevent x/0
    theta_calc = theta_min + (theta_max - theta_min) * (c_pow_n / (Kd_pow_n + c_pow_n + epsilon))

    # Scatter to the full-sized tensor
    if data.scatter_theta == 1:
        theta_calc = theta_calc.ravel()[data.map_theta]
    
    return theta_calc


def get_hyperparameters() -> Dict[str, Any]:
    """
    Gets default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names (as strings) to their
        default values.
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

    parameters["theta_hill_n_hyper_loc_loc"] = 2.0
    parameters["theta_hill_n_hyper_loc_scale"] = 1.0
    parameters["theta_hill_n_hyper_scale"] = 1.0

    return parameters


def get_guesses(name: str, data: DataClass) -> Dict[str, Any]:
    """
    Gets initial guess values for model parameters.

    These are used to initialize the MCMC sampler (e.g., via
    ``numpyro.infer.init_to_value``).

    Parameters
    ----------
    name : str
        The prefix for the parameter names (e.g., "theta").
    data : DataClass
        A data object containing metadata, primarily:
        - ``data.num_titrant_name`` : (int) Number of titrants.
        - ``data.num_genotype`` : (int) Number of genotypes.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping parameter names to their initial
        guess values.
    """
    
    guesses = {}

    guesses[f"{name}_logit_min_hyper_loc"] = -1
    guesses[f"{name}_logit_min_hyper_scale"] = 1.5
    guesses[f"{name}_logit_max_hyper_loc"] = 1
    guesses[f"{name}_logit_max_hyper_scale"] = 1.5
    
    guesses[f"{name}_log_hill_K_hyper_loc"] = -4.14433344452323 # ln(0.017 mM)
    guesses[f"{name}_log_hill_K_hyper_scale"] = 1
    guesses[f"{name}_hill_n_hyper_loc"] = 2
    guesses[f"{name}_hill_n_hyper_scale"] = 0.3

    guesses[f"{name}_logit_min_offset"] = jnp.ones((data.num_titrant_name,data.num_genotype))*0.076
    guesses[f"{name}_logit_max_offset"] = jnp.ones((data.num_titrant_name,data.num_genotype))*0.924
    guesses[f"{name}_log_hill_K_offset"] = jnp.ones((data.num_titrant_name,data.num_genotype))*(-4.144) #ln(0.017 mM)
    guesses[f"{name}_hill_n_offset"] = jnp.ones((data.num_titrant_name,data.num_genotype))*2

    return guesses

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Returns
    -------
    ModelPriors
        A populated Pytree (Flax dataclass) of hyperparameters.
    """
    return ModelPriors(**get_hyperparameters())