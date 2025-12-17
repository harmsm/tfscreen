import jax
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
    theta_logit_low_hyper_loc_loc : float
    theta_logit_low_hyper_loc_scale : float
    theta_logit_low_hyper_scale : float
    theta_logit_delta_hyper_loc_loc : float
    theta_logit_delta_hyper_loc_scale : float
    theta_logit_delta_hyper_scale : float
    theta_log_hill_K_hyper_loc_loc : float
    theta_log_hill_K_hyper_loc_scale : float
    theta_log_hill_K_hyper_scale : float
    theta_log_hill_n_hyper_loc_loc : float
    theta_log_hill_n_hyper_loc_scale : float
    theta_log_hill_n_hyper_scale : float
    """

    theta_logit_low_hyper_loc_loc: float
    theta_logit_low_hyper_loc_scale: float
    theta_logit_low_hyper_scale: float
    theta_logit_delta_hyper_loc_loc: float
    theta_logit_delta_hyper_loc_scale: float
    theta_logit_delta_hyper_scale: float

    theta_log_hill_K_hyper_loc_loc: float
    theta_log_hill_K_hyper_loc_scale: float
    theta_log_hill_K_hyper_scale: float

    theta_log_hill_n_hyper_loc_loc: float
    theta_log_hill_n_hyper_loc_scale: float
    theta_log_hill_n_hyper_scale: float


@dataclass(frozen=True)
class ThetaParam:
    """
    JAX Pytree holding the sampled Hill equation parameters.
    
    These are the parameters sampled in their natural scale.

    Attributes
    ----------
    theta_low : jnp.ndarray
        The minimum fractional occupancy (baseline).
    theta_high : jnp.ndarray
        The maximum fractional occupancy (saturation).
    log_hill_K : jnp.ndarray
        The Hill constant (K_D) in log-space.
    hill_n : jnp.ndarray
        The Hill coefficient.
    """

    theta_low: jnp.ndarray
    theta_high: jnp.ndarray
    log_hill_K: jnp.ndarray
    hill_n: jnp.ndarray


def define_model(name: str,
                 data: DataClass,
                 priors: ModelPriors) -> ThetaParam:
    """
    Defines the hierarchical Hill model parameters.
    
    This function defines the Numpyro ``sample`` sites for a non-centered
    hierarchical model of Hill parameters (theta_low, theta_high, K, and n).
    
    - ``theta_low`` and ``theta_delta`` use pooled logit-scaled hyperpriors. 
      We convert ``theta_low`` and ``theta_delta`` into ``theta_high`` prior
      to the sigmoid transform to enforce [0,1] bounds on both.
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
        - ``priors.theta_logit_low_hyper_loc_loc``
        - ``priors.theta_logit_low_hyper_loc_scale``
        - ``priors.theta_logit_low_hyper_scale``
        - ``priors.theta_logit_delta_hyper_loc_loc``
        - ``priors.theta_logit_delta_hyper_loc_scale``
        - ``priors.theta_logit_delta_hyper_scale``
        - ``priors.theta_log_hill_K_hyper_loc_loc``
        - ``priors.theta_log_hill_K_hyper_loc_scale``
        - ``priors.theta_log_hill_K_hyper_scale``
        - ``priors.theta_log_hill_n_hyper_loc_loc``
        - ``priors.theta_log_hill_n_hyper_loc_scale``
        - ``priors.theta_log_hill_n_hyper_scale``

    Returns
    -------
    ThetaParam
        A Pytree containing the sampled Hill parameters (theta_low,
        theta_high, log_hill_K, hill_n), each with shape
        ``[num_titrant_name, num_genotype]``.
    """

    # --------------------------------------------------------------------------
    # Hyperpriors for the Hill model parameters to be inferred 
    
    # hyperpriors for the min theta (logit scale)
    logit_theta_low_hyper_loc = pyro.sample(
        f"{name}_logit_low_hyper_loc",
        dist.Normal(priors.theta_logit_low_hyper_loc_loc,
                    priors.theta_logit_low_hyper_loc_scale)
    )
    logit_theta_low_hyper_scale = pyro.sample(
        f"{name}_logit_low_hyper_scale",
        dist.HalfNormal(priors.theta_logit_low_hyper_scale)
    )

    # hyperpriors for delta theta (logit scale)
    logit_theta_delta_hyper_loc = pyro.sample(
        f"{name}_logit_delta_hyper_loc",
        dist.Normal(priors.theta_logit_delta_hyper_loc_loc,
                    priors.theta_logit_delta_hyper_loc_scale)
    )
    logit_theta_delta_hyper_scale = pyro.sample(
        f"{name}_logit_delta_hyper_scale",
        dist.HalfNormal(priors.theta_logit_delta_hyper_scale)
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
    log_hill_n_hyper_loc = pyro.sample(
        f"{name}_log_hill_n_hyper_loc",
        dist.Normal(priors.theta_log_hill_n_hyper_loc_loc,
                    priors.theta_log_hill_n_hyper_loc_scale)
    )
    log_hill_n_hyper_scale = pyro.sample(
        f"{name}_log_hill_n_hyper_scale",
        dist.HalfNormal(priors.theta_log_hill_n_hyper_scale)
    )

    # --------------------------------------------------------------------------
    # Sample curve parameters for each (genotype, titrant_name) group 

    with pyro.plate(f"{name}_titrant_name_plate",data.num_titrant_name,dim=-2):
        with pyro.plate("shared_genotype_plate", size=data.batch_size,dim=-1):
            with pyro.handlers.scale(scale=data.scale_vector):
                logit_theta_low_offset = pyro.sample(f"{name}_logit_low_offset", dist.Normal(0.0, 1.0))
                logit_theta_delta_offset = pyro.sample(f"{name}_logit_delta_offset", dist.Normal(0.0, 1.0))
                log_hill_K_offset = pyro.sample(f"{name}_log_hill_K_offset", dist.Normal(0.0, 1.0))
                log_hill_n_offset = pyro.sample(f"{name}_log_hill_n_offset", dist.Normal(0.0, 1.0))

    logit_theta_low = logit_theta_low_hyper_loc + logit_theta_low_offset * logit_theta_low_hyper_scale
    logit_theta_delta = logit_theta_delta_hyper_loc + logit_theta_delta_offset * logit_theta_delta_hyper_scale
    logit_theta_high = logit_theta_low + logit_theta_delta
    log_hill_K = log_hill_K_hyper_loc + log_hill_K_offset * log_hill_K_hyper_scale
    log_hill_n = log_hill_n_hyper_loc + log_hill_n_offset * log_hill_n_hyper_scale

    # --------------------------------------------------------------------------
    # Expand parameters 

    # Transform parameters to their natural scale
    theta_low = dist.transforms.SigmoidTransform()(logit_theta_low)
    theta_high = dist.transforms.SigmoidTransform()(logit_theta_high)
    # log_hill_K is already on its natural scale
    hill_n = jnp.exp(log_hill_n)

    # Register parameter values
    pyro.deterministic(f"{name}_theta_low",theta_low)
    pyro.deterministic(f"{name}_theta_high",theta_high)
    pyro.deterministic(f"{name}_log_hill_K",log_hill_K)
    pyro.deterministic(f"{name}_hill_n",hill_n)

    theta_param = ThetaParam(theta_low=theta_low,
                             theta_high=theta_high,
                             log_hill_K=log_hill_K,
                             hill_n=hill_n)
    
    return theta_param

def guide(name: str,
          data: DataClass,
          priors: ModelPriors) -> ThetaParam:
    """
    Guide corresponding to the hierarchical Hill model.

    This guide defines the variational family for the Hill model parameters,
    using:
    - Normal distributions for location hyperparameters (`_loc`).
    - LogNormal distributions for scale hyperparameters (`_scale`) and positive
      variables.
    - An amortized/offset parameterization for the local (per-group) parameters.
    """

    # ==========================================================================
    # 1. Global Hyperparameters
    # ==========================================================================

    # --- Theta Low (Logit Scale) ---
    # Loc
    h_low_loc_loc = pyro.param(f"{name}_logit_low_hyper_loc_loc", jnp.array(priors.theta_logit_low_hyper_loc_loc))
    h_low_loc_scale = pyro.param(f"{name}_logit_low_hyper_loc_scale", jnp.array(priors.theta_logit_low_hyper_loc_scale),
                                 constraint=dist.constraints.positive)
    logit_theta_low_hyper_loc = pyro.sample(f"{name}_logit_low_hyper_loc", 
                                            dist.Normal(h_low_loc_loc, h_low_loc_scale))

    # Scale (LogNormal guide)
    h_low_scale_loc = pyro.param(f"{name}_logit_low_hyper_scale_loc", jnp.array(-1.0))
    h_low_scale_scale = pyro.param(f"{name}_logit_low_hyper_scale_scale", jnp.array(0.1),
                                   constraint=dist.constraints.positive)
    logit_theta_low_hyper_scale = pyro.sample(f"{name}_logit_low_hyper_scale", 
                                              dist.LogNormal(h_low_scale_loc, h_low_scale_scale))

    # --- Theta Delta (Logit Scale) ---
    # Loc
    h_delta_loc_loc = pyro.param(f"{name}_logit_delta_hyper_loc_loc", jnp.array(priors.theta_logit_delta_hyper_loc_loc))
    h_delta_loc_scale = pyro.param(f"{name}_logit_delta_hyper_loc_scale", jnp.array(priors.theta_logit_delta_hyper_loc_scale),
                                   constraint=dist.constraints.positive)
    logit_theta_delta_hyper_loc = pyro.sample(f"{name}_logit_delta_hyper_loc", 
                                              dist.Normal(h_delta_loc_loc, h_delta_loc_scale))

    # Scale (LogNormal guide)
    h_delta_scale_loc = pyro.param(f"{name}_logit_delta_hyper_scale_loc", jnp.array(-1.0))
    h_delta_scale_scale = pyro.param(f"{name}_logit_delta_hyper_scale_scale", jnp.array(0.1),
                                     constraint=dist.constraints.positive)
    logit_theta_delta_hyper_scale = pyro.sample(f"{name}_logit_delta_hyper_scale", 
                                                dist.LogNormal(h_delta_scale_loc, h_delta_scale_scale))

    # --- Hill K (Log Scale) ---
    # Loc
    h_K_loc_loc = pyro.param(f"{name}_log_hill_K_hyper_loc_loc", jnp.array(priors.theta_log_hill_K_hyper_loc_loc))
    h_K_loc_scale = pyro.param(f"{name}_log_hill_K_hyper_loc_scale", jnp.array(priors.theta_log_hill_K_hyper_loc_scale),
                               constraint=dist.constraints.positive)
    log_hill_K_hyper_loc = pyro.sample(f"{name}_log_hill_K_hyper_loc", 
                                       dist.Normal(h_K_loc_loc, h_K_loc_scale))

    # Scale (LogNormal guide)
    h_K_scale_loc = pyro.param(f"{name}_log_hill_K_hyper_scale_loc", jnp.array(-1.0))
    h_K_scale_scale = pyro.param(f"{name}_log_hill_K_hyper_scale_scale", jnp.array(0.1),
                                 constraint=dist.constraints.positive)
    log_hill_K_hyper_scale = pyro.sample(f"{name}_log_hill_K_hyper_scale", 
                                         dist.LogNormal(h_K_scale_loc, h_K_scale_scale))

    # --- Hill n (Log Scale) ---
    # Loc
    h_n_loc_loc = pyro.param(f"{name}_log_hill_n_hyper_loc_loc", jnp.array(priors.theta_log_hill_n_hyper_loc_loc))
    h_n_loc_scale = pyro.param(f"{name}_log_hill_n_hyper_loc_scale", jnp.array(priors.theta_log_hill_n_hyper_loc_scale),
                               constraint=dist.constraints.positive)
    log_hill_n_hyper_loc = pyro.sample(f"{name}_log_hill_n_hyper_loc", 
                                       dist.Normal(h_n_loc_loc, h_n_loc_scale))

    # Scale (LogNormal guide)
    h_n_scale_loc = pyro.param(f"{name}_log_hill_n_hyper_scale_loc", jnp.array(-1.0))
    h_n_scale_scale = pyro.param(f"{name}_log_hill_n_hyper_scale_scale", jnp.array(0.1),
                                 constraint=dist.constraints.positive)
    log_hill_n_hyper_scale = pyro.sample(f"{name}_log_hill_n_hyper_scale", 
                                         dist.LogNormal(h_n_scale_loc, h_n_scale_scale))


    # ==========================================================================
    # 2. Local Parameters (Offset Variational Params)
    # ==========================================================================
    
    # Shape: (NumTitrants, NumGenotypes)
    local_shape = (data.num_titrant_name, data.num_genotype)

    # Low Offsets
    low_offset_locs = pyro.param(f"{name}_logit_low_offset_locs", jnp.zeros(local_shape,dtype=float))
    low_offset_scales = pyro.param(f"{name}_logit_low_offset_scales", jnp.ones(local_shape,dtype=float), 
                                   constraint=dist.constraints.positive)

    # Delta Offsets
    delta_offset_locs = pyro.param(f"{name}_logit_delta_offset_locs", jnp.zeros(local_shape,dtype=float))
    delta_offset_scales = pyro.param(f"{name}_logit_delta_offset_scales", jnp.ones(local_shape,dtype=float), 
                                     constraint=dist.constraints.positive)

    # K Offsets
    K_offset_locs = pyro.param(f"{name}_log_hill_K_offset_locs", jnp.zeros(local_shape,dtype=float))
    K_offset_scales = pyro.param(f"{name}_log_hill_K_offset_scales", jnp.ones(local_shape,dtype=float), 
                                 constraint=dist.constraints.positive)

    # n Offsets
    n_offset_locs = pyro.param(f"{name}_log_hill_n_offset_locs", jnp.zeros(local_shape,dtype=float))
    n_offset_scales = pyro.param(f"{name}_log_hill_n_offset_scales", jnp.ones(local_shape,dtype=float), 
                                 constraint=dist.constraints.positive)


    # ==========================================================================
    # 3. Sampling (Sliced by Genotype)
    # ==========================================================================

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-2):
        # Batching on Genotype (dim=-1)
        with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
            
            # Scale data for sub-sampling
            with pyro.handlers.scale(scale=data.scale_vector):
            
                # Low
                logit_theta_low_offset = pyro.sample(f"{name}_logit_low_offset", 
                    dist.Normal(low_offset_locs[..., data.batch_idx], low_offset_scales[..., data.batch_idx]))
                
                # Delta
                logit_theta_delta_offset = pyro.sample(f"{name}_logit_delta_offset", 
                    dist.Normal(delta_offset_locs[..., data.batch_idx], delta_offset_scales[..., data.batch_idx]))
                
                # K
                log_hill_K_offset = pyro.sample(f"{name}_log_hill_K_offset", 
                    dist.Normal(K_offset_locs[..., data.batch_idx], K_offset_scales[..., data.batch_idx]))
                
                # n
                log_hill_n_offset = pyro.sample(f"{name}_log_hill_n_offset", 
                    dist.Normal(n_offset_locs[..., data.batch_idx], n_offset_scales[..., data.batch_idx]))

    # ==========================================================================
    # 4. Reconstruction
    # ==========================================================================

    logit_theta_low = logit_theta_low_hyper_loc + logit_theta_low_offset * logit_theta_low_hyper_scale
    logit_theta_delta = logit_theta_delta_hyper_loc + logit_theta_delta_offset * logit_theta_delta_hyper_scale
    logit_theta_high = logit_theta_low + logit_theta_delta
    log_hill_K = log_hill_K_hyper_loc + log_hill_K_offset * log_hill_K_hyper_scale
    log_hill_n = log_hill_n_hyper_loc + log_hill_n_offset * log_hill_n_hyper_scale

    # Transform
    theta_low = dist.transforms.SigmoidTransform()(logit_theta_low)
    theta_high = dist.transforms.SigmoidTransform()(logit_theta_high)
    hill_n = jnp.exp(log_hill_n)

    theta_param = ThetaParam(theta_low=theta_low,
                             theta_high=theta_high,
                             log_hill_K=log_hill_K,
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
        - ``data.log_titrant_conc``: (jnp.ndarray) Titrant concentrations,
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
    theta_low = theta_param.theta_low[:,None,data.geno_theta_idx]
    theta_high = theta_param.theta_high[:,None,data.geno_theta_idx]
    log_hill_K = theta_param.log_hill_K[:,None,data.geno_theta_idx]
    hill_n = theta_param.hill_n[:,None,data.geno_theta_idx]

    log_titrant = data.log_titrant_conc[None,:,None]

    occupancy = jax.nn.sigmoid(hill_n * (log_titrant - log_hill_K))
    theta_calc = theta_low + (theta_high - theta_low)*occupancy

    # Broadcast to the full-sized tensor
    if data.scatter_theta == 1:
        theta_calc = theta_calc[None,None,None,None,:,:,:]
    
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

    parameters["theta_logit_low_hyper_loc_loc"] = 2.0
    parameters["theta_logit_low_hyper_loc_scale"] = 2.0
    parameters["theta_logit_low_hyper_scale"] = 1.0

    parameters["theta_logit_delta_hyper_loc_loc"] = -4.0
    parameters["theta_logit_delta_hyper_loc_scale"] = 2.0
    parameters["theta_logit_delta_hyper_scale"] = 1.0

    parameters["theta_log_hill_K_hyper_loc_loc"] = -4.1
    parameters["theta_log_hill_K_hyper_loc_scale"] = 2.0
    parameters["theta_log_hill_K_hyper_scale"] = 1.0

    parameters["theta_log_hill_n_hyper_loc_loc"] = 0.7
    parameters["theta_log_hill_n_hyper_loc_scale"] = 0.3
    parameters["theta_log_hill_n_hyper_scale"] = 1.0

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

    guesses[f"{name}_logit_low_hyper_loc"] = 2.0
    guesses[f"{name}_logit_low_hyper_scale"] = 2.0
    guesses[f"{name}_logit_delta_hyper_loc"] = -4.0
    guesses[f"{name}_logit_delta_hyper_scale"] = 2.0
    
    guesses[f"{name}_log_hill_K_hyper_loc"] = -4.1 # ln(0.017 mM)
    guesses[f"{name}_log_hill_K_hyper_scale"] = 1.0
    guesses[f"{name}_log_hill_n_hyper_loc"] = 0.7
    guesses[f"{name}_log_hill_n_hyper_scale"] = 0.3

    guesses[f"{name}_logit_low_offset"] = jnp.zeros((data.num_titrant_name,data.num_genotype),dtype=float)
    guesses[f"{name}_logit_delta_offset"] = jnp.zeros((data.num_titrant_name,data.num_genotype),dtype=float)
    guesses[f"{name}_log_hill_K_offset"] = jnp.zeros((data.num_titrant_name,data.num_genotype),dtype=float)
    guesses[f"{name}_log_hill_n_offset"] = jnp.zeros((data.num_titrant_name,data.num_genotype),dtype=float)

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