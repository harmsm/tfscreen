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
    JAX Pytree holding hyperparameters for the categorical theta model.
    
    Attributes
    ----------
    logit_theta_hyper_loc_loc : float
        Mean of the prior for the hyper-location of logit(theta).
    logit_theta_hyper_loc_scale : float
        Std dev of the prior for the hyper-location of logit(theta).
    logit_theta_hyper_scale : float
        Scale of the HalfNormal prior for the hyper-scale of logit(theta).
    """

    logit_theta_hyper_loc_loc: float
    logit_theta_hyper_loc_scale: float
    logit_theta_hyper_scale: float


@dataclass(frozen=True)
class ThetaParam:
    """
    JAX Pytree holding the sampled categorical theta parameters.

    Attributes
    ----------
    theta : jnp.ndarray
    mu : jnp.ndarray
    sigma : jnp.ndarray
        A tensor of sampled fractional occupancy values, with shape
        ``[num_titrant_name, num_titrant_conc, num_genotype]``.
    """

    theta: jnp.ndarray
    mu: jnp.ndarray
    sigma: jnp.ndarray


def define_model(name: str,
                 data: DataClass,
                 priors: ModelPriors) -> ThetaParam:
    """
    Defines the hierarchical categorical model for theta.
    
    This function defines a unique, sampled ``theta`` parameter for every
    ``(titrant_name, titrant_conc, genotype)`` combination.
    
    The parameters are sampled from a pooled Normal distribution in
    logit-space using a non-centered parameterization.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample sites (e.g., "theta").
    data : DataClass
        A data object (e.g., `GrowthData`) containing metadata, primarily:
        - ``data.num_titrant_name`` : (int) Number of titrants.
        - ``data.num_titrant_conc`` : (int) Number of titrant concentrations.
        - ``data.num_genotype`` : (int) Number of genotypes.
    priors : ModelPriors
        A Pytree (Flax dataclass) containing all hyperparameters for the
        pooled logit-space prior.

    Returns
    -------
    ThetaParam
        A Pytree containing the sampled theta parameters in their
        natural scale, with shape
        ``[num_titrant_name, num_titrant_conc, num_genotype]``.
    """

    # --------------------------------------------------------------------------
    # Hyperpriors for the logit(theta) parameters (now condition specific)
    
    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
    
            logit_theta_hyper_loc = pyro.sample(
                f"{name}_logit_theta_hyper_loc",
                dist.Normal(priors.logit_theta_hyper_loc_loc,
                            priors.logit_theta_hyper_loc_scale)
            )
            logit_theta_hyper_scale = pyro.sample(
                f"{name}_logit_theta_hyper_scale",
                dist.HalfNormal(priors.logit_theta_hyper_scale)
            )

    # --------------------------------------------------------------------------
    # Sample parameters for each (titrant_name, titrant_conc, genotype) group 

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name,dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc,dim=-2):
            with pyro.plate("theta_genotype_plate", size=data.batch_size,dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):
                
                    logit_theta_offset = pyro.sample(
                        f"{name}_logit_theta_offset", 
                        dist.Normal(0.0, 1.0)
                    )

    # Guard against full-sized array substitution during initialization or re-runs 
    # with full-sized initial values
    if logit_theta_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        logit_theta_offset = logit_theta_offset[..., data.batch_idx]

    # Calculate parameters in logit-space
    logit_theta = logit_theta_hyper_loc + logit_theta_offset * logit_theta_hyper_scale

    # --------------------------------------------------------------------------
    # Transform parameters to natural scale
    
    theta = dist.transforms.SigmoidTransform()(logit_theta)

    # Register parameter values
    pyro.deterministic(f"{name}_theta", theta)

    theta_param = ThetaParam(theta=theta,
                             mu=logit_theta_hyper_loc,
                             sigma=logit_theta_hyper_scale)
    
    return theta_param

def guide(name: str,
          data: DataClass,
          priors: ModelPriors) -> ThetaParam:
    """
    Guide corresponding to the categorical theta model.

    This guide defines the variational family for the categorical theta
    parameters. It uses:
    - Normal/LogNormal distributions for global location/scale hyperparameters.
    - An amortized parameterization for the local 3D tensor of parameters
      (titrant x conc x genotype).
    """

    # --- 1. Global Hypers (now plated) ---
    
    local_shape_global = (data.num_titrant_name, data.num_titrant_conc, 1)

    # Logit Theta Hyper Loc (Normal)
    h_loc_loc = pyro.param(f"{name}_logit_theta_hyper_loc_loc", 
                           jnp.full(local_shape_global, priors.logit_theta_hyper_loc_loc))
    h_loc_scale = pyro.param(f"{name}_logit_theta_hyper_loc_scale", 
                             jnp.full(local_shape_global, priors.logit_theta_hyper_loc_scale),
                             constraint=dist.constraints.positive)
    
    # Logit Theta Hyper Scale (LogNormal guide)
    h_scale_loc = pyro.param(f"{name}_logit_theta_hyper_scale_loc", jnp.full(local_shape_global, -1.0))
    h_scale_scale = pyro.param(f"{name}_logit_theta_hyper_scale_scale", jnp.full(local_shape_global, 0.1),
                               constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
            logit_theta_hyper_loc = pyro.sample(
                f"{name}_logit_theta_hyper_loc",
                dist.Normal(h_loc_loc, h_loc_scale)
            )
            logit_theta_hyper_scale = pyro.sample(
                f"{name}_logit_theta_hyper_scale",
                dist.LogNormal(h_scale_loc, h_scale_scale)
            )

    # --- 2. Local Parameters (3D Tensor) ---
    
    # Shape: (NumTitrantName, NumTitrantConc, NumGenotype)
    param_shape = (data.num_titrant_name, data.num_titrant_conc, data.num_genotype)

    offset_locs = pyro.param(f"{name}_logit_theta_offset_locs", 
                             jnp.zeros(param_shape,dtype=float))
    offset_scales = pyro.param(f"{name}_logit_theta_offset_scales", 
                               jnp.ones(param_shape,dtype=float),
                               constraint=dist.constraints.positive)

    # --- 3. Sampling (Sliced by Genotype) ---

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
            # Batching on Genotype (dim=-1)
            with pyro.plate("theta_genotype_plate", size=data.batch_size,dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):
                
                    # Slice the last dimension (Genotype) using the batch indices
                    # The ellipsis (...) preserves the TitrantName and TitrantConc dimensions
                    batch_locs = offset_locs[..., data.batch_idx]
                    batch_scales = offset_scales[..., data.batch_idx]
                    
                    logit_theta_offset = pyro.sample(
                        f"{name}_logit_theta_offset", 
                        dist.Normal(batch_locs, batch_scales)
                    )

    # Guard against full-sized array substitution during initialization or re-runs 
    # with full-sized initial values
    if logit_theta_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        logit_theta_offset = logit_theta_offset[..., data.batch_idx]
        

    # --- 4. Reconstruction (Deterministic) ---
    
    # Calculate parameters in logit-space
    logit_theta = logit_theta_hyper_loc + logit_theta_offset * logit_theta_hyper_scale

    # Transform parameters to natural scale
    theta = dist.transforms.SigmoidTransform()(logit_theta)

    theta_param = ThetaParam(theta=theta,
                             mu=logit_theta_hyper_loc,
                             sigma=logit_theta_hyper_scale)
    
    return theta_param

def run_model(theta_param: ThetaParam, data: DataClass) -> jnp.ndarray:
    """
    "Calculates" fractional occupancy (theta) by looking it up.

    This is a pure JAX function that returns the sampled categorical
    theta values. Unlike the Hill model, no calculation is needed.
    It simply passes the sampled tensor and handles the final optional
    scattering.

    Parameters
    ----------
    theta_param : ThetaParam
        A Pytree generated by ``define_model`` containing the sampled
        theta parameters. ``theta_param.theta`` has dimensions
        ``[titrant_name, titrant_conc, genotype]``.
    data : DataClass
        A data object (e.g., ``GrowthData`` or ``BindingData``) containing:
        - ``data.map_theta``: (jnp.ndarray) Mapper with dimensions
          ``[replicate, time, treatment, genotype]``, used for scattering.
        - ``data.scatter_theta``: (int) A flag (0 or 1) indicating
          whether to scatter the final tensor.

    Returns
    -------
    jnp.ndarray
        A tensor of theta values.
        - If ``data.scatter_theta == 0``, shape is
          ``[titrant_name, titrant_conc, genotype]``.
        - If ``data.scatter_theta == 1``, shape is
          ``[replicate, time, treatment, genotype]``.
    """
    
    # The parameters are the calculated values
    theta_calc = theta_param.theta

    # Broadcast to the full-sized tensor if required
    if data.scatter_theta == 1:
        theta_calc = theta_calc[None,None,None,None,:,:,:]
    
    return theta_calc


def get_population_moments(theta_param: ThetaParam, data: DataClass) -> tuple:
    """
    Returns the expected population moments (mu, sigma) in logit-space.
    
    For the categorical model, these are simply the hyper-parameters.
    
    Returns
    -------
    tuple
        (mu, sigma) with shape broadcastable to (titrant_name, titrant_conc, 1).
    """
    return theta_param.mu, theta_param.sigma


def get_hyperparameters() -> Dict[str, Any]:
    """
    Gets default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        A dictionary of hyperparameter names and their default values.
    """

    parameters = {}

    # Center logit(theta) prior on 0.0 (i.e., theta = 0.5)
    parameters["logit_theta_hyper_loc_loc"] = 0.0
    parameters["logit_theta_hyper_loc_scale"] = 1.5
    parameters["logit_theta_hyper_scale"] = 1.0

    return parameters


def get_guesses(name: str, data: DataClass) -> Dict[str, Any]:
    """
    Gets initial guess values for model parameters.

    Parameters
    ----------
    name : str
        The prefix for the parameter names (e.g., "theta").
    data : DataClass
        A data object (e.g., `GrowthData`) containing metadata, primarily:
        - ``data.num_titrant_name``
        - ``data.num_titrant_conc``
        - ``data.num_genotype``

    Returns
    -------
    dict[str, Any]
        A dictionary mapping parameter names to their initial
        guess values.
    """
    
    guesses = {}

    # Guess hyperparams
    guesses[f"{name}_logit_theta_hyper_loc"] = jnp.zeros((data.num_titrant_name, data.num_titrant_conc, 1))
    guesses[f"{name}_logit_theta_hyper_scale"] = jnp.ones((data.num_titrant_name, data.num_titrant_conc, 1))

    # Guess offsets (all zeros)
    shape = (data.num_titrant_name, data.num_titrant_conc, data.num_genotype)
    guesses[f"{name}_logit_theta_offset"] = jnp.zeros(shape,dtype=float)

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