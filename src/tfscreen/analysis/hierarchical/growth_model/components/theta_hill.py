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
from .genotype_utils import (sample_genotype_parameter, 
                             sample_genotype_parameter_guide,
                             get_genotype_parameter_guesses)

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
    
    Attributes
    ----------
    theta_low : jnp.ndarray
    theta_high : jnp.ndarray
    log_hill_K : jnp.ndarray
    hill_n : jnp.ndarray
    """

    theta_low: jnp.ndarray
    theta_high: jnp.ndarray
    log_hill_K: jnp.ndarray
    hill_n: jnp.ndarray
    mu: jnp.ndarray = None
    sigma: jnp.ndarray = None


def define_model(name: str,
                 data: DataClass,
                 priors: ModelPriors) -> ThetaParam:
    """
    Defines the hierarchical Hill model parameters.
    """

    # --------------------------------------------------------------------------
    # Hyperpriors
    
    logit_theta_low_hyper_loc = pyro.sample(
        f"{name}_logit_low_hyper_loc",
        dist.Normal(priors.theta_logit_low_hyper_loc_loc,
                    priors.theta_logit_low_hyper_loc_scale)
    )
    logit_theta_low_hyper_scale = pyro.sample(
        f"{name}_logit_low_hyper_scale",
        dist.HalfNormal(priors.theta_logit_low_hyper_scale)
    )

    logit_theta_delta_hyper_loc = pyro.sample(
        f"{name}_logit_delta_hyper_loc",
        dist.Normal(priors.theta_logit_delta_hyper_loc_loc,
                    priors.theta_logit_delta_hyper_loc_scale)
    )
    logit_theta_delta_hyper_scale = pyro.sample(
        f"{name}_logit_delta_hyper_scale",
        dist.HalfNormal(priors.theta_logit_delta_hyper_scale)
    )
    
    log_hill_K_hyper_loc = pyro.sample(
        f"{name}_log_hill_K_hyper_loc",
        dist.Normal(priors.theta_log_hill_K_hyper_loc_loc,
                    priors.theta_log_hill_K_hyper_loc_scale)
    )
    log_hill_K_hyper_scale = pyro.sample(
        f"{name}_log_hill_K_hyper_scale",
        dist.HalfNormal(priors.theta_log_hill_K_hyper_scale)
    )

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
        
        # We define a helper to sample each Hill parameter
        def sample_hill_param(site_suffix, h_loc, h_scale):
            def sample_fn(site_name, size):
                offset = pyro.sample(f"{site_name}_offset", dist.Normal(0.0, 1.0))
                if data.epistasis_mode == "genotype":
                    if offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
                        offset = offset[..., data.batch_idx]
                return h_loc + offset * h_scale
            
            return sample_genotype_parameter(f"{name}_{site_suffix}", data, sample_fn)

        logit_theta_low = sample_hill_param("logit_low", logit_theta_low_hyper_loc, logit_theta_low_hyper_scale)
        logit_theta_delta = sample_hill_param("logit_delta", logit_theta_delta_hyper_loc, logit_theta_delta_hyper_scale)
        log_hill_K = sample_hill_param("log_hill_K", log_hill_K_hyper_loc, log_hill_K_hyper_scale)
        log_hill_n = sample_hill_param("log_hill_n", log_hill_n_hyper_loc, log_hill_n_hyper_scale)

    logit_theta_high = logit_theta_low + logit_theta_delta

    # --------------------------------------------------------------------------
    # Calculate population moments
    
    n_ghost = 100
    ghost_low = logit_theta_low_hyper_loc + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(0), (n_ghost,)) * logit_theta_low_hyper_scale
    ghost_delta = logit_theta_delta_hyper_loc + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(1), (n_ghost,)) * logit_theta_delta_hyper_scale
    ghost_high = ghost_low + ghost_delta
    ghost_K = log_hill_K_hyper_loc + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(2), (n_ghost,)) * log_hill_K_hyper_scale
    ghost_n = log_hill_n_hyper_loc + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(3), (n_ghost,)) * log_hill_n_hyper_scale
    
    log_conc = data.log_titrant_conc[None, :, None] 
    
    g_low = ghost_low[None, None, :]
    g_high = ghost_high[None, None, :]
    g_K = ghost_K[None, None, :]
    g_n = jnp.exp(ghost_n[None, None, :])
    
    eps = 1e-6
    g_occ = jax.nn.sigmoid(g_n * (log_conc - g_K))
    g_theta = jnp.clip(dist.transforms.SigmoidTransform()(g_low) + (dist.transforms.SigmoidTransform()(g_high) - dist.transforms.SigmoidTransform()(g_low)) * g_occ, eps, 1.0 - eps)
    g_logit_theta = jax.scipy.special.logit(g_theta)
    
    mu_pop = jnp.mean(g_logit_theta, axis=-1, keepdims=True)
    sigma_pop = jnp.std(g_logit_theta, axis=-1, keepdims=True)

    # --------------------------------------------------------------------------
    # Expand parameters 

    theta_low = dist.transforms.SigmoidTransform()(logit_theta_low)
    theta_high = dist.transforms.SigmoidTransform()(logit_theta_high)
    hill_n = jnp.exp(log_hill_n)

    pyro.deterministic(f"{name}_theta_low",theta_low)
    pyro.deterministic(f"{name}_theta_high",theta_high)
    pyro.deterministic(f"{name}_log_hill_K",log_hill_K)
    pyro.deterministic(f"{name}_hill_n",hill_n)

    theta_param = ThetaParam(theta_low=theta_low,
                             theta_high=theta_high,
                             log_hill_K=log_hill_K,
                             hill_n=hill_n,
                             mu=mu_pop,
                             sigma=sigma_pop)
    
    return theta_param


def guide(name: str,
          data: DataClass,
          priors: ModelPriors) -> ThetaParam:
    """
    Guide corresponding to the hierarchical Hill model.
    """

    # Global Hyperparameters
    def sample_hyper(h_name, h_p_loc, h_p_scale, is_scale=False):
        g_loc = pyro.param(f"{name}_{h_name}_loc", jnp.array(h_p_loc))
        g_scale = pyro.param(f"{name}_{h_name}_scale", jnp.array(h_p_scale if not is_scale else 0.1), 
                             constraint=dist.constraints.positive)
        if is_scale:
            return pyro.sample(f"{name}_{h_name}", dist.LogNormal(g_loc, g_scale))
        else:
            return pyro.sample(f"{name}_{h_name}", dist.Normal(g_loc, g_scale))

    logit_theta_low_hyper_loc = sample_hyper("logit_low_hyper_loc", priors.theta_logit_low_hyper_loc_loc, priors.theta_logit_low_hyper_loc_scale)
    logit_theta_low_hyper_scale = sample_hyper("logit_low_hyper_scale", -1.0, 0.1, is_scale=True)

    logit_theta_delta_hyper_loc = sample_hyper("logit_delta_hyper_loc", priors.theta_logit_delta_hyper_loc_loc, priors.theta_logit_delta_hyper_loc_scale)
    logit_theta_delta_hyper_scale = sample_hyper("logit_delta_hyper_scale", -1.0, 0.1, is_scale=True)

    log_hill_K_hyper_loc = sample_hyper("log_hill_K_hyper_loc", priors.theta_log_hill_K_hyper_loc_loc, priors.theta_log_hill_K_hyper_loc_scale)
    log_hill_K_hyper_scale = sample_hyper("log_hill_K_hyper_scale", -1.0, 0.1, is_scale=True)

    log_hill_n_hyper_loc = sample_hyper("log_hill_n_hyper_loc", priors.theta_log_hill_n_hyper_loc_loc, priors.theta_log_hill_n_hyper_loc_scale)
    log_hill_n_hyper_scale = sample_hyper("log_hill_n_hyper_scale", -1.0, 0.1, is_scale=True)

    # Local Parameters
    actual_size = data.num_genotype if data.epistasis_mode == "genotype" else data.num_mutation
    local_shape = (data.num_titrant_name, actual_size)

    def guide_hill_param(site_suffix, h_loc, h_scale):
        g_offset_locs = pyro.param(f"{name}_{site_suffix}_offset_locs", jnp.zeros(local_shape))
        g_offset_scales = pyro.param(f"{name}_{site_suffix}_offset_scales", jnp.ones(local_shape), 
                                     constraint=dist.constraints.positive)
        
        def guide_fn(site_name, size):
            if data.epistasis_mode == "genotype":
                batch_locs = g_offset_locs[..., data.batch_idx]
                batch_scales = g_offset_scales[..., data.batch_idx]
            else:
                batch_locs = g_offset_locs
                batch_scales = g_offset_scales

            offset = pyro.sample(f"{site_name}_offset", dist.Normal(batch_locs, batch_scales))
            return h_loc + offset * h_scale
        
        return sample_genotype_parameter_guide(f"{name}_{site_suffix}", data, guide_fn)

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-2):
        logit_theta_low = guide_hill_param("logit_low", logit_theta_low_hyper_loc, logit_theta_low_hyper_scale)
        logit_theta_delta = guide_hill_param("logit_delta", logit_theta_delta_hyper_loc, logit_theta_delta_hyper_scale)
        log_hill_K = guide_hill_param("log_hill_K", log_hill_K_hyper_loc, log_hill_K_hyper_scale)
        log_hill_n = guide_hill_param("log_hill_n", log_hill_n_hyper_loc, log_hill_n_hyper_scale)

    logit_theta_high = logit_theta_low + logit_theta_delta

    # Transform
    theta_low = dist.transforms.SigmoidTransform()(logit_theta_low)
    theta_high = dist.transforms.SigmoidTransform()(logit_theta_high)
    hill_n = jnp.exp(log_hill_n)

    # Ghost population calculation (copied from model)
    n_ghost = 100
    ghost_low = logit_theta_low_hyper_loc + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(0), (n_ghost,)) * logit_theta_low_hyper_scale
    ghost_delta = logit_theta_delta_hyper_loc + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(1), (n_ghost,)) * logit_theta_delta_hyper_scale
    ghost_high = ghost_low + ghost_delta
    ghost_K = log_hill_K_hyper_loc + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(2), (n_ghost,)) * log_hill_K_hyper_scale
    ghost_n = log_hill_n_hyper_loc + dist.Normal(0.0, 1.0).sample(jax.random.PRNGKey(3), (n_ghost,)) * log_hill_n_hyper_scale
    
    log_conc = data.log_titrant_conc[None, :, None] 
    
    g_low = ghost_low[None, None, :]
    g_high = ghost_high[None, None, :]
    g_K = ghost_K[None, None, :]
    g_n = jnp.exp(ghost_n[None, None, :])
    
    eps = 1e-6
    g_occ = jax.nn.sigmoid(g_n * (log_conc - g_K))
    g_theta = jnp.clip(dist.transforms.SigmoidTransform()(g_low) + (dist.transforms.SigmoidTransform()(g_high) - dist.transforms.SigmoidTransform()(g_low)) * g_occ, eps, 1.0 - eps)
    g_logit_theta = jax.scipy.special.logit(g_theta)
    
    mu_pop = jnp.mean(g_logit_theta, axis=-1, keepdims=True)
    sigma_pop = jnp.std(g_logit_theta, axis=-1, keepdims=True)

    theta_param = ThetaParam(theta_low=theta_low,
                             theta_high=theta_high,
                             log_hill_K=log_hill_K,
                             hill_n=hill_n,
                             mu=mu_pop,
                             sigma=sigma_pop)
    
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


def get_population_moments(theta_param: ThetaParam, data: DataClass) -> tuple:
    """
    Returns the expected population moments (mu, sigma) in logit-space.
    """
    return theta_param.mu, theta_param.sigma


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

    parameters["theta_logit_low_hyper_loc_loc"] = 0.0
    parameters["theta_logit_low_hyper_loc_scale"] = 2.0
    parameters["theta_logit_low_hyper_scale"] = 1.0

    parameters["theta_logit_delta_hyper_loc_loc"] = -2.0
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
    """
    
    guesses = {}

    guesses[f"{name}_logit_low_hyper_loc"] = 0.0
    guesses[f"{name}_logit_low_hyper_scale"] = 2.0
    guesses[f"{name}_logit_delta_hyper_loc"] = -2.0
    guesses[f"{name}_logit_delta_hyper_scale"] = 2.0
    
    guesses[f"{name}_log_hill_K_hyper_loc"] = -4.1 # ln(0.017 mM)
    guesses[f"{name}_log_hill_K_hyper_scale"] = 1.0
    guesses[f"{name}_log_hill_n_hyper_loc"] = 0.7
    guesses[f"{name}_log_hill_n_hyper_scale"] = 0.3

    def guess_hill_per_unit(site_name, size):
        return {f"{site_name}_offset": jnp.zeros((data.num_titrant_name, size))}

    # For Hill, we have four parameters, each needs its own guesses via genotype_utils
    hill_param_sites = ["logit_low", "logit_delta", "log_hill_K", "log_hill_n"]
    for suffix in hill_param_sites:
        guesses.update(get_genotype_parameter_guesses(f"{name}_{suffix}", data, guess_hill_per_unit))

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