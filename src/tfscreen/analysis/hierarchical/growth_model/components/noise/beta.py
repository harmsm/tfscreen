import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import (
    dataclass
)
from typing import Dict, Any
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the Beta noise model.

    Attributes
    ----------
    beta_kappa_loc : float
        The 'shape' parameter for the Gamma prior on kappa.
    beta_kappa_scale : float
        The 'rate' parameter for the Gamma prior on kappa.
    """

    beta_kappa_loc: float
    beta_kappa_scale: float


def define_model(name: str, 
                 fx_calc: jnp.ndarray, 
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Applies Beta-distributed noise to a deterministic value.

    This function models noise for a value `fx_calc` (presumed to be
    between 0 and 1) using a Beta distribution. The Beta distribution
    is reparameterized by its mean (`fx_calc`) and a concentration
    parameter, `kappa`.

    mean = fx_calc
    kappa = alpha + beta

    A high `kappa` results in a narrow distribution (low noise), while
    a low `kappa` results in a wide distribution (high noise).

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample/deterministic sites.
    fx_calc : jnp.ndarray
        The deterministically calculated mean value (e.g., fractional
        occupancy), which must be in the range (0, 1).
    priors : ModelPriors
        A Pytree (Flax dataclass) containing the hyperparameters for
        the `kappa` prior.

    Returns
    -------
    jnp.ndarray
        The noisy value, `fx_noisy`, sampled from `Beta(alpha, beta)`.
    """

    # kappa is the concentration parameter for the beta distribution.
    # It is sampled from a Gamma distribution.
    kappa = pyro.sample(
        f"{name}_beta_kappa",
        dist.Gamma(priors.beta_kappa_loc,
                   priors.beta_kappa_scale)
    ) 
    
    # Reparameterize: alpha = mean * concentration
    alpha = fx_calc * kappa
    # Reparameterize: beta = (1.0 - mean) * concentration
    beta = (1.0 - fx_calc) * kappa

    # Clip alpha and beta for stability
    # The Beta distribution requires alpha > 0 and beta > 0.
    alpha = jnp.clip(alpha, min=1e-10, max=1e10)
    beta = jnp.clip(beta, min=1e-10, max=1e10)

    # Sample from beta distribution centered on fx_calc with spread dictated
    # by kappa
    fx_noisy = pyro.sample(f"{name}_dist", dist.Beta(alpha, beta))
    
    # Register final tensors (optional, as fx_noisy is already sampled)
    pyro.deterministic(name, fx_noisy)
    
    return fx_noisy

def guide(name: str, 
          fx_calc: jnp.ndarray, 
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide for the Beta noise model.
    
    Since 'fx_calc' is passed without indices/plates, we cannot store 
    variational parameters for each individual 'fx_noisy' value. 
    Instead, we use an amortized strategy: we learn the global 'kappa' 
    parameter, and then sample 'fx_noisy' from a Beta distribution 
    derived from that learned 'kappa' and the input 'fx_calc'.
    """

    # --- 1. Global Parameter (Kappa) ---
    
    # We use a LogNormal guide for the Gamma prior (positive support).
    # Initialize loc to log(prior_loc) to start near the prior mean.
    # (Adding a small epsilon safety or defaulting to log(10.0) if priors are complex)
    kappa_init = jnp.log(priors.beta_kappa_loc)
    
    kappa_loc = pyro.param(f"{name}_beta_kappa_loc", kappa_init)
    kappa_scale = pyro.param(f"{name}_beta_kappa_scale", jnp.array(0.1),
                             constraint=dist.constraints.positive)

    kappa = pyro.sample(
        f"{name}_beta_kappa",
        dist.LogNormal(kappa_loc, kappa_scale)
    )

    # --- 2. Local Latent Variable (Amortized fx_noisy) ---
    
    # Calculate the variational alpha/beta using the input fx_calc
    # and the sampled variational kappa. 
    alpha = fx_calc * kappa
    beta = (1.0 - fx_calc) * kappa

    # Clip for stability (exactly mirroring the model)
    alpha = jnp.clip(alpha, min=1e-10, max=1e10)
    beta = jnp.clip(beta, min=1e-10, max=1e10)

    # Sample fx_noisy
    # Note: Because we are mirroring the model's structure exactly, 
    # the KL divergence contribution from this term will largely cancel,
    # focusing the optimization on learning 'kappa'.
    fx_noisy = pyro.sample(f"{name}_dist", dist.Beta(alpha, beta))
    
    return fx_noisy

def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names to their
        default values.
    """

    parameters = {}

    # Prior for kappa (concentration parameter)
    # Gamma(shape, rate) has mean = shape / rate
    # Mean = 25.0 / 0.05 = 500.0
    parameters["beta_kappa_loc"] = 25.0
    parameters["beta_kappa_scale"] = 0.05

    return parameters


def get_guesses(name: str,data: GrowthData) -> Dict[str, Any]:
    """
    Get guess values for the model's latent parameters.

    These values are used in `numpyro.handlers.substitute` for testing
    or initializing inference.

    Parameters
    ----------
    name : str
        The prefix used for all sample sites (e.g., "my_model").

    Returns
    -------
    dict[str, Any]
        A dictionary mapping sample site names to guess values.
    """
    
    guesses = {}

    # Guess the mean of the prior distribution for kappa
    hyperparams = get_hyperparameters()
    mean_kappa = hyperparams["beta_kappa_loc"] / hyperparams["beta_kappa_scale"]
    guesses[f"{name}_beta_kappa"] = mean_kappa

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