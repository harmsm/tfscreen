import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import (
    dataclass
)

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    beta_kappa_loc: float
    beta_kappa_scale: float


def define_model(name,fx_calc,priors):

    # Parameter for the beta distribution modeling the stochastic nature of an
    # experiment that measures [0-1] fractional outputs
    kappa = pyro.sample(
        f"{name}_beta_kappa",
        dist.Gamma(priors.beta_kappa_loc,
                   priors.beta_kappa_scale)
    ) 
    
    alpha = fx_calc * kappa
    beta = (1.0 - fx_calc) * kappa

    # Clip alpha and beta for stability
    alpha = jnp.clip(alpha, a_min=1e-10, a_max=1e10)
    beta = jnp.clip(beta, a_min=1e-10, a_max=1e10)

    # Sample from beta distribution centered on fx_calc with spread dictated
    # by kappa
    fx_noisy = pyro.sample(f"{name}_dist", dist.Beta(alpha, beta))
    
    # Register final tensors
    pyro.deterministic(name, fx_noisy)
    
    return fx_noisy

def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}

    # Center gamma on 300
    parameters["beta_kappa_loc"] = 25.0
    parameters["beta_kappa_scale"] = 0.05

    return parameters


def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """
    
    guesses = {}

    guesses[f"{name}_beta_log_hill_n_hyper_scale"] = 0.3

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())