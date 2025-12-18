import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

def _kumaraswamy_cdf(x, a, b):
    """
    Cumulative Distribution Function: F(x) = 1 - (1 - x^a)^b
    
    Refactored for numerical stability using log1p and expm1.

    Parameters
    ----------
    x : jax.array
        occupancy value between 0 and 1
    a : float
        CDF shape parameter
    b : float
        CDF shape parameter

    Returns
    -------
    jax.array
        cdf array
    """
    # Clip x to [0, 1] for safety
    x = jnp.clip(x, 0.0, 1.0)
    
    # We use a small epsilon to avoid log(0) and gradient singularities.
    # 1e-6 is safer for float32 precision.
    eps = 1e-6
    x_safe = jnp.clip(x, eps, 1.0 - eps)

    # F(x) = 1 - (1 - x^a)^b
    # Let u = log(1 - x^a). Then F(x) = 1 - exp(b * u).
    # To calculate u = log(1 - x^a) stably near x=1:
    # 1 - x^a = 1 - exp(a * log(x)) = -expm1(a * log(x))
    # u = log(-expm1(a * log(x)))
    
    u = jnp.log(-jnp.expm1(a * jnp.log(x_safe)))
    return -jnp.expm1(b * u)



def update_thetas(theta, params, mask=None):
    """
    Corrects theta values for co-transformation using the method of 
    re-sampling from the background distribution.

    Parameters
    ----------
    theta : jnp.array
        Array of theta values. 
        Expected shape: (..., num_genotype)
    params : tuple
        Tuple of (lam, a, b) parameters from the model/guide.
        lam : co-transformation rate
        a : Kumaraswamy background shape parameter a
        b : Kumaraswamy background shape parameter b
    mask : jnp.array, optional
        Boolean array of shape (num_genotype,) where True indicates the 
        genotype should be corrected for congression. If None, all genotypes
        are corrected.

    Returns
    -------
    jnp.array
        Corrected theta array with shape broadcasted from inputs.
    """
    
    lam, a, b = params

    # 1. Broadcast shapes against each other (excluding the last dimension)
    # Theta shape: (..., titrant_name, titrant_conc, geno) or similar
    # a, b shape: (titrant_name, titrant_conc, 1) or broadcastable
    
    # Calculate target shape (batch dims)
    target_shape = theta.shape[:-1]
    
    # Broadcast theta
    theta_b = jnp.broadcast_to(theta, target_shape + (theta.shape[-1],))
    
    # Broadcast params to match target_shape (excluding genotype dim)
    # lam is scalar, a/b are (name, conc, 1)
    # We broadcast them to (target_shape + (1,)) to align with theta rows
    
    lam_b = jnp.broadcast_to(lam, target_shape + (1,))
    a_b   = jnp.broadcast_to(a,   target_shape + (1,))
    b_b   = jnp.broadcast_to(b,   target_shape + (1,))


    # Flatten batch dimensions for vmap
    # We process each genotype curve (dim -1) independently across the batch
    flat_theta = theta_b.reshape(-1, theta_b.shape[-1]) # (TotalBatch, Genotype)
    
    flat_lam = lam_b.reshape(-1) # (TotalBatch,)
    flat_a   = a_b.reshape(-1)   # (TotalBatch,)
    flat_b   = b_b.reshape(-1)   # (TotalBatch,)
    
    # 2. Correct each theta value
    # We use vmap to apply correction over the batch.
    
    # Correction function for a single row (vectorized over batch)
    def correct_row(t_row, l_val, a_val, b_val):
        # Apply correction to each element in the row (vectorized over genotype)
        # Note: calculate_expected... is scalar-wise, so we vmap it over the row
        calc_max = lambda t: calculate_expected_observed_max(t, a_val, b_val, l_val)
        return jax.vmap(calc_max)(t_row)

    corrected_flat = jax.vmap(correct_row)(flat_theta, flat_lam, flat_a, flat_b)
    
    # Reshape back
    full_theta_shape = target_shape + (theta.shape[-1],)
    corrected_theta = corrected_flat.reshape(full_theta_shape)

    # 3. Apply mask if provided
    if mask is not None:
        # mask shape: (num_genotype,)
        # corrected_theta shape: (..., num_genotype)
        # We need to broadcast the mask to match corrected_theta
        corrected_theta = jnp.where(mask, corrected_theta, theta_b)
    
    return corrected_theta

def calculate_expected_observed_max(x_val, a, b, lam, n_grid=100):
    """
    Calculate E[max(x, M)] where M is the maximum of a Poisson(lam) number
    of samples from the background Kumaraswamy(a, b) distribution.
    
    Use the stable formula:
    E[max(x, M)] = 1 - integrate_{x}^1 exp(lam * (F(t) - 1)) dt

    Parameters
    ----------
    x_val : float
        the true value of the object (float)
    a, b : float, float
        shape parameters of the background population
    lam : float
        poisson parameter lambda
    n_grid : int
        number of grid points for integration

    Returns
    -------
    float
        expected observed value of x
    """
    x_safe = jnp.clip(x_val, 0.0, 1.0)
    
    # 1. Integration part: integrate_{x}^1 exp(lam * (F(t) - 1)) dt
    def integrand(t):
        Ft = _kumaraswamy_cdf(t, a, b)
        return jnp.exp(lam * (Ft - 1.0))

    t_grid = jnp.linspace(x_safe, 1.0, n_grid)
    y_vals = integrand(t_grid)
    integral_val = jnp.trapezoid(y_vals, t_grid)
    
    return 1.0 - integral_val


def calculate_expected_observed_min(x_val, a, b, lam, n_grid=100):
    """
    Calculate E[min(x, M)] where M is the minimum of a Poisson(lam) number
    of samples from the background Kumaraswamy(a, b) distribution.
    
    Use the stable formula:
    E[min(x, M)] = integrate_{0}^x exp(-lam * F(t)) dt

    Parameters
    ----------
    x_val : float
        the true value of the object (float)
    a, b : float, float
        shape parameters of the background population
    lam : float
        poisson parameter lambda
    n_grid : int
        number of grid points for integration

    Returns
    -------
    float
        expected observed value of x
    """
    x_safe = jnp.clip(x_val, 0.0, 1.0)
    
    # 1. Integration part: integrate_{0}^x exp(-lam * F(t)) dt
    def integrand(t):
        Ft = _kumaraswamy_cdf(t, a, b)
        return jnp.exp(-lam * Ft)

    t_grid = jnp.linspace(0.0, x_safe, n_grid)
    y_vals = integrand(t_grid)
    integral_val = jnp.trapezoid(y_vals, t_grid)
    
    return integral_val


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.

    Attributes
    ----------
    lam_loc : float
        Mean of the LogNormal prior for lambda.
    lam_scale : float
        Scale (sigma) of the LogNormal prior for lambda.
    a_loc : float
        Mean of LogNormal prior for a.
    a_scale : float
        Scale for a.
    b_loc : float
        Mean of LogNormal prior for b.
    b_scale : float
        Scale for b.
    """

    lam_loc: float
    lam_scale: float
    a_loc: float
    a_scale: float
    b_loc: float
    b_scale: float
    

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Defines the hierarchical model for the co-transformation parameter (lambda)
    and background parameters (a, b).

    Parameters
    ----------
    name : str
        The prefix for the Numpyro sample sites.
    data : GrowthData
        A data object containing metadata (not currently used for lambda 
        dimensions, but required by interface).
    priors : ModelPriors
        A Pytree containing the hyperparameters.

    Returns
    -------
    tuple
        (lam, a, b) sampled values.
    """
    
    # Global shared lambda
    # Prior: LogNormal(loc, scale)
    # Using LogNormal ensures lambda is positive
    
    lam = pyro.sample(
        f"{name}_lam",
        dist.LogNormal(priors.lam_loc, priors.lam_scale)
    )
    pyro.deterministic(f"{name}_lam_value", lam)

    # Nested plates for titrant_name and titrant_conc
    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
    
            a = pyro.sample(
                f"{name}_a",
                dist.LogNormal(priors.a_loc, priors.a_scale)
            )
            pyro.deterministic(f"{name}_a_value", a)

            b = pyro.sample(
                f"{name}_b",
                dist.LogNormal(priors.b_loc, priors.b_scale)
            )
            pyro.deterministic(f"{name}_b_value", b)
    
    return lam, a, b

def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide corresponding to the co-transformation model.
    Learns variational posteriors for lam, a, and b.

    Parameters
    ----------
    name : str
        The prefix for the Numpyro sample sites.
    data : GrowthData
        A data object containing metadata.
    priors : ModelPriors
        A Pytree containing the hyperparameters for initialization.

    Returns
    -------
    tuple
        (lam, a, b) sampled values from guide.
    """
    
    # Variational approximation for lambda
    # Using LogNormal family
    
    h_lam_loc = pyro.param(f"{name}_lam_loc", jnp.array(priors.lam_loc))
    h_lam_scale = pyro.param(f"{name}_lam_scale", jnp.array(priors.lam_scale),
                             constraint=dist.constraints.positive)
    lam = pyro.sample(f"{name}_lam", dist.LogNormal(h_lam_loc, h_lam_scale))

    # Variational approximation for a and b (plated)
    # Shape: (num_name, num_conc, 1) to broadcast against (..., num_name, num_conc, num_geno)
    
    local_shape = (data.num_titrant_name, data.num_titrant_conc, 1)

    h_a_loc = pyro.param(f"{name}_a_loc", jnp.full(local_shape, priors.a_loc))
    h_a_scale = pyro.param(f"{name}_a_scale", jnp.full(local_shape, priors.a_scale),
                           constraint=dist.constraints.positive)
    
    h_b_loc = pyro.param(f"{name}_b_loc", jnp.full(local_shape, priors.b_loc))
    h_b_scale = pyro.param(f"{name}_b_scale", jnp.full(local_shape, priors.b_scale),
                           constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
            
            a = pyro.sample(f"{name}_a", dist.LogNormal(h_a_loc, h_a_scale))
            b = pyro.sample(f"{name}_b", dist.LogNormal(h_b_loc, h_b_scale))

    return lam, a, b


def get_hyperparameters():
    """
    Gets default values for the model hyperparameters.

    Returns
    -------
    dict
        A dictionary of hyperparameter names and their default values.
    """

    parameters = {}
    
    # Lambda prior: centered on 1.0
    parameters["lam_loc"] = 0.0 
    parameters["lam_scale"] = 1.0 
    
    # Background priors (a, b)
    # Centered on 1.0 (Uniform-ish)
    parameters["a_loc"] = 0.0
    parameters["a_scale"] = 1.0
    parameters["b_loc"] = 0.0
    parameters["b_scale"] = 1.0
               
    return parameters
    

def get_guesses(name,data):
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
        - ``data.num_genotype`` : (int) number of non-wt genotypes

    Returns
    -------
    dict
        A dictionary mapping parameter names to their initial
        guess values.
    """


    guesses = {}
    guesses[f"{name}_lam"] = 1.0
    guesses[f"{name}_a"] = 1.0
    guesses[f"{name}_b"] = 1.0
    
    return guesses

def get_priors():
    """
    Utility function to create a populated ModelPriors object.

    Returns
    -------
    ModelPriors
        A populated Pytree (Flax dataclass) of hyperparameters.
    """
    return ModelPriors(**get_hyperparameters())
