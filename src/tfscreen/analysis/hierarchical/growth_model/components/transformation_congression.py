import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

def _logit_normal_cdf(x, mu, sigma):
    """
    Cumulative Distribution Function for Logit-Normal distribution:
    F(x) = Phi((logit(x) - mu) / sigma)
    
    Parameters
    ----------
    x : jax.array
        occupancy value between 0 and 1
    mu : float
        Mean of the underlying Normal distribution in logit-space.
    sigma : float
        Standard deviation of the underlying Normal distribution in logit-space.

    Returns
    -------
    jax.array
        cdf array
    """
    # Clip x to [eps, 1-eps] for stability in logit
    eps = 1e-6
    x_safe = jnp.clip(x, eps, 1.0 - eps)
    
    # Calculate logit(x)
    logit_x = jax.scipy.special.logit(x_safe)
    
    # Return Phi((logit(x) - mu) / sigma)
    return jax.scipy.stats.norm.cdf(logit_x, loc=mu, scale=sigma)


def update_thetas(theta, params, mask=None, n_grid=512):
    """
    Corrects theta values for co-transformation using the method of 
    re-sampling from the background distribution.

    Parameters
    ----------
    theta : jnp.array
        Array of theta values. 
        Expected shape: (..., num_genotype)
    params : tuple
        Tuple of (lam, mu, sigma) parameters from the model/guide.
        lam : co-transformation rate
        mu : Logit-Normal background location parameter
        sigma : Logit-Normal background scale parameter
    mask : jnp.array, optional
        Boolean array of shape (num_genotype,) where True indicates the 
        genotype should be corrected for congression. If None, all genotypes
        are corrected.
    n_grid : int, optional
        Number of points for grid-based numerical integration (default 512).

    Returns
    -------
    jnp.array
        Corrected theta array with shape broadcasted from inputs.
    """
    
    lam, mu, sigma = params

    # 1. Broadcast shapes against each other (excluding the last dimension)
    # Theta shape: (..., titrant_name, titrant_conc, geno) or similar
    
    # Calculate target shape (batch dims)
    target_shape = theta.shape[:-1]
    num_genotypes = theta.shape[-1]
    
    # Broadcast theta
    theta_b = jnp.broadcast_to(theta, target_shape + (num_genotypes,))
    
    # Broadcast params to match target_shape (excluding genotype dim)
    lam_b = jnp.broadcast_to(lam, target_shape + (1,))
    mu_b    = jnp.broadcast_to(mu,    target_shape + (1,))
    sigma_b = jnp.broadcast_to(sigma, target_shape + (1,))

    # Flatten batch dimensions
    flat_theta = theta_b.reshape(-1, num_genotypes) # (TotalBatch, Genotype)
    flat_lam   = lam_b.reshape(-1) # (TotalBatch,)
    flat_mu    = mu_b.reshape(-1)   # (TotalBatch,)
    flat_sigma = sigma_b.reshape(-1)   # (TotalBatch,)
    
    num_batches = flat_theta.shape[0]

    # 2. Pre-calculate integration on a grid for each batch element
    # Grid of occupancy values
    t_grid = jnp.linspace(0.0, 1.0, n_grid)
    
    # Calculate logit-normal CDF on grid for all batch elements
    # Ft_grid shape: (TotalBatch, n_grid)
    Ft_grid = jax.vmap(lambda m, s: _logit_normal_cdf(t_grid, m, s))(flat_mu, flat_sigma)
    
    # Calculate integrand: exp(lam * (F(t) - 1))
    # integrand_grid shape: (TotalBatch, n_grid)
    integrand_grid = jnp.exp(flat_lam[:, None] * (Ft_grid - 1.0))
    
    # Cumulative integration using trapezoidal rule
    # J(x) = integrate_0^x I(t) dt
    h = 1.0 / (n_grid - 1)
    f_mid = (integrand_grid[:, :-1] + integrand_grid[:, 1:]) * h / 2.0
    J_grid = jnp.concatenate([jnp.zeros((num_batches, 1)), jnp.cumsum(f_mid, axis=1)], axis=1)
    
    # Expected value calculation part 1: G(x) = integrate_x^1 I(t) dt
    # G(x) = J(1) - J(x)
    G1 = J_grid[:, -1:]
    Gx_grid = G1 - J_grid
    
    # 3. Interpolate G(x) for each genotype in the batch
    # Use vmap to apply interpolation over batch rows
    def interp_row(g_row, th_row):
        return jnp.interp(th_row, t_grid, g_row)
    
    integral_vals = jax.vmap(interp_row)(Gx_grid, flat_theta)
    
    # Expected observed value E[max(x, M)] = 1 - G(x)
    corrected_flat = 1.0 - integral_vals
    
    # Reshape back to original dimensions
    corrected_theta = corrected_flat.reshape(target_shape + (num_genotypes,))

    # 4. Apply mask if provided
    if mask is not None:
        corrected_theta = jnp.where(mask, corrected_theta, theta_b)
    
    return corrected_theta

def calculate_expected_observed_max(x_val, mu, sigma, lam, n_grid=100):
    """
    Calculate E[max(x, M)] where M is the maximum of a Poisson(lam) number
    of samples from the background Logit-Normal(mu, sigma) distribution.
    
    Use the stable formula:
    E[max(x, M)] = 1 - integrate_{x}^1 exp(lam * (F(t) - 1)) dt

    Parameters
    ----------
    x_val : float
        the true value of the object (float)
    mu, sigma : float, float
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
        Ft = _logit_normal_cdf(t, mu, sigma)
        return jnp.exp(lam * (Ft - 1.0))

    t_grid = jnp.linspace(x_safe, 1.0, n_grid)
    y_vals = integrand(t_grid)
    integral_val = jnp.trapezoid(y_vals, t_grid)
    
    return 1.0 - integral_val


def calculate_expected_observed_min(x_val, mu, sigma, lam, n_grid=100):
    """
    Calculate E[min(x, M)] where M is the minimum of a Poisson(lam) number
    of samples from the background Logit-Normal(mu, sigma) distribution.
    
    Use the stable formula:
    E[min(x, M)] = integrate_{0}^x exp(-lam * F(t)) dt

    Parameters
    ----------
    x_val : float
        the true value of the object (float)
    mu, sigma : float, float
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
        Ft = _logit_normal_cdf(t, mu, sigma)
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
    mu_anchoring_scale : float
        Scale factor for the consistency prior anchoring mu_bg to prior_mu.
    sigma_anchoring_scale : float
        Scale factor for the consistency prior anchoring sigma_bg to prior_sigma.
    """

    lam_loc: float
    lam_scale: float
    mu_anchoring_scale: float
    sigma_anchoring_scale: float
    

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors,
                 anchors: tuple = None) -> jnp.ndarray:
    """
    Defines the hierarchical model for the co-transformation parameter (lambda)
    and background parameters (mu, sigma).

    Parameters
    ----------
    name : str
        The prefix for the Numpyro sample sites.
    data : GrowthData
        A data object containing metadata.
    priors : ModelPriors
        A Pytree containing the hyperparameters.
    anchors : tuple, optional
        (prior_mu, prior_sigma) expected population moments from the theta model.
        Shape: (num_titrant_name, num_titrant_conc, 1)

    Returns
    -------
    tuple
        (lam, mu, sigma) sampled values.
    """
    
    # Global shared lambda
    # Prior: LogNormal(loc, scale)
    # Using LogNormal ensures lambda is positive
    
    lam = pyro.sample(
        f"{name}_lam",
        dist.LogNormal(priors.lam_loc, priors.lam_scale)
    )
    pyro.deterministic(f"{name}_lam_value", lam)

    if anchors is None:
        # Fallback to defaults if no anchors provided
        anc_mu = 0.0
        anc_sigma = 1.0
    else:
        anc_mu, anc_sigma = anchors

    # Nested plates for titrant_name and titrant_conc
    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
    
            mu = pyro.sample(
                f"{name}_mu",
                dist.Normal(anc_mu, priors.mu_anchoring_scale)
            )
            pyro.deterministic(f"{name}_mu_value", mu)

            sigma = pyro.sample(
                f"{name}_sigma",
                dist.LogNormal(jnp.log(anc_sigma), priors.sigma_anchoring_scale)
            )
            pyro.deterministic(f"{name}_sigma_value", sigma)
    
    return lam, mu, sigma

def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors,
          anchors: tuple = None) -> jnp.ndarray:
    """
    Guide corresponding to the co-transformation model.
    Learns variational posteriors for lam, mu, and sigma.

    Parameters
    ----------
    name : str
        The prefix for the Numpyro sample sites.
    data : GrowthData
        A data object containing metadata.
    priors : ModelPriors
        A Pytree containing the hyperparameters for initialization.
    anchors : tuple, optional
        Population anchors (dummy here, but needed for interface).

    Returns
    -------
    tuple
        (lam, mu, sigma) sampled values from guide.
    """
    
    # Variational approximation for lambda
    # Using LogNormal family
    
    h_lam_loc = pyro.param(f"{name}_lam_loc", jnp.array(priors.lam_loc))
    h_lam_scale = pyro.param(f"{name}_lam_scale", jnp.array(priors.lam_scale),
                             constraint=dist.constraints.positive)
    lam = pyro.sample(f"{name}_lam", dist.LogNormal(h_lam_loc, h_lam_scale))

    # Variational approximation for mu and sigma (plated)
    
    local_shape = (data.num_titrant_name, data.num_titrant_conc, 1)

    if anchors is None:
        init_mu = 0.0
        init_sigma = 1.0
    else:
        init_mu, init_sigma = anchors

    h_mu_loc = pyro.param(f"{name}_mu_loc", jnp.full(local_shape, init_mu))
    h_mu_scale = pyro.param(f"{name}_mu_scale", jnp.full(local_shape, 0.1),
                            constraint=dist.constraints.positive)
    
    h_sigma_loc = pyro.param(f"{name}_sigma_loc", jnp.full(local_shape, jnp.log(init_sigma)))
    h_sigma_scale = pyro.param(f"{name}_sigma_scale", jnp.full(local_shape, 0.1),
                               constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
            
            mu = pyro.sample(f"{name}_mu", dist.Normal(h_mu_loc, h_mu_scale))
            sigma = pyro.sample(f"{name}_sigma", dist.LogNormal(h_sigma_loc, h_sigma_scale))

    return lam, mu, sigma


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
    parameters["lam_scale"] = 0.5
    
    # Anchoring scales (tightening this makes the background track the prior more closely)
    parameters["mu_anchoring_scale"] = 0.5
    parameters["sigma_anchoring_scale"] = 0.2
               
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
    guesses[f"{name}_mu"] = 0.0
    guesses[f"{name}_sigma"] = 1.0
    
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
