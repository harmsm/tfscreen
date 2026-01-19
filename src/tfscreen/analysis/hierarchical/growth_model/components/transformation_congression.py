import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import (
    dataclass,
    field
)
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from typing import Optional, Tuple, Union

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


def _empirical_cdf(theta, t_grid):
    """
    Cumulative Distribution Function estimated from the observed population
    of theta values.
    """
    n = theta.shape[-1]
    sorted_theta = jnp.sort(theta, axis=-1)
    
    # Empirical CDF values at the sorted points.
    # We use (i + 0.5) / n to be unbiased for a continuous distribution.
    y = (jnp.arange(n) + 0.5) / n
    
    # Interpolate to the integration grid
    shape = theta.shape[:-1]
    flat_sorted = sorted_theta.reshape(-1, n)
    
    def get_cdf(s):
        return jnp.interp(t_grid, s, y)
    
    flat_cdf = jax.vmap(get_cdf)(flat_sorted)
    return flat_cdf.reshape(shape + (len(t_grid),))


def update_thetas(theta, params, theta_dist=None, mask=None, n_grid=256):
    """
    Corrects theta values for co-transformation using the method of 
    re-sampling from the background distribution.

    Parameters
    ----------
    theta : jnp.array
        Array of theta values. 
        Expected shape: (..., num_genotype)
    params : tuple
        Tuple of parameters defining the background distribution.
        If theta_dist is "logit_norm" (default): (lam, mu, sigma)
        If theta_dist is "empirical": (lam,)
    theta_dist : str, optional
        One of "logit_norm" or "empirical".
        If None, the distribution is inferred from the length of params.
    mask : jnp.array, optional
        Boolean array of shape (num_genotype,) where True indicates the 
        genotype should be corrected for congression. If None, all genotypes
        are corrected.
    n_grid : int, optional
        Number of points for grid-based numerical integration (default 256).

    Returns
    -------
    jnp.array
        Corrected theta array with shape broadcasted from inputs.
    """
    # Extract parameters. We assume the first is always lam.
    lam = params[0]
    bg_params = params[1:]

    # Infer theta_dist if not provided
    if theta_dist is None:
        if len(bg_params) == 2:
            theta_dist = "logit_norm"
        elif len(bg_params) == 0:
            theta_dist = "empirical"
        else:
            raise ValueError(f"Ambiguous parameter count: {len(params)}. Please specify theta_dist.")

    # Integration grid
    t_grid = jnp.linspace(0.0, 1.0, n_grid)

    if theta_dist == "logit_norm":
        # (lam, mu, sigma)
        b_arrays = jnp.broadcast_arrays(*params[:3])
        flat_params = [jnp.reshape(a, -1) for a in b_arrays]
        num_p_batches = flat_params[0].shape[0]
        integration_shape = b_arrays[0].shape
        flat_lam = flat_params[0]
        
        Ft_grid = jax.vmap(lambda m, s: _logit_normal_cdf(t_grid, m, s))(flat_params[1], flat_params[2])
        
    elif theta_dist == "empirical":
        Ft_grid_raw = _empirical_cdf(theta, t_grid)
        num_p_batches = Ft_grid_raw.reshape(-1, n_grid).shape[0]
        
        # Broadcast lambda to match theta's batch dimension
        lam_b = jnp.broadcast_to(lam, theta.shape[:-1])
        flat_lam = lam_b.reshape(-1)
        
        Ft_grid = Ft_grid_raw.reshape(-1, n_grid)
        integration_shape = theta.shape[:-1]
        
    else:
        raise ValueError(f"Unsupported theta_dist: {theta_dist}")

    # Calculate integrand: exp(lam * (F(t) - 1))
    # integrand_grid shape: (num_p_batches, n_grid)
    integrand_grid = jnp.exp(flat_lam[:, None] * (Ft_grid - 1.0))
    
    # Cumulative integration using trapezoidal rule
    # J(x) = integrate_0^x I(t) dt
    h = 1.0 / (n_grid - 1)
    f_mid = (integrand_grid[:, :-1] + integrand_grid[:, 1:]) * h / 2.0
    J_grid = jnp.concatenate([jnp.zeros((num_p_batches, 1)), jnp.cumsum(f_mid, axis=1)], axis=1)
    
    # Expected value calculation part 1: G(x) = integrate_x^1 I(t) dt
    # G(x) = J(1) - J(x)
    G1 = J_grid[:, -1:]
    Gx_grid_p = G1 - J_grid
    
    # Reshape Gx_grid back to broadcasting-ready shape
    # If integration_shape has a trailing 1 (genotype dim), remove it so we can broadcast
    # to (target_shape + n_grid) correctly.
    if len(integration_shape) > 0 and integration_shape[-1] == 1:
        res_batch_shape = integration_shape[:-1]
    else:
        res_batch_shape = integration_shape
        
    Gx_grid_final = Gx_grid_p.reshape(res_batch_shape + (n_grid,))
    
    # 3. Broadcast integration results to match theta's batch dimensions
    target_shape = theta.shape[:-1]
    num_genotypes = theta.shape[-1]
    
    # Gx_grid_final is already (integration_shape, n_grid)
    # If integration_shape matches target_shape, we are good.
    Gx_grid_b = jnp.broadcast_to(Gx_grid_final, target_shape + (n_grid,))
    
    # 4. Interpolate G(x) for each genotype in the full batch
    flat_theta = theta.reshape(-1, num_genotypes)
    flat_Gx = Gx_grid_b.reshape(-1, n_grid)
    
    def interp_row(g_row, th_row):
        return jnp.interp(th_row, t_grid, g_row)
    
    integral_vals = jax.vmap(interp_row)(flat_Gx, flat_theta)
    
    # Expected observed value E[max(x, M)] = 1 - G(x)
    corrected_flat = 1.0 - integral_vals
    
    # Reshape back to original dimensions
    corrected_theta = corrected_flat.reshape(target_shape + (num_genotypes,))

    # 5. Apply mask if provided
    if mask is not None:
        corrected_theta = jnp.where(mask, corrected_theta, theta)
    
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
    mode: str = field(default="logit_norm", pytree_node=False)
    
def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors,
                 anchors: tuple = None) -> tuple:
    """
    Defines the hierarchical model for the co-transformation parameter (lambda)
    and background parameters (mu, sigma, etc.).

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
        (lam, ...) sampled values depending on priors.mode.
    """
    
    # Global shared lambda
    lam = pyro.sample(
        f"{name}_lam",
        dist.LogNormal(priors.lam_loc, priors.lam_scale)
    )
    pyro.deterministic(f"{name}_lam_value", lam)

    if priors.mode == "empirical":
        return (lam,)

    if anchors is None:
        anc_mu = 0.0
        anc_sigma = 1.0
    else:
        anc_mu, anc_sigma = anchors

    if priors.mode == "logit_norm":
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
          anchors: tuple = None) -> tuple:
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
        (lam, ...) sampled values from guide.
    """
    
    # Variational approximation for lambda
    h_lam_loc = pyro.param(f"{name}_lam_loc", jnp.array(priors.lam_loc))
    h_lam_scale = pyro.param(f"{name}_lam_scale", jnp.array(priors.lam_scale),
                             constraint=dist.constraints.positive)
    lam = pyro.sample(f"{name}_lam", dist.LogNormal(h_lam_loc, h_lam_scale))

    if priors.mode == "empirical":
        return (lam,)

    # Variational approximation for mu and sigma (plated)
    local_shape = (data.num_titrant_name, data.num_titrant_conc, 1)

    if anchors is None:
        init_mu = 0.0
        init_sigma = 1.0
    else:
        init_mu, init_sigma = anchors

    if priors.mode == "logit_norm":
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

    raise ValueError(f"Unsupported mode: {priors.mode}")


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
    parameters["lam_scale"] = 0.01
    
    # Anchoring scales (tightening this makes the background track the prior more closely)
    parameters["mu_anchoring_scale"] = 0.5
    parameters["sigma_anchoring_scale"] = 0.2
    
    # Background model mode: 'logit_normal' (default) or 'empirical'
    parameters["mode"] = "logit_norm"
               
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
    
    # Only add these if we are not in empirical mode. 
    # NOTE: get_guesses currently doesn't know the mode from the config easily 
    # during initialization in model_class.py unless we pass it. 
    # For now, we'll keep them as guesses; they just won't be used if not 
    # in the model/guide.
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
