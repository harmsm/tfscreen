"""
The expected-maximum theta operator for co-transformation.

``update_thetas`` replaces a genotype's theta by ``E[max(theta, M)]``, where M
is the largest theta among Poisson(lambda) co-resident plasmids drawn from a
background distribution (logit-normal or the empirical population). It was
the fit's congression correction until step 3.3c of
``planning/congression-physics-plan.md`` replaced it with the observable-level
mixture (``mixture.py``). It is kept as a plain function for Stage 1.5 of
``tfs-fit-genotypes`` / ``tfs-build-empirical``
(``tfmodel/genotype_fit/congression.py``), which de-attenuates theta curves
with it; it is not a registered transformation component.
"""

import jax
import jax.numpy as jnp

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


def update_thetas(theta, params, theta_dist=None, mask=None, n_grid=256,
                  population_theta=None):
    """
    Corrects theta values for co-transformation using the method of
    re-sampling from the background distribution.

    Parameters
    ----------
    theta : jnp.array
        Array of theta values to correct.
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
    population_theta : jnp.array, optional
        Array of theta values used to build the background (empirical) CDF,
        expected shape (..., num_population_genotype), broadcastable against
        ``theta`` on every axis except the last.  Only consulted when
        ``theta_dist == "empirical"``.  If None (default), ``theta`` itself is
        used as the population sample, which is only statistically valid when
        ``theta`` already spans the full genotype population — callers that
        only ever see a genotype subset or minibatch (e.g. batched training or
        single-genotype prediction) must pass the true population sample here
        or the empirical CDF silently degenerates to whatever subset is
        present, biasing the correction. Ignored for "logit_norm", which uses
        the smooth analytic CDF in ``params`` instead of raw samples.

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
        # Build the background CDF from population_theta when supplied (the
        # full genotype population) rather than theta itself (which may only
        # cover a training minibatch or a handful of requested genotypes).
        # See the population_theta docstring above.
        pop_theta = theta if population_theta is None else population_theta

        Ft_grid_raw = _empirical_cdf(pop_theta, t_grid)
        num_p_batches = Ft_grid_raw.reshape(-1, n_grid).shape[0]

        # Broadcast lambda to match the population's leading (non-genotype)
        # dimensions, which must agree with theta's leading dimensions.
        lam_b = jnp.broadcast_to(lam, pop_theta.shape[:-1])
        flat_lam = lam_b.reshape(-1)

        Ft_grid = Ft_grid_raw.reshape(-1, n_grid)
        integration_shape = pop_theta.shape[:-1]

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
