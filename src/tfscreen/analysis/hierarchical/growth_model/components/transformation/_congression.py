import math
import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass, field
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from typing import Optional, Tuple, Union


def _torch_interp(x, xp, fp):
    """
    1-D linear interpolation with boundary clamping (equivalent to numpy.interp).

    Parameters
    ----------
    x : torch.Tensor
        Points at which to interpolate.
    xp : torch.Tensor
        Known x-coordinates (must be sorted ascending).
    fp : torch.Tensor
        Known y-coordinates corresponding to xp.

    Returns
    -------
    torch.Tensor
        Interpolated values at x.
    """
    idx = torch.searchsorted(xp.contiguous(), x.contiguous())
    lo = torch.clamp(idx - 1, 0, xp.shape[0] - 1)
    hi = torch.clamp(idx,     0, xp.shape[0] - 1)
    x0, x1 = xp[lo], xp[hi]
    y0, y1 = fp[lo], fp[hi]
    denom = x1 - x0
    t = torch.where(denom > 0, (x - x0) / denom, torch.zeros_like(x))
    return y0 + t * (y1 - y0)


def _logit_normal_cdf(x, mu, sigma):
    """
    Cumulative Distribution Function for Logit-Normal distribution:
    F(x) = Phi((logit(x) - mu) / sigma)

    Parameters
    ----------
    x : torch.Tensor
        occupancy value between 0 and 1
    mu : float or torch.Tensor
        Mean of the underlying Normal distribution in logit-space.
    sigma : float or torch.Tensor
        Standard deviation of the underlying Normal distribution in logit-space.

    Returns
    -------
    torch.Tensor
        cdf array
    """
    # Clip x to [eps, 1-eps] for stability in logit
    eps = 1e-6
    x_safe = torch.clamp(torch.as_tensor(x, dtype=torch.float32), eps, 1.0 - eps)

    # Calculate logit(x)
    logit_x = torch.logit(x_safe)

    # Return Phi((logit(x) - mu) / sigma)
    return torch.distributions.Normal(mu, sigma).cdf(logit_x)


def _empirical_cdf(theta, t_grid):
    """
    Cumulative Distribution Function estimated from the observed population
    of theta values.
    """
    theta = torch.as_tensor(theta, dtype=torch.float32)
    t_grid = torch.as_tensor(t_grid, dtype=torch.float32)
    n = theta.shape[-1]
    sorted_theta, _ = torch.sort(theta, dim=-1)

    # Empirical CDF values at the sorted points.
    # We use (i + 0.5) / n to be unbiased for a continuous distribution.
    y = (torch.arange(n, dtype=torch.float32) + 0.5) / n

    # Interpolate to the integration grid
    shape = theta.shape[:-1]
    flat_sorted = sorted_theta.reshape(-1, n)

    rows = []
    for i in range(flat_sorted.shape[0]):
        rows.append(_torch_interp(t_grid, flat_sorted[i], y))

    flat_cdf = torch.stack(rows, dim=0)
    return flat_cdf.reshape(shape + (len(t_grid),))


def update_thetas(theta, params, theta_dist=None, mask=None, n_grid=256):
    """
    Corrects theta values for co-transformation using the method of
    re-sampling from the background distribution.

    Parameters
    ----------
    theta : torch.Tensor
        Array of theta values.
        Expected shape: (..., num_genotype)
    params : tuple
        Tuple of parameters defining the background distribution.
        If theta_dist is "logit_norm" (default): (lam, mu, sigma)
        If theta_dist is "empirical": (lam,)
    theta_dist : str, optional
        One of "logit_norm" or "empirical".
        If None, the distribution is inferred from the length of params.
    mask : torch.Tensor, optional
        Boolean array of shape (num_genotype,) where True indicates the
        genotype should be corrected for congression. If None, all genotypes
        are corrected.
    n_grid : int, optional
        Number of points for grid-based numerical integration (default 256).

    Returns
    -------
    torch.Tensor
        Corrected theta array with shape broadcasted from inputs.
    """
    theta = torch.as_tensor(theta, dtype=torch.float32)

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
    t_grid = torch.linspace(0.0, 1.0, n_grid)

    if theta_dist == "logit_norm":
        # (lam, mu, sigma) — broadcast together
        p_tensors = [torch.as_tensor(p, dtype=torch.float32) for p in params[:3]]
        b_arrays = torch.broadcast_tensors(*p_tensors)
        integration_shape = b_arrays[0].shape
        flat_params = [a.reshape(-1) for a in b_arrays]
        num_p_batches = flat_params[0].shape[0]
        flat_lam = flat_params[0]

        # Vectorised CDF: shape (num_p_batches, n_grid)
        m = flat_params[1][:, None]   # (num_p_batches, 1)
        s = flat_params[2][:, None]   # (num_p_batches, 1)
        t = t_grid[None, :]           # (1, n_grid)
        Ft_grid = _logit_normal_cdf(t, m, s)

    elif theta_dist == "empirical":
        Ft_grid_raw = _empirical_cdf(theta, t_grid)
        num_p_batches = Ft_grid_raw.reshape(-1, n_grid).shape[0]

        # Broadcast lambda to match theta's batch dimension
        lam_tensor = torch.as_tensor(lam, dtype=torch.float32)
        lam_b = lam_tensor.expand(theta.shape[:-1])
        flat_lam = lam_b.reshape(-1)

        Ft_grid = Ft_grid_raw.reshape(-1, n_grid)
        integration_shape = theta.shape[:-1]

    else:
        raise ValueError(f"Unsupported theta_dist: {theta_dist}")

    # Calculate integrand: exp(lam * (F(t) - 1))
    # integrand_grid shape: (num_p_batches, n_grid)
    integrand_grid = torch.exp(flat_lam[:, None] * (Ft_grid - 1.0))

    # Cumulative integration using trapezoidal rule
    # J(x) = integrate_0^x I(t) dt
    h = 1.0 / (n_grid - 1)
    f_mid = (integrand_grid[:, :-1] + integrand_grid[:, 1:]) * h / 2.0
    J_grid = torch.cat([torch.zeros(num_p_batches, 1), torch.cumsum(f_mid, dim=1)], dim=1)

    # Expected value calculation part 1: G(x) = integrate_x^1 I(t) dt
    # G(x) = J(1) - J(x)
    G1 = J_grid[:, -1:]
    Gx_grid_p = G1 - J_grid

    # Reshape Gx_grid back to broadcasting-ready shape
    if len(integration_shape) > 0 and integration_shape[-1] == 1:
        res_batch_shape = integration_shape[:-1]
    else:
        res_batch_shape = integration_shape

    Gx_grid_final = Gx_grid_p.reshape(res_batch_shape + (n_grid,))

    # Broadcast integration results to match theta's batch dimensions
    target_shape = theta.shape[:-1]
    num_genotypes = theta.shape[-1]

    Gx_grid_b = Gx_grid_final.expand(target_shape + (n_grid,))

    # Interpolate G(x) for each genotype in the full batch
    flat_theta = theta.reshape(-1, num_genotypes)
    flat_Gx = Gx_grid_b.reshape(-1, n_grid)

    rows = []
    for i in range(flat_Gx.shape[0]):
        rows.append(_torch_interp(flat_theta[i], t_grid, flat_Gx[i]))
    integral_vals = torch.stack(rows, dim=0)

    # Expected observed value E[max(x, M)] = 1 - G(x)
    corrected_flat = 1.0 - integral_vals

    # Reshape back to original dimensions
    corrected_theta = corrected_flat.reshape(target_shape + (num_genotypes,))

    # Apply mask if provided
    if mask is not None:
        corrected_theta = torch.where(mask, corrected_theta, theta)

    return corrected_theta


def calculate_expected_observed_max(x_val, mu, sigma, lam, n_grid=100):
    """
    Calculate E[max(x, M)] where M is the maximum of a Poisson(lam) number
    of samples from the background Logit-Normal(mu, sigma) distribution.

    Use the stable formula:
    E[max(x, M)] = 1 - integrate_{x}^1 exp(lam * (F(t) - 1)) dt
    """
    x_safe = torch.clamp(torch.as_tensor(float(x_val)), 0.0, 1.0)

    t_grid = torch.linspace(x_safe.item(), 1.0, n_grid)
    Ft = _logit_normal_cdf(t_grid, mu, sigma)
    y_vals = torch.exp(lam * (Ft - 1.0))
    integral_val = torch.trapezoid(y_vals, t_grid)

    return 1.0 - integral_val


def calculate_expected_observed_min(x_val, mu, sigma, lam, n_grid=100):
    """
    Calculate E[min(x, M)] where M is the minimum of a Poisson(lam) number
    of samples from the background Logit-Normal(mu, sigma) distribution.

    Use the stable formula:
    E[min(x, M)] = integrate_{0}^x exp(-lam * F(t)) dt
    """
    x_safe = torch.clamp(torch.as_tensor(float(x_val)), 0.0, 1.0)

    t_grid = torch.linspace(0.0, x_safe.item(), n_grid)
    Ft = _logit_normal_cdf(t_grid, mu, sigma)
    y_vals = torch.exp(-lam * Ft)
    integral_val = torch.trapezoid(y_vals, t_grid)

    return integral_val


@dataclass
class ModelPriors:
    """
    Holds hyperparameters for the co-transformation model.

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
    mode : str
        Background distribution mode: "logit_norm" or "empirical".
    """

    lam_loc: float
    lam_scale: float
    mu_anchoring_scale: float
    sigma_anchoring_scale: float
    mode: str = "logit_norm"


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors,
                 anchors: tuple = None) -> tuple:
    """
    Defines the hierarchical model for the co-transformation parameter (lambda)
    and background parameters (mu, sigma, etc.).
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
        with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
            with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):

                mu = pyro.sample(
                    f"{name}_mu",
                    dist.Normal(anc_mu, priors.mu_anchoring_scale)
                )
                pyro.deterministic(f"{name}_mu_value", mu)

                log_anc_sigma = torch.log(torch.as_tensor(anc_sigma, dtype=torch.float32)) if not isinstance(anc_sigma, float) else math.log(anc_sigma) if anc_sigma > 0 else 0.0
                sigma = pyro.sample(
                    f"{name}_sigma",
                    dist.LogNormal(log_anc_sigma, priors.sigma_anchoring_scale)
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
    """

    # Variational approximation for lambda
    h_lam_loc = pyro.param(f"{name}_lam_loc", torch.tensor(float(priors.lam_loc)))
    h_lam_scale = pyro.param(f"{name}_lam_scale", torch.tensor(float(priors.lam_scale)),
                             constraint=torch.distributions.constraints.positive)
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
        h_mu_loc = pyro.param(f"{name}_mu_loc",
                              torch.full(local_shape, float(init_mu) if not hasattr(init_mu, 'item') else init_mu.mean().item()))
        h_mu_scale = pyro.param(f"{name}_mu_scale",
                                torch.full(local_shape, 0.1),
                                constraint=torch.distributions.constraints.positive)

        init_log_sigma = math.log(float(init_sigma)) if isinstance(init_sigma, (int, float)) else torch.log(torch.as_tensor(init_sigma, dtype=torch.float32)).mean().item()
        h_sigma_loc = pyro.param(f"{name}_sigma_loc",
                                 torch.full(local_shape, init_log_sigma))
        h_sigma_scale = pyro.param(f"{name}_sigma_scale",
                                   torch.full(local_shape, 0.1),
                                   constraint=torch.distributions.constraints.positive)

        with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
            with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):

                mu = pyro.sample(f"{name}_mu", dist.Normal(h_mu_loc, h_mu_scale))
                sigma = pyro.sample(f"{name}_sigma", dist.LogNormal(h_sigma_loc, h_sigma_scale))

        return lam, mu, sigma

    raise ValueError(f"Unsupported mode: {priors.mode}")


def get_hyperparameters():
    """
    Gets default values for the model hyperparameters.
    """

    parameters = {}

    # Lambda prior: centered on 1.0
    parameters["lam_loc"] = 0.0
    parameters["lam_scale"] = 0.01

    # Anchoring scales
    parameters["mu_anchoring_scale"] = 0.5
    parameters["sigma_anchoring_scale"] = 0.2

    # Background model mode
    parameters["mode"] = "logit_norm"

    return parameters


def get_guesses(name, data):
    """
    Gets initial guess values for model parameters.
    """

    guesses = {}
    guesses[f"{name}_lam"] = 1.0
    guesses[f"{name}_mu"] = 0.0
    guesses[f"{name}_sigma"] = 1.0

    return guesses


def get_priors():
    """
    Utility function to create a populated ModelPriors object.
    """
    return ModelPriors(**get_hyperparameters())
