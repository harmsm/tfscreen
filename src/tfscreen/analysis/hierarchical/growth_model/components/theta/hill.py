import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass
from typing import Dict, Any, Optional

from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass

@dataclass
class ModelPriors:
    """
    Holds hyperparameters for the Hill model priors.

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


@dataclass
class ThetaParam:
    """
    Holds the sampled Hill equation parameters.

    These are the parameters sampled in their natural scale.

    Attributes
    ----------
    theta_low : torch.Tensor
        The minimum fractional occupancy (baseline).
    theta_high : torch.Tensor
        The maximum fractional occupancy (saturation).
    log_hill_K : torch.Tensor
        The Hill constant (K_D) in log-space.
    hill_n : torch.Tensor
        The Hill coefficient.
    """

    theta_low: torch.Tensor
    theta_high: torch.Tensor
    log_hill_K: torch.Tensor
    hill_n: torch.Tensor
    mu: Optional[torch.Tensor] = None
    sigma: Optional[torch.Tensor] = None


def define_model(name: str,
                 data: DataClass,
                 priors: ModelPriors) -> ThetaParam:
    """
    Defines the hierarchical Hill model parameters.

    This function defines the Pyro ``sample`` sites for a non-centered
    hierarchical model of Hill parameters (theta_low, theta_high, K, and n).

    - ``theta_low`` and ``theta_delta`` use pooled logit-scaled hyperpriors.
      We convert ``theta_low`` and ``theta_delta`` into ``theta_high`` prior
      to the sigmoid transform to enforce [0,1] bounds on both.
    - ``hill_K`` and ``hill_n`` use pooled log-scaled hyperpriors.

    Parameters
    ----------
    name : str
        The prefix for all Pyro sample sites (e.g., "theta").
    data : DataClass
        A data object containing metadata, primarily:
        - ``data.num_titrant_name`` : (int) Number of titrants.
        - ``data.num_genotype`` : (int) Number of genotypes.
    priors : ModelPriors
        A dataclass containing all hyperparameters for the model.

    Returns
    -------
    ThetaParam
        A dataclass containing the sampled Hill parameters (theta_low,
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

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-2):
        with pyro.plate(f"{name}_genotype_plate", size=data.batch_size, dim=-1):
            with pyro.poutine.scale(scale=data.scale_vector):
                logit_theta_low_offset = pyro.sample(f"{name}_logit_low_offset", dist.Normal(0.0, 1.0))
                logit_theta_delta_offset = pyro.sample(f"{name}_logit_delta_offset", dist.Normal(0.0, 1.0))
                log_hill_K_offset = pyro.sample(f"{name}_log_hill_K_offset", dist.Normal(0.0, 1.0))
                log_hill_n_offset = pyro.sample(f"{name}_log_hill_n_offset", dist.Normal(0.0, 1.0))

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if logit_theta_low_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        logit_theta_low_offset = logit_theta_low_offset[..., data.batch_idx]
        logit_theta_delta_offset = logit_theta_delta_offset[..., data.batch_idx]
        log_hill_K_offset = log_hill_K_offset[..., data.batch_idx]
        log_hill_n_offset = log_hill_n_offset[..., data.batch_idx]

    logit_theta_low = logit_theta_low_hyper_loc + logit_theta_low_offset * logit_theta_low_hyper_scale
    logit_theta_delta = logit_theta_delta_hyper_loc + logit_theta_delta_offset * logit_theta_delta_hyper_scale
    logit_theta_high = logit_theta_low + logit_theta_delta
    log_hill_K = log_hill_K_hyper_loc + log_hill_K_offset * log_hill_K_hyper_scale
    log_hill_n = log_hill_n_hyper_loc + log_hill_n_offset * log_hill_n_hyper_scale

    # --------------------------------------------------------------------------
    # Calculate population moments (mu, sigma) using a "ghost population"

    # We sample a fixed-size population from the hyper-priors to estimate
    # the distribution of logit(theta) at each concentration.
    n_ghost = 100
    ghost_low = logit_theta_low_hyper_loc + torch.randn(n_ghost) * logit_theta_low_hyper_scale
    ghost_delta = logit_theta_delta_hyper_loc + torch.randn(n_ghost) * logit_theta_delta_hyper_scale
    ghost_high = ghost_low + ghost_delta
    ghost_K = log_hill_K_hyper_loc + torch.randn(n_ghost) * log_hill_K_hyper_scale
    ghost_n = log_hill_n_hyper_loc + torch.randn(n_ghost) * log_hill_n_hyper_scale

    # Calculate logit(theta) for ghost population across all concentrations
    log_conc = torch.as_tensor(data.log_titrant_conc).float()[None, :, None]  # (1, Conc, 1)

    # Ghost parameters shape: (1, 1, Ghost)
    g_low = ghost_low[None, None, :]
    g_high = ghost_high[None, None, :]
    g_K = ghost_K[None, None, :]
    g_n = torch.exp(ghost_n[None, None, :])

    eps = 1e-6
    g_occ = torch.sigmoid(g_n * (log_conc - g_K))
    g_theta = torch.clamp(
        torch.sigmoid(g_low) + (torch.sigmoid(g_high) - torch.sigmoid(g_low)) * g_occ,
        eps, 1.0 - eps)
    g_logit_theta = torch.logit(g_theta)

    mu_pop = g_logit_theta.mean(dim=-1, keepdim=True)    # (1, Conc, 1)
    sigma_pop = g_logit_theta.std(dim=-1, keepdim=True)  # (1, Conc, 1)

    # --------------------------------------------------------------------------
    # Expand parameters

    # Transform parameters to their natural scale
    theta_low = torch.sigmoid(logit_theta_low)
    theta_high = torch.sigmoid(logit_theta_high)
    # log_hill_K is already on its natural scale
    hill_n = torch.exp(log_hill_n)

    # Register parameter values
    pyro.deterministic(f"{name}_theta_low", theta_low)
    pyro.deterministic(f"{name}_theta_high", theta_high)
    pyro.deterministic(f"{name}_log_hill_K", log_hill_K)
    pyro.deterministic(f"{name}_hill_n", hill_n)

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
    h_low_loc_loc = pyro.param(f"{name}_logit_low_hyper_loc_loc", torch.tensor(priors.theta_logit_low_hyper_loc_loc))
    h_low_loc_scale = pyro.param(f"{name}_logit_low_hyper_loc_scale", torch.tensor(priors.theta_logit_low_hyper_loc_scale),
                                 constraint=torch.distributions.constraints.positive)
    logit_theta_low_hyper_loc = pyro.sample(f"{name}_logit_low_hyper_loc",
                                            dist.Normal(h_low_loc_loc, h_low_loc_scale))

    # Scale (LogNormal guide)
    h_low_scale_loc = pyro.param(f"{name}_logit_low_hyper_scale_loc", torch.tensor(-1.0))
    h_low_scale_scale = pyro.param(f"{name}_logit_low_hyper_scale_scale", torch.tensor(0.1),
                                   constraint=torch.distributions.constraints.positive)
    logit_theta_low_hyper_scale = pyro.sample(f"{name}_logit_low_hyper_scale",
                                              dist.LogNormal(h_low_scale_loc, h_low_scale_scale))

    # --- Theta Delta (Logit Scale) ---
    # Loc
    h_delta_loc_loc = pyro.param(f"{name}_logit_delta_hyper_loc_loc", torch.tensor(priors.theta_logit_delta_hyper_loc_loc))
    h_delta_loc_scale = pyro.param(f"{name}_logit_delta_hyper_loc_scale", torch.tensor(priors.theta_logit_delta_hyper_loc_scale),
                                   constraint=torch.distributions.constraints.positive)
    logit_theta_delta_hyper_loc = pyro.sample(f"{name}_logit_delta_hyper_loc",
                                              dist.Normal(h_delta_loc_loc, h_delta_loc_scale))

    # Scale (LogNormal guide)
    h_delta_scale_loc = pyro.param(f"{name}_logit_delta_hyper_scale_loc", torch.tensor(-1.0))
    h_delta_scale_scale = pyro.param(f"{name}_logit_delta_hyper_scale_scale", torch.tensor(0.1),
                                     constraint=torch.distributions.constraints.positive)
    logit_theta_delta_hyper_scale = pyro.sample(f"{name}_logit_delta_hyper_scale",
                                                dist.LogNormal(h_delta_scale_loc, h_delta_scale_scale))

    # --- Hill K (Log Scale) ---
    # Loc
    h_K_loc_loc = pyro.param(f"{name}_log_hill_K_hyper_loc_loc", torch.tensor(priors.theta_log_hill_K_hyper_loc_loc))
    h_K_loc_scale = pyro.param(f"{name}_log_hill_K_hyper_loc_scale", torch.tensor(priors.theta_log_hill_K_hyper_loc_scale),
                               constraint=torch.distributions.constraints.positive)
    log_hill_K_hyper_loc = pyro.sample(f"{name}_log_hill_K_hyper_loc",
                                       dist.Normal(h_K_loc_loc, h_K_loc_scale))

    # Scale (LogNormal guide)
    h_K_scale_loc = pyro.param(f"{name}_log_hill_K_hyper_scale_loc", torch.tensor(-1.0))
    h_K_scale_scale = pyro.param(f"{name}_log_hill_K_hyper_scale_scale", torch.tensor(0.1),
                                 constraint=torch.distributions.constraints.positive)
    log_hill_K_hyper_scale = pyro.sample(f"{name}_log_hill_K_hyper_scale",
                                         dist.LogNormal(h_K_scale_loc, h_K_scale_scale))

    # --- Hill n (Log Scale) ---
    # Loc
    h_n_loc_loc = pyro.param(f"{name}_log_hill_n_hyper_loc_loc", torch.tensor(priors.theta_log_hill_n_hyper_loc_loc))
    h_n_loc_scale = pyro.param(f"{name}_log_hill_n_hyper_loc_scale", torch.tensor(priors.theta_log_hill_n_hyper_loc_scale),
                               constraint=torch.distributions.constraints.positive)
    log_hill_n_hyper_loc = pyro.sample(f"{name}_log_hill_n_hyper_loc",
                                       dist.Normal(h_n_loc_loc, h_n_loc_scale))

    # Scale (LogNormal guide)
    h_n_scale_loc = pyro.param(f"{name}_log_hill_n_hyper_scale_loc", torch.tensor(-1.0))
    h_n_scale_scale = pyro.param(f"{name}_log_hill_n_hyper_scale_scale", torch.tensor(0.1),
                                 constraint=torch.distributions.constraints.positive)
    log_hill_n_hyper_scale = pyro.sample(f"{name}_log_hill_n_hyper_scale",
                                         dist.LogNormal(h_n_scale_loc, h_n_scale_scale))


    # ==========================================================================
    # 2. Local Parameters (Offset Variational Params)
    # ==========================================================================

    # Shape: (NumTitrants, NumGenotypes)
    local_shape = (data.num_titrant_name, data.num_genotype)

    # Low Offsets
    low_offset_locs = pyro.param(f"{name}_logit_low_offset_locs", torch.zeros(local_shape))
    low_offset_scales = pyro.param(f"{name}_logit_low_offset_scales", torch.ones(local_shape),
                                   constraint=torch.distributions.constraints.positive)

    # Delta Offsets
    delta_offset_locs = pyro.param(f"{name}_logit_delta_offset_locs", torch.zeros(local_shape))
    delta_offset_scales = pyro.param(f"{name}_logit_delta_offset_scales", torch.ones(local_shape),
                                     constraint=torch.distributions.constraints.positive)

    # K Offsets
    K_offset_locs = pyro.param(f"{name}_log_hill_K_offset_locs", torch.zeros(local_shape))
    K_offset_scales = pyro.param(f"{name}_log_hill_K_offset_scales", torch.ones(local_shape),
                                 constraint=torch.distributions.constraints.positive)

    # n Offsets
    n_offset_locs = pyro.param(f"{name}_log_hill_n_offset_locs", torch.zeros(local_shape))
    n_offset_scales = pyro.param(f"{name}_log_hill_n_offset_scales", torch.ones(local_shape),
                                 constraint=torch.distributions.constraints.positive)


    # ==========================================================================
    # 3. Sampling (Sliced by Genotype)
    # ==========================================================================

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-2):
        # Batching on Genotype (dim=-1)
        with pyro.plate(f"{name}_genotype_plate", size=data.batch_size, dim=-1):

            # Scale data for sub-sampling
            with pyro.poutine.scale(scale=data.scale_vector):

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

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if logit_theta_low_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        logit_theta_low_offset = logit_theta_low_offset[..., data.batch_idx]
        logit_theta_delta_offset = logit_theta_delta_offset[..., data.batch_idx]
        log_hill_K_offset = log_hill_K_offset[..., data.batch_idx]
        log_hill_n_offset = log_hill_n_offset[..., data.batch_idx]

    # ==========================================================================
    # 4. Reconstruction
    # ==========================================================================

    logit_theta_low = logit_theta_low_hyper_loc + logit_theta_low_offset * logit_theta_low_hyper_scale
    logit_theta_delta = logit_theta_delta_hyper_loc + logit_theta_delta_offset * logit_theta_delta_hyper_scale
    logit_theta_high = logit_theta_low + logit_theta_delta
    log_hill_K = log_hill_K_hyper_loc + log_hill_K_offset * log_hill_K_hyper_scale
    log_hill_n = log_hill_n_hyper_loc + log_hill_n_offset * log_hill_n_hyper_scale

    # Transform
    theta_low = torch.sigmoid(logit_theta_low)
    theta_high = torch.sigmoid(logit_theta_high)
    hill_n = torch.exp(log_hill_n)

    # --------------------------------------------------------------------------
    # Calculate population moments (mu, sigma) using a "ghost population"

    n_ghost = 100
    ghost_low = logit_theta_low_hyper_loc + torch.randn(n_ghost) * logit_theta_low_hyper_scale
    ghost_delta = logit_theta_delta_hyper_loc + torch.randn(n_ghost) * logit_theta_delta_hyper_scale
    ghost_high = ghost_low + ghost_delta
    ghost_K = log_hill_K_hyper_loc + torch.randn(n_ghost) * log_hill_K_hyper_scale
    ghost_n = log_hill_n_hyper_loc + torch.randn(n_ghost) * log_hill_n_hyper_scale

    log_conc = torch.as_tensor(data.log_titrant_conc).float()[None, :, None]  # (1, Conc, 1)

    g_low = ghost_low[None, None, :]
    g_high = ghost_high[None, None, :]
    g_K = ghost_K[None, None, :]
    g_n = torch.exp(ghost_n[None, None, :])

    eps = 1e-6
    g_occ = torch.sigmoid(g_n * (log_conc - g_K))
    g_theta = torch.clamp(
        torch.sigmoid(g_low) + (torch.sigmoid(g_high) - torch.sigmoid(g_low)) * g_occ,
        eps, 1.0 - eps)
    g_logit_theta = torch.logit(g_theta)

    mu_pop = g_logit_theta.mean(dim=-1, keepdim=True)    # (1, Conc, 1)
    sigma_pop = g_logit_theta.std(dim=-1, keepdim=True)  # (1, Conc, 1)

    theta_param = ThetaParam(theta_low=theta_low,
                             theta_high=theta_high,
                             log_hill_K=log_hill_K,
                             hill_n=hill_n,
                             mu=mu_pop,
                             sigma=sigma_pop)

    return theta_param

def run_model(theta_param: ThetaParam, data: DataClass) -> torch.Tensor:
    """
    Calculates fractional occupancy (theta) using the Hill equation.

    This is a pure PyTorch function that deterministically calculates theta
    values using the sampled parameters from ``define_model``.

    Parameters
    ----------
    theta_param : ThetaParam
        A dataclass generated by ``define_model`` containing the sampled
        Hill parameters. Tensors within (e.g., ``theta_param.hill_K``)
        are expected to have dimensions ``[titrant_name, genotype]``.
    data : DataClass
        A data object containing:
        - ``data.log_titrant_conc``: (torch.Tensor) Titrant concentrations.
        - ``data.scatter_theta``: (int) A flag (0 or 1) indicating
          whether to scatter the final tensor.
        - ``data.geno_theta_idx``: (torch.Tensor) Indices of genotypes to select.

    Returns
    -------
    torch.Tensor
        A tensor of calculated theta values.
        - If ``data.scatter_theta == 0``, shape is
          ``[titrant_name, titrant_conc, genotype]``.
        - If ``data.scatter_theta == 1``, shape is
          ``[replicate, time, treatment, genotype]``.
    """

    # Select genotypes from the last dimension using [..., geno_idx].
    # This handles both 2D [titrant_name, genotype] during normal inference
    # and 3D [num_samples, titrant_name, genotype] during posterior prediction
    # (when Predictive adds a leading sample dimension).
    geno_idx = torch.as_tensor(data.geno_theta_idx, dtype=torch.long)
    theta_low = torch.as_tensor(theta_param.theta_low, dtype=torch.float32)[..., geno_idx]
    theta_high = torch.as_tensor(theta_param.theta_high, dtype=torch.float32)[..., geno_idx]
    log_hill_K = torch.as_tensor(theta_param.log_hill_K, dtype=torch.float32)[..., geno_idx]
    hill_n = torch.as_tensor(theta_param.hill_n, dtype=torch.float32)[..., geno_idx]

    # Insert titrant_conc dim at -2 position:
    # 2D: [T, B_geno] -> [T, 1, B_geno]
    # 3D: [S, T, B_geno] -> [S, T, 1, B_geno]
    theta_low = theta_low.unsqueeze(-2)
    theta_high = theta_high.unsqueeze(-2)
    log_hill_K = log_hill_K.unsqueeze(-2)
    hill_n = hill_n.unsqueeze(-2)

    # log_titrant [C] -> [1, C, 1] to broadcast at the -2 position
    log_titrant = torch.as_tensor(data.log_titrant_conc, dtype=torch.float32)[None, :, None]

    occupancy = torch.sigmoid(hill_n * (log_titrant - log_hill_K))
    theta_calc = theta_low + (theta_high - theta_low) * occupancy

    # Broadcast to the full-sized tensor by inserting 4 leading size-1 dims
    # (for rep, time, cond_pre, cond_sel).
    # Extra leading singleton dims may be added by AutoDelta/Predictive's plate
    # depth tracking. Collapse all leading dims into a single batch dim, then
    # insert 4 spatial singletons. Result shape: (..., 1, 1, 1, 1, T, C, G).
    if data.scatter_theta == 1:
        # Insert 4 spatial singleton dims between any leading batch dims and the
        # trailing (titrant_name, titrant_conc, genotype) dims.
        # 3D input (T, C, G) -> (1, 1, 1, 1, T, C, G)
        # 4D input (S, T, C, G) -> (S, 1, 1, 1, 1, T, C, G)
        leading = theta_calc.shape[:-3]
        trailing = theta_calc.shape[-3:]
        theta_calc = theta_calc.reshape(*leading, 1, 1, 1, 1, *trailing)

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
        A dictionary mapping parameter names to their initial guess values.
    """

    guesses = {}

    guesses[f"{name}_logit_low_hyper_loc"] = 2.0
    guesses[f"{name}_logit_low_hyper_scale"] = 2.0
    guesses[f"{name}_logit_delta_hyper_loc"] = -4.0
    guesses[f"{name}_logit_delta_hyper_scale"] = 2.0

    guesses[f"{name}_log_hill_K_hyper_loc"] = -4.1  # ln(0.017 mM)
    guesses[f"{name}_log_hill_K_hyper_scale"] = 1.0
    guesses[f"{name}_log_hill_n_hyper_loc"] = 0.7
    guesses[f"{name}_log_hill_n_hyper_scale"] = 0.3

    guesses[f"{name}_logit_low_offset"] = torch.zeros(data.num_titrant_name, data.num_genotype)
    guesses[f"{name}_logit_delta_offset"] = torch.zeros(data.num_titrant_name, data.num_genotype)
    guesses[f"{name}_log_hill_K_offset"] = torch.zeros(data.num_titrant_name, data.num_genotype)
    guesses[f"{name}_log_hill_n_offset"] = torch.zeros(data.num_titrant_name, data.num_genotype)

    return guesses

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Returns
    -------
    ModelPriors
        A populated dataclass of hyperparameters.
    """
    return ModelPriors(**get_hyperparameters())
