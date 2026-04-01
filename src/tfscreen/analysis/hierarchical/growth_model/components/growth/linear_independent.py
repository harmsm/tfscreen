import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import (
    GrowthData
)

@dataclass
class LinearParams:
    """
    Holds linear growth parameters (intercept and slope) for pre-selection
    and selection phases.
    """
    k_pre: torch.Tensor
    m_pre: torch.Tensor
    k_sel: torch.Tensor
    m_sel: torch.Tensor

@dataclass
class ModelPriors:
    """
    Holds hyperparameters for the independent growth model.

    Attributes
    ----------
    growth_k_hyper_loc_loc : torch.Tensor
        Mean of the prior for the hyper-location of k (per-condition).
    growth_k_hyper_loc_scale : torch.Tensor
        Standard deviation of the prior for the hyper-location of k
        (per-condition).
    growth_k_hyper_scale : torch.Tensor
        Scale of the HalfNormal prior for the hyper-scale of k
        (per-condition).
    growth_m_hyper_loc_loc : torch.Tensor
        Mean of the prior for the hyper-location of m (per-condition).
    growth_m_hyper_loc_scale : torch.Tensor
        Standard deviation of the prior for the hyper-location of m
        (per-condition).
    growth_m_hyper_scale : torch.Tensor
        Scale of the HalfNormal prior for the hyper-scale of m
        (per-condition).
    """

    # dims are num_conditions long
    growth_k_hyper_loc_loc: torch.Tensor
    growth_k_hyper_loc_scale: torch.Tensor
    growth_k_hyper_scale: torch.Tensor

    growth_m_hyper_loc_loc: torch.Tensor
    growth_m_hyper_loc_scale: torch.Tensor
    growth_m_hyper_scale: torch.Tensor

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> LinearParams:
    """
    Defines growth parameters k and m with independent priors per condition.

    This model defines growth parameters k (basal growth) and m (theta-dependent
    growth) where k and m are modeled as `k = k_hyper_loc + k_offset * k_hyper_scale` (and similarly for m).

    In this "independent" model, the hyper-parameters (`_hyper_loc`,
    `_hyper_scale`) are sampled independently for each experimental condition,
    and then all replicates within that condition share those hyper-parameters.

    Parameters
    ----------
    name : str
        The prefix for all Pyro sample/deterministic sites in this
        component.
    data : GrowthData
        A dataclass containing experimental data and metadata.
        This function primarily uses:
        - ``data.num_condition_rep`` : (int) Number of experimental conditions.
        - ``data.num_replicate`` : (int) Number of replicates per condition.
        - ``data.map_condition_pre`` : (torch.Tensor) Index array to map
          per-condition/replicate parameters to pre-selection observations.
        - ``data.map_condition_sel`` : (torch.Tensor) Index array to map
          per-condition/replicate parameters to post-selection observations.
    priors : ModelPriors
        A dataclass containing the hyperparameters for the
        priors. All attributes are ``torch.Tensor``s of shape
        ``(data.num_condition_rep,)``.
        - priors.growth_k_hyper_loc_loc
        - priors.growth_k_hyper_loc_scale
        - priors.growth_k_hyper_scale
        - priors.growth_m_hyper_loc_loc
        - priors.growth_m_hyper_loc_scale
        - priors.growth_m_hyper_scale

    Returns
    -------
    params : LinearParams
        A dataclass containing k_pre, m_pre, k_sel, and m_sel.
    """

    # Data assertions
    if data.growth_shares_replicates:
        raise ValueError("linear_independent cannot be used with growth_shares_replicates=True. Use 'linear' instead.")

    # Loop over conditions. NOTE THE FLIPPED PLATES. I need each condition to
    # have its own priors (outer loop) for each replicate (inner loop). The
    # data are ordered in the parameters as rep0, cond0 \ rep0, cond1 \ etc.
    # which means they ravel with these dimensions.
    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep, dim=-1):

        growth_k_hyper_loc = pyro.sample(
            f"{name}_k_hyper_loc",
            dist.Normal(priors.growth_k_hyper_loc_loc,
                        priors.growth_k_hyper_loc_scale)
        )
        growth_k_hyper_scale = pyro.sample(
            f"{name}_k_hyper_scale",
            dist.HalfNormal(priors.growth_k_hyper_scale)
        )

        growth_m_hyper_loc = pyro.sample(
            f"{name}_m_hyper_loc",
            dist.Normal(priors.growth_m_hyper_loc_loc,
                        priors.growth_m_hyper_loc_scale)
        )
        growth_m_hyper_scale = pyro.sample(
            f"{name}_m_hyper_scale",
            dist.HalfNormal(priors.growth_m_hyper_scale)
        )

        # Loop over replicates
        with pyro.plate(f"{name}_replicate_parameters", data.num_replicate, dim=-2):
            k_offset = pyro.sample(f"{name}_k_offset", dist.Normal(0.0, 1.0))
            m_offset = pyro.sample(f"{name}_m_offset", dist.Normal(0.0, 1.0))

        growth_k_dist = growth_k_hyper_loc + k_offset * growth_k_hyper_scale
        growth_m_dist = growth_m_hyper_loc + m_offset * growth_m_hyper_scale

    # Flatten array
    growth_k_dist_1d = growth_k_dist.reshape(-1)
    growth_m_dist_1d = growth_m_dist.reshape(-1)

    # Register dists
    pyro.deterministic(f"{name}_k", growth_k_dist_1d)
    pyro.deterministic(f"{name}_m", growth_m_dist_1d)

    # Expand to full-sized tensors
    k_pre = growth_k_dist_1d[data.map_condition_pre]
    m_pre = growth_m_dist_1d[data.map_condition_pre]
    k_sel = growth_k_dist_1d[data.map_condition_sel]
    m_sel = growth_m_dist_1d[data.map_condition_sel]

    return LinearParams(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)

def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> LinearParams:
    """
    Guide function for the independent condition/replicate growth model.

    This function defines the variational distributions (guide) for the
    independent growth model, specifying the parameterization of the
    variational family for SVI inference. It registers variational parameters
    for all global (per-condition) and local (per-replicate and per-condition)
    latent variables, and samples from the corresponding distributions using
    nested plates.

    Parameters
    ----------
    name : str
        Prefix for all Pyro sample and parameter sites in this guide.
    data : GrowthData
        Dataclass containing experimental data and metadata.
        Used to determine the number of conditions and replicates, and to
        provide mapping arrays for expanding parameters to observations.
    priors : ModelPriors
        Dataclass containing the prior hyperparameters for the
        model. Used to initialize the variational parameters.

    Returns
    -------
    params : LinearParams
        A dataclass containing k_pre, m_pre, k_sel, and m_sel.

    Notes
    -----
    - The guide uses nested plates: the outer plate is over experimental
      conditions, and the inner plate is over replicates within each condition.
    - All variational parameters are registered using `pyro.param` and are
      initialized from the provided priors or with default values.
    - The returned arrays are flattened and then expanded to match the
      observation indices using the mapping arrays in `data`.
    """

    # --- 1. Global Parameters (Per Condition) ---
    if data.growth_shares_replicates:
        raise ValueError("linear_independent cannot be used with growth_shares_replicates=True. Use 'linear' instead.")

    # K Hyper Loc (Normal)
    k_hl_loc = pyro.param(f"{name}_k_hyper_loc_loc", torch.as_tensor(priors.growth_k_hyper_loc_loc))
    k_hl_scale = pyro.param(f"{name}_k_hyper_loc_scale", torch.as_tensor(priors.growth_k_hyper_loc_scale),
                            constraint=torch.distributions.constraints.positive)

    # K Hyper Scale (LogNormal guide for HalfNormal prior)
    k_hs_loc = pyro.param(f"{name}_k_hyper_scale_loc",
                          torch.full((data.num_condition_rep,), -1.0))
    k_hs_scale = pyro.param(f"{name}_k_hyper_scale_scale",
                            torch.full((data.num_condition_rep,), 0.1),
                            constraint=torch.distributions.constraints.positive)

    # M Hyper Loc (Normal)
    m_hl_loc = pyro.param(f"{name}_m_hyper_loc_loc", torch.as_tensor(priors.growth_m_hyper_loc_loc))
    m_hl_scale = pyro.param(f"{name}_m_hyper_loc_scale", torch.as_tensor(priors.growth_m_hyper_loc_scale),
                            constraint=torch.distributions.constraints.positive)

    # M Hyper Scale (LogNormal guide for HalfNormal prior)
    m_hs_loc = pyro.param(f"{name}_m_hyper_scale_loc",
                          torch.full((data.num_condition_rep,), -1.0))
    m_hs_scale = pyro.param(f"{name}_m_hyper_scale_scale",
                            torch.full((data.num_condition_rep,), 0.1),
                            constraint=torch.distributions.constraints.positive)

    # --- 2. Local Parameters (Per Replicate AND Condition) ---
    # Shape: (num_replicate, num_condition_rep)
    # Note: dim 0 is replicate (-2), dim 1 is condition (-1)

    local_shape = (data.num_replicate, data.num_condition_rep)

    k_offset_locs = pyro.param(f"{name}_k_offset_locs", torch.zeros(local_shape))
    k_offset_scales = pyro.param(f"{name}_k_offset_scales", torch.ones(local_shape),
                                 constraint=torch.distributions.constraints.positive)

    m_offset_locs = pyro.param(f"{name}_m_offset_locs", torch.zeros(local_shape))
    m_offset_scales = pyro.param(f"{name}_m_offset_scales", torch.ones(local_shape),
                                 constraint=torch.distributions.constraints.positive)


    # --- 3. Sampling with Nested Plates ---

    # Outer Loop: Conditions (dim=-1)
    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep, dim=-1) as idx_c:

        # Sample Hypers (Sliced by Condition)
        growth_k_hyper_loc = pyro.sample(f"{name}_k_hyper_loc",
                                         dist.Normal(k_hl_loc[idx_c], k_hl_scale[idx_c]))

        growth_k_hyper_scale = pyro.sample(f"{name}_k_hyper_scale",
                                           dist.LogNormal(k_hs_loc[idx_c], k_hs_scale[idx_c]))

        growth_m_hyper_loc = pyro.sample(f"{name}_m_hyper_loc",
                                         dist.Normal(m_hl_loc[idx_c], m_hl_scale[idx_c]))

        growth_m_hyper_scale = pyro.sample(f"{name}_m_hyper_scale",
                                           dist.LogNormal(m_hs_loc[idx_c], m_hs_scale[idx_c]))

        # Inner Loop: Replicates (dim=-2)
        with pyro.plate(f"{name}_replicate_parameters", data.num_replicate, dim=-2) as idx_r:

            # Slice Locals:
            # We must broadcast row indices (idx_r) against col indices (idx_c)
            # idx_r[:, None] gives shape (Batch_R, 1)
            # idx_c          gives shape (Batch_C)
            # Result         gives shape (Batch_R, Batch_C) matching the plates

            k_batch_locs = k_offset_locs[idx_r[:, None], idx_c]
            k_batch_scales = k_offset_scales[idx_r[:, None], idx_c]
            k_offset = pyro.sample(f"{name}_k_offset", dist.Normal(k_batch_locs, k_batch_scales))

            m_batch_locs = m_offset_locs[idx_r[:, None], idx_c]
            m_batch_scales = m_offset_scales[idx_r[:, None], idx_c]
            m_offset = pyro.sample(f"{name}_m_offset", dist.Normal(m_batch_locs, m_batch_scales))

    # --- 4. Reconstruction ---
    # Note: Broadcasting handles the shape mismatch between Hypers (Batch_C,) and Offsets (Batch_R, Batch_C)
    growth_k_dist = growth_k_hyper_loc + k_offset * growth_k_hyper_scale
    growth_m_dist = growth_m_hyper_loc + m_offset * growth_m_hyper_scale

    # Flatten array (reshape uses C-style order: row0, row1...)
    # This matches the "rep0, cond0 \ rep0, cond1" order if cond is the last axis.
    growth_k_dist_1d = growth_k_dist.reshape(-1)
    growth_m_dist_1d = growth_m_dist.reshape(-1)

    # Expand to full-sized tensors
    k_pre = growth_k_dist_1d[data.map_condition_pre]
    m_pre = growth_m_dist_1d[data.map_condition_pre]
    k_sel = growth_k_dist_1d[data.map_condition_sel]
    m_sel = growth_m_dist_1d[data.map_condition_sel]

    return LinearParams(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)

def calculate_growth(params: LinearParams,
                     dk_geno: torch.Tensor,
                     activity: torch.Tensor,
                     theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the growth rates for pre-selection and selection phases.

    Parameters
    ----------
    params : LinearParams
        A dataclass containing k_pre, m_pre, k_sel, m_sel.
    dk_geno : torch.Tensor
        Genotype-specific death rate.
    activity : torch.Tensor
        Genotype activity.
    theta : torch.Tensor
        Occupancy/binding probability.

    Returns
    -------
    g_pre : torch.Tensor
        Pre-selection growth rate tensor.
    g_sel : torch.Tensor
        Selection growth rate tensor.
    """

    g_pre = params.k_pre + dk_geno + activity * params.m_pre * theta
    g_sel = params.k_sel + dk_geno + activity * params.m_sel * theta

    return g_pre, g_sel


def get_hyperparameters(num_condition_rep: int=1) -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Parameters
    ----------
    num_condition_rep : int
        The number of experimental conditions, used to shape the
        hyperparameter arrays.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names (as strings) to their
        default values (torch tensors).
    """

    parameters = {}
    parameters["growth_k_hyper_loc_loc"] = torch.ones(num_condition_rep) * 0.025
    parameters["growth_k_hyper_loc_scale"] = torch.ones(num_condition_rep) * 0.1
    parameters["growth_k_hyper_scale"] = torch.ones(num_condition_rep)
    parameters["growth_m_hyper_loc_loc"] = torch.zeros(num_condition_rep)
    parameters["growth_m_hyper_loc_scale"] = torch.ones(num_condition_rep) * 0.01
    parameters["growth_m_hyper_scale"] = torch.ones(num_condition_rep)

    return parameters


def get_guesses(name: str, data: GrowthData) -> Dict[str, torch.Tensor]:
    """
    Get guess values for the model's latent parameters.

    These values are used in `pyro.poutine.condition` for testing
    or initializing inference.

    Parameters
    ----------
    name : str
        The prefix used for all sample sites (e.g., "my_model").
    data : GrowthData
        A dataclass containing data metadata, used to determine the
        shape of the guess arrays. Requires:
        - ``data.num_condition_rep``
        - ``data.num_replicate``

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary mapping sample site names (e.g., "my_model_k_offset")
        to torch tensors of guess values.

    Notes
    -----
    The shapes of the guesses are critical:
    - ``_hyper_loc``/``_hyper_scale`` sites are sampled within the
      ``condition_parameters`` plate, so their shape must be
      ``(data.num_condition_rep, 1)``.
    - ``_offset`` sites are sampled within both plates, so their shape
      must be ``(data.num_condition_rep, data.num_replicate)``.
    """

    shape = (data.num_condition_rep, data.num_replicate)

    # Shape for hyper-parameters sampled inside the condition plate
    hyper_shape = (data.num_condition_rep, 1)

    guesses = {}
    guesses[f"{name}_k_hyper_loc"] = torch.ones(hyper_shape)
    guesses[f"{name}_k_hyper_scale"] = torch.ones(hyper_shape) * 0.1
    guesses[f"{name}_m_hyper_loc"] = torch.ones(hyper_shape)
    guesses[f"{name}_m_hyper_scale"] = torch.ones(hyper_shape) * 0.1

    guesses[f"{name}_k_offset"] = torch.zeros(shape)
    guesses[f"{name}_m_offset"] = torch.zeros(shape)

    return guesses

def get_priors(num_condition_rep: int=1) -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Parameters
    ----------
    num_condition_rep : int, optional
        The number of experimental conditions, which is required by
        `get_hyperparameters`. Default is 1.

    Returns
    -------
    ModelPriors
        A populated dataclass of hyperparameters.
    """
    params = get_hyperparameters(num_condition_rep)
    return ModelPriors(**params)
