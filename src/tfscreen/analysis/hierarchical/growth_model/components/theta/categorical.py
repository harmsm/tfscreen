import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass
from typing import Dict, Any
from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass


@dataclass
class ModelPriors:
    """
    Holds hyperparameters for the categorical theta model.

    Attributes
    ----------
    logit_theta_hyper_loc_loc : float
        Mean of the prior for the hyper-location of logit(theta).
    logit_theta_hyper_loc_scale : float
        Std dev of the prior for the hyper-location of logit(theta).
    logit_theta_hyper_scale : float
        Scale of the HalfNormal prior for the hyper-scale of logit(theta).
    """

    logit_theta_hyper_loc_loc: float
    logit_theta_hyper_loc_scale: float
    logit_theta_hyper_scale: float


@dataclass
class ThetaParam:
    """
    Holds the sampled categorical theta parameters.

    Attributes
    ----------
    theta : torch.Tensor
    mu : torch.Tensor
    sigma : torch.Tensor
    concentrations : torch.Tensor
        The concentrations associated with the categories in theta.
    """

    theta: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    concentrations: torch.Tensor


def define_model(name: str,
                 data: DataClass,
                 priors: ModelPriors) -> ThetaParam:
    """
    Defines the hierarchical categorical model for theta.

    This function defines a unique, sampled ``theta`` parameter for every
    ``(titrant_name, titrant_conc, genotype)`` combination.

    The parameters are sampled from a pooled Normal distribution in
    logit-space using a non-centered parameterization.

    Parameters
    ----------
    name : str
        The prefix for all Pyro sample sites (e.g., "theta").
    data : DataClass
        A data object containing metadata, primarily:
        - ``data.num_titrant_name`` : (int) Number of titrants.
        - ``data.num_titrant_conc`` : (int) Number of titrant concentrations.
        - ``data.num_genotype`` : (int) Number of genotypes.
    priors : ModelPriors
        A dataclass containing all hyperparameters for the pooled logit-space prior.

    Returns
    -------
    ThetaParam
        A dataclass containing the sampled theta parameters in their
        natural scale, with shape
        ``[num_titrant_name, num_titrant_conc, num_genotype]``.
    """

    # --------------------------------------------------------------------------
    # Hyperpriors for the logit(theta) parameters (titrant name x conc)
    # We use a 3D shape (Name, Conc, 1) to ensure correct broadcasting
    # and plating at -3, -2, -1.

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):

            logit_theta_hyper_loc = pyro.sample(
                f"{name}_logit_theta_hyper_loc",
                dist.Normal(priors.logit_theta_hyper_loc_loc,
                            priors.logit_theta_hyper_loc_scale)
            )
            logit_theta_hyper_scale = pyro.sample(
                f"{name}_logit_theta_hyper_scale",
                dist.HalfNormal(priors.logit_theta_hyper_scale)
            )

    # --------------------------------------------------------------------------
    # Sample parameters for each (titrant_name, titrant_conc, genotype) group

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
            with pyro.plate("theta_genotype_plate", size=data.batch_size, dim=-1):
                with pyro.poutine.scale(scale=data.scale_vector):

                    logit_theta_offset = pyro.sample(
                        f"{name}_logit_theta_offset",
                        dist.Normal(0.0, 1.0)
                    )

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if logit_theta_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        logit_theta_offset = logit_theta_offset[..., data.batch_idx]

    # Calculate parameters in logit-space
    # Reshape to (Name, Conc, 1) for broadcast with offset (Name, Conc, batch_size)
    lh_loc = logit_theta_hyper_loc.reshape(data.num_titrant_name, data.num_titrant_conc, 1)
    lh_scale = logit_theta_hyper_scale.reshape(data.num_titrant_name, data.num_titrant_conc, 1)

    logit_theta = lh_loc + logit_theta_offset * lh_scale

    # --------------------------------------------------------------------------
    # Transform parameters to natural scale

    theta = torch.sigmoid(logit_theta)

    # Register parameter values
    pyro.deterministic(f"{name}_theta", theta)

    theta_param = ThetaParam(theta=theta,
                             mu=lh_loc,
                             sigma=lh_scale,
                             concentrations=torch.as_tensor(data.titrant_conc).float())

    return theta_param


def guide(name: str,
          data: DataClass,
          priors: ModelPriors) -> ThetaParam:
    """
    Guide corresponding to the categorical theta model.

    This guide defines the variational family for the categorical theta
    parameters. It uses:
    - Normal/LogNormal distributions for global location/scale hyperparameters.
    - An amortized parameterization for the local 3D tensor of parameters
      (titrant x conc x genotype).
    """

    # --- 1. Global Hypers (now plated) ---

    # We use (Name, Conc, 1) to match the expected plate structure exactly.
    local_shape_global = (data.num_titrant_name, data.num_titrant_conc, 1)

    # Logit Theta Hyper Loc (Normal)
    h_loc_loc = pyro.param(f"{name}_logit_theta_hyper_loc_loc",
                           torch.full(local_shape_global, float(priors.logit_theta_hyper_loc_loc)))
    h_loc_scale = pyro.param(f"{name}_logit_theta_hyper_loc_scale",
                             torch.full(local_shape_global, float(priors.logit_theta_hyper_loc_scale)),
                             constraint=torch.distributions.constraints.positive)

    # Logit Theta Hyper Scale (LogNormal guide)
    h_scale_loc = pyro.param(f"{name}_logit_theta_hyper_scale_loc", torch.full(local_shape_global, -1.0))
    h_scale_scale = pyro.param(f"{name}_logit_theta_hyper_scale_scale", torch.full(local_shape_global, 0.1),
                               constraint=torch.distributions.constraints.positive)

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
            logit_theta_hyper_loc = pyro.sample(
                f"{name}_logit_theta_hyper_loc",
                dist.Normal(h_loc_loc, h_loc_scale)
            )
            logit_theta_hyper_scale = pyro.sample(
                f"{name}_logit_theta_hyper_scale",
                dist.LogNormal(h_scale_loc, h_scale_scale)
            )

    # --- 2. Local Parameters (3D Tensor) ---

    # Shape: (NumTitrantName, NumTitrantConc, NumGenotype)
    param_shape = (data.num_titrant_name, data.num_titrant_conc, data.num_genotype)

    offset_locs = pyro.param(f"{name}_logit_theta_offset_locs",
                             torch.zeros(param_shape))
    offset_scales = pyro.param(f"{name}_logit_theta_offset_scales",
                               torch.ones(param_shape),
                               constraint=torch.distributions.constraints.positive)

    # --- 3. Sampling (Sliced by Genotype) ---

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
            # Batching on Genotype (dim=-1)
            with pyro.plate("theta_genotype_plate", size=data.batch_size, dim=-1):
                with pyro.poutine.scale(scale=data.scale_vector):

                    # Slice the last dimension (Genotype) using the batch indices
                    batch_locs = offset_locs[..., data.batch_idx]
                    batch_scales = offset_scales[..., data.batch_idx]

                    logit_theta_offset = pyro.sample(
                        f"{name}_logit_theta_offset",
                        dist.Normal(batch_locs, batch_scales)
                    )

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if logit_theta_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        logit_theta_offset = logit_theta_offset[..., data.batch_idx]

    # --- 4. Reconstruction (Deterministic) ---

    # Calculate parameters in logit-space
    lh_loc = logit_theta_hyper_loc.reshape(data.num_titrant_name, data.num_titrant_conc, 1)
    lh_scale = logit_theta_hyper_scale.reshape(data.num_titrant_name, data.num_titrant_conc, 1)

    logit_theta = lh_loc + logit_theta_offset * lh_scale

    # Transform parameters to natural scale
    theta = torch.sigmoid(logit_theta)

    theta_param = ThetaParam(theta=theta,
                             mu=lh_loc,
                             sigma=lh_scale,
                             concentrations=torch.as_tensor(data.titrant_conc).float())

    return theta_param


def run_model(theta_param: ThetaParam, data: DataClass) -> torch.Tensor:
    """
    "Calculates" fractional occupancy (theta) by looking it up.

    This is a pure PyTorch function that returns the sampled categorical
    theta values. Unlike the Hill model, no calculation is needed.
    It simply passes the sampled tensor and handles the final optional
    scattering.

    Parameters
    ----------
    theta_param : ThetaParam
        A dataclass generated by ``define_model`` containing the sampled
        theta parameters. ``theta_param.theta`` has dimensions
        ``[titrant_name, titrant_conc, genotype]``.
    data : DataClass
        A data object containing:
        - ``data.scatter_theta``: (int) A flag (0 or 1) indicating
          whether to scatter the final tensor.
        - ``data.geno_theta_idx``: (torch.Tensor) Indices of genotypes to select.
        - ``data.titrant_conc``: (torch.Tensor) Titrant concentrations in the data.

    Returns
    -------
    torch.Tensor
        A tensor of theta values.
        - If ``data.scatter_theta == 0``, shape is
          ``[titrant_name, titrant_conc, genotype]``.
        - If ``data.scatter_theta == 1``, shape is
          ``[replicate, time, treatment, genotype]``.
    """

    # 1. Select the correct genotypes for this dataset
    # theta_param.theta: (Name, Conc, Genotypes)
    theta_base = theta_param.theta[..., data.geno_theta_idx]

    # 2. Map concentrations
    titrant_conc = torch.as_tensor(data.titrant_conc).float()
    conc_idx = torch.searchsorted(theta_param.concentrations.contiguous(),
                                  titrant_conc.contiguous())

    # Clip to avoid out-of-bounds (fallback to nearest category)
    conc_idx = torch.clamp(conc_idx, 0, theta_param.concentrations.shape[0] - 1)

    # Select the mapped concentrations
    # theta_base is (Name, OrigConc, Geno)
    # conc_idx is (NewConc,)
    # theta_calc should be (Name, NewConc, Geno)
    theta_calc = theta_base[:, conc_idx, :]

    # Broadcast to the full-sized tensor if required (for growth experiment)
    if data.scatter_theta == 1:
        theta_calc = theta_calc[None, None, None, None, :, :, :]

    return theta_calc


def get_population_moments(theta_param: ThetaParam, data: DataClass) -> tuple:
    """
    Returns the expected population moments (mu, sigma) in logit-space.

    For the categorical model, these are simply the hyper-parameters.

    Returns
    -------
    tuple
        (mu, sigma) with shape broadcastable to (titrant_name, titrant_conc, 1).
    """
    return theta_param.mu, theta_param.sigma


def get_hyperparameters() -> Dict[str, Any]:
    """
    Gets default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        A dictionary of hyperparameter names and their default values.
    """

    parameters = {}

    # Center logit(theta) prior on 0.0 (i.e., theta = 0.5)
    parameters["logit_theta_hyper_loc_loc"] = 0.0
    parameters["logit_theta_hyper_loc_scale"] = 1.5
    parameters["logit_theta_hyper_scale"] = 1.0

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
        - ``data.num_titrant_name``
        - ``data.num_titrant_conc``
        - ``data.num_genotype``

    Returns
    -------
    dict[str, Any]
        A dictionary mapping parameter names to their initial guess values.
    """

    guesses = {}

    # Guess hyperparams
    guesses[f"{name}_logit_theta_hyper_loc"] = torch.zeros(data.num_titrant_name, data.num_titrant_conc)
    guesses[f"{name}_logit_theta_hyper_scale"] = torch.ones(data.num_titrant_name, data.num_titrant_conc)

    # Guess offsets (all zeros)
    shape = (data.num_titrant_name, data.num_titrant_conc, data.num_genotype)
    guesses[f"{name}_logit_theta_offset"] = torch.zeros(shape)

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
