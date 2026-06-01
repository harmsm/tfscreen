import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Dict, Any
from tfscreen.tfmodel.data_class import DataClass


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the categorical theta model.

    Attributes
    ----------
    logit_theta_hyper_loc_loc : float
        Mean of the prior for the hyper-location of logit(theta).
    logit_theta_hyper_loc_scale : float
        Std dev of the prior for the hyper-location of logit(theta).
    logit_theta_hyper_scale_loc : float
        Scale of the HalfNormal prior for the hyper-scale of logit(theta).
    """

    logit_theta_hyper_loc_loc: float
    logit_theta_hyper_loc_scale: float
    logit_theta_hyper_scale_loc: float


@dataclass(frozen=True)
class ThetaParam:
    """
    JAX Pytree holding the sampled categorical theta parameters.

    Attributes
    ----------
    theta : jnp.ndarray
    mu : jnp.ndarray
    sigma : jnp.ndarray
    concentrations : jnp.ndarray
        The concentrations associated with the categories in theta.
    """

    theta: jnp.ndarray
    mu: jnp.ndarray
    sigma: jnp.ndarray
    concentrations: jnp.ndarray


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
        The prefix for all Numpyro sample sites (e.g., "theta").
    data : DataClass
        A data object (e.g., `GrowthData`) containing metadata, primarily:
        - ``data.num_titrant_name`` : (int) Number of titrants.
        - ``data.num_titrant_conc`` : (int) Number of titrant concentrations.
        - ``data.num_genotype`` : (int) Number of genotypes.
    priors : ModelPriors
        A Pytree (Flax dataclass) containing all hyperparameters for the
        pooled logit-space prior.

    Returns
    -------
    ThetaParam
        A Pytree containing the sampled theta parameters in their
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
                dist.HalfNormal(priors.logit_theta_hyper_scale_loc)
            )

    # --------------------------------------------------------------------------
    # Sample parameters for each (titrant_name, titrant_conc, genotype) group

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
            with pyro.plate("theta_genotype_plate", size=data.batch_size, dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):

                    logit_theta_offset = pyro.sample(
                        f"{name}_logit_theta_offset",
                        dist.Normal(0.0, 1.0)
                    )

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if logit_theta_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        logit_theta_offset = logit_theta_offset[..., data.batch_idx]

    # Calculate parameters in logit-space
    # logit_theta_hyper_loc result from plating is (Name, Conc) if rank-minimized,
    # or (Name, Conc, 1) if plated at -3, -2 with some other dim at -1.
    # We ensure it's (Name, Conc, 1) for the broadcast.
    lh_loc = jnp.reshape(logit_theta_hyper_loc, (data.num_titrant_name, data.num_titrant_conc, 1))
    lh_scale = jnp.reshape(logit_theta_hyper_scale, (data.num_titrant_name, data.num_titrant_conc, 1))
    
    logit_theta = lh_loc + logit_theta_offset * lh_scale

    # --------------------------------------------------------------------------
    # Transform parameters to natural scale

    theta = dist.transforms.SigmoidTransform()(logit_theta)

    # Register parameter values
    pyro.deterministic(f"{name}_theta", theta)

    theta_param = ThetaParam(theta=theta,
                             mu=lh_loc,
                             sigma=lh_scale,
                             concentrations=data.titrant_conc)

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
                           jnp.full(local_shape_global, priors.logit_theta_hyper_loc_loc))
    h_loc_scale = pyro.param(f"{name}_logit_theta_hyper_loc_scale",
                             jnp.full(local_shape_global, priors.logit_theta_hyper_loc_scale),
                             constraint=dist.constraints.greater_than(1e-4))

    # Logit Theta Hyper Scale (LogNormal guide)
    h_scale_loc = pyro.param(f"{name}_logit_theta_hyper_scale_loc", jnp.full(local_shape_global, -1.0))
    h_scale_scale = pyro.param(f"{name}_logit_theta_hyper_scale_scale", jnp.full(local_shape_global, 0.1),
                               constraint=dist.constraints.greater_than(1e-4))

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
                             jnp.zeros(param_shape, dtype=float))
    offset_scales = pyro.param(f"{name}_logit_theta_offset_scales",
                               jnp.ones(param_shape, dtype=float),
                               constraint=dist.constraints.positive)

    # --- 3. Sampling (Sliced by Genotype) ---

    with pyro.plate(f"{name}_titrant_name_plate", data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc_plate", data.num_titrant_conc, dim=-2):
            # Batching on Genotype (dim=-1)
            with pyro.plate("theta_genotype_plate", size=data.batch_size, dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):

                    # Slice the last dimension (Genotype) using the batch indices
                    # The ellipsis (...) preserves the TitrantName and TitrantConc dimensions
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
    lh_loc = jnp.reshape(logit_theta_hyper_loc, (data.num_titrant_name, data.num_titrant_conc, 1))
    lh_scale = jnp.reshape(logit_theta_hyper_scale, (data.num_titrant_name, data.num_titrant_conc, 1))
    
    logit_theta = lh_loc + logit_theta_offset * lh_scale

    # Transform parameters to natural scale
    theta = dist.transforms.SigmoidTransform()(logit_theta)

    theta_param = ThetaParam(theta=theta,
                             mu=lh_loc,
                             sigma=lh_scale,
                             concentrations=data.titrant_conc)

    return theta_param


def run_model(theta_param: ThetaParam, data: DataClass) -> jnp.ndarray:
    """
    "Calculates" fractional occupancy (theta) by looking it up.

    This is a pure JAX function that returns the sampled categorical
    theta values. Unlike the Hill model, no calculation is needed.
    It simply passes the sampled tensor and handles the final optional
    scattering.

    Parameters
    ----------
    theta_param : ThetaParam
        A Pytree generated by ``define_model`` containing the sampled
        theta parameters. ``theta_param.theta`` has dimensions
        ``[titrant_name, titrant_conc, genotype]``.
    data : DataClass
        A data object (e.g., ``GrowthData`` or ``BindingData``) containing:
        - ``data.map_theta``: (jnp.ndarray) Mapper with dimensions
          ``[replicate, time, treatment, genotype]``, used for scattering.
        - ``data.scatter_theta``: (int) A flag (0 or 1) indicating
          whether to scatter the final tensor.
        - ``data.geno_theta_idx``: (jnp.ndarray) Indices of genotypes to select.
        - ``data.titrant_conc``: (jnp.ndarray) Titrant concentrations in the data.

    Returns
    -------
    jnp.ndarray
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
    # We find the indices of data.titrant_conc in theta_param.concentrations.
    # This assumes that the concentrations in theta_param.concentrations are ordered.
    # In practice, they are the labels of the growth_tm's conc dimension.
    conc_idx = jnp.searchsorted(theta_param.concentrations, data.titrant_conc)
    
    # Clip to avoid out-of-bounds (fallback to nearest category)
    conc_idx = jnp.clip(conc_idx, 0, theta_param.concentrations.shape[0] - 1)
    
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
    parameters["logit_theta_hyper_scale_loc"] = 1.0

    return parameters


def get_guesses(name: str, data: DataClass) -> Dict[str, Any]:
    """
    Gets initial guess values for model parameters.

    Parameters
    ----------
    name : str
        The prefix for the parameter names (e.g., "theta").
    data : DataClass
        A data object (e.g., `GrowthData`) containing metadata, primarily:
        - ``data.num_titrant_name``
        - ``data.num_titrant_conc``
        - ``data.num_genotype``

    Returns
    -------
    dict[str, Any]
        A dictionary mapping parameter names to their initial
        guess values.
    """

    guesses = {}

    # Guess hyperparams
    guesses[f"{name}_logit_theta_hyper_loc"] = jnp.zeros((data.num_titrant_name, data.num_titrant_conc))
    guesses[f"{name}_logit_theta_hyper_scale"] = jnp.ones((data.num_titrant_name, data.num_titrant_conc))

    # Guess offsets (all zeros)
    shape = (data.num_titrant_name, data.num_titrant_conc, data.num_genotype)
    guesses[f"{name}_logit_theta_offset"] = jnp.zeros(shape, dtype=float)

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


def get_extract_specs(ctx):
    return [dict(
        input_df=ctx.growth_tm.df,
        params_to_get=["theta"],
        map_column="map_theta",
        get_columns=["genotype", "titrant_name", "titrant_conc"],
        in_run_prefix="theta_",
    )]


def build_calc_df(model, manual_titrant_df):
    """
    Build the lookup DataFrame for categorical theta extraction.

    The categorical model stores one theta value per
    (titrant_name, titrant_conc, genotype) triple seen during training.
    Interpolation to new concentrations or genotypes is not possible.

    Returns
    -------
    calc_df : pd.DataFrame
        Rows for each (genotype, titrant_name, titrant_conc) combination,
        including internal index columns used by compute_theta_samples.
    internal_cols : list of str
        Columns to strip before returning results to the caller.
    extra_kwargs : dict
        Empty; no extra arguments are needed by compute_theta_samples.

    Raises
    ------
    ValueError
        If manual_titrant_df requests (titrant_name, titrant_conc) pairs not
        present in the training data.
    """
    training_df = (model.training_tm.df[["genotype", "titrant_name", "titrant_conc",
                                        "titrant_name_idx", "titrant_conc_idx",
                                        "genotype_idx"]]
                   .drop_duplicates()
                   .reset_index(drop=True))

    if manual_titrant_df is None:
        calc_df = training_df.copy()
    else:
        required = {"titrant_name", "titrant_conc"}
        missing = required - set(manual_titrant_df.columns)
        if missing:
            raise ValueError(f"manual_titrant_df is missing columns: {missing}")

        # Restrict to concentrations that exist in the training data.
        training_titrant_pairs = (training_df[["titrant_name", "titrant_conc"]]
                                  .drop_duplicates())
        merged = manual_titrant_df.merge(training_titrant_pairs,
                                         on=["titrant_name", "titrant_conc"],
                                         how="left",
                                         indicator=True)
        bad = merged[merged["_merge"] == "left_only"][["titrant_name", "titrant_conc"]]
        if len(bad):
            raise ValueError(
                "The categorical theta model cannot predict at concentrations "
                "not seen during training. The following requested "
                "(titrant_name, titrant_conc) pairs were not in the training "
                f"data:\n{bad.drop_duplicates().to_string(index=False)}"
            )
        requested_titrant_df = manual_titrant_df[["titrant_name", "titrant_conc"]].drop_duplicates()

        if "genotype" in manual_titrant_df.columns:
            genotypes = manual_titrant_df["genotype"].unique()
        else:
            genotypes = training_df["genotype"].unique()

        # Cross genotypes with the requested titrant pairs, then join indices.
        geno_df = pd.DataFrame({"genotype": genotypes})
        cross = geno_df.merge(requested_titrant_df, how="cross")
        calc_df = cross.merge(
            training_df[["genotype", "titrant_name", "titrant_conc",
                          "titrant_name_idx", "titrant_conc_idx", "genotype_idx"]],
            on=["genotype", "titrant_name", "titrant_conc"],
            how="left",
        ).reset_index(drop=True)

    internal_cols = ["titrant_name_idx", "titrant_conc_idx", "genotype_idx"]
    return calc_df, internal_cols, {}


def compute_theta_samples(calc_df, param_posteriors):
    """
    Reconstruct posterior theta samples for the categorical model.

    Does NOT use the ``theta_theta`` deterministic site, which is computed
    inside a forward batch and only covers ``batch_size`` genotypes in the
    HDF5 file (not the full genotype set).  Instead, reconstructs theta from
    its three constituent posterior arrays, all of which are correctly stored
    for every genotype:

    * ``theta_logit_theta_hyper_loc``  – global hyperparameter, shape
      ``(S, N_name, N_conc[, 1])``.
    * ``theta_logit_theta_hyper_scale`` – global hyperparameter, same shape.
    * ``theta_logit_theta_offset``     – per-genotype non-centred offset,
      shape ``(S, N_name, N_conc, N_geno)``.

    theta = sigmoid(hyper_loc + offset * hyper_scale)

    Parameters
    ----------
    calc_df : pd.DataFrame
        Output of build_calc_df; must contain titrant_name_idx,
        titrant_conc_idx, and genotype_idx columns.
    param_posteriors : dict-like
        Posterior samples keyed by parameter name (with ``theta_`` prefix).

    Returns
    -------
    theta_samples : np.ndarray, shape (S, N)
        Posterior theta at each row of calc_df.
    """
    import warnings
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples

    def _load_global(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = np.asarray(v)
        v = np.asarray(v)
        # Drop trailing size-1 dim added by the (N_name, N_conc, 1) plate shape
        if v.ndim >= 4 and v.shape[-1] == 1:
            v = v[..., 0]
        return v  # (S, N_name, N_conc)

    hyper_loc   = _load_global("theta_logit_theta_hyper_loc")    # (S, N_name, N_conc)
    hyper_scale = _load_global("theta_logit_theta_hyper_scale")  # (S, N_name, N_conc)

    # theta_logit_theta_offset is a sample site inside the genotype plate, so
    # get_posteriors correctly concatenates it across forward batches to shape
    # (S, N_name, N_conc, N_geno).  It may be an h5py Dataset (lazy).
    offset_raw = get_posterior_samples(param_posteriors, "theta_logit_theta_offset")

    # MAP checkpoints trained with batch_size < N_geno store only batch_size
    # offset parameters.  On GPU, jnp.take with out-of-bounds indices returns
    # NaN rather than clipping, so the HDF5 contains NaN for genotypes beyond
    # the first batch_size.  Detect this and fall back to offset=0 (i.e.
    # theta = sigmoid(hyper_loc), the population mean) for those genotypes.
    stored_n_geno = offset_raw.shape[-1]
    required_n_geno = int(calc_df["genotype_idx"].max()) + 1
    partial_offset = stored_n_geno < required_n_geno
    if partial_offset:
        warnings.warn(
            f"theta_logit_theta_offset has {stored_n_geno} genotype entries "
            f"but {required_n_geno} are needed.  This happens when MAP "
            f"training uses batch_size ({stored_n_geno}) < N_geno "
            f"({required_n_geno}): AutoDelta only stores parameters for the "
            f"initial batch.  Genotypes beyond the first {stored_n_geno} will "
            f"be assigned theta = sigmoid(hyper_loc) (the population mean). "
            f"For per-genotype theta estimates, use SVI and tfs-sample-posterior.",
            UserWarning,
            stacklevel=2,
        )

    S = hyper_loc.shape[0]
    N = len(calc_df)
    result = np.empty((S, N))

    # Process one (titrant_name_idx, titrant_conc_idx) group at a time so that
    # the HDF5 read for `offset_raw` is one contiguous (S, N_geno) slice per
    # group rather than loading the full tensor at once.
    for (ni, ci), group in calc_df.groupby(["titrant_name_idx", "titrant_conc_idx"]):
        ni, ci = int(ni), int(ci)
        gi      = group["genotype_idx"].values.astype(int)
        row_pos = group.index.values

        if partial_offset:
            in_range  = gi < stored_n_geno
            gi_safe   = np.clip(gi, 0, stored_n_geno - 1)
        else:
            in_range  = np.ones(len(gi), dtype=bool)
            gi_safe   = gi

        if isinstance(offset_raw, np.ndarray):
            off_sel = offset_raw[:, ni, ci, gi_safe]            # (S, n_group)
        else:
            # HDF5: load the (S, stored_n_geno) slice for this (ni, ci), then index
            off_sel = np.asarray(offset_raw[:, ni, ci, :])[:, gi_safe]  # (S, n_group)

        # Zero out offsets for genotypes beyond the stored range so they get
        # theta = sigmoid(hyper_loc) rather than a clipped/wrong value.
        if partial_offset and not in_range.all():
            off_sel = off_sel.copy()
            off_sel[:, ~in_range] = 0.0

        lh_loc   = hyper_loc[:, ni, ci, np.newaxis]    # (S, 1)
        lh_scale = hyper_scale[:, ni, ci, np.newaxis]  # (S, 1)

        logit_theta = lh_loc + off_sel * lh_scale       # (S, n_group)
        result[:, row_pos] = 1.0 / (1.0 + np.exp(-logit_theta))

    return result