import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

from typing import Dict, Any
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the ln_cfu0 model.

    Attributes
    ----------
    ln_cfu0_hyper_loc_loc : float
        Mean of the prior for the hyper-location of ln_cfu0 (library genotypes).
    ln_cfu0_hyper_loc_scale : float
        Standard deviation of the prior for the hyper-location of ln_cfu0.
    ln_cfu0_hyper_scale_loc : float
        Scale of the HalfNormal prior for the hyper-scale of ln_cfu0.
    ln_cfu0_spiked_loc_loc : float
        Mean of the Normal prior for the ln_cfu0 location of spiked genotypes.
    ln_cfu0_spiked_loc_scale : float
        Standard deviation of the Normal prior for the spiked genotype location.
    """

    ln_cfu0_hyper_loc_loc: float
    ln_cfu0_hyper_loc_scale: float
    ln_cfu0_hyper_scale_loc: float
    ln_cfu0_spiked_loc_loc: float
    ln_cfu0_spiked_loc_scale: float
    ln_cfu0_wt_loc_loc: float
    ln_cfu0_wt_loc_scale: float

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Defines the hierarchical model for initial cell counts (ln_cfu0).

    Library genotypes share a pooled Normal distribution whose location and
    scale are learned hyper-parameters.  Spiked genotypes (flagged via
    ``data.ln_cfu0_spiked_mask``) share a separate scalar location parameter,
    allowing them to have a very different prior mean (e.g. ln_cfu0 ~ 10)
    without distorting the library hierarchy.  All genotypes share the same
    hyper-scale and per-genotype non-centered offsets.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample/deterministic sites.
    data : GrowthData
        Pytree containing experimental data and metadata. Uses:
        - ``data.num_replicate``, ``data.num_condition_pre``, ``data.batch_size``
        - ``data.batch_idx`` : index array for the current mini-batch
        - ``data.scale_vector`` : importance-weight vector for subsampling
        - ``data.ln_cfu0_spiked_mask`` : bool array (num_genotype,), True = spiked
        - ``data.ln_cfu0_wt_mask`` : bool array (num_genotype,), True = wildtype
    priors : ModelPriors
        Pytree containing the hyperparameters.

    Returns
    -------
    jnp.ndarray
        The sampled ``ln_cfu0`` values expanded to observation shape
        ``(num_replicate, 1, num_condition_pre, 1, 1, 1, batch_size)``.
    """

    # Hyper-priors for the library-genotype pooled distribution
    ln_cfu0_hyper_loc = pyro.sample(
        f"{name}_hyper_loc",
        dist.Normal(priors.ln_cfu0_hyper_loc_loc,
                    priors.ln_cfu0_hyper_loc_scale)
    )
    ln_cfu0_hyper_scale = pyro.sample(
        f"{name}_hyper_scale",
        dist.HalfNormal(priors.ln_cfu0_hyper_scale_loc)
    )

    # Separate scalar location for spiked genotypes
    ln_cfu0_spiked_loc = pyro.sample(
        f"{name}_spiked_loc",
        dist.Normal(priors.ln_cfu0_spiked_loc_loc,
                    priors.ln_cfu0_spiked_loc_scale)
    )

    # Separate scalar location for wildtype genotype
    ln_cfu0_wt_loc = pyro.sample(
        f"{name}_wt_loc",
        dist.Normal(priors.ln_cfu0_wt_loc_loc,
                    priors.ln_cfu0_wt_loc_scale)
    )

    # Sample non-centered offsets for each ln_cfu0 group
    with pyro.plate(f"{name}_replicate",data.num_replicate,dim=-3):
        with pyro.plate(f"{name}_condition_pre",data.num_condition_pre,dim=-2):
            with pyro.plate("shared_genotype_plate", size=data.batch_size,dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):
                    ln_cfu0_offsets = pyro.sample(f"{name}_offset", dist.Normal(0.0, 1.0))

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if ln_cfu0_offsets.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        ln_cfu0_offsets = ln_cfu0_offsets[..., data.batch_idx]

    # Per-genotype location: wt uses wt_loc, spiked uses spiked_loc, library uses hyper_loc
    batch_spiked_mask = data.ln_cfu0_spiked_mask[data.batch_idx]
    batch_wt_mask = data.ln_cfu0_wt_mask[data.batch_idx]
    per_geno_loc = jnp.where(batch_wt_mask, ln_cfu0_wt_loc,
                   jnp.where(batch_spiked_mask, ln_cfu0_spiked_loc, ln_cfu0_hyper_loc))

    # Calculate the per-group ln_cfu0 values
    ln_cfu0_per_rep_cond_geno = per_geno_loc + ln_cfu0_offsets * ln_cfu0_hyper_scale

    # Register deterministic values for inspection
    pyro.deterministic(name, ln_cfu0_per_rep_cond_geno)

    # Expand tensor to match all observations
    ln_cfu0 = ln_cfu0_per_rep_cond_geno[:,None,:,None,None,None,:]

    return ln_cfu0

def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide corresponding to the hierarchical ln_cfu0 model.

    Uses Normal variational distributions for the hyper-location and spiked
    location, LogNormal for the hyper-scale, and Normal for per-genotype offsets.
    """

    # -------------------------------------------------------------------------
    # Global parameters

    # Hyper Loc — library genotypes (Normal posterior)
    h_loc_loc = pyro.param(f"{name}_hyper_loc_loc", jnp.array(priors.ln_cfu0_hyper_loc_loc))
    h_loc_scale = pyro.param(f"{name}_hyper_loc_scale", jnp.array(priors.ln_cfu0_hyper_loc_scale), constraint=dist.constraints.greater_than(1e-4))
    ln_cfu0_hyper_loc = pyro.sample(f"{name}_hyper_loc", dist.Normal(h_loc_loc, h_loc_scale))

    # Hyper Scale (LogNormal posterior approximation for positive variable)
    h_scale_loc = pyro.param(f"{name}_hyper_scale_loc", jnp.array(-1.0)) # Init small
    h_scale_scale = pyro.param(f"{name}_hyper_scale_scale", jnp.array(0.1), constraint=dist.constraints.greater_than(1e-4))
    ln_cfu0_hyper_scale = pyro.sample(f"{name}_hyper_scale", dist.LogNormal(h_scale_loc, h_scale_scale))

    # Spiked Loc — spiked genotypes (Normal posterior)
    s_loc_loc = pyro.param(f"{name}_spiked_loc_loc", jnp.array(priors.ln_cfu0_spiked_loc_loc))
    s_loc_scale = pyro.param(f"{name}_spiked_loc_scale", jnp.array(priors.ln_cfu0_spiked_loc_scale), constraint=dist.constraints.greater_than(1e-4))
    ln_cfu0_spiked_loc = pyro.sample(f"{name}_spiked_loc", dist.Normal(s_loc_loc, s_loc_scale))

    # WT Loc — wildtype genotype (Normal posterior)
    w_loc_loc = pyro.param(f"{name}_wt_loc_loc", jnp.array(priors.ln_cfu0_wt_loc_loc))
    w_loc_scale = pyro.param(f"{name}_wt_loc_scale", jnp.array(priors.ln_cfu0_wt_loc_scale), constraint=dist.constraints.greater_than(1e-4))
    ln_cfu0_wt_loc = pyro.sample(f"{name}_wt_loc", dist.Normal(w_loc_loc, w_loc_scale))

    # -------------------------------------------------------------------------
    # Genotype-specific parameter

    param_shape = (data.num_replicate, data.num_condition_pre, data.num_genotype)
    offset_locs = pyro.param(f"{name}_offset_locs",
                             jnp.zeros(param_shape,dtype=float))
    offset_scales = pyro.param(f"{name}_offset_scales",
                               jnp.ones(param_shape,dtype=float),
                               constraint=dist.constraints.positive)

    # Sample non-centered offsets for each ln_cfu0 group
    with pyro.plate(f"{name}_replicate",data.num_replicate,dim=-3):
        with pyro.plate(f"{name}_condition_pre",data.num_condition_pre,dim=-2):
            with pyro.plate("shared_genotype_plate", size=data.batch_size,dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):

                    batch_locs = offset_locs[...,data.batch_idx]
                    batch_scales = offset_scales[...,data.batch_idx]
                    ln_cfu0_offsets = pyro.sample(f"{name}_offset", dist.Normal(batch_locs,batch_scales))

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if ln_cfu0_offsets.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        ln_cfu0_offsets = ln_cfu0_offsets[..., data.batch_idx]

    # Per-genotype location: wt uses wt_loc, spiked uses spiked_loc, library uses hyper_loc
    batch_spiked_mask = data.ln_cfu0_spiked_mask[data.batch_idx]
    batch_wt_mask = data.ln_cfu0_wt_mask[data.batch_idx]
    per_geno_loc = jnp.where(batch_wt_mask, ln_cfu0_wt_loc,
                   jnp.where(batch_spiked_mask, ln_cfu0_spiked_loc, ln_cfu0_hyper_loc))

    # Calculate the per-group ln_cfu0 values
    ln_cfu0_per_rep_cond_geno = per_geno_loc + ln_cfu0_offsets * ln_cfu0_hyper_scale

    # Expand tensor to match all observations
    ln_cfu0 = ln_cfu0_per_rep_cond_geno[:,None,:,None,None,None,:]

    return ln_cfu0

def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names to their default values.
    """

    parameters = {}

    parameters["ln_cfu0_hyper_loc_loc"] = -2.5
    parameters["ln_cfu0_hyper_loc_scale"] = 3.0
    parameters["ln_cfu0_hyper_scale_loc"] = 2.0
    parameters["ln_cfu0_spiked_loc_loc"] = 12.0
    parameters["ln_cfu0_spiked_loc_scale"] = 3.0
    parameters["ln_cfu0_wt_loc_loc"] = 13.0
    parameters["ln_cfu0_wt_loc_scale"] = 3.0

    return parameters


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get initial guess values derived empirically from observed ln_cfu data.

    For each (replicate, condition_pre, genotype), the median ln_cfu across
    all valid observations (masked by ``data.good_mask``) is used as an
    estimate of the starting cell density.  Group-level medians over spiked,
    wildtype, and library genotypes provide estimates for ``spiked_loc``,
    ``wt_loc``, and ``hyper_loc``, respectively.  Per-genotype offsets are
    derived by centering on the group-level estimate and dividing by a
    default ``hyper_scale``.

    Falls back to hard-coded defaults for any group that has no valid
    observations (e.g. no spiked genotypes in the library).

    Parameters
    ----------
    name : str
        The prefix used for all sample sites.
    data : GrowthData
        Pytree containing data and metadata.  Uses ``data.ln_cfu``,
        ``data.good_mask``, ``data.ln_cfu0_spiked_mask``,
        ``data.ln_cfu0_wt_mask``, ``data.num_replicate``,
        ``data.num_condition_pre``, and ``data.num_genotype``.

    Returns
    -------
    dict[str, jnp.ndarray]
        A dictionary mapping sample site names to initial guess values.

    Notes
    -----
    ``data.ln_cfu`` is expected to have shape
    ``(num_replicate, num_time, num_condition_pre, num_condition_sel,
    num_titrant_name, num_titrant_conc, num_genotype)``.
    The median is taken over the time, condition_sel, titrant_name, and
    titrant_conc axes (axes 1, 3, 4, 5), yielding a
    ``(num_replicate, num_condition_pre, num_genotype)`` array of
    per-group empirical estimates.
    """

    _DEFAULT_HYPER_SCALE = 3.0
    _FALLBACK_HYPER_LOC  = -2.5
    _FALLBACK_SPIKED_LOC = 12.0
    _FALLBACK_WT_LOC     = 13.0

    ln_cfu      = np.array(data.ln_cfu)    # (rep, time, cond_pre, cond_sel, tname, tconc, geno)
    good_mask   = np.array(data.good_mask)
    spiked_mask = np.array(data.ln_cfu0_spiked_mask)  # (num_genotype,)
    wt_mask     = np.array(data.ln_cfu0_wt_mask)      # (num_genotype,)

    # Guard: if data are not the expected 7-D tensor (e.g. mocked in tests),
    # fall back to hard-coded defaults rather than crashing.
    if ln_cfu.ndim != 7:
        guesses = {}
        guesses[f"{name}_hyper_loc"]   = _FALLBACK_HYPER_LOC
        guesses[f"{name}_hyper_scale"] = _DEFAULT_HYPER_SCALE
        guesses[f"{name}_spiked_loc"]  = _FALLBACK_SPIKED_LOC
        guesses[f"{name}_wt_loc"]      = _FALLBACK_WT_LOC
        guesses[f"{name}_offset"] = jnp.zeros(
            (data.num_replicate, data.num_condition_pre, data.num_genotype),
            dtype=float)
        return guesses

    # Replace invalid observations with NaN before computing medians
    ln_cfu_valid = np.where(good_mask, ln_cfu, np.nan)

    # Reduce over (time=1, cond_sel=3, titrant_name=4, titrant_conc=5)
    # Result shape: (num_replicate, num_condition_pre, num_genotype)
    per_rep_cond_geno = np.nanmedian(ln_cfu_valid, axis=(1, 3, 4, 5))

    # Helper: median over all (rep, cond_pre) for genotypes in a mask group
    def _group_median(mask, fallback):
        if not mask.any():
            return fallback
        vals = per_rep_cond_geno[:, :, mask]   # (rep, cond_pre, n_in_group)
        result = np.nanmedian(vals)
        return fallback if np.isnan(result) else float(result)

    library_mask = ~spiked_mask & ~wt_mask
    spiked_loc   = _group_median(spiked_mask, _FALLBACK_SPIKED_LOC)
    wt_loc       = _group_median(wt_mask,     _FALLBACK_WT_LOC)
    hyper_loc    = _group_median(library_mask, _FALLBACK_HYPER_LOC)

    # Per-genotype group location used to centre the offsets
    per_geno_loc = np.where(wt_mask, wt_loc,
                   np.where(spiked_mask, spiked_loc, hyper_loc))  # (num_genotype,)

    # Non-centred offset: (empirical estimate - group loc) / hyper_scale
    # Replace any remaining NaN (fully-masked genotypes) with 0.
    diff   = per_rep_cond_geno - per_geno_loc[np.newaxis, np.newaxis, :]
    offset = diff / _DEFAULT_HYPER_SCALE
    offset = np.where(np.isnan(offset), 0.0, offset)

    guesses = {}
    guesses[f"{name}_hyper_loc"]  = float(hyper_loc)
    guesses[f"{name}_hyper_scale"] = _DEFAULT_HYPER_SCALE
    guesses[f"{name}_spiked_loc"] = float(spiked_loc)
    guesses[f"{name}_wt_loc"]     = float(wt_loc)
    guesses[f"{name}_offset"]     = jnp.array(offset, dtype=float)

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
