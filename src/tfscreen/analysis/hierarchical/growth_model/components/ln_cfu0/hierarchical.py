import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass, field

from typing import Dict, Any, Mapping, Optional
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from tfscreen.analysis.hierarchical.growth_model.components._pinning import (
    _hyper,
    _pinned_value,
)


# Base names of hyperparameter suffixes that may be pinned via
# ModelPriors.pinned.  The actual per-class site suffixes are
# ``{base}_{class_index}``, e.g. ``"hyper_loc_0"``, ``"hyper_scale_1"``.
# Only the *library* subgroup hyperpriors are eligible — the wt and spiked
# subgroups are non-hierarchical (single-member or near-single-member with
# fixed per-genotype scales), so pinning them would not address any
# bilinear pathology.
_PINNABLE_SUFFIXES = (
    "hyper_loc", "hyper_scale",
)


# ---------------------------------------------------------------------------
# Defaults & helpers
# ---------------------------------------------------------------------------

# Floor on data-derived scale estimates.  Stops a degenerate group (single
# observation, identical replicates) from collapsing the prior to zero, which
# would lock the corresponding ln_cfu0 parameter at its loc.
_SCALE_FLOOR = 0.5

# Fall-back values used when no data is available (default get_priors call,
# or a group has no members in the supplied data).
_FALLBACK_HYPER_LOC   = 6.0
_FALLBACK_SPIKED_LOC  = 12.0
_FALLBACK_WT_LOC      = 13.0
_FALLBACK_GROUP_SCALE = 3.0


def _mad_scale(values: np.ndarray, fallback: float) -> float:
    """
    Median-absolute-deviation based scale estimate, with a floor.

    Parameters
    ----------
    values : np.ndarray
        Array of observations (NaNs are ignored).
    fallback : float
        Value to return when ``values`` is empty or all-NaN.

    Returns
    -------
    float
        Robust scale estimate (``1.4826 * MAD``), clipped from below by
        ``_SCALE_FLOOR``.  Returns ``fallback`` if no usable values exist.
    """
    if values.size == 0:
        return fallback
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return fallback
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    scale = 1.4826 * float(mad)
    return max(scale, _SCALE_FLOOR)


def _per_geno_loc_scale(data: GrowthData,
                        priors: "ModelPriors",
                        ln_cfu0_hyper_locs,
                        ln_cfu0_hyper_scales,
                        ln_cfu0_spiked_loc,
                        ln_cfu0_wt_loc):
    """
    Build per-genotype location and scale arrays for the current batch.

    Library genotypes are assigned to one of ``num_classes`` subgroups via
    ``data.ln_cfu0_library_masks``.  Class 0 is the default; later classes
    override it for their masked genotypes.  Spiked and wildtype genotypes
    use their own (sampled) locations and fixed prior-supplied scales,
    overriding any library-class assignment.
    """
    batch_spiked_mask = data.ln_cfu0_spiked_mask[data.batch_idx]
    batch_wt_mask = data.ln_cfu0_wt_mask[data.batch_idx]

    num_classes = getattr(data, "num_ln_cfu0_library_classes", 1)

    # Initialise from class 0 (covers all library genotypes not in a later class)
    per_geno_loc   = jnp.full(data.batch_size, ln_cfu0_hyper_locs[0])
    per_geno_scale = jnp.full(data.batch_size, ln_cfu0_hyper_scales[0])

    # Apply subsequent classes — each overrides the default for its members
    for i in range(1, num_classes):
        class_mask = data.ln_cfu0_library_masks[i][data.batch_idx]
        per_geno_loc   = jnp.where(class_mask, ln_cfu0_hyper_locs[i],   per_geno_loc)
        per_geno_scale = jnp.where(class_mask, ln_cfu0_hyper_scales[i], per_geno_scale)

    # Spiked and wt genotypes always override the library assignment
    per_geno_loc = jnp.where(
        batch_wt_mask, ln_cfu0_wt_loc,
        jnp.where(batch_spiked_mask, ln_cfu0_spiked_loc, per_geno_loc)
    )
    per_geno_scale = jnp.where(
        batch_wt_mask, priors.ln_cfu0_wt_scale,
        jnp.where(batch_spiked_mask,
                  priors.ln_cfu0_spiked_scale,
                  per_geno_scale)
    )
    return per_geno_loc, per_geno_scale


# ---------------------------------------------------------------------------
# Priors dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the ln_cfu0 model.

    Attributes
    ----------
    ln_cfu0_hyper_loc_locs : jnp.ndarray, shape (num_classes,)
        Mean of the Normal prior for each library-class hyper-location.
    ln_cfu0_hyper_loc_scales : jnp.ndarray, shape (num_classes,)
        Standard deviation of the Normal prior for each library-class
        hyper-location.
    ln_cfu0_hyper_scale_locs : jnp.ndarray, shape (num_classes,)
        Scale of the HalfNormal prior for each library-class hyper-scale.
    ln_cfu0_spiked_loc_loc : float
        Mean of the Normal prior for the ln_cfu0 location of spiked genotypes.
    ln_cfu0_spiked_loc_scale : float
        Standard deviation of the Normal prior for the spiked genotype location.
    ln_cfu0_spiked_scale : float
        Fixed scale used as the per-genotype noise for spiked genotypes.
    ln_cfu0_wt_loc_loc : float
        Mean of the Normal prior for the ln_cfu0 location of the wildtype
        genotype.
    ln_cfu0_wt_loc_scale : float
        Standard deviation of the Normal prior for the wildtype location.
    ln_cfu0_wt_scale : float
        Fixed scale used as the per-genotype noise for the wildtype genotype.
    """

    ln_cfu0_hyper_loc_locs: jnp.ndarray
    ln_cfu0_hyper_loc_scales: jnp.ndarray
    ln_cfu0_hyper_scale_locs: jnp.ndarray
    ln_cfu0_spiked_loc_loc: float
    ln_cfu0_spiked_loc_scale: float
    ln_cfu0_spiked_scale: float
    ln_cfu0_wt_loc_loc: float
    ln_cfu0_wt_loc_scale: float
    ln_cfu0_wt_scale: float

    pinned: Mapping[str, float] = field(
        pytree_node=False, default_factory=dict
    )


# ---------------------------------------------------------------------------
# Model and guide
# ---------------------------------------------------------------------------

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Defines the hierarchical model for initial cell counts (ln_cfu0).

    Library genotypes are split into ``num_classes`` subgroups, each with its
    own pooled Normal hyper-location and hyper-scale.  Spiked genotypes and
    the wildtype genotype have their own scalar location parameters and use
    *fixed* per-genotype scales drawn from ``priors``.

    Sample sites for library hyper-parameters are named
    ``{name}_hyper_loc_{i}`` and ``{name}_hyper_scale_{i}`` for class ``i``.
    These can be pinned via ``priors.pinned`` using the same suffix strings
    (e.g. ``"hyper_loc_0"``).

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
        - ``data.ln_cfu0_library_masks`` : bool array (num_classes, num_genotype)
        - ``data.num_ln_cfu0_library_classes`` : int, number of library classes
    priors : ModelPriors
        Pytree containing the hyperparameters.

    Returns
    -------
    jnp.ndarray
        The sampled ``ln_cfu0`` values expanded to observation shape
        ``(num_replicate, 1, num_condition_pre, 1, 1, 1, batch_size)``.
    """

    pinned = priors.pinned
    num_classes = getattr(data, "num_ln_cfu0_library_classes", 1)

    # Hyper-priors for each library-class pooled distribution
    hyper_locs_list   = []
    hyper_scales_list = []
    for i in range(num_classes):
        hl = _hyper(
            name, f"hyper_loc_{i}",
            dist.Normal(priors.ln_cfu0_hyper_loc_locs[i],
                        priors.ln_cfu0_hyper_loc_scales[i]),
            pinned,
        )
        hs = _hyper(
            name, f"hyper_scale_{i}",
            dist.HalfNormal(priors.ln_cfu0_hyper_scale_locs[i]),
            pinned,
        )
        hyper_locs_list.append(hl)
        hyper_scales_list.append(hs)

    ln_cfu0_hyper_locs   = jnp.stack(hyper_locs_list)
    ln_cfu0_hyper_scales = jnp.stack(hyper_scales_list)

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
    with pyro.plate(f"{name}_replicate", data.num_replicate, dim=-3):
        with pyro.plate(f"{name}_condition_pre", data.num_condition_pre, dim=-2):
            with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):
                    ln_cfu0_offsets = pyro.sample(f"{name}_offset", dist.Normal(0.0, 1.0))

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if ln_cfu0_offsets.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        ln_cfu0_offsets = ln_cfu0_offsets[..., data.batch_idx]

    per_geno_loc, per_geno_scale = _per_geno_loc_scale(
        data, priors,
        ln_cfu0_hyper_locs, ln_cfu0_hyper_scales,
        ln_cfu0_spiked_loc, ln_cfu0_wt_loc,
    )

    # Calculate the per-group ln_cfu0 values (per-genotype scale eliminates
    # the bilinear pathology for single-member subgroups).
    ln_cfu0_per_rep_cond_geno = per_geno_loc + ln_cfu0_offsets * per_geno_scale

    # Register deterministic values for inspection
    pyro.deterministic(name, ln_cfu0_per_rep_cond_geno)

    # Expand tensor to match all observations
    ln_cfu0 = ln_cfu0_per_rep_cond_geno[:, None, :, None, None, None, :]

    return ln_cfu0


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide corresponding to the hierarchical ln_cfu0 model.

    Uses Normal variational distributions for each per-class hyper-location
    and spiked location, LogNormal for each per-class hyper-scale, and Normal
    for per-genotype offsets.
    """

    pinned = priors.pinned
    num_classes = getattr(data, "num_ln_cfu0_library_classes", 1)

    # -------------------------------------------------------------------------
    # Global parameters — one per library class

    hyper_locs_list   = []
    hyper_scales_list = []
    for i in range(num_classes):
        # Hyper Loc — Normal posterior
        pinned_hl = _pinned_value(f"hyper_loc_{i}", pinned)
        if pinned_hl is not None:
            hl = pinned_hl
        else:
            h_loc_loc = pyro.param(
                f"{name}_hyper_loc_{i}_loc",
                jnp.array(priors.ln_cfu0_hyper_loc_locs[i])
            )
            h_loc_scale = pyro.param(
                f"{name}_hyper_loc_{i}_scale",
                jnp.array(priors.ln_cfu0_hyper_loc_scales[i]),
                constraint=dist.constraints.greater_than(1e-4)
            )
            hl = pyro.sample(f"{name}_hyper_loc_{i}", dist.Normal(h_loc_loc, h_loc_scale))

        # Hyper Scale — LogNormal posterior approximation for positive variable
        pinned_hs = _pinned_value(f"hyper_scale_{i}", pinned)
        if pinned_hs is not None:
            hs = pinned_hs
        else:
            h_scale_loc = pyro.param(
                f"{name}_hyper_scale_{i}_loc",
                jnp.array(-1.0)  # Init small
            )
            h_scale_scale = pyro.param(
                f"{name}_hyper_scale_{i}_scale",
                jnp.array(0.1),
                constraint=dist.constraints.greater_than(1e-4)
            )
            hs = pyro.sample(f"{name}_hyper_scale_{i}", dist.LogNormal(h_scale_loc, h_scale_scale))

        hyper_locs_list.append(hl)
        hyper_scales_list.append(hs)

    ln_cfu0_hyper_locs   = jnp.stack(hyper_locs_list)
    ln_cfu0_hyper_scales = jnp.stack(hyper_scales_list)

    # Spiked Loc — Normal posterior
    s_loc_loc = pyro.param(f"{name}_spiked_loc_loc",
                           jnp.array(priors.ln_cfu0_spiked_loc_loc))
    s_loc_scale = pyro.param(f"{name}_spiked_loc_scale",
                             jnp.array(priors.ln_cfu0_spiked_loc_scale),
                             constraint=dist.constraints.greater_than(1e-4))
    ln_cfu0_spiked_loc = pyro.sample(f"{name}_spiked_loc",
                                     dist.Normal(s_loc_loc, s_loc_scale))

    # WT Loc — Normal posterior
    w_loc_loc = pyro.param(f"{name}_wt_loc_loc",
                           jnp.array(priors.ln_cfu0_wt_loc_loc))
    w_loc_scale = pyro.param(f"{name}_wt_loc_scale",
                             jnp.array(priors.ln_cfu0_wt_loc_scale),
                             constraint=dist.constraints.greater_than(1e-4))
    ln_cfu0_wt_loc = pyro.sample(f"{name}_wt_loc",
                                 dist.Normal(w_loc_loc, w_loc_scale))

    # -------------------------------------------------------------------------
    # Genotype-specific parameter

    param_shape = (data.num_replicate, data.num_condition_pre, data.num_genotype)
    offset_locs = pyro.param(f"{name}_offset_locs",
                             jnp.zeros(param_shape, dtype=float))
    offset_scales = pyro.param(f"{name}_offset_scales",
                               jnp.ones(param_shape, dtype=float),
                               constraint=dist.constraints.positive)

    # Sample non-centered offsets for each ln_cfu0 group
    with pyro.plate(f"{name}_replicate", data.num_replicate, dim=-3):
        with pyro.plate(f"{name}_condition_pre", data.num_condition_pre, dim=-2):
            with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):

                    batch_locs   = offset_locs[..., data.batch_idx]
                    batch_scales = offset_scales[..., data.batch_idx]
                    ln_cfu0_offsets = pyro.sample(f"{name}_offset",
                                                  dist.Normal(batch_locs, batch_scales))

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if ln_cfu0_offsets.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        ln_cfu0_offsets = ln_cfu0_offsets[..., data.batch_idx]

    per_geno_loc, per_geno_scale = _per_geno_loc_scale(
        data, priors,
        ln_cfu0_hyper_locs, ln_cfu0_hyper_scales,
        ln_cfu0_spiked_loc, ln_cfu0_wt_loc,
    )

    # Calculate the per-group ln_cfu0 values
    ln_cfu0_per_rep_cond_geno = per_geno_loc + ln_cfu0_offsets * per_geno_scale

    # Expand tensor to match all observations
    ln_cfu0 = ln_cfu0_per_rep_cond_geno[:, None, :, None, None, None, :]

    return ln_cfu0


# ---------------------------------------------------------------------------
# Hyperparameters / priors / guesses
# ---------------------------------------------------------------------------

def get_hyperparameters(num_classes: int = 1) -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Parameters
    ----------
    num_classes : int, optional
        Number of library genotype classes.  Defaults to 1 (all library
        genotypes share one pooled distribution).

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names to their default values.
        ``ln_cfu0_hyper_loc_locs``, ``ln_cfu0_hyper_loc_scales``, and
        ``ln_cfu0_hyper_scale_locs`` are 1-D jnp arrays of length
        ``num_classes``.
    """

    parameters = {}

    parameters["ln_cfu0_hyper_loc_locs"]   = jnp.full(num_classes, _FALLBACK_HYPER_LOC)
    parameters["ln_cfu0_hyper_loc_scales"]  = jnp.full(num_classes, 3.0)
    parameters["ln_cfu0_hyper_scale_locs"]  = jnp.full(num_classes, 2.0)
    parameters["ln_cfu0_spiked_loc_loc"]    = _FALLBACK_SPIKED_LOC
    parameters["ln_cfu0_spiked_loc_scale"]  = 3.0
    parameters["ln_cfu0_spiked_scale"]      = _FALLBACK_GROUP_SCALE
    parameters["ln_cfu0_wt_loc_loc"]        = _FALLBACK_WT_LOC
    parameters["ln_cfu0_wt_loc_scale"]      = 3.0
    parameters["ln_cfu0_wt_scale"]          = _FALLBACK_GROUP_SCALE

    return parameters


def _empirical_group_estimates(data: GrowthData):
    """
    Compute per-group and per-class empirical (loc, scale) estimates and the
    per-(rep, cond_pre, geno) median tensor.

    Returns ``None`` if ``data.ln_cfu`` is not the expected 7-D tensor (e.g.
    in mocked tests or when called with a stub data object).  Otherwise
    returns a dict with keys:
      - ``per_rep_cond_geno`` : (rep, cond_pre, geno) ndarray of medians
      - ``hyper_locs``, ``hyper_scales`` : lists of floats, one per class
      - ``hyper_loc``, ``hyper_scale``   : class-0 aliases (backward compat)
      - ``spiked_loc``, ``spiked_scale``
      - ``wt_loc``, ``wt_scale``
      - ``library_mask``, ``spiked_mask``, ``wt_mask`` : (geno,) bool ndarrays
    """

    ln_cfu      = np.asarray(data.ln_cfu)
    good_mask   = np.asarray(data.good_mask)
    spiked_mask = np.asarray(data.ln_cfu0_spiked_mask)
    wt_mask     = np.asarray(data.ln_cfu0_wt_mask)

    if ln_cfu.ndim != 7:
        return None

    # Replace invalid observations with NaN before computing medians
    ln_cfu_valid = np.where(good_mask, ln_cfu, np.nan)

    # Reduce over (time=1, cond_sel=3, titrant_name=4, titrant_conc=5)
    # Result shape: (num_replicate, num_condition_pre, num_genotype)
    per_rep_cond_geno = np.nanmedian(ln_cfu_valid, axis=(1, 3, 4, 5))

    library_mask = ~spiked_mask & ~wt_mask

    def _group_loc(mask, fallback):
        if not mask.any():
            return fallback
        vals   = per_rep_cond_geno[:, :, mask]
        finite = vals[~np.isnan(vals)]
        if finite.size == 0:
            return fallback
        return float(np.median(finite))

    def _group_scale(mask, group_loc, fallback_scale):
        if not mask.any():
            return fallback_scale
        vals       = per_rep_cond_geno[:, :, mask]
        deviations = (vals - group_loc).flatten()
        return _mad_scale(deviations, fallback_scale)

    spiked_loc = _group_loc(spiked_mask, _FALLBACK_SPIKED_LOC)
    wt_loc     = _group_loc(wt_mask,     _FALLBACK_WT_LOC)

    spiked_scale = _group_scale(spiked_mask, spiked_loc, _FALLBACK_GROUP_SCALE)
    wt_scale     = _group_scale(wt_mask,     wt_loc,     _FALLBACK_GROUP_SCALE)

    # Per-class library estimates
    num_classes = getattr(data, "num_ln_cfu0_library_classes", 1)
    library_masks_raw = getattr(data, "ln_cfu0_library_masks", None)

    if num_classes > 1 and library_masks_raw is not None:
        all_class_masks = [
            np.asarray(library_masks_raw[i]) & library_mask
            for i in range(num_classes)
        ]
    else:
        all_class_masks = [library_mask]

    hyper_locs   = []
    hyper_scales = []
    for class_mask in all_class_masks:
        loc   = _group_loc(class_mask, _FALLBACK_HYPER_LOC)
        scale = _group_scale(class_mask, loc, _FALLBACK_GROUP_SCALE)
        hyper_locs.append(loc)
        hyper_scales.append(scale)

    return {
        "per_rep_cond_geno": per_rep_cond_geno,
        "hyper_locs":        hyper_locs,
        "hyper_scales":      hyper_scales,
        "hyper_loc":         hyper_locs[0],   # class-0 alias
        "hyper_scale":       hyper_scales[0], # class-0 alias
        "spiked_loc":        spiked_loc,
        "spiked_scale":      spiked_scale,
        "wt_loc":            wt_loc,
        "wt_scale":          wt_scale,
        "library_mask":      library_mask,
        "spiked_mask":       spiked_mask,
        "wt_mask":           wt_mask,
    }


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get initial guess values derived empirically from observed ln_cfu data.

    For each (replicate, condition_pre, genotype), the median ln_cfu across
    all valid observations (masked by ``data.good_mask``) is used as an
    estimate of the starting cell density.  Group-level medians over spiked,
    wildtype, and each library class provide estimates for ``spiked_loc``,
    ``wt_loc``, and ``hyper_loc_{i}`` respectively.  Per-genotype offsets are
    derived by centring on the matching group-level estimate and dividing by
    the group-level MAD-based scale.

    Sample-site keys for library hyper-parameters follow the naming
    ``{name}_hyper_loc_{i}`` and ``{name}_hyper_scale_{i}``.

    Falls back to hard-coded defaults for any group that has no valid
    observations.
    """

    estimates = _empirical_group_estimates(data)

    num_classes = getattr(data, "num_ln_cfu0_library_classes", 1)

    # Fallback path: data is not the expected 7-D tensor
    if estimates is None:
        guesses = {
            f"{name}_spiked_loc": _FALLBACK_SPIKED_LOC,
            f"{name}_wt_loc":     _FALLBACK_WT_LOC,
            f"{name}_offset": jnp.zeros(
                (data.num_replicate, data.num_condition_pre, data.num_genotype),
                dtype=float),
        }
        for i in range(num_classes):
            guesses[f"{name}_hyper_loc_{i}"]   = _FALLBACK_HYPER_LOC
            guesses[f"{name}_hyper_scale_{i}"] = _FALLBACK_GROUP_SCALE
        return guesses

    per_rep_cond_geno = estimates["per_rep_cond_geno"]
    spiked_mask       = estimates["spiked_mask"]
    wt_mask           = estimates["wt_mask"]
    library_mask      = estimates["library_mask"]

    # Build per-class masks (intersected with the library mask)
    library_masks_raw = getattr(data, "ln_cfu0_library_masks", None)
    if num_classes > 1 and library_masks_raw is not None:
        class_masks = [
            np.asarray(library_masks_raw[i]) & library_mask
            for i in range(num_classes)
        ]
    else:
        class_masks = [library_mask]

    # Per-genotype group location and scale for offset centring.
    # Each library genotype belongs to the first class whose mask is True;
    # later classes override earlier ones (matching _per_geno_loc_scale logic).
    per_geno_loc   = np.full(data.num_genotype, estimates["hyper_locs"][0])
    per_geno_scale = np.full(data.num_genotype, estimates["hyper_scales"][0])
    for i in range(1, len(class_masks)):
        m = class_masks[i]
        per_geno_loc[m]   = estimates["hyper_locs"][i]
        per_geno_scale[m] = estimates["hyper_scales"][i]

    # Spiked and wt override library assignments
    per_geno_loc   = np.where(wt_mask,     estimates["wt_loc"],
                    np.where(spiked_mask,  estimates["spiked_loc"], per_geno_loc))
    per_geno_scale = np.where(wt_mask,     estimates["wt_scale"],
                    np.where(spiked_mask,  estimates["spiked_scale"], per_geno_scale))

    # Non-centred offset: (empirical estimate - group loc) / group scale.
    # Replace any remaining NaN (fully-masked genotypes) with 0.
    diff   = per_rep_cond_geno - per_geno_loc[np.newaxis, np.newaxis, :]
    offset = diff / per_geno_scale[np.newaxis, np.newaxis, :]
    offset = np.where(np.isnan(offset), 0.0, offset)

    guesses = {
        f"{name}_spiked_loc": float(estimates["spiked_loc"]),
        f"{name}_wt_loc":     float(estimates["wt_loc"]),
        f"{name}_offset":     jnp.array(offset, dtype=float),
    }
    for i in range(len(class_masks)):
        guesses[f"{name}_hyper_loc_{i}"]   = float(estimates["hyper_locs"][i])
        guesses[f"{name}_hyper_scale_{i}"] = float(estimates["hyper_scales"][i])

    return guesses


def get_priors(data: Optional[GrowthData] = None) -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    When ``data`` is provided:
    - The fixed per-genotype scales for the wt and spiked subgroups
      (``ln_cfu0_wt_scale`` and ``ln_cfu0_spiked_scale``) are derived
      empirically via the MAD-based estimator.
    - The per-class hyper-location priors
      (``ln_cfu0_hyper_loc_locs``) are centred on empirical class medians.
    - The number of classes is read from ``data.num_ln_cfu0_library_classes``
      (defaults to 1 when the attribute is absent).

    Without data, every value is a single-class default.

    Parameters
    ----------
    data : GrowthData, optional
        Experimental data pytree.

    Returns
    -------
    ModelPriors
        A populated Pytree (Flax dataclass) of hyperparameters.
    """
    num_classes = getattr(data, "num_ln_cfu0_library_classes", 1) if data is not None else 1
    params = get_hyperparameters(num_classes=num_classes)

    if data is not None:
        estimates = _empirical_group_estimates(data)
        if estimates is not None:
            params["ln_cfu0_wt_scale"]     = float(estimates["wt_scale"])
            params["ln_cfu0_spiked_scale"] = float(estimates["spiked_scale"])
            # Centre each class's hyper-loc prior on the empirical class median
            params["ln_cfu0_hyper_loc_locs"] = jnp.array(estimates["hyper_locs"])

    return ModelPriors(**params)


def get_extract_specs(ctx):
    if "map_ln_cfu0" not in ctx.growth_tm.df.columns:
        return []
    return [dict(
        input_df=ctx.growth_tm.df,
        params_to_get=["ln_cfu0"],
        map_column="map_ln_cfu0",
        get_columns=["replicate", "condition_pre", "genotype"],
        in_run_prefix="",
    )]
