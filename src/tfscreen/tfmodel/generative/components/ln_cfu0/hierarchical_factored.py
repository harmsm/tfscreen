import warnings

import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass, field

from typing import Dict, Any, Mapping, Optional
from tfscreen.tfmodel.data_class import GrowthData
from tfscreen.tfmodel.generative.components._pinning import (
    _hyper,
    _pinned_value,
)
from tfscreen.tfmodel.generative.components.ln_cfu0.hierarchical import (
    _per_geno_loc_scale,
    _empirical_group_estimates,
    _mad_scale,
    _FALLBACK_HYPER_LOC,
    _FALLBACK_SPIKED_LOC,
    _FALLBACK_WT_LOC,
    _FALLBACK_GROUP_SCALE,
)


# Only the library hyperpriors are pinnable; tube_scale is not pinnable.
_PINNABLE_SUFFIXES = (
    "hyper_loc", "hyper_scale",
)

_FALLBACK_TUBE_SCALE = 0.5


# ---------------------------------------------------------------------------
# Priors dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the hierarchical_factored ln_cfu0 model.

    This model factors ln_cfu0[r, c, g] = geno_baseline[r, g] + tube_offset[r, c],
    where geno_baseline captures genotype-level abundance (shared across all
    pre-expression conditions) and tube_offset captures systematic dilution
    differences between individual tubes.

    Attributes
    ----------
    ln_cfu0_hyper_loc_locs : jnp.ndarray, shape (num_classes,)
    ln_cfu0_hyper_loc_scales : jnp.ndarray, shape (num_classes,)
    ln_cfu0_hyper_scale_locs : jnp.ndarray, shape (num_classes,)
    ln_cfu0_spiked_loc_loc : float
    ln_cfu0_spiked_loc_scale : float
    ln_cfu0_spiked_scale : float
    ln_cfu0_wt_loc_loc : float
    ln_cfu0_wt_loc_scale : float
    ln_cfu0_wt_scale : float
    ln_cfu0_tube_scale_loc : float
        Scale of the HalfNormal prior on the between-tube standard deviation.
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
    ln_cfu0_tube_scale_loc: float

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
    Factored hierarchical model for initial cell counts (ln_cfu0).

    Parameterises ln_cfu0 as the sum of two independent terms::

        ln_cfu0[r, c, g] = geno_baseline[r, g] + tube_offset[r, c]

    ``geno_baseline`` uses the same non-centred hierarchical structure as the
    ``hierarchical`` component (pooled Normal per library class, separate
    locations for spiked/wt genotypes) but without the condition_pre
    dimension — reflecting the biology that all tubes were split from the
    same stock at the same OD600.  ``tube_offset`` is a small centred Normal
    per (replicate, condition_pre) pair that absorbs any remaining systematic
    tube-to-tube dilution differences.

    Parameters
    ----------
    name : str
    data : GrowthData
    priors : ModelPriors

    Returns
    -------
    jnp.ndarray
        Shape ``(num_replicate, 1, num_condition_pre, 1, 1, 1, batch_size)``.
    """

    pinned = priors.pinned
    num_classes = getattr(data, "num_ln_cfu0_library_classes", 1)

    # Per-library-class hyperpriors
    hyper_locs_list = []
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

    ln_cfu0_spiked_loc = pyro.sample(
        f"{name}_spiked_loc",
        dist.Normal(priors.ln_cfu0_spiked_loc_loc, priors.ln_cfu0_spiked_loc_scale)
    )
    ln_cfu0_wt_loc = pyro.sample(
        f"{name}_wt_loc",
        dist.Normal(priors.ln_cfu0_wt_loc_loc, priors.ln_cfu0_wt_loc_scale)
    )

    # Hyperprior on between-tube variation
    tube_scale = pyro.sample(
        f"{name}_tube_scale",
        dist.HalfNormal(priors.ln_cfu0_tube_scale_loc)
    )

    # Per-(replicate, genotype) non-centred offsets — shape (R, batch)
    with pyro.plate(f"{name}_geno_replicate", data.num_replicate, dim=-2):
        with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
            with pyro.handlers.scale(scale=data.scale_vector):
                offset_geno = pyro.sample(f"{name}_offset_geno", dist.Normal(0.0, 1.0))

    # Guard against full-sized array substitution during initialisation
    if offset_geno.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        offset_geno = offset_geno[..., data.batch_idx]

    # Per-(replicate, condition_pre) tube offsets — shape (R, C)
    with pyro.plate(f"{name}_tube_replicate", data.num_replicate, dim=-2):
        with pyro.plate(f"{name}_condition_pre", data.num_condition_pre, dim=-1):
            tube_offset = pyro.sample(
                f"{name}_tube_offset",
                dist.Normal(0.0, tube_scale)
            )

    per_geno_loc, per_geno_scale = _per_geno_loc_scale(
        data, priors,
        ln_cfu0_hyper_locs, ln_cfu0_hyper_scales,
        ln_cfu0_spiked_loc, ln_cfu0_wt_loc,
    )

    # geno_baseline: (R, batch)
    geno_baseline = per_geno_loc + offset_geno * per_geno_scale

    # Combined: (R, C, batch)
    ln_cfu0_per_rep_cond_geno = geno_baseline[:, None, :] + tube_offset[:, :, None]

    pyro.deterministic(name, ln_cfu0_per_rep_cond_geno)

    return ln_cfu0_per_rep_cond_geno[:, None, :, None, None, None, :]


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide for the hierarchical_factored ln_cfu0 model.

    Variational parameters:
    - ``{name}_offset_geno_locs/scales``: shape ``(num_replicate, num_genotype)``
    - ``{name}_tube_offset_locs/scales``: shape ``(num_replicate, num_condition_pre)``
    - Per-class hyper_loc/hyper_scale variational params (skipped when pinned)
    - ``{name}_tube_scale``: LogNormal variational posterior for positive scalar
    """

    pinned = priors.pinned
    num_classes = getattr(data, "num_ln_cfu0_library_classes", 1)

    hyper_locs_list = []
    hyper_scales_list = []
    for i in range(num_classes):
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

        pinned_hs = _pinned_value(f"hyper_scale_{i}", pinned)
        if pinned_hs is not None:
            hs = pinned_hs
        else:
            h_scale_loc = pyro.param(
                f"{name}_hyper_scale_{i}_loc",
                jnp.array(-1.0)
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

    s_loc_loc = pyro.param(f"{name}_spiked_loc_loc",
                           jnp.array(priors.ln_cfu0_spiked_loc_loc))
    s_loc_scale = pyro.param(f"{name}_spiked_loc_scale",
                             jnp.array(priors.ln_cfu0_spiked_loc_scale),
                             constraint=dist.constraints.greater_than(1e-4))
    ln_cfu0_spiked_loc = pyro.sample(f"{name}_spiked_loc",
                                     dist.Normal(s_loc_loc, s_loc_scale))

    w_loc_loc = pyro.param(f"{name}_wt_loc_loc",
                           jnp.array(priors.ln_cfu0_wt_loc_loc))
    w_loc_scale = pyro.param(f"{name}_wt_loc_scale",
                             jnp.array(priors.ln_cfu0_wt_loc_scale),
                             constraint=dist.constraints.greater_than(1e-4))
    ln_cfu0_wt_loc = pyro.sample(f"{name}_wt_loc",
                                 dist.Normal(w_loc_loc, w_loc_scale))

    # Tube scale — LogNormal variational posterior
    ts_loc = pyro.param(f"{name}_tube_scale_loc", jnp.array(-1.0))
    ts_scale = pyro.param(f"{name}_tube_scale_scale", jnp.array(0.1),
                          constraint=dist.constraints.greater_than(1e-4))
    pyro.sample(f"{name}_tube_scale", dist.LogNormal(ts_loc, ts_scale))

    # Per-(replicate, genotype) offset variational params — shape (R, G)
    geno_param_shape = (data.num_replicate, data.num_genotype)
    offset_geno_locs = pyro.param(f"{name}_offset_geno_locs",
                                  jnp.zeros(geno_param_shape, dtype=float))
    offset_geno_scales = pyro.param(f"{name}_offset_geno_scales",
                                    jnp.ones(geno_param_shape, dtype=float),
                                    constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_geno_replicate", data.num_replicate, dim=-2):
        with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
            with pyro.handlers.scale(scale=data.scale_vector):
                batch_geno_locs   = offset_geno_locs[..., data.batch_idx]
                batch_geno_scales = offset_geno_scales[..., data.batch_idx]
                offset_geno = pyro.sample(f"{name}_offset_geno",
                                          dist.Normal(batch_geno_locs, batch_geno_scales))

    # Guard against full-sized substitution
    if offset_geno.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        offset_geno = offset_geno[..., data.batch_idx]

    # Per-(replicate, condition_pre) tube offset variational params — shape (R, C)
    tube_param_shape = (data.num_replicate, data.num_condition_pre)
    tube_offset_locs = pyro.param(f"{name}_tube_offset_locs",
                                  jnp.zeros(tube_param_shape, dtype=float))
    tube_offset_scales = pyro.param(f"{name}_tube_offset_scales",
                                    jnp.full(tube_param_shape, 0.1, dtype=float),
                                    constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_tube_replicate", data.num_replicate, dim=-2):
        with pyro.plate(f"{name}_condition_pre", data.num_condition_pre, dim=-1):
            tube_offset = pyro.sample(f"{name}_tube_offset",
                                      dist.Normal(tube_offset_locs, tube_offset_scales))

    per_geno_loc, per_geno_scale = _per_geno_loc_scale(
        data, priors,
        ln_cfu0_hyper_locs, ln_cfu0_hyper_scales,
        ln_cfu0_spiked_loc, ln_cfu0_wt_loc,
    )

    geno_baseline = per_geno_loc + offset_geno * per_geno_scale
    ln_cfu0_per_rep_cond_geno = geno_baseline[:, None, :] + tube_offset[:, :, None]

    return ln_cfu0_per_rep_cond_geno[:, None, :, None, None, None, :]


# ---------------------------------------------------------------------------
# Hyperparameters / priors / guesses
# ---------------------------------------------------------------------------

def get_hyperparameters(num_classes: int = 1) -> Dict[str, Any]:
    """Default hyperparameter values for the hierarchical_factored ln_cfu0 model."""

    params = {}
    params["ln_cfu0_hyper_loc_locs"]   = jnp.full(num_classes, _FALLBACK_HYPER_LOC)
    params["ln_cfu0_hyper_loc_scales"]  = jnp.full(num_classes, 3.0)
    params["ln_cfu0_hyper_scale_locs"]  = jnp.full(num_classes, 2.0)
    params["ln_cfu0_spiked_loc_loc"]    = _FALLBACK_SPIKED_LOC
    params["ln_cfu0_spiked_loc_scale"]  = 3.0
    params["ln_cfu0_spiked_scale"]      = _FALLBACK_GROUP_SCALE
    params["ln_cfu0_wt_loc_loc"]        = _FALLBACK_WT_LOC
    params["ln_cfu0_wt_loc_scale"]      = 3.0
    params["ln_cfu0_wt_scale"]          = _FALLBACK_GROUP_SCALE
    params["ln_cfu0_tube_scale_loc"]    = _FALLBACK_TUBE_SCALE
    return params


def get_priors(data: Optional[GrowthData] = None,
              presplit: Optional[Any] = None) -> ModelPriors:
    """
    Build a ModelPriors instance, optionally informed by empirical data.

    The per-class hyper-loc priors and wt/spiked scales are derived from
    data when available, using the same MAD-based estimator as the
    ``hierarchical`` component.  If ``presplit`` is also provided, its
    direct t=-t_pre measurements are preferred over the ln_cfu median
    wherever available (see
    :func:`~tfscreen.tfmodel.generative.components.ln_cfu0.hierarchical._empirical_group_estimates`).
    """
    num_classes = getattr(data, "num_ln_cfu0_library_classes", 1) if data is not None else 1
    params = get_hyperparameters(num_classes=num_classes)

    if data is not None:
        estimates = _empirical_group_estimates(data, presplit=presplit)
        if estimates is not None:
            params["ln_cfu0_wt_scale"]       = float(estimates["wt_scale"])
            params["ln_cfu0_spiked_scale"]   = float(estimates["spiked_scale"])
            params["ln_cfu0_hyper_loc_locs"] = jnp.array(estimates["hyper_locs"])

    return ModelPriors(**params)


def get_guesses(name: str, data: GrowthData,
               presplit: Optional[Any] = None) -> Dict[str, jnp.ndarray]:
    """
    Empirically derived initial values for all model sample sites.

    The genotype baseline is estimated as the per-(replicate, genotype)
    median across condition_pre, unless a direct pre-split measurement is
    available (see ``presplit``), in which case it is preferred wherever
    valid.  The tube offset is the per-(replicate, condition_pre) median
    residual after subtracting that baseline.  The tube scale is estimated
    from the MAD of the tube offsets.

    Falls back to hard-coded defaults for any group or dimension with no
    valid observations.

    Parameters
    ----------
    name : str
    data : GrowthData
    presplit : PreSplitData, optional
        Optional direct t=-t_pre measurement of ln_cfu0; see
        :func:`~tfscreen.tfmodel.generative.components.ln_cfu0.hierarchical._empirical_group_estimates`.
    """

    estimates = _empirical_group_estimates(data, presplit=presplit)
    num_classes = getattr(data, "num_ln_cfu0_library_classes", 1)

    if estimates is None:
        guesses = {
            f"{name}_spiked_loc":  _FALLBACK_SPIKED_LOC,
            f"{name}_wt_loc":      _FALLBACK_WT_LOC,
            f"{name}_tube_scale":  _FALLBACK_TUBE_SCALE,
            f"{name}_offset_geno": jnp.zeros(
                (data.num_replicate, data.num_genotype), dtype=float),
            f"{name}_tube_offset": jnp.zeros(
                (data.num_replicate, data.num_condition_pre), dtype=float),
        }
        for i in range(num_classes):
            guesses[f"{name}_hyper_loc_{i}"]   = _FALLBACK_HYPER_LOC
            guesses[f"{name}_hyper_scale_{i}"] = _FALLBACK_GROUP_SCALE
        return guesses

    per_rep_cond_geno = estimates["per_rep_cond_geno"]  # (R, C, G)
    spiked_mask       = estimates["spiked_mask"]
    wt_mask           = estimates["wt_mask"]
    library_mask      = estimates["library_mask"]

    library_masks_raw = getattr(data, "ln_cfu0_library_masks", None)
    if num_classes > 1 and library_masks_raw is not None:
        class_masks = [
            np.asarray(library_masks_raw[i]) & library_mask
            for i in range(num_classes)
        ]
    else:
        class_masks = [library_mask]

    per_geno_loc   = np.full(data.num_genotype, estimates["hyper_locs"][0])
    per_geno_scale = np.full(data.num_genotype, estimates["hyper_scales"][0])
    for i in range(1, len(class_masks)):
        m = class_masks[i]
        per_geno_loc[m]   = estimates["hyper_locs"][i]
        per_geno_scale[m] = estimates["hyper_scales"][i]
    per_geno_loc   = np.where(wt_mask,     estimates["wt_loc"],
                    np.where(spiked_mask,  estimates["spiked_loc"],   per_geno_loc))
    per_geno_scale = np.where(wt_mask,     estimates["wt_scale"],
                    np.where(spiked_mask,  estimates["spiked_scale"], per_geno_scale))

    # Geno baseline: median over condition_pre — shape (R, G)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        geno_baseline_est = np.nanmedian(per_rep_cond_geno, axis=1)

    # Tube offset: per-tube median residual — shape (R, C)
    residuals = per_rep_cond_geno - geno_baseline_est[:, None, :]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        tube_offset_est = np.nanmedian(residuals, axis=2)
    tube_offset_est = np.where(np.isnan(tube_offset_est), 0.0, tube_offset_est)

    # Tube scale from the spread of tube offsets
    tube_scale_est = _mad_scale(tube_offset_est.flatten(), fallback=_FALLBACK_TUBE_SCALE)

    # Non-centred geno offset: (geno_baseline - per_geno_loc) / per_geno_scale — shape (R, G)
    diff       = geno_baseline_est - per_geno_loc[np.newaxis, :]
    offset_geno = diff / per_geno_scale[np.newaxis, :]
    offset_geno = np.where(np.isnan(offset_geno), 0.0, offset_geno)

    guesses = {
        f"{name}_spiked_loc":  float(estimates["spiked_loc"]),
        f"{name}_wt_loc":      float(estimates["wt_loc"]),
        f"{name}_tube_scale":  float(tube_scale_est),
        f"{name}_offset_geno": jnp.array(offset_geno, dtype=float),
        f"{name}_tube_offset": jnp.array(tube_offset_est, dtype=float),
    }
    for i in range(len(class_masks)):
        guesses[f"{name}_hyper_loc_{i}"]   = float(estimates["hyper_locs"][i])
        guesses[f"{name}_hyper_scale_{i}"] = float(estimates["hyper_scales"][i])

    return guesses


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
