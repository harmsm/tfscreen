"""
Pre-fit accounting of model parameters and observations.

This module answers, before any inference is run: how many latent parameters
does the configured model have, of what kinds, how many observations are there
to constrain them, and which genotypes are individually under-determined.

Nothing here is hard-coded per component.  Every latent is discovered by
tracing the Numpyro model, so a newly-registered component is counted
automatically.  Each ``pyro.sample`` site yields four pieces of information:

- its scalar count, from the traced value's shape;
- its **plate membership**, which says what the parameter is indexed by
  (genotype / mutation / condition / ...) and therefore what it scales with;
- its **support**, which distinguishes scale-like latents (positive support:
  hierarchical sigmas, horseshoe local/global scales) from location-like ones;
- its **component**, from the site-name prefix, which is the registry key each
  component is handed as its ``name`` argument.

Observation counts come from the ``good_mask`` arrays rather than from tensor
shapes, so padding in the ragged tensors is never counted as data.

The trace is run under ``jax.eval_shape`` -- abstract evaluation, no FLOPs and
no allocation -- so this is free even on a full-size library.  If a component
does something abstract evaluation cannot handle, the trace falls back to a
concrete forward pass.

The main entry point is :func:`count_model_dimensions`, which takes a
``ModelOrchestrator`` and returns a :class:`ModelStats`.
"""

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from numpyro.handlers import seed, trace

# -----------------------------------------------------------------------------
# Site classification vocabulary
#
# Plate names are built by the components as f"{name}_<something>", so these
# match on substrings of the plate name.  Entities are the axes a parameter
# count can scale along as the library grows; design axes are fixed by the
# experiment.
# -----------------------------------------------------------------------------

# Checked in order -- a site inside both a mutation and a pair plate is a
# pair-level parameter.
_ENTITY_PATTERNS = (
    ("pair", "per_pair"),
    ("mutation", "per_mutation"),
    ("genotype", "per_genotype"),
)

# Axes along which a single genotype's phenotype is *measured*.  A latent
# indexed by an entity AND one of these grows with the number of observations
# (e.g. the per-genotype-per-concentration epsilon of the logit_normal noise
# component), which changes how a parameter count should be read.
_DATUM_KEYS = ("titrant_conc", "time")

# Fallback scope for sites with no entity plate, checked in order.
_DESIGN_PATTERNS = (
    ("condition", "per_condition"),
    ("replicate", "per_replicate"),
    ("tube", "per_sample"),
    ("sample", "per_sample"),
    ("titrant_conc", "per_titrant_conc"),
    ("titrant_name", "per_titrant_name"),
    ("time", "per_time"),
)

# Registry keys handed to components as their `name`, plus the four observers.
# Longest-first so that "theta_growth_noise_epsilon" resolves to
# "theta_growth_noise" rather than to "theta".
_COMPONENT_KEYS = (
    "condition_growth",
    "growth_transition",
    "theta_growth_noise",
    "theta_binding_noise",
    "sample_offset",
    "transformation",
    "growth_noise",
    "base_growth",
    "ln_cfu0",
    "dk_geno",
    "activity",
    "presplit",
    "binding",
    "growth",
    "theta",
)


@dataclass
class ModelStats:
    """
    Container for the results of :func:`count_model_dimensions`.

    Attributes
    ----------
    sites : pandas.DataFrame
        One row per latent sample site.  Columns: ``site``, ``component``,
        ``scope``, ``level``, ``kind``, ``n_param``, ``scales_with_library``,
        ``per_datum``, ``shape``, ``plates``.
    per_genotype : pandas.DataFrame
        One row per genotype, describing how much data constrains it.  Not
        written to disk (it can be large); summarized into ``summary`` and
        kept here for plotting and further analysis.
    summary : dict
        Nested dictionary of headline counts, ratios, and coverage quantiles.
    warnings : list of str
        Human-readable flags raised by the coverage and anchor checks.
    """

    sites: pd.DataFrame
    per_genotype: pd.DataFrame
    summary: dict
    warnings: list = field(default_factory=list)


# -----------------------------------------------------------------------------
# Tracing
# -----------------------------------------------------------------------------

def _run_trace(model, data, priors):
    """
    Trace `model`, preferring abstract (zero-cost) evaluation.

    Parameters
    ----------
    model : callable
        A Numpyro model taking ``data`` and ``priors`` keyword arguments.
    data : DataClass
        Full-batch data pytree.
    priors : PriorsClass
        Priors pytree.

    Returns
    -------
    trace : dict
        The Numpyro trace.
    mode : str
        Either 'abstract' or 'concrete', recording how the trace was obtained.
    """

    box = {}

    def _traced(d, p):
        box["trace"] = trace(seed(model, rng_seed=0)).get_trace(data=d, priors=p)
        return jnp.zeros(())

    try:
        jax.eval_shape(_traced, data, priors)
        return box["trace"], "abstract"
    except Exception:
        # Some component did something abstract evaluation cannot handle
        # (e.g. a data-dependent Python branch).  Pay for a real forward pass.
        box.clear()
        _traced(data, priors)
        return box["trace"], "concrete"


def _site_component(site_name):
    """Resolve a site name to the component that registered it."""

    for key in sorted(_COMPONENT_KEYS, key=len, reverse=True):
        if site_name == key or site_name.startswith(f"{key}_"):
            return key

    return "unknown"


def _plate_frames(site):
    """Return [(plate_name, size), ...] for a traced site."""

    return [(f.name, int(f.size)) for f in site.get("cond_indep_stack", [])]


def _site_entity(frames):
    """
    Return (entity, scope) for the widest entity plate a site sits in.

    Returns (None, None) when the site is in no entity plate.
    """

    for pattern, scope in _ENTITY_PATTERNS:
        for plate_name, _ in frames:
            if pattern in plate_name:
                return pattern, scope

    return None, None


def _site_scope(frames):
    """
    Classify what a site is indexed by.

    Returns
    -------
    scope : str
        'per_pair' / 'per_mutation' / 'per_genotype' when the site is in an
        entity plate, with a '_datum' suffix when it is also indexed by a
        measurement axis; otherwise the most specific design axis it sits in
        ('per_condition', 'per_replicate', ...); 'global' when it is in no
        plate of size > 1.
    entity : str or None
        The entity pattern matched, or None.
    per_datum : bool
        True when the site is indexed by an entity AND by a measurement axis
        (concentration or time) of size > 1 -- i.e. its parameter count grows
        with the number of observations, not just with the library.
    """

    entity, entity_scope = _site_entity(frames)

    per_datum = False
    if entity is not None:
        per_datum = any(
            key in plate_name and size > 1
            for plate_name, size in frames
            for key in _DATUM_KEYS
        )
        if per_datum:
            # Indexed by an entity *and* by a measurement axis: this grows
            # with the data, not just with the library.
            entity_scope = f"{entity_scope}_datum"
        return entity_scope, entity, per_datum

    for pattern, scope in _DESIGN_PATTERNS:
        for plate_name, size in frames:
            if pattern in plate_name and size > 1:
                return scope, None, False

    return "global", None, False


def _is_scale_like(fn):
    """
    True when a distribution's support is positive.

    Positive support is the signal that a latent is a dispersion or shrinkage
    parameter (hierarchical sigmas, horseshoe local/global scales, StudentT nu)
    rather than a location.  Matched on the constraint's type name so this
    survives numpyro version changes.
    """

    support = getattr(fn, "support", None)
    if support is None:
        return False

    name = type(support).__name__.lower()

    return ("positive" in name) or ("greaterthan" in name)


def _build_site_table(model_trace):
    """
    Turn a Numpyro trace into a one-row-per-latent DataFrame.

    Observed sites and non-sample sites (plates, deterministics) are dropped.
    """

    rows = []
    for name, site in model_trace.items():

        if site["type"] != "sample":
            continue
        if site.get("is_observed", False):
            continue

        frames = _plate_frames(site)
        shape = tuple(int(s) for s in np.shape(site["value"]))
        scope, entity, per_datum = _site_scope(frames)

        rows.append({
            "site": name,
            "component": _site_component(name),
            "scope": scope,
            "entity": entity,
            "kind": "scale" if _is_scale_like(site["fn"]) else "location",
            "n_param": int(np.prod(shape)) if shape else 1,
            "scales_with_library": entity is not None,
            "per_datum": per_datum,
            "shape": str(shape),
            "plates": "; ".join(f"{n}={s}" for n, s in frames),
        })

    sites = pd.DataFrame(rows)
    if len(sites) == 0:
        sites["level"] = []
        return sites

    # A site is a hyperparameter when its component also has entity-indexed
    # sites but this site is not one of them -- i.e. it sits above the
    # individual-level latents in the hierarchy.  Components with no
    # entity-level sites at all (condition_growth, say) are all 'individual'.
    has_entity = sites.groupby("component")["scales_with_library"].transform("any")
    sites["level"] = np.where(
        has_entity & (~sites["scales_with_library"]),
        "hyperparameter",
        "individual",
    )

    return sites[["site", "component", "scope", "level", "kind", "n_param",
                  "scales_with_library", "per_datum", "entity", "shape",
                  "plates"]]


# -----------------------------------------------------------------------------
# Observation counting
#
# These mirror the masking each observer applies.  A test asserts they match
# the masks recovered from a concrete trace, so they cannot silently drift.
# -----------------------------------------------------------------------------

def _observation_counts(full_data):
    """
    Count unmasked observations per likelihood channel.

    Returns a dict mapping channel name to
    ``{"n_obs": int, "n_allocated": int}``, where n_allocated is the padded
    tensor size.
    """

    counts = {}

    def _record(key, mask):
        mask = np.asarray(mask)
        counts[key] = {"n_obs": int(mask.sum()), "n_allocated": int(mask.size)}

    growth = getattr(full_data, "growth", None)
    if growth is not None and getattr(growth, "good_mask", None) is not None:
        _record("growth", growth.good_mask)

    binding = getattr(full_data, "binding", None)
    if binding is not None and getattr(binding, "good_mask", None) is not None:
        _record("binding", binding.good_mask)

    # presplit and base_growth borrow the growth genotype batch state; index
    # their masks exactly as their observers do.
    if growth is not None:
        batch_idx = np.asarray(growth.batch_idx)

        presplit = getattr(full_data, "presplit", None)
        if presplit is not None and getattr(presplit, "good_mask", None) is not None:
            _record("presplit", np.asarray(presplit.good_mask)[:, :, batch_idx])

        base_growth = getattr(full_data, "base_growth", None)
        if base_growth is not None and getattr(base_growth, "good_mask", None) is not None:
            _record("base_growth", np.asarray(base_growth.good_mask)[batch_idx])

    return counts


# -----------------------------------------------------------------------------
# Per-genotype coverage
# -----------------------------------------------------------------------------

def _axis_reduction(mask, dim_names, keep):
    """
    Collapse a mask down to the axes named in `keep`, using .any().

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean mask whose axes are named by `dim_names`.
    dim_names : list of str
        Axis names, same length as ``mask.ndim``.
    keep : tuple of str
        Names of the axes to retain.

    Returns
    -------
    numpy.ndarray
        Boolean array over the kept axes, in the order they appear in
        `dim_names`.
    """

    keep_idx = tuple(i for i, n in enumerate(dim_names) if n in keep)
    drop_idx = tuple(i for i in range(mask.ndim) if i not in keep_idx)

    return mask.any(axis=drop_idx)


def _per_genotype_coverage(orchestrator, full_data):
    """
    Build the per-genotype coverage table.

    One row per genotype in the growth library, recording how many
    observations constrain it and how much of the titration / condition /
    replicate design it actually spans.  A genotype measured at only two
    distinct titrant concentrations cannot support a four-parameter Hill
    curve no matter how many timepoints back it up, which is why the
    distinct-level counts are tracked separately from the raw observation
    count.
    """

    growth_tm = orchestrator.growth_tm
    if growth_tm is None:
        return pd.DataFrame()

    dim_names = list(growth_tm.tensor_dim_names)
    labels = list(growth_tm.tensor_dim_labels[-1])
    mask = np.asarray(full_data.growth.good_mask).astype(bool)

    n_geno = mask.shape[-1]
    out = {"genotype": labels[:n_geno]}

    # Raw observation count per genotype.
    out["n_obs_growth"] = mask.reshape(-1, n_geno).sum(axis=0)

    # Distinct design levels covered, per genotype.  Each is "how many
    # distinct values of this axis have at least one good observation".
    for axis_name, col in (("titrant_conc", "n_titrant_conc"),
                           ("replicate", "n_replicate"),
                           ("time", "n_timepoint")):
        if axis_name in dim_names:
            reduced = _axis_reduction(mask, dim_names, (axis_name, "genotype"))
            # reduced axes are ordered as in dim_names; genotype is last
            out[col] = reduced.sum(axis=0)

    # Conditions are the (condition_pre, condition_sel) cross.
    cond_axes = tuple(n for n in ("condition_pre", "condition_sel") if n in dim_names)
    if cond_axes:
        reduced = _axis_reduction(mask, dim_names, cond_axes + ("genotype",))
        out["n_condition"] = reduced.reshape(-1, n_geno).sum(axis=0)

    coverage = pd.DataFrame(out)

    # Binding observations, joined on genotype label (the binding tensor
    # carries its own, smaller, genotype axis).
    binding_tm = getattr(orchestrator, "binding_tm", None)
    coverage["n_obs_binding"] = 0
    if binding_tm is not None and getattr(full_data, "binding", None) is not None:
        b_mask = np.asarray(full_data.binding.good_mask).astype(bool)
        b_labels = list(binding_tm.tensor_dim_labels[-1])
        b_counts = b_mask.reshape(-1, b_mask.shape[-1]).sum(axis=0)
        b_map = dict(zip(b_labels, b_counts))
        coverage["n_obs_binding"] = [
            int(b_map.get(g, 0)) for g in coverage["genotype"]
        ]

    # Side-channel anchors, per genotype.
    batch_idx = np.asarray(full_data.growth.batch_idx)

    coverage["n_obs_presplit"] = 0
    presplit = getattr(full_data, "presplit", None)
    if presplit is not None and getattr(presplit, "good_mask", None) is not None:
        p_mask = np.asarray(presplit.good_mask)[:, :, batch_idx].astype(bool)
        coverage["n_obs_presplit"] = p_mask.reshape(-1, n_geno).sum(axis=0)

    coverage["n_obs_base_growth"] = 0
    base_growth = getattr(full_data, "base_growth", None)
    if base_growth is not None and getattr(base_growth, "good_mask", None) is not None:
        bg_mask = np.asarray(base_growth.good_mask)[batch_idx].astype(bool)
        coverage["n_obs_base_growth"] = bg_mask.reshape(n_geno, -1).sum(axis=1)

    spiked = orchestrator.settings.get("spiked_genotypes") or []
    coverage["is_spiked"] = coverage["genotype"].isin(list(spiked))

    return coverage


def _anchor_inventory(orchestrator, coverage):
    """
    Count the measurements that pin the model's weakly-identified directions.

    The k / dk_geno / k_ref additive slide is only broken by data that
    measures one of those terms directly: binding curves (which pin theta and
    hence the growth slope m), direct reference-condition growth rates, and
    externally pinned dk_geno values.  Congression-free (spiked) binding
    genotypes and in-library ones play different roles, so they are counted
    separately.
    """

    settings = orchestrator.settings
    spiked = set(settings.get("spiked_genotypes") or [])

    inventory = {
        "n_genotype": int(len(coverage)) if len(coverage) else 0,
        "n_binding_genotype": 0,
        "n_binding_genotype_spiked": 0,
        "n_binding_genotype_in_library": 0,
        "n_base_growth_genotype": 0,
        "n_presplit_genotype": 0,
        "n_spiked_genotype": len(spiked),
        "n_dk_geno_pinned": 0,
    }

    if len(coverage):
        has_binding = coverage["n_obs_binding"] > 0
        inventory["n_binding_genotype"] = int(has_binding.sum())
        inventory["n_binding_genotype_spiked"] = \
            int((has_binding & coverage["is_spiked"]).sum())
        inventory["n_binding_genotype_in_library"] = \
            int((has_binding & (~coverage["is_spiked"])).sum())
        inventory["n_base_growth_genotype"] = \
            int((coverage["n_obs_base_growth"] > 0).sum())
        inventory["n_presplit_genotype"] = \
            int((coverage["n_obs_presplit"] > 0).sum())

    dk_values = getattr(orchestrator, "_dk_geno_values", None)
    if dk_values is not None:
        inventory["n_dk_geno_pinned"] = int(np.size(np.asarray(dk_values)))

    return inventory


# -----------------------------------------------------------------------------
# Summary assembly
# -----------------------------------------------------------------------------

def _entity_params_per_unit(sites, entity, n_unit):
    """
    Parameters carried by each instance of `entity` (0 if none/undefined).

    Per-datum latents are excluded: they are random effects that grow with the
    number of measurements, so counting them against a genotype's observation
    budget would double-count the data on both sides of the comparison.
    """

    if len(sites) == 0 or not n_unit:
        return 0.0

    keep = (sites["entity"] == entity) & (~sites["per_datum"])
    total = sites.loc[keep, "n_param"].sum()

    return float(total) / float(n_unit)


def _quantiles(values, levels=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)):
    """Quantile summary of an integer coverage vector, as a plain dict."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}

    return {f"q{level}": float(np.quantile(values, level)) for level in levels}


def _build_summary(orchestrator, sites, coverage, obs_counts, trace_mode,
                   n_variational):
    """Assemble the nested summary dictionary."""

    settings = orchestrator.settings
    data = orchestrator.data

    n_obs_total = int(sum(c["n_obs"] for c in obs_counts.values()))
    n_allocated = int(sum(c["n_allocated"] for c in obs_counts.values()))

    n_param_total = int(sites["n_param"].sum()) if len(sites) else 0
    n_per_datum = int(sites.loc[sites["per_datum"], "n_param"].sum()) if len(sites) else 0
    n_scale = int(sites.loc[sites["kind"] == "scale", "n_param"].sum()) if len(sites) else 0

    if len(sites):
        pooled = sites["scales_with_library"] & (~sites["per_datum"])
        p_lower = int(sites.loc[~(pooled | sites["per_datum"]), "n_param"].sum())
    else:
        p_lower = 0
    p_upper = n_param_total - n_per_datum

    def _ratio(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else float("nan")

    n_genotype = int(getattr(data, "num_genotype", 0) or 0)
    n_mutation = int(getattr(data.growth, "num_mutation", 0) or 0) \
        if getattr(data, "growth", None) is not None else 0

    params_per_genotype = _entity_params_per_unit(sites, "genotype", n_genotype)
    params_per_mutation = _entity_params_per_unit(sites, "mutation", n_mutation)

    theta_sites = sites[sites["component"] == "theta"] if len(sites) else sites
    theta_per_genotype = _entity_params_per_unit(theta_sites, "genotype", n_genotype)

    summary = {
        "components": {k: v for k, v in settings.items()},
        "trace_mode": trace_mode,
        "observations": {
            "by_channel": obs_counts,
            "n_obs_total": n_obs_total,
            "n_allocated_total": n_allocated,
            "padding_fraction": 1.0 - _ratio(n_obs_total, n_allocated),
        },
        "parameters": {
            "n_param_total": n_param_total,
            "n_param_scale_like": n_scale,
            "n_param_location_like": n_param_total - n_scale,
            "n_param_per_datum": n_per_datum,
            "by_component": (
                sites.groupby("component")["n_param"].sum().astype(int).to_dict()
                if len(sites) else {}
            ),
            "by_scope": (
                sites.groupby("scope")["n_param"].sum().astype(int).to_dict()
                if len(sites) else {}
            ),
            "by_level": (
                sites.groupby("level")["n_param"].sum().astype(int).to_dict()
                if len(sites) else {}
            ),
            "n_variational_param": n_variational,
        },
        "effective_parameters": {
            "note": (
                "Partial pooling means each entity-level latent costs strictly "
                "less than one degree of freedom, so the effective parameter "
                "count lies inside this bracket. A real p_eff (WAIC / PSIS-LOO) "
                "requires a fitted model. Per-datum latents are excluded from "
                "both bounds -- they are random effects that grow with the "
                "data, not free parameters."
            ),
            "p_eff_lower": p_lower,
            "p_eff_upper": p_upper,
            "obs_per_param_lower_bound": _ratio(n_obs_total, p_upper),
            "obs_per_param_upper_bound": _ratio(n_obs_total, p_lower),
        },
        "library": {
            "n_genotype": n_genotype,
            "n_mutation": n_mutation,
            "params_per_genotype": params_per_genotype,
            "params_per_mutation": params_per_mutation,
            "theta_params_per_genotype": theta_per_genotype,
        },
        "batching": {
            "batch_size": settings.get("batch_size"),
            "n_batch_per_epoch": (
                int(np.ceil(n_genotype / settings["batch_size"]))
                if settings.get("batch_size") else 1
            ),
        },
    }

    if len(coverage):
        n_under = int((coverage["n_obs_growth"] < params_per_genotype).sum())
        n_no_data = int((coverage["n_obs_growth"] == 0).sum())
        conc = coverage.get("n_titrant_conc")
        n_theta_under = (
            int((conc < theta_per_genotype).sum())
            if conc is not None and theta_per_genotype > 0 else 0
        )

        summary["coverage"] = {
            "n_obs_growth_per_genotype": _quantiles(coverage["n_obs_growth"]),
            "n_titrant_conc_per_genotype": (
                _quantiles(conc) if conc is not None else {}
            ),
            "n_genotype_no_growth_data": n_no_data,
            "n_genotype_under_determined": n_under,
            "frac_genotype_under_determined": _ratio(n_under, len(coverage)),
            "n_genotype_theta_under_determined": n_theta_under,
            "frac_genotype_theta_under_determined": _ratio(n_theta_under, len(coverage)),
        }

    summary["anchors"] = _anchor_inventory(orchestrator, coverage)

    return summary


def _build_warnings(sites, coverage, summary):
    """Turn the summary into a list of human-readable flags."""

    warnings = []
    cov = summary.get("coverage", {})
    anchors = summary.get("anchors", {})
    lib = summary["library"]

    if cov.get("n_genotype_no_growth_data", 0):
        warnings.append(
            f"{cov['n_genotype_no_growth_data']} genotype(s) have no unmasked "
            f"growth observations at all."
        )

    if cov.get("n_genotype_under_determined", 0):
        warnings.append(
            f"{cov['n_genotype_under_determined']} genotype(s) "
            f"({100 * cov['frac_genotype_under_determined']:.1f}%) have fewer "
            f"growth observations than the "
            f"{lib['params_per_genotype']:.3g} parameters carried per genotype."
        )

    if cov.get("n_genotype_theta_under_determined", 0):
        warnings.append(
            f"{cov['n_genotype_theta_under_determined']} genotype(s) "
            f"({100 * cov['frac_genotype_theta_under_determined']:.1f}%) are "
            f"measured at fewer distinct titrant concentrations than the "
            f"{lib['theta_params_per_genotype']:.3g} theta parameters they "
            f"carry; their theta curve is not identified by growth data alone."
        )

    if anchors.get("n_binding_genotype", 0) == 0:
        warnings.append(
            "No genotype has binding observations. Nothing pins the growth "
            "slope m independently of theta."
        )

    if anchors.get("n_base_growth_genotype", 0) == 0:
        warnings.append(
            "No base_growth measurements. The k / dk_geno / k_ref additive "
            "direction is anchored only by priors and the wt dk_geno pin."
        )

    n_per_datum = summary["parameters"]["n_param_per_datum"]
    if n_per_datum:
        per_datum_sites = ", ".join(sorted(sites.loc[sites["per_datum"], "component"].unique()))
        warnings.append(
            f"{n_per_datum} latent(s) are indexed per observation (from: "
            f"{per_datum_sites}). These are random effects, not free "
            f"parameters; they are excluded from the effective-parameter "
            f"bracket."
        )

    padding = summary["observations"]["padding_fraction"]
    if padding > 0.5:
        warnings.append(
            f"{100 * padding:.1f}% of the allocated observation tensor is "
            f"padding. The design is ragged; memory use is dominated by cells "
            f"with no data."
        )

    return warnings


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def count_model_dimensions(orchestrator):
    """
    Count the parameters and observations of a configured model.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        A fully constructed orchestrator. Its component choices, data, and
        priors define everything counted here.

    Returns
    -------
    ModelStats
        Per-site parameter table, per-genotype coverage table, summary
        dictionary, and warning list.

    Notes
    -----
    The model is traced with the *full* genotype batch, never a mini-batch.
    Tracing a mini-batch would report every per-genotype parameter count as
    the batch size rather than the library size.
    """

    data = jax.device_put(orchestrator.data)
    full_data = orchestrator.get_batch(
        data, jnp.arange(orchestrator.data.num_genotype)
    )

    model_trace, trace_mode = _run_trace(
        orchestrator.jax_model, full_data, orchestrator.priors
    )
    sites = _build_site_table(model_trace)

    # The guide's pyro.param sites are the values SVI actually optimizes --
    # typically a loc and a scale per latent. Counting them is the honest
    # answer for "how big is the thing being fit".
    n_variational = None
    try:
        guide_trace, _ = _run_trace(
            orchestrator.jax_model_guide, full_data, orchestrator.priors
        )
        n_variational = int(sum(
            np.prod(np.shape(s["value"])) if np.shape(s["value"]) else 1
            for s in guide_trace.values() if s["type"] == "param"
        ))
    except Exception:
        pass

    obs_counts = _observation_counts(full_data)
    coverage = _per_genotype_coverage(orchestrator, full_data)

    summary = _build_summary(orchestrator, sites, coverage, obs_counts,
                             trace_mode, n_variational)
    warnings = _build_warnings(sites, coverage, summary)

    return ModelStats(sites=sites, per_genotype=coverage, summary=summary,
                      warnings=warnings)


def format_model_stats(stats):
    """
    Render a ModelStats as a compact block of text for stdout.

    Parameters
    ----------
    stats : ModelStats

    Returns
    -------
    str
    """

    s = stats.summary
    lines = []

    def _header(text):
        lines.append("")
        lines.append(text)
        lines.append("-" * len(text))

    lines.append("")
    lines.append("=" * 66)
    lines.append("MODEL DIMENSION SUMMARY")
    lines.append("=" * 66)

    _header("Observations")
    for channel, counts in s["observations"]["by_channel"].items():
        lines.append(f"  {channel:<16s} {counts['n_obs']:>12,d}")
    lines.append(f"  {'TOTAL':<16s} {s['observations']['n_obs_total']:>12,d}"
                 f"   ({100 * s['observations']['padding_fraction']:.1f}% of the "
                 f"allocated tensor is padding)")

    _header("Parameters by component")
    for component, n in sorted(s["parameters"]["by_component"].items(),
                               key=lambda kv: -kv[1]):
        lines.append(f"  {component:<24s} {n:>12,d}")
    lines.append(f"  {'TOTAL':<24s} {s['parameters']['n_param_total']:>12,d}")

    _header("Parameters by what they are indexed by")
    for scope, n in sorted(s["parameters"]["by_scope"].items(),
                           key=lambda kv: -kv[1]):
        lines.append(f"  {scope:<24s} {n:>12,d}")

    _header("Parameters by role")
    for level, n in sorted(s["parameters"]["by_level"].items()):
        lines.append(f"  {level:<24s} {n:>12,d}")
    lines.append(f"  {'scale-like (of total)':<24s} "
                 f"{s['parameters']['n_param_scale_like']:>12,d}")
    if s["parameters"]["n_variational_param"] is not None:
        lines.append(f"  {'SVI variational params':<24s} "
                     f"{s['parameters']['n_variational_param']:>12,d}")

    ep = s["effective_parameters"]
    _header("Effective parameters (bracket)")
    lines.append(f"  p_eff is between {ep['p_eff_lower']:,d} and "
                 f"{ep['p_eff_upper']:,d}")
    lines.append(f"  observations per parameter: "
                 f"{ep['obs_per_param_lower_bound']:.1f} to "
                 f"{ep['obs_per_param_upper_bound']:.1f}")
    lines.append("  Partial pooling makes each entity-level latent cost less")
    lines.append("  than a full degree of freedom, so the truth is inside the")
    lines.append("  bracket. A real p_eff needs a fitted model.")

    if "coverage" in s:
        cov = s["coverage"]
        lib = s["library"]
        _header("Per-genotype coverage")
        lines.append(f"  parameters carried per genotype: "
                     f"{lib['params_per_genotype']:.3g} "
                     f"(theta: {lib['theta_params_per_genotype']:.3g})")
        q = cov["n_obs_growth_per_genotype"]
        if q:
            lines.append(f"  growth obs per genotype: min {q['q0.0']:.0f}, "
                         f"median {q['q0.5']:.0f}, max {q['q1.0']:.0f}")
        q = cov["n_titrant_conc_per_genotype"]
        if q:
            lines.append(f"  distinct [titrant] per genotype: min {q['q0.0']:.0f}, "
                         f"median {q['q0.5']:.0f}, max {q['q1.0']:.0f}")
        lines.append(f"  under-determined genotypes: "
                     f"{cov['n_genotype_under_determined']:,d} "
                     f"({100 * cov['frac_genotype_under_determined']:.1f}%)")

    _header("Anchors")
    a = s["anchors"]
    lines.append(f"  binding genotypes: {a['n_binding_genotype']:,d} "
                 f"({a['n_binding_genotype_spiked']} spiked, "
                 f"{a['n_binding_genotype_in_library']} in-library)")
    lines.append(f"  base_growth genotypes: {a['n_base_growth_genotype']:,d}")
    lines.append(f"  presplit genotypes: {a['n_presplit_genotype']:,d}")
    lines.append(f"  pinned dk_geno values: {a['n_dk_geno_pinned']:,d}")

    if stats.warnings:
        _header("Flags")
        for warning in stats.warnings:
            lines.append(f"  * {warning}")

    lines.append("")
    lines.append("=" * 66)
    lines.append("")

    return "\n".join(lines)


def write_model_stats(stats, out_prefix):
    """
    Write the per-site table and the summary to disk.

    Writes ``{out_prefix}_model_stats.csv`` (one row per latent sample site)
    and ``{out_prefix}_model_stats.json`` (headline counts, coverage
    quantiles, anchors, and warnings).

    Parameters
    ----------
    stats : ModelStats
    out_prefix : str

    Returns
    -------
    tuple of str
        The two paths written.
    """

    csv_path = f"{out_prefix}_model_stats.csv"
    json_path = f"{out_prefix}_model_stats.json"

    stats.sites.to_csv(csv_path, index=False)

    payload = dict(stats.summary)
    payload["warnings"] = stats.warnings
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return csv_path, json_path
