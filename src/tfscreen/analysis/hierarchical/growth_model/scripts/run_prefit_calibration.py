"""
Pre-fit calibration runner for the two-stage hierarchical workflow.

The pre-fit's job is to produce empirical-Bayes pinned priors for the
*linking-function* components of the production model — currently
``condition_growth`` and ``growth_transition``.  All other components
(activity, dk_geno, ln_cfu0, theta, transformation, noise) are
deliberately replaced by their simplest forms during the pre-fit so the
linking-function MAP doesn't have to fight the full hierarchy:

- ``theta``                → ``simple`` (theta values pinned from binding data)
- ``activity``             → ``hierarchical`` with hyperparams *pinned* to their
                             prior locs (degenerate, no learning)
- ``dk_geno``              → ``hierarchical`` with hyperparams *pinned*
- ``ln_cfu0``              → ``hierarchical`` with hyperparams *pinned*
- ``transformation``       → ``single`` (no learning)
- ``theta_*_noise``        → ``zero`` (no learning)

The data is filtered to the (genotype, titrant_name, titrant_conc)
intersection of the production growth and binding inputs so the
calibration runs only against fully-observed cells.

After MAP convergence, the script:

1. Estimates a per-site 1-sigma uncertainty from the Hessian of the
   negative log-joint at the MAP point (delta-method propagated through
   any constrained-support bijections).
2. Identifies which fields of the production ``ModelPriors`` provide
   ``dist.loc`` / ``dist.scale`` for each calibrated sample site, using
   a sentinel-trace introspection so we don't have to hard-code the
   field-naming conventions of every component.
3. Updates the production ``{out_root}_guesses.csv`` *in place*, writing
   a ``.bak`` backup before overwriting.  For simple-prior
   ``condition_growth`` and ``growth_transition`` components, per-condition
   MAP estimates are written directly as per-condition guess values
   (``{site}_locs`` rows in the guesses CSV), giving the production SVI a
   warm start from the calibration fit.  Only rows belonging to
   ``condition_growth`` or ``growth_transition`` are touched.

   For legacy hierarchical components (if any), scalar hyper-site MAP
   estimates also update the corresponding rows in the priors CSV.
"""

import dataclasses
import os
import shutil
import sys

import numpy as np
import pandas as pd
import jax.numpy as jnp
import optax

import yaml

from numpyro.handlers import seed, trace

import tfscreen
from tfscreen.util.cli.generalized_main import generalized_main
from tfscreen.analysis.hierarchical.run_inference import RunInference
from tfscreen.analysis.hierarchical.growth_model import GrowthModel
from tfscreen.analysis.hierarchical.growth_model.configuration_io import (
    read_configuration,
)


# ---------------------------------------------------------------------------
# Hardcoded calibration component overrides
#
# These are the components that get *replaced* relative to the production
# YAML.  ``condition_growth`` and ``growth_transition`` are intentionally
# absent from this dict — their production choices flow through unchanged
# so the calibration MAP can refine the production priors.
# ---------------------------------------------------------------------------
_CALIBRATION_OVERRIDES = {
    "theta": "simple",
    "activity": "hierarchical",
    "dk_geno": "hierarchical",
    "ln_cfu0": "hierarchical",
    "transformation": "single",
    "theta_growth_noise": "zero",
    "theta_binding_noise": "zero",
}

# Components that contribute pinnable hyperparameters during the pre-fit.
# Each entry maps the component-name → ((sample-site suffix, prior field
# providing the pinned value), ...).  These match the
# ``_PINNABLE_SUFFIXES`` declared in each component module.
_PINNED_COMPONENTS = {
    "activity": (
        ("hyper_loc",   "hyper_loc_loc"),
        ("hyper_scale", "hyper_scale_loc"),
    ),
    "dk_geno": (
        ("hyper_loc",   "hyper_loc_loc"),
        ("hyper_scale", "hyper_scale_loc"),
        ("hyper_shift", "hyper_shift_loc"),
    ),
    "ln_cfu0": (
        ("hyper_loc",   "ln_cfu0_hyper_loc_loc"),
        ("hyper_scale", "ln_cfu0_hyper_scale_loc"),
    ),
}

# Theta values are clipped to ``[_THETA_EPS, 1 - _THETA_EPS]`` so the
# downstream logit transform (in the simple-theta component) is finite.
_THETA_EPS = 1e-6


# ---------------------------------------------------------------------------
# Data filtering
# ---------------------------------------------------------------------------

def _intersect_data(growth_df, binding_df):
    """
    Filter the production growth and binding DataFrames to their shared
    (genotype, titrant_name, titrant_conc) cells.

    Both inputs are read with :func:`tfscreen.util.io.read_dataframe` so a
    file path or DataFrame is acceptable.  Genotypes are *not* coerced to
    a categorical here — that happens inside ``GrowthModel`` — so any
    string/category mismatch surfaces as a missing intersection rather
    than a silent miscompare.

    Returns
    -------
    (growth_df_cal, binding_df_cal) : tuple[pd.DataFrame, pd.DataFrame]
        Filtered copies, with the original index preserved.

    Raises
    ------
    ValueError
        If the intersection is empty.
    """
    growth_df = tfscreen.util.io.read_dataframe(growth_df)
    binding_df = tfscreen.util.io.read_dataframe(binding_df)

    cols = ["genotype", "titrant_name", "titrant_conc"]
    for c in cols:
        if c not in growth_df.columns:
            raise ValueError(
                f"growth_df is missing required column '{c}' for calibration intersection."
            )
        if c not in binding_df.columns:
            raise ValueError(
                f"binding_df is missing required column '{c}' for calibration intersection."
            )

    growth_keys = (growth_df[cols]
                   .astype({"genotype": str, "titrant_name": str})
                   .drop_duplicates()
                   .set_index(cols).index)
    binding_keys = (binding_df[cols]
                    .astype({"genotype": str, "titrant_name": str})
                    .drop_duplicates()
                    .set_index(cols).index)
    shared = growth_keys.intersection(binding_keys)

    if len(shared) == 0:
        raise ValueError(
            "Calibration intersection is empty: no (genotype, titrant_name, "
            "titrant_conc) cell is present in both growth_df and binding_df."
        )

    growth_idx = growth_df.set_index(cols).index.isin(shared)
    binding_idx = binding_df.set_index(cols).index.isin(shared)

    return growth_df.loc[growth_idx].copy(), binding_df.loc[binding_idx].copy()


# ---------------------------------------------------------------------------
# theta_values for the simple component
# ---------------------------------------------------------------------------

def _compute_theta_values(gm_cal, binding_df_cal):
    """
    Build a (num_titrant_name, num_titrant_conc) theta tensor for the
    simple-theta component of the calibration model.

    Each cell is the inverse-variance weighted mean of ``theta_obs``
    across the genotypes that contributed observations at that
    (titrant_name, titrant_conc) cell.  Cells with no usable observations
    fall back to a plain mean of ``theta_obs``; if that is also empty the
    cell is set to 0.5 (uninformative midpoint).

    The dimension ordering is taken from
    ``gm_cal.binding_tm.tensor_dim_labels`` so the resulting array
    matches the layout the simple-theta component expects.
    """
    tn_idx = gm_cal.binding_tm.tensor_dim_names.index("titrant_name")
    tc_idx = gm_cal.binding_tm.tensor_dim_names.index("titrant_conc")
    titrant_name_labels = list(gm_cal.binding_tm.tensor_dim_labels[tn_idx])
    titrant_conc_labels = list(gm_cal.binding_tm.tensor_dim_labels[tc_idx])

    n_name = len(titrant_name_labels)
    n_conc = len(titrant_conc_labels)
    theta_values = np.full((n_name, n_conc), 0.5, dtype=float)

    name_to_i = {str(n): i for i, n in enumerate(titrant_name_labels)}
    conc_to_j = {float(c): j for j, c in enumerate(titrant_conc_labels)}

    grouped = binding_df_cal.groupby(["titrant_name", "titrant_conc"],
                                     observed=True)
    for (tn, tc), grp in grouped:
        i = name_to_i.get(str(tn))
        j = conc_to_j.get(float(tc))
        if i is None or j is None:
            continue

        theta_obs = grp["theta_obs"].to_numpy(dtype=float)
        theta_std = grp["theta_std"].to_numpy(dtype=float)
        valid = (np.isfinite(theta_obs) & np.isfinite(theta_std)
                 & (theta_std > 0))
        if valid.any():
            w = 1.0 / np.square(theta_std[valid])
            theta_values[i, j] = float(np.sum(theta_obs[valid] * w) / np.sum(w))
        else:
            usable = np.isfinite(theta_obs)
            if usable.any():
                theta_values[i, j] = float(np.mean(theta_obs[usable]))

    np.clip(theta_values, _THETA_EPS, 1.0 - _THETA_EPS, out=theta_values)
    return jnp.asarray(theta_values, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Calibration model construction
# ---------------------------------------------------------------------------

def _build_calibration_model(gm_prod, growth_df_cal, binding_df_cal):
    """
    Construct the calibration ``GrowthModel`` with hardcoded calibration
    overrides applied on top of the production component selections.

    ``condition_growth`` and ``growth_transition`` carry through from the
    production config (the whole point of the pre-fit is to refine
    *those* priors).  Spiked genotypes are dropped: the calibration only
    sees the (calibration-genotype × calibration-condition) intersection,
    which by construction does not include the production's spiked rows.
    """
    settings = dict(gm_prod.settings)
    for k, v in _CALIBRATION_OVERRIDES.items():
        settings[k] = v
    settings["spiked_genotypes"] = None
    batch_size = settings.pop("batch_size", None)

    return GrowthModel(growth_df_cal,
                       binding_df_cal,
                       batch_size=batch_size,
                       **settings)


def _inject_calibration_priors(gm_cal, gm_prod, theta_values):
    """
    Wire the calibration model's priors into shape:

    - ``condition_growth`` and ``growth_transition`` get the production
      priors (so the MAP starts from production hyperparam settings),
      with ``pinned`` cleared so MAP can refine them.
    - ``theta`` gets the supplied empirical ``theta_values`` (simple
      component is always-pinned).
    - ``activity``, ``dk_geno`` and ``ln_cfu0`` get every entry in
      ``_PINNED_COMPONENTS`` written into their ``pinned`` dict, fixing
      those hyperparameters to their prior loc / scale defaults so the
      MAP doesn't try to learn them from calibration-only data.

    Mutates ``gm_cal._priors`` in place.
    """
    prod_growth = gm_prod.priors.growth
    cal_growth = gm_cal.priors.growth

    cg_prod = prod_growth.condition_growth
    cg_cal = cal_growth.condition_growth
    if hasattr(cg_prod, "replace") and hasattr(cg_cal, "replace"):
        # Carry the production scalar fields onto the calibration object
        # without disturbing whatever pytree-static fields cal added
        # (e.g. its empty `pinned` dict).
        cg_updates = {}
        for f in dataclasses.fields(cg_prod):
            if f.name == "pinned":
                continue
            if hasattr(cg_cal, f.name):
                cg_updates[f.name] = getattr(cg_prod, f.name)
        if any(f.name == "pinned" for f in dataclasses.fields(cg_cal)):
            cg_cal_new = cg_cal.replace(**cg_updates, pinned={})
        elif cg_updates:
            cg_cal_new = cg_cal.replace(**cg_updates)
        else:
            cg_cal_new = cg_cal
    else:
        cg_cal_new = cg_cal

    gt_prod = prod_growth.growth_transition
    gt_cal = cal_growth.growth_transition
    if hasattr(gt_prod, "replace") and hasattr(gt_cal, "replace"):
        gt_updates = {}
        for f in dataclasses.fields(gt_prod):
            if f.name == "pinned":
                continue
            if hasattr(gt_cal, f.name):
                gt_updates[f.name] = getattr(gt_prod, f.name)
        # Some growth_transition variants (e.g. instant) have no `pinned`
        # field at all; only set it when the dataclass exposes it.
        if any(f.name == "pinned" for f in dataclasses.fields(gt_cal)):
            gt_cal_new = gt_cal.replace(**gt_updates, pinned={})
        elif gt_updates:
            gt_cal_new = gt_cal.replace(**gt_updates)
        else:
            gt_cal_new = gt_cal
    else:
        gt_cal_new = gt_cal

    # Pin activity / dk_geno / ln_cfu0 hyperparams to their prior locs.
    # This produces a degenerate (delta) hierarchy that contributes
    # nothing to the calibration MAP gradient.
    pinned_components = {}
    for comp_name, suffix_field_pairs in _PINNED_COMPONENTS.items():
        comp = getattr(cal_growth, comp_name, None)
        if comp is None:
            continue
        pinned_dict = {}
        for suffix, field_name in suffix_field_pairs:
            if hasattr(comp, field_name):
                pinned_dict[suffix] = float(getattr(comp, field_name))
        if pinned_dict and hasattr(comp, "replace"):
            pinned_components[comp_name] = comp.replace(pinned=pinned_dict)

    growth_updates = {"condition_growth": cg_cal_new,
                      "growth_transition": gt_cal_new}
    growth_updates.update(pinned_components)
    new_growth = cal_growth.replace(**growth_updates)

    # theta priors live on PriorsClass.theta
    theta_priors = gm_cal.priors.theta
    if hasattr(theta_priors, "replace"):
        new_theta = theta_priors.replace(theta_values=theta_values)
    else:
        new_theta = theta_priors

    new_priors = gm_cal.priors.replace(growth=new_growth, theta=new_theta)
    gm_cal._priors = new_priors


# ---------------------------------------------------------------------------
# Field mapping introspection
# ---------------------------------------------------------------------------

def _identify_field_mapping(gm_cal):
    """
    Enumerate the ``condition_growth`` and ``growth_transition`` sample sites
    and derive their ``ModelPriors`` field names.

    Simple-prior components expose per-condition array sites following the
    convention:

    - Site ``{component}_{x}`` → Normal distribution;
      ``loc_field = {x}_loc``, ``scale_field = {x}_scale``.
      ``is_array`` is True when the site holds a per-condition array.

    Legacy hierarchical components (if any remain) instead expose scalar
    hyper-parameter sites:

    - Site ``{component}_{x}_hyper_loc`` → Normal;
      ``loc_field = {x}_hyper_loc_loc``, ``scale_field = {x}_hyper_loc_scale``.
    - Site ``{component}_{x}_hyper_scale`` → HalfNormal;
      ``scale_field = {x}_hyper_scale_loc``.

    Returns
    -------
    dict[str, dict]
        Site name → ``{"component", "dist_class", "is_array", ...field names...}``.
    """
    model_trace = trace(seed(gm_cal.jax_model, rng_seed=0)).get_trace(
        data=gm_cal.data, priors=gm_cal.priors
    )

    out = {}
    for site_name, site in model_trace.items():
        if site["type"] != "sample" or site.get("is_observed", False):
            continue

        if site_name.startswith("condition_growth_"):
            component = "condition_growth"
            suffix = site_name[len("condition_growth_"):]
        elif site_name.startswith("growth_transition_"):
            component = "growth_transition"
            suffix = site_name[len("growth_transition_"):]
        else:
            continue

        try:
            is_array = np.shape(np.asarray(site["value"])) != ()
        except Exception:
            continue

        if suffix.endswith("_hyper_loc"):
            # Legacy hierarchical scalar site
            out[site_name] = {
                "component": component,
                "dist_class": "Normal",
                "loc_field": f"{suffix}_loc",
                "scale_field": f"{suffix}_scale",
                "is_array": False,
            }
        elif suffix.endswith("_hyper_scale"):
            # Legacy hierarchical scalar site
            out[site_name] = {
                "component": component,
                "dist_class": "HalfNormal",
                "scale_field": f"{suffix}_loc",
                "is_array": False,
            }
        else:
            # Simple-prior per-condition array site
            out[site_name] = {
                "component": component,
                "dist_class": "Normal",
                "loc_field": f"{suffix}_loc",
                "scale_field": f"{suffix}_scale",
                "is_array": is_array,
            }

    return out


# ---------------------------------------------------------------------------
# CSV update logic
# ---------------------------------------------------------------------------

def _csv_row_name(component, field_name):
    """Produce the dotted row name used in the production priors CSV."""
    return f"growth.{component}.{field_name}"


def _build_csv_updates(field_mapping, hessian_results):
    """
    Translate the sentinel-trace mapping plus per-site MAP / sigma
    estimates into two flat dicts of in-place updates:

    Returns
    -------
    prior_updates : dict[str, float]
        ``{csv_row_name → new_value}`` for the priors CSV.
        Normal sites contribute two rows (loc field ← MAP, scale field ←
        Hessian sigma); HalfNormal contributes one (scale field ← MAP,
        recentering the prior on the MAP point).
    guess_updates : dict[str, float]
        ``{site_name → MAP value}`` for the guesses CSV.  Only scalar
        sites are included.
    """
    prior_updates = {}
    guess_updates = {}

    for site_name, info in field_mapping.items():
        if site_name not in hessian_results:
            continue
        result = hessian_results[site_name]
        map_val_arr = np.asarray(result["map"])
        sigma_arr = np.asarray(result["sigma"])

        component = info["component"]
        dist_class = info["dist_class"]
        loc_field = info.get("loc_field")
        scale_field = info.get("scale_field")
        is_array = info.get("is_array", False)

        if is_array:
            # Simple-prior per-condition array site.
            # Write per-condition MAP estimates to guesses so the production
            # SVI starts from the calibration fit.  Priors are left unchanged.
            if loc_field is not None:
                guess_updates[f"{site_name}_locs"] = map_val_arr
        else:
            # Scalar site (legacy hierarchical components).
            if map_val_arr.shape != ():
                continue
            map_val = float(map_val_arr)
            sigma_val = float(sigma_arr) if sigma_arr.shape == () else None

            if dist_class == "Normal":
                if loc_field is not None:
                    prior_updates[_csv_row_name(component, loc_field)] = map_val
                if scale_field is not None and sigma_val is not None:
                    prior_updates[_csv_row_name(component, scale_field)] = sigma_val
            elif dist_class == "HalfNormal":
                if scale_field is not None:
                    # Recenter the HalfNormal on the MAP point.
                    prior_updates[_csv_row_name(component, scale_field)] = map_val
            else:
                if loc_field is not None:
                    prior_updates[_csv_row_name(component, loc_field)] = map_val
                if scale_field is not None and sigma_val is not None:
                    prior_updates[_csv_row_name(component, scale_field)] = sigma_val

            guess_updates[site_name] = map_val

    return prior_updates, guess_updates


def _apply_priors_updates(priors_path, prior_updates):
    """
    Overwrite ``parameter == row_name`` rows of the production priors
    CSV with the new values.  Writes a ``.bak`` copy first.  Rows whose
    ``parameter`` is not present in ``prior_updates`` are preserved
    unchanged.  Logs a warning for any update key that has no matching
    row.
    """
    if not prior_updates:
        return
    df = pd.read_csv(priors_path)
    if "parameter" not in df.columns or "value" not in df.columns:
        raise ValueError(
            f"Priors CSV {priors_path} is missing required 'parameter' / "
            "'value' columns."
        )

    matched = set()
    for row_name, new_val in prior_updates.items():
        mask = df["parameter"] == row_name
        if mask.any():
            df.loc[mask, "value"] = new_val
            matched.add(row_name)

    missing = sorted(set(prior_updates) - matched)
    if missing:
        print(
            f"  warning: {len(missing)} prior update(s) had no matching row "
            f"in {priors_path}: {missing}",
            file=sys.stderr,
        )

    shutil.copy2(priors_path, priors_path + ".bak")
    df.to_csv(priors_path, index=False)
    print(f"Updated {len(matched)} priors row(s) in {priors_path}")


def _apply_guesses_updates(guesses_path, guess_updates):
    """
    Overwrite rows in the production guesses CSV with new MAP values.

    Scalar values (0-d) target the ``flat_index``-is-NaN row for that
    parameter.  Array values target the ``flat_index == i`` rows for
    ``i`` in ``range(len(value))``, giving the production SVI a warm
    start from the per-condition calibration estimates.  Writes a
    ``.bak`` copy first.
    """
    if not guess_updates:
        return
    df = pd.read_csv(guesses_path)
    if "parameter" not in df.columns or "value" not in df.columns:
        raise ValueError(
            f"Guesses CSV {guesses_path} is missing required 'parameter' / "
            "'value' columns."
        )

    has_flat_index = "flat_index" in df.columns
    if has_flat_index:
        scalar_mask = df["flat_index"].isna()
    else:
        scalar_mask = pd.Series(True, index=df.index)

    matched = set()
    for site_name, new_val in guess_updates.items():
        new_val_arr = np.asarray(new_val)
        if new_val_arr.ndim == 0:
            # Scalar update: target the flat_index-is-NaN row.
            row_mask = scalar_mask & (df["parameter"] == site_name)
            if row_mask.any():
                df.loc[row_mask, "value"] = float(new_val_arr)
                matched.add(site_name)
        else:
            # Array update: match each element by flat_index.
            if not has_flat_index:
                continue
            any_matched = False
            for i, val in enumerate(new_val_arr):
                row_mask = (df["parameter"] == site_name) & (df["flat_index"] == float(i))
                if row_mask.any():
                    df.loc[row_mask, "value"] = float(val)
                    any_matched = True
            if any_matched:
                matched.add(site_name)

    missing = sorted(set(guess_updates) - matched)
    if missing:
        print(
            f"  warning: {len(missing)} guess update(s) had no matching "
            f"row in {guesses_path}: {missing}",
            file=sys.stderr,
        )

    shutil.copy2(guesses_path, guesses_path + ".bak")
    df.to_csv(guesses_path, index=False)
    print(f"Updated {len(matched)} guesses row(s) in {guesses_path}")


# ---------------------------------------------------------------------------
# CSV-path resolution from the production config
# ---------------------------------------------------------------------------

def _resolve_csv_paths(config_file):
    """
    Read the production YAML config and return the absolute paths of its
    priors and guesses CSV files.  Raises if either is missing.
    """
    with open(config_file, "r") as fh:
        cfg = yaml.safe_load(fh)
    cfg_dir = os.path.dirname(os.path.abspath(config_file))
    priors_file = cfg.get("priors_file")
    guesses_file = cfg.get("guesses_file")
    if priors_file is None:
        raise ValueError(f"priors_file not specified in {config_file}")
    if guesses_file is None:
        raise ValueError(f"guesses_file not specified in {config_file}")

    priors_path = priors_file if os.path.isabs(priors_file) \
        else os.path.join(cfg_dir, priors_file)
    guesses_path = guesses_file if os.path.isabs(guesses_file) \
        else os.path.join(cfg_dir, guesses_file)

    if not os.path.exists(priors_path):
        raise FileNotFoundError(f"Priors file not found: {priors_path}")
    if not os.path.exists(guesses_path):
        raise FileNotFoundError(f"Guesses file not found: {guesses_path}")

    return priors_path, guesses_path


# ---------------------------------------------------------------------------
# MAP execution (mirrors run_growth_analysis._run_map but trimmed)
# ---------------------------------------------------------------------------

def _run_calibration_map(ri,
                         init_params,
                         out_root,
                         checkpoint_file,
                         adam_step_size,
                         adam_final_step_size,
                         adam_clip_norm,
                         elbo_num_particles,
                         convergence_tolerance,
                         convergence_window,
                         patience,
                         convergence_check_interval,
                         checkpoint_interval,
                         max_num_epochs,
                         init_param_jitter):
    """Set up a MAP SVI optimizer and run it; return ``(svi_state, params,
    converged)``.  Behaviour mirrors ``run_growth_analysis._run_map`` but
    the ``always_get_posterior`` plumbing is dropped (the pre-fit only
    consumes MAP point estimates and Hessian-derived sigmas)."""
    schedule = optax.exponential_decay(
        init_value=adam_step_size,
        transition_steps=float(max_num_epochs * ri._iterations_per_epoch),
        decay_rate=adam_final_step_size / adam_step_size,
    )
    map_obj = ri.setup_svi(adam_step_size=schedule,
                           adam_clip_norm=adam_clip_norm,
                           elbo_num_particles=elbo_num_particles,
                           guide_type="delta")

    svi_state, params, converged = ri.run_optimization(
        map_obj,
        init_params=init_params,
        out_root=out_root,
        svi_state=checkpoint_file,
        convergence_tolerance=convergence_tolerance,
        convergence_window=convergence_window,
        patience=patience,
        convergence_check_interval=convergence_check_interval,
        checkpoint_interval=checkpoint_interval,
        max_num_epochs=max_num_epochs,
        init_param_jitter=init_param_jitter,
    )

    ri.write_params(params, out_root=out_root)

    if converged:
        print("Calibration MAP run converged.", flush=True)
    else:
        print("Calibration MAP run has not yet converged.", flush=True)

    return svi_state, params, converged


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _make_calibration_plots(gm_cal, params, out_root):
    """
    Generate per-genotype calibration diagnostic plots as PDFs.

    For each genotype in the calibration data, writes one PDF containing
    one subplot per (condition_pre, condition_sel, titrant_name, titrant_conc)
    combination.  Each subplot shows:

    * **Observed** — the experimental ``ln_cfu ± ln_cfu_std`` data points
      plotted at their ``t_sel`` x-coordinates (selection-phase time).
    * **Model prediction** — a smooth 100-point trajectory computed from
      the MAP parameter estimates:

      - Pre-selection phase (x from ``-t_pre`` to 0): a straight line from
        ``(−t_pre, ln_cfu0)`` to ``(0, ln_cfu0 + g_pre·t_pre)``, where
        ``ln_cfu0`` comes from the ``ln_cfu0`` deterministic site and the
        pre-selection slope ``g_pre`` is derived from the selection-phase
        intercept.
      - Selection phase (x from 0 to ``max(t_sel)``): a straight line with
        slope ``g_sel`` estimated by fitting the MAP ``growth_pred`` values
        at the observed ``t_sel`` times.

    The linear-in-time approximation for the smooth trajectory is exact for
    the ``instant`` growth-transition component and a reasonable visual
    approximation for non-linear variants.

    Parameters
    ----------
    gm_cal : GrowthModel
        Calibration GrowthModel (exposes ``growth_tm``, ``data``,
        ``priors``, ``jax_model``).
    params : dict
        MAP parameter dict from the SVI optimiser (keys follow the
        ``{site}_auto_loc`` convention).
    out_root : str
        File-name prefix; each genotype's PDF is written to
        ``{out_root}_calib_{genotype}.pdf``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  warning: matplotlib not available; skipping calibration plots.",
            file=sys.stderr,
        )
        return

    from numpyro.infer import Predictive
    from numpyro.infer.autoguide import AutoDelta
    import jax

    print("Generating calibration quality plots ...", flush=True)

    # --- Run Predictive at the MAP point to recover growth_pred and ln_cfu0 ---
    guide = AutoDelta(gm_cal.jax_model)
    all_indices = jnp.arange(gm_cal.data.num_genotype)
    full_data = gm_cal.get_batch(gm_cal.data, all_indices)
    pred_fn = Predictive(gm_cal.jax_model, guide=guide, params=params, num_samples=1)
    map_samples = pred_fn(
        jax.random.PRNGKey(0), data=full_data, priors=gm_cal.priors
    )

    if "growth_pred" not in map_samples:
        print(
            "  warning: growth_pred not in model trace; skipping plots.",
            file=sys.stderr,
        )
        return

    # Remove leading sample dimension (num_samples=1).
    # growth_pred: (R, T, CP, CS, TN, TC, G) at the original observed timepoints.
    # Kept separately from growth_pred_fine (fine-grid) so the CSV can record
    # predictions that align row-for-row with the observed growth_df entries.
    growth_pred = np.asarray(map_samples["growth_pred"][0])

    # ln_cfu0 deterministic: (R, CP, G) — registered by the hierarchical component
    ln_cfu0_map = (
        np.asarray(map_samples["ln_cfu0"][0])
        if "ln_cfu0" in map_samples
        else None
    )

    # --- Data tensors from the calibration model ---
    gd = gm_cal.data.growth
    good_mask    = np.asarray(gd.good_mask)        # (R, T, CP, CS, TN, TC, G)
    t_pre_tensor = np.asarray(gd.t_pre)            # (R, T, CP, CS, TN, TC, G)
    t_sel_tensor = np.asarray(gd.t_sel)            # (R, T, CP, CS, TN, TC, G)
    ln_cfu_obs   = np.asarray(gd.ln_cfu)           # (R, T, CP, CS, TN, TC, G)
    ln_cfu_std   = np.asarray(gd.ln_cfu_std)       # (R, T, CP, CS, TN, TC, G)

    n_rep, n_t, n_cp, n_cs, n_tn, n_tc, n_geno = good_mask.shape

    # --- Fine-grid Predictive for exact smooth selection-phase trajectories ---
    # Replace the T dimension with T_FINE evenly-spaced points from 0 to the
    # global max t_sel.  t_pre is constant over T within each condition cell,
    # so we broadcast from the first observed timepoint.  The same pred_fn is
    # reused with the new data argument; JAX will retrace for the new shape.
    T_FINE = 50
    global_max_t_sel = float(np.nanmax(t_sel_tensor[good_mask]))
    t_fine_1d = np.linspace(0.0, global_max_t_sel, T_FINE)
    fine_shape = (n_rep, T_FINE, n_cp, n_cs, n_tn, n_tc, n_geno)

    # Helper: broadcast a 7-D array from T=1 (time-constant) to T_FINE.
    # All per-condition tensors are constant along the T axis; slicing any
    # timepoint and broadcasting is therefore exact.
    def _bc_t(arr, *, dtype=None):
        a = np.asarray(arr)
        r = np.broadcast_to(a[:, 0:1, ...], fine_shape).copy()
        return jnp.array(r if dtype is None else r.astype(dtype))

    t_sel_fine = np.broadcast_to(
        t_fine_1d[None, :, None, None, None, None, None], fine_shape
    ).copy()

    # Active wherever any observed timepoint was valid for that cell.
    has_data_bc = good_mask.any(axis=1, keepdims=True)   # (R,1,CP,CS,TN,TC,G)
    good_mask_fine = np.broadcast_to(has_data_bc, fine_shape).copy()

    # map_condition_pre / map_condition_sel are also shape (R,T,CP,CS,TN,TC,G);
    # each element is an index into per-condition-rep arrays, constant over T.
    gd_full = full_data.growth
    fine_gd = gd_full.replace(
        num_time=T_FINE,
        t_sel=jnp.array(t_sel_fine),
        t_pre=_bc_t(gd_full.t_pre),
        good_mask=jnp.array(good_mask_fine),
        map_condition_pre=_bc_t(gd_full.map_condition_pre, dtype=int),
        map_condition_sel=_bc_t(gd_full.map_condition_sel, dtype=int),
        ln_cfu=jnp.zeros(fine_shape),
        ln_cfu_std=jnp.ones(fine_shape),
    )
    fine_data = full_data.replace(growth=fine_gd)
    map_samples_fine = pred_fn(
        jax.random.PRNGKey(1), data=fine_data, priors=gm_cal.priors
    )
    # shape: (R, T_FINE, CP, CS, TN, TC, G)
    growth_pred_fine = np.asarray(map_samples_fine["growth_pred"][0])

    # --- Dimension labels from the TensorManager ---
    tm  = gm_cal.growth_tm
    dn  = tm.tensor_dim_names

    geno_labels = list(tm.tensor_dim_labels[dn.index("genotype")])
    rep_labels  = list(tm.tensor_dim_labels[dn.index("replicate")])
    cp_labels   = list(tm.tensor_dim_labels[dn.index("condition_pre")])
    tn_labels   = list(tm.tensor_dim_labels[dn.index("titrant_name")])
    tc_labels   = list(tm.tensor_dim_labels[dn.index("titrant_conc")])

    # Map (cp_idx, cs_idx) → actual condition_sel name.  The tensor uses the
    # reduced (integer) condition_sel dimension; the original string name lives
    # in the processed DataFrame alongside the integer index column.
    df = tm.df
    cs_name_map = {}
    for _, row in df.drop_duplicates(
        ["condition_pre_idx", "condition_sel_idx"]
    ).iterrows():
        cs_name_map[
            (int(row["condition_pre_idx"]), int(row["condition_sel_idx"]))
        ] = str(row["condition_sel"])

    prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for g_i, geno_name in enumerate(geno_labels):

        # Collect valid (cp, cs, tn, tc) combinations and the replicates that
        # contribute observations.
        condition_combos: dict = {}
        for r_i in range(n_rep):
            for cp_i in range(n_cp):
                for cs_i in range(n_cs):
                    for tn_i in range(n_tn):
                        for tc_i in range(n_tc):
                            if good_mask[r_i, :, cp_i, cs_i, tn_i, tc_i, g_i].any():
                                condition_combos.setdefault(
                                    (cp_i, cs_i, tn_i, tc_i), []
                                ).append(r_i)

        if not condition_combos:
            continue

        n_combos = len(condition_combos)
        n_cols   = min(3, n_combos)
        n_rows   = (n_combos + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            squeeze=False,
            sharey=True,
        )
        fig.suptitle(
            f"Calibration fit — genotype: {geno_name}",
            fontsize=13,
            fontweight="bold",
        )

        for combo_i, ((cp_i, cs_i, tn_i, tc_i), rep_list) in enumerate(
            condition_combos.items()
        ):
            ax = axes[combo_i // n_cols][combo_i % n_cols]

            cp_name = str(cp_labels[cp_i])
            cs_name = cs_name_map.get((cp_i, cs_i), f"sel_{cs_i}")
            tn_name = str(tn_labels[tn_i])
            tc_val  = float(tc_labels[tc_i])

            ax.set_title(
                f"{cp_name} → {cs_name}\n{tn_name} = {tc_val:.3g}", fontsize=9
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("ln(CFU)")
            ax.axvline(0.0, color="0.6", lw=0.8, ls="--")

            # Collect valid per-replicate data for this condition combination.
            rep_data = []
            for r_i in rep_list:
                mask    = good_mask[r_i, :, cp_i, cs_i, tn_i, tc_i, g_i]
                valid_t = np.where(mask)[0]
                if len(valid_t) == 0:
                    continue
                rep_data.append({
                    "r_i":   r_i,
                    "t_sel": t_sel_tensor[r_i, valid_t, cp_i, cs_i, tn_i, tc_i, g_i],
                    "obs":   ln_cfu_obs[r_i, valid_t, cp_i, cs_i, tn_i, tc_i, g_i],
                    "std":   ln_cfu_std[r_i, valid_t, cp_i, cs_i, tn_i, tc_i, g_i],
                    "t_pre": float(np.nanmedian(
                        t_pre_tensor[r_i, valid_t, cp_i, cs_i, tn_i, tc_i, g_i]
                    )),
                })

            if not rep_data:
                ax.set_visible(False)
                continue

            max_t_sel = float(max(np.nanmax(rd["t_sel"]) for rd in rep_data))
            max_t_pre = float(max(rd["t_pre"] for rd in rep_data))

            # Plot each replicate in its own colour: data points + model line.
            for rd in rep_data:
                r_i       = rd["r_i"]
                rep_color = prop_colors[r_i % len(prop_colors)]
                rep_label = str(rep_labels[r_i])

                # Observed data with error bars
                ax.errorbar(
                    rd["t_sel"], rd["obs"], yerr=rd["std"],
                    fmt="o", color=rep_color, ms=5, lw=1, capsize=3,
                    label=rep_label, zorder=3,
                )

                # Exact model prediction for this replicate.
                # growth_pred_fine: (R, T_FINE, CP, CS, TN, TC, G)
                y_fine_r    = growth_pred_fine[r_i, :, cp_i, cs_i, tn_i, tc_i, g_i]
                calc_at_0_r = float(y_fine_r[0])

                # ln_cfu0 anchor; fall back to calc_at_0 when site is absent.
                if ln_cfu0_map is not None and ln_cfu0_map.ndim == 3:
                    ln_cfu0_r = float(ln_cfu0_map[r_i, cp_i, g_i])
                else:
                    ln_cfu0_r = calc_at_0_r

                # Pre-selection: two-point line (-t_pre, ln_cfu0) → (0, calc_at_0)
                # Selection: exact fine-grid predictions
                t_smooth = np.concatenate([np.array([-rd["t_pre"], 0.0]), t_fine_1d])
                y_smooth = np.concatenate([np.array([ln_cfu0_r, calc_at_0_r]), y_fine_r])
                ax.plot(t_smooth, y_smooth, "-", color=rep_color, lw=1.8, zorder=4)

            ax.legend(fontsize=8, loc="best")
            x_pad = (max_t_sel + max_t_pre) * 0.03
            ax.set_xlim(-max_t_pre - x_pad, max_t_sel + x_pad)

        # Hide unused subplots
        for extra_i in range(n_combos, n_rows * n_cols):
            axes[extra_i // n_cols][extra_i % n_cols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf_path = f"{out_root}_calib_{geno_name}.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {pdf_path}", flush=True)

    # --- Write predictions CSV ---
    # Extract the MAP-predicted ln_cfu at every valid (observed) tensor cell,
    # then attach it as a new column alongside the original growth_df rows.
    #
    # The tensor has 7 dimensions: (R, T, CP, CS, TN, TC, G).
    # tm.df carries _idx columns for 6 of them (all except T); the T dimension
    # is identified by the t_sel float value, which is the same in both the
    # tensor and the DataFrame.
    r_idx, t_idx, cp_idx, cs_idx, tn_idx, tc_idx, g_idx = np.where(good_mask)

    pred_lookup = pd.DataFrame({
        "replicate_idx":     r_idx.astype(int),
        "condition_pre_idx": cp_idx.astype(int),
        "condition_sel_idx": cs_idx.astype(int),
        "titrant_name_idx":  tn_idx.astype(int),
        "titrant_conc_idx":  tc_idx.astype(int),
        "genotype_idx":      g_idx.astype(int),
        "t_sel": t_sel_tensor[r_idx, t_idx, cp_idx, cs_idx, tn_idx, tc_idx, g_idx],
        "ln_cfu_pred":
            growth_pred[r_idx, t_idx, cp_idx, cs_idx, tn_idx, tc_idx, g_idx],
    })

    merge_cols = [
        "replicate_idx", "condition_pre_idx", "condition_sel_idx",
        "titrant_name_idx", "titrant_conc_idx", "genotype_idx", "t_sel",
    ]
    growth_df_out = df.merge(pred_lookup, on=merge_cols, how="left")
    csv_path = f"{out_root}_calib_growth_df.csv"
    growth_df_out.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}", flush=True)

    print("Calibration plots complete.", flush=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_prefit_calibration(config_file,
                           seed=None,
                           checkpoint_file=None,
                           out_root="prefit",
                           adam_step_size=1e-3,
                           adam_final_step_size=1e-6,
                           adam_clip_norm=1.0,
                           elbo_num_particles=2,
                           convergence_tolerance=0.01,
                           convergence_window=10,
                           patience=10,
                           convergence_check_interval=2,
                           checkpoint_interval=10,
                           max_num_epochs=100000,
                           init_param_jitter=0.0):
    """
    Run the calibration pre-fit and update the production priors / guesses
    CSVs in place.

    The script:

    1. Reads the production ``config_file`` to obtain the production
       ``GrowthModel`` (so we know which priors / guesses files to
       update later) and the production data paths.
    2. Filters the growth and binding inputs to their shared
       (genotype, titrant_name, titrant_conc) cells.
    3. Builds an in-process calibration ``GrowthModel`` whose
       ``condition_growth`` and ``growth_transition`` components match
       the production choices but whose other components are pinned /
       collapsed (see module docstring).
    4. Runs MAP, then computes per-site Hessian sigmas at the MAP
       point.
    5. Writes ``.bak`` copies of the production priors and guesses CSVs
       and overwrites only the rows that belong to ``condition_growth``
       or ``growth_transition``.

    Parameters
    ----------
    config_file : str
        Path to the *production* YAML configuration file.  The pre-fit
        does not write a calibration YAML — it only updates the
        existing production CSVs.
    seed : int, optional
        Random seed for reproducibility.  Required unless resuming from
        a checkpoint.
    checkpoint_file : str or None, optional
        Path to a previously written pre-fit checkpoint to resume from.
    out_root : str, optional
        Prefix for calibration MAP artefacts (``{out_root}_params.npz``,
        ``{out_root}_checkpoint.pkl``, etc.).  Defaults to ``"prefit"``.
        These are diagnostic outputs; the user-facing artefact is the
        in-place update of the production CSVs.
    adam_step_size, adam_final_step_size, adam_clip_norm,
    elbo_num_particles, convergence_tolerance, convergence_window,
    patience, convergence_check_interval, checkpoint_interval,
    max_num_epochs, init_param_jitter : optional
        Standard MAP optimizer / convergence kwargs; see
        :func:`run_growth_analysis.run_growth_analysis` for details.
        ``init_param_jitter`` defaults to ``0.0`` because the pre-fit
        benefits from being deterministic given a seed.

    Returns
    -------
    svi_state : Any
        Final MAP SVI optimizer state.
    params : dict
        MAP point estimates (raw ``{site}_auto_loc`` keys).
    converged : bool
        Whether the MAP run converged.
    """
    if seed is None and checkpoint_file is None:
        raise ValueError("seed must be provided unless loading from a checkpoint.")

    # 1. Resolve production config and CSV targets.
    priors_path, guesses_path = _resolve_csv_paths(config_file)
    gm_prod, _ = read_configuration(config_file)

    # 2. Filter to the calibration (genotype, titrant_name, titrant_conc)
    # intersection.  read_configuration already loaded the production
    # data paths into gm_prod; pull them straight off the config so we
    # don't depend on an internal attribute.
    with open(config_file, "r") as fh:
        cfg = yaml.safe_load(fh)
    growth_path = cfg["data"]["growth"]
    binding_path = cfg["data"]["binding"]
    growth_df_cal, binding_df_cal = _intersect_data(growth_path, binding_path)

    # 3. Build the calibration model with overrides applied.
    gm_cal = _build_calibration_model(gm_prod, growth_df_cal, binding_df_cal)
    theta_values = _compute_theta_values(gm_cal, binding_df_cal)
    _inject_calibration_priors(gm_cal, gm_prod, theta_values)

    # 4. MAP fit + Hessian sigmas.
    effective_seed = seed if seed is not None else 0
    ri = RunInference(gm_cal, effective_seed)

    svi_state, params, converged = _run_calibration_map(
        ri,
        init_params=gm_cal.init_params,
        out_root=out_root,
        checkpoint_file=checkpoint_file,
        adam_step_size=adam_step_size,
        adam_final_step_size=adam_final_step_size,
        adam_clip_norm=adam_clip_norm,
        elbo_num_particles=elbo_num_particles,
        convergence_tolerance=convergence_tolerance,
        convergence_window=convergence_window,
        patience=patience,
        convergence_check_interval=convergence_check_interval,
        checkpoint_interval=checkpoint_interval,
        max_num_epochs=max_num_epochs,
        init_param_jitter=init_param_jitter,
    )

    print("Computing Hessian-based per-site uncertainties ...", flush=True)
    hessian_results = ri.compute_hessian_sigmas(params)

    # 5. Map sample sites → CSV fields and apply in-place updates.
    field_mapping = _identify_field_mapping(gm_cal)
    prior_updates, guess_updates = _build_csv_updates(field_mapping,
                                                      hessian_results)
    _apply_priors_updates(priors_path, prior_updates)
    _apply_guesses_updates(guesses_path, guess_updates)

    # 6. Write per-genotype diagnostic plots.
    _make_calibration_plots(gm_cal, params, out_root)

    return svi_state, params, converged


def main():
    return generalized_main(
        run_prefit_calibration,
        manual_arg_types={"config_file": str,
                          "seed": int,
                          "checkpoint_file": str,
                          "init_param_jitter": float},
    )


if __name__ == "__main__":
    main()
