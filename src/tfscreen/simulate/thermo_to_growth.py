"""
Generate phenotypes from a genotype library via prior-predictive theta sampling.
"""

from tfscreen.genetics import (
    set_categorical_genotype,
    standardize_genotypes,
    argsort_genotypes,
)
from tfscreen.simulate.sample_theta import sample_theta_prior
from tfscreen.simulate.growth.growth_linkage import get_growth_model

import pandas as pd
import numpy as np
from numpy.random import Generator

from typing import Iterable, Optional

_EPS = 1e-6

_THETA_RESCALE = {
    "passthrough": lambda x: x,
    "logit": lambda x: np.log(
        np.clip(x, _EPS, 1.0 - _EPS) / (1.0 - np.clip(x, _EPS, 1.0 - _EPS))
    ),
}


def _assign_activity(unique_genotypes,
                     activity_wt=1.0,
                     activity_mut_scale=0.0,
                     rng: Generator | None = None):
    """
    Assign per-genotype TF activity values.

    The wild-type genotype receives ``activity_wt``.  Each mutant genotype
    draws independently from ``LogNormal(log(activity_wt), activity_mut_scale)``.
    When ``activity_mut_scale == 0`` every genotype gets ``activity_wt``.
    """
    if rng is None:
        rng = np.random.default_rng()

    log_a_wt = np.log(activity_wt)
    activities = {}
    for g in unique_genotypes:
        if g == "wt" or activity_mut_scale == 0.0:
            activities[g] = float(activity_wt)
        else:
            activities[g] = float(np.exp(rng.normal(log_a_wt, activity_mut_scale)))

    return pd.Series(activities)


def _sample_horseshoe_activity(unique_genotypes,
                               params: Optional[dict] = None,
                               rng: Generator | None = None):
    """
    Sample per-genotype TF activity from a horseshoe prior in log-activity space.

    Mirrors the ``horseshoe`` activity component used in tfmodel inference:

        tau      ~ HalfNormal(global_scale_tau_scale)
        lambda_i ~ HalfNormal(1)
        z_i      ~ Normal(0, 1)
        log(A_i) = z_i * tau * lambda_i
        A_i      = exp(log(A_i))

    Wild-type genotypes always receive A = 1.0.

    Parameters
    ----------
    unique_genotypes : array-like of str
    params : dict or None
        Override any of the default hyperparameters:
        ``global_scale_tau_scale`` (default 0.1).
    rng : numpy.random.Generator or None

    Returns
    -------
    pandas.Series
        Per-genotype activity values indexed by genotype string.
    """
    if rng is None:
        rng = np.random.default_rng()
    p = {"global_scale_tau_scale": 0.1}
    if params:
        p.update(params)

    G = len(unique_genotypes)
    tau = abs(rng.standard_normal()) * p["global_scale_tau_scale"]
    lambdas = np.abs(rng.standard_normal(G))
    offsets = rng.standard_normal(G)
    log_activities = offsets * tau * lambdas
    activities = np.exp(log_activities)

    result = pd.Series(activities, index=list(unique_genotypes), dtype=float)
    result[result.index == "wt"] = 1.0
    return result


def _sample_hierarchical_activity(unique_genotypes,
                                   params: Optional[dict] = None,
                                   rng: Generator | None = None):
    """
    Sample per-genotype TF activity from a hierarchical log-normal prior.

    Mirrors the ``hierarchical`` activity component used in tfmodel inference:

        hyper_loc   ~ Normal(hyper_loc_loc, hyper_loc_scale)
        hyper_scale ~ HalfNormal(hyper_scale_loc)
        offset_i    ~ Normal(0, 1)
        log(A_i)    = hyper_loc + offset_i * hyper_scale
        A_i         = exp(log(A_i))

    Wild-type genotypes always receive A = 1.0.

    Parameters
    ----------
    unique_genotypes : array-like of str
    params : dict or None
        Override any of the default hyperparameters:
        ``hyper_loc_loc`` (default 0.0), ``hyper_loc_scale`` (default 0.01),
        ``hyper_scale_loc`` (default 0.1).
    rng : numpy.random.Generator or None

    Returns
    -------
    pandas.Series
        Per-genotype activity values indexed by genotype string.
    """
    if rng is None:
        rng = np.random.default_rng()
    p = {"hyper_loc_loc": 0.0, "hyper_loc_scale": 0.01, "hyper_scale_loc": 0.1}
    if params:
        p.update(params)

    G = len(unique_genotypes)
    hyper_loc = rng.normal(p["hyper_loc_loc"], p["hyper_loc_scale"])
    hyper_scale = abs(rng.standard_normal()) * p["hyper_scale_loc"]
    offsets = rng.standard_normal(G)
    log_activities = hyper_loc + offsets * hyper_scale
    activities = np.clip(np.exp(log_activities), a_min=None, a_max=1e30)

    result = pd.Series(activities, index=list(unique_genotypes), dtype=float)
    result[result.index == "wt"] = 1.0
    return result


# Components supported by the numpy simulation path.
# (The mutation-decomposed variants horseshoe_mut / hierarchical_mut operate
# in mutation space and require the tfmodel component interface; add them here
# if a numpy equivalent is implemented in future.)
_ACTIVITY_COMPONENTS = {"fixed", "horseshoe_geno", "hierarchical_geno"}


def _theta_param_to_df(theta_param, unique_genotypes, sim_indices):
    """
    Extract per-genotype scalar fields from a ThetaParam pytree.

    Iterates over the dataclass fields of ``theta_param``.  Fields whose
    array has exactly two dimensions (T × G_sim) are treated as
    per-genotype parameters: columns are selected via ``sim_indices`` and
    a leading titrant dimension of size 1 is squeezed away; when T > 1 each
    titrant gets its own column suffixed ``_T{t}``.  Fields with any other
    dimensionality (e.g. population moments ``mu``/``sigma`` with shape
    T × C × 1) are silently skipped.

    If ``theta_param`` is not a standard or Flax dataclass (e.g. a mock
    during testing) the function returns a DataFrame containing only the
    ``genotype`` column, to which ``dk_geno`` and ``activity`` are then
    added by the caller.

    Parameters
    ----------
    theta_param : pytree
        Returned by ``sample_theta_prior``.  Per-genotype fields have shape
        ``(num_titrant_name, num_genotype_sim)``.
    unique_genotypes : array-like of str
        Unique genotype strings in the desired output row order.
    sim_indices : array-like of int
        For each entry in ``unique_genotypes``, its column index in the
        sim_data / ``library_df`` order stored in ``theta_param``.

    Returns
    -------
    pandas.DataFrame
        One row per genotype.  Columns are the extracted ``theta_param``
        field names (with a ``_T{t}`` suffix when ``num_titrant_name > 1``).
        The ``genotype`` column is always present as the first column.
    """
    # Use the class-level __dataclass_fields__ so that MagicMock instances
    # (which intercept instance-level attribute access) safely return an
    # empty dict rather than triggering mock attribute creation.
    field_names = list(
        getattr(type(theta_param), "__dataclass_fields__", {}).keys()
    )

    col_data = {}
    for fname in field_names:
        try:
            val = np.array(getattr(theta_param, fname))
        except Exception:
            continue
        if val.ndim != 2:
            continue  # skip mu, sigma (3-D) and any scalars
        selected = val[:, np.asarray(sim_indices)]   # (T, n_unique)
        if selected.shape[0] == 1:
            col_data[fname] = selected[0]
        else:
            for t in range(selected.shape[0]):
                col_data[f"{fname}_T{t}"] = selected[t]

    df = pd.DataFrame(col_data)
    df.insert(0, "genotype", list(unique_genotypes))
    return df


def _assign_dk_geno(unique_genotypes,
                    hyper_loc=-3.5,
                    hyper_scale=1.0,
                    hyper_shift=0.02,
                    rng: Generator | None = None):
    """
    Assign a pleiotropic growth-rate effect (dk_geno) to each genotype.

    Each genotype draws independently from the shifted lognormal:
        dk_geno = hyper_shift - exp(Normal(hyper_loc, hyper_scale))

    Wild-type receives dk_geno = 0.  This matches the prior used in the
    hierarchical tfmodel inference component.
    """
    if rng is None:
        rng = np.random.default_rng()

    dk_geno = {}
    for g in unique_genotypes:
        if g == "wt":
            dk_geno[g] = 0.0
        else:
            offset = rng.normal(hyper_loc, hyper_scale)
            dk_geno[g] = float(hyper_shift - np.exp(offset))

    return pd.Series(dk_geno)


def _apply_growth_params(condition_array, theta_array, growth_params,
                         activity_array=None):
    """
    Compute per-row growth rate k given theta and per-condition model parameters.

    Parameters
    ----------
    condition_array : numpy.ndarray
        1D array of condition strings.
    theta_array : numpy.ndarray
        1D array of fractional occupancy values (same length).
    growth_params : dict
        Mapping from condition string to a parameter dict.  Must contain a
        ``'model'`` key (``'linear'``, ``'power'``, or ``'saturation'``);
        remaining keys are forwarded to the model's ``predict()`` method.
    activity_array : numpy.ndarray or None
        Per-row TF activity scaling factors.  Defaults to 1.0.

    Returns
    -------
    numpy.ndarray
        Growth rate k for each row.
    """
    if activity_array is None:
        activity_array = np.ones(len(theta_array), dtype=float)
    result = np.zeros(len(theta_array), dtype=float)
    for cond in np.unique(condition_array):
        mask = condition_array == cond
        params = growth_params[cond].copy()
        model_name = params.pop("model", "linear")
        model = get_growth_model(model_name)
        result[mask] = model.predict(theta_array[mask],
                                     activity=activity_array[mask], **params)
    return result


def thermo_to_growth(
    genotypes: Iterable[str],
    sim_data,
    sample_df: pd.DataFrame,
    theta_component: str,
    theta_rng_key,
    growth_params: dict,
    theta_priors_overrides: Optional[dict] = None,
    theta_noise_sigma_logit: float = 0.0,
    dk_geno_hyper_loc: float = -3.5,
    dk_geno_hyper_scale: float = 1.0,
    dk_geno_hyper_shift: float = 0.02,
    activity_wt: float = 1.0,
    activity_mut_scale: float = 0.0,
    rng: Generator | None = None,
    activity_component: str = "fixed",
    activity_priors_overrides: Optional[dict] = None,
    theta_rescale: str = "passthrough",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate phenotypes from genotypes via prior-predictive theta sampling.

    Parameters
    ----------
    genotypes : Iterable[str]
        Genotype strings in the same order as ``sim_data`` was built from.
    sim_data : SimData
        Built by ``build_sim_data`` from the same library and sample_df.
    sample_df : pd.DataFrame
        Experimental conditions (titrant_conc, condition_pre, condition_sel,
        t_pre, t_sel, …).
    theta_component : str
        Registered theta component name (e.g. ``"mwc_dimer_lnK_mut"``).
    theta_rng_key : jax.random.PRNGKey
        Seed for prior-predictive theta sampling.
    growth_params : dict
        Per-condition growth model parameters keyed by condition string.
    theta_priors_overrides : dict or None
        Overrides to the theta component's default hyperparameters.
    theta_noise_sigma_logit : float, default 0.0
        Standard deviation of additive Normal noise applied on the logit
        scale after prior-predictive theta sampling, giving:
            theta_noisy = sigmoid(logit(theta_pred) + Normal(0, sigma_logit))
        Set to 0.0 (default) for deterministic theta. Mirrors the
        'logit_normal' theta_growth_noise component in tfmodel inference.
    dk_geno_hyper_loc : float
        Mean of the normal distribution in log-space for dk_geno sampling.
        Matches the ``hyper_loc`` hyperparameter of the hierarchical tfmodel
        dk_geno component.
    dk_geno_hyper_scale : float
        Std dev of the normal distribution in log-space for dk_geno sampling.
        Matches the ``hyper_scale`` hyperparameter of the hierarchical tfmodel
        dk_geno component.
    dk_geno_hyper_shift : float
        Shift applied after exponentiation: dk_geno = hyper_shift - exp(offset).
        Controls the maximum possible (beneficial) growth-rate effect.
    activity_wt : float
        TF activity of the wild-type genotype.  Used only when
        ``activity_component == "fixed"``.
    activity_mut_scale : float
        Std dev of log(activity) for mutant genotypes under the plain
        LogNormal path.  0 gives all genotypes identical activity.  Used
        only when ``activity_component == "fixed"``.
    rng : numpy.random.Generator or None
        NumPy RNG used for all stochastic sampling (dk_geno, activity).
    activity_component : str, default ``"fixed"``
        Activity distribution to use.  One of ``"fixed"``, ``"horseshoe"``,
        or ``"hierarchical"``.  When ``"fixed"``, ``activity_wt`` and
        ``activity_mut_scale`` control the draw; the other two components
        are pure-numpy implementations of the matching tfmodel priors and
        ignore ``activity_wt`` / ``activity_mut_scale``.
    activity_priors_overrides : dict or None
        Hyperparameter overrides forwarded to the horseshoe or hierarchical
        sampler.  Keys and defaults:

        * ``"horseshoe"``     — ``global_scale_tau_scale`` (default 0.1)
        * ``"hierarchical"``  — ``hyper_loc_loc`` (0.0),
          ``hyper_loc_scale`` (0.01), ``hyper_scale_loc`` (0.1)

        Ignored when ``activity_component == "fixed"``.
    theta_rescale : str, default ``"passthrough"``
        Transformation applied to theta before it is passed to the growth
        model.  Mirrors the ``theta_rescale`` component in ``tfmodel``.
        ``"passthrough"`` leaves theta in (0, 1); ``"logit"`` maps theta to
        the logit scale ``log(θ/(1−θ))``.  The ``"theta"`` column in
        ``phenotype_df`` and ``genotype_theta_df`` always store the
        pre-rescale (0–1) value.

    Returns
    -------
    phenotype_df : pd.DataFrame
        Long-form dataframe with one row per (genotype, sample condition).
        Columns include ``theta``, ``dk_geno``, ``activity``, ``k_pre``,
        ``k_sel``.  ``dk_geno`` and ``activity`` are retained here so that
        ``selection_experiment`` can use them downstream; they are also
        collected in ``parameters_df`` for file output.
    genotype_theta_df : pd.DataFrame
        Wide-form dataframe with one row per genotype and one column per
        unique effector concentration, giving the ground-truth theta value.
        Use this to compare simulation inputs against fitted posteriors.
    parameters_df : pd.DataFrame
        One row per unique genotype.  Always contains ``genotype``,
        ``dk_geno``, and ``activity``.  Also contains any scalar
        per-genotype fields extracted from the ``theta_param`` pytree
        (e.g. ``theta_low``, ``theta_high``, ``log_hill_K``, ``hill_n``
        for ``hill_geno``).
    """

    print("Sampling theta from prior... ", end="", flush=True)

    if rng is None:
        rng = np.random.default_rng()

    if theta_rescale not in _THETA_RESCALE:
        raise ValueError(
            f"theta_rescale '{theta_rescale}' not recognized. "
            f"It should be one of: {list(_THETA_RESCALE.keys())}"
        )

    # ── Sort genotypes and sample_df in a stereotyped way ────────────────────

    unique_unsorted = np.unique(standardize_genotypes(genotypes))
    genotype_order = argsort_genotypes(unique_unsorted)
    unique_genotypes = unique_unsorted[genotype_order]

    standard_sort_order = ["replicate", "library", "condition_sel",
                           "titrant_name", "titrant_conc", "t_sel"]
    sort_on = [s for s in standard_sort_order if s in sample_df.columns]
    if len(sort_on) > 0:
        sample_df = sample_df.sort_values(sort_on).reset_index(drop=True)

    # ── Prior-predictive theta sampling ──────────────────────────────────────
    # theta_gc shape: (G, C) where G = num_genotype, C = num_unique_conc
    # Genotype order matches sim_data / library_df order, not unique_genotypes.
    # We need to build a mapping from unique_genotypes to the sim_data index.

    theta_gc, theta_param = sample_theta_prior(
        component_name=theta_component,
        sim_data=sim_data,
        rng_key=theta_rng_key,
        priors_overrides=theta_priors_overrides,
    )

    print("Done.", flush=True)

    # ── Apply logit-normal noise to theta ─────────────────────────────────────
    if theta_noise_sigma_logit > 0.0:
        theta_safe = np.clip(theta_gc, 1e-6, 1.0 - 1e-6)
        logit_theta = np.log(theta_safe / (1.0 - theta_safe))
        epsilon = rng.normal(0.0, theta_noise_sigma_logit, size=theta_gc.shape)
        theta_gc = 1.0 / (1.0 + np.exp(-(logit_theta + epsilon)))

    # ── Index lookups (shared by genotype_theta_df, phenotype_df, parameters_df) ─
    # Computed once here so that genotype_theta_df can be built from unique
    # genotypes rather than the full sim_data order (which may contain duplicates
    # when the same sequence appears in more than one sub-library).

    all_genotypes = list(genotypes)   # sim_data order (may have duplicates)
    geno_to_sim_idx = {g: i for i, g in enumerate(all_genotypes)}
    unique_sim_indices = np.array([geno_to_sim_idx[g] for g in unique_genotypes])

    # ── Build genotype_theta_df (ground-truth theta per unique genotype) ──────
    # Wide form: one row per UNIQUE genotype (in sorted unique_genotypes order),
    # one column per concentration.  unique_sim_indices maps each unique genotype
    # to its representative row in theta_gc, eliminating duplicates.

    conc_vals = np.array(sim_data.titrant_conc)
    conc_col_names = [f"theta_at_{c:.6g}mM" for c in conc_vals]
    genotype_theta_df = pd.DataFrame(
        theta_gc[unique_sim_indices],
        index=list(unique_genotypes),
        columns=conc_col_names,
    )
    genotype_theta_df.index.name = "genotype"
    genotype_theta_df = genotype_theta_df.reset_index()
    genotype_theta_df = set_categorical_genotype(genotype_theta_df)

    # ── Map sample_df concentrations to unique-concentration indices ──────────
    # sample_df may have duplicate concentrations across rows; each row maps to
    # one column of theta_gc.
    # Use float64 values from sample_df directly (same source as build_sim_data)
    # rather than conc_vals, which is float32 from the JAX array and would cause
    # dict-lookup misses due to float32→float64 precision loss.
    sorted_concs_f64 = np.sort(sample_df["titrant_conc"].unique()).astype(float)
    conc_to_col = {c: i for i, c in enumerate(sorted_concs_f64)}
    sample_conc_idx = sample_df["titrant_conc"].map(conc_to_col).values.astype(int)

    print("Calculating growth rates and building phenotype dataframe... ",
          end="", flush=True)

    # ── Expand theta to (genotype × sample) long form ────────────────────────
    # Build arrays aligned to unique_genotypes for later dk_geno / activity
    # assignment, then merge with sample_df.

    n_geno = len(unique_genotypes)
    n_samples = len(sample_df)

    # Theta matrix aligned to unique_genotypes order, shape (n_geno, n_samples)
    # (unique_sim_indices already computed above)
    theta_ordered = theta_gc[unique_sim_indices]          # (n_geno, C)
    theta_for_samples = theta_ordered[:, sample_conc_idx]  # (n_geno, n_samples)

    # Build long-form theta dataframe matching the old theta_out.stack() shape
    theta_long_df = (
        pd.DataFrame(
            theta_for_samples,
            index=unique_genotypes,
            columns=range(n_samples),
        )
        .stack()
        .reset_index()
    )
    theta_long_df.columns = ["genotype", "feature_id", "theta"]

    phenotype_df = pd.merge(theta_long_df, sample_df,
                            left_on="feature_id", right_index=True)

    # ── Per-genotype fitness cost and activity ────────────────────────────────

    genotype_dk_geno_series = _assign_dk_geno(
        unique_genotypes, dk_geno_hyper_loc, dk_geno_hyper_scale,
        dk_geno_hyper_shift, rng,
    )
    phenotype_df["dk_geno"] = phenotype_df["genotype"].map(genotype_dk_geno_series)

    if activity_component not in _ACTIVITY_COMPONENTS:
        raise ValueError(
            f"activity_component '{activity_component}' not recognized. "
            f"Must be one of: {sorted(_ACTIVITY_COMPONENTS)}"
        )

    if activity_component == "horseshoe_geno":
        genotype_activity_series = _sample_horseshoe_activity(
            unique_genotypes, params=activity_priors_overrides, rng=rng,
        )
    elif activity_component == "hierarchical_geno":
        genotype_activity_series = _sample_hierarchical_activity(
            unique_genotypes, params=activity_priors_overrides, rng=rng,
        )
    else:  # "fixed"
        genotype_activity_series = _assign_activity(
            unique_genotypes, activity_wt=activity_wt,
            activity_mut_scale=activity_mut_scale, rng=rng,
        )
    phenotype_df["activity"] = phenotype_df["genotype"].map(genotype_activity_series)

    # ── Validate growth_params coverage ──────────────────────────────────────

    used_conditions = (set(phenotype_df["condition_pre"].unique()) |
                       set(phenotype_df["condition_sel"].unique()))
    missing = used_conditions - set(growth_params.keys())
    if missing:
        err = "The following conditions are required but missing from growth_params:\n"
        for c in sorted(missing):
            err += f"    {c}\n"
        raise ValueError(err)

    theta = phenotype_df["theta"].to_numpy()
    activity = phenotype_df["activity"].to_numpy()

    growth_theta = _THETA_RESCALE[theta_rescale](theta)

    k_pre = _apply_growth_params(phenotype_df["condition_pre"].to_numpy(),
                                 growth_theta, growth_params, activity_array=activity)
    phenotype_df["k_pre"] = k_pre + phenotype_df["dk_geno"].to_numpy()

    k_sel = _apply_growth_params(phenotype_df["condition_sel"].to_numpy(),
                                 growth_theta, growth_params, activity_array=activity)
    phenotype_df["k_sel"] = k_sel + phenotype_df["dk_geno"].to_numpy()

    # ── Final column ordering ─────────────────────────────────────────────────

    final_columns = list(phenotype_df.columns)
    final_columns.remove("feature_id")
    final_columns.remove("theta")
    final_columns.append("theta")
    phenotype_df = phenotype_df.loc[:, final_columns]

    # ── Build parameters_df (one row per unique genotype) ────────────────────
    # Extracts per-genotype theta_param fields (e.g. theta_low, log_hill_K)
    # and pairs them with the already-computed dk_geno and activity values.
    # phenotype_df keeps dk_geno/activity so selection_experiment still works.

    parameters_df = _theta_param_to_df(theta_param, unique_genotypes,
                                        unique_sim_indices)
    parameters_df["dk_geno"] = genotype_dk_geno_series.reindex(
        list(unique_genotypes)
    ).values
    parameters_df["activity"] = genotype_activity_series.reindex(
        list(unique_genotypes)
    ).values
    _front = ["genotype", "dk_geno", "activity"]
    _other = [c for c in parameters_df.columns if c not in _front]
    parameters_df = parameters_df[_front + _other]
    parameters_df = set_categorical_genotype(parameters_df)

    print("Done.", flush=True)

    return phenotype_df, genotype_theta_df, parameters_df
