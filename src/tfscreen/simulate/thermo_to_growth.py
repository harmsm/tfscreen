"""
Generate phenotypes from a genotype library via prior-predictive theta sampling.
"""

from tfscreen.genetics import (
    combine_mutation_effects,
    set_categorical_genotype,
    standardize_genotypes,
    argsort_genotypes,
)
from tfscreen.simulate.sample_theta import sample_theta_prior
from tfscreen.simulate.growth.growth_linkage import get_growth_model
from tfscreen.simulate.growth.transition_linkage import get_transition_model

import pandas as pd
import numpy as np
from numpy.random import Generator
from tqdm.auto import tqdm
from scipy.stats import gamma

from typing import Iterable, Union, Callable, Optional


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


def _assign_dk_geno(unique_genotypes,
                    shape_param=3,
                    scale_param=0.002,
                    mut_combine_fcn="sum",
                    rng: Generator | None = None):
    """
    Assign a global fitness cost (dk_geno) to each genotype.

    Per-mutation costs are drawn from a gamma distribution and combined for
    multi-mutation genotypes, producing a distribution skewed towards
    deleterious (negative) values.
    """
    if rng is None:
        rng = np.random.default_rng()

    g_lookup = pd.Series(list(unique_genotypes),
                         index=unique_genotypes).str.split("/", expand=True)
    if "wt" in g_lookup.index:
        g_lookup.loc["wt", :] = np.nan
    single_mutations = g_lookup.stack().unique()

    dk_geno_values = scale_param / 2 - gamma.rvs(
        a=shape_param,
        scale=scale_param,
        size=len(single_mutations),
        random_state=rng,
    )

    mut_dk_mapper = pd.Series(data=dk_geno_values, index=single_mutations)

    return combine_mutation_effects(
        unique_genotypes=unique_genotypes,
        single_mutant_effects=mut_dk_mapper,
        mut_combine_fcn=mut_combine_fcn,
    )


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
    mut_growth_rate_shape: float = 3,
    mut_growth_rate_scale: float = 0.002,
    activity_wt: float = 1.0,
    activity_mut_scale: float = 0.0,
    dk_geno_combine_fcn: Optional[Union[str, Callable]] = "sum",
    rng: Generator | None = None,
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
    mut_growth_rate_shape : float
        Shape of the gamma distribution for dk_geno sampling.
    mut_growth_rate_scale : float
        Scale of the gamma distribution for dk_geno sampling.
    activity_wt : float
        TF activity of the wild-type genotype.
    activity_mut_scale : float
        Std dev of log(activity) for mutant genotypes.  0 gives all genotypes
        identical activity.
    dk_geno_combine_fcn : str or callable
        Method for combining per-mutation dk_geno values.
    rng : numpy.random.Generator or None
        NumPy RNG for dk_geno / activity sampling.

    Returns
    -------
    phenotype_df : pd.DataFrame
        Long-form dataframe with one row per (genotype, sample condition).
        Columns include ``theta``, ``dk_geno``, ``activity``, ``k_pre``,
        ``k_sel``.
    genotype_theta_df : pd.DataFrame
        Wide-form dataframe with one row per genotype and one column per
        unique effector concentration, giving the ground-truth theta value.
        Use this to compare simulation inputs against fitted posteriors.
    """

    print("Sampling theta from prior... ", end="", flush=True)

    if rng is None:
        rng = np.random.default_rng()

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

    # ── Build genotype_theta_df (ground-truth parameters for comparison) ─────
    # Wide form: rows = genotypes (in sim_data order), cols = concentrations.

    all_genotypes = list(genotypes)   # sim_data order
    conc_vals = np.array(sim_data.titrant_conc)
    conc_col_names = [f"theta_at_{c:.6g}mM" for c in conc_vals]
    genotype_theta_df = pd.DataFrame(
        theta_gc,
        index=all_genotypes,
        columns=conc_col_names,
    )
    genotype_theta_df.index.name = "genotype"
    genotype_theta_df = genotype_theta_df.reset_index()
    genotype_theta_df = set_categorical_genotype(genotype_theta_df)

    # ── Map sample_df concentrations to unique-concentration indices ──────────
    # sample_df may have duplicate concentrations across rows; each row maps to
    # one column of theta_gc.
    conc_to_col = {float(c): i for i, c in enumerate(conc_vals)}
    sample_conc_idx = sample_df["titrant_conc"].map(conc_to_col).values

    # ── Build genotype-index lookup into sim_data order ───────────────────────
    geno_to_sim_idx = {g: i for i, g in enumerate(all_genotypes)}

    print("Calculating growth rates and building phenotype dataframe... ",
          end="", flush=True)

    # ── Expand theta to (genotype × sample) long form ────────────────────────
    # Build arrays aligned to unique_genotypes for later dk_geno / activity
    # assignment, then merge with sample_df.

    n_geno = len(unique_genotypes)
    n_samples = len(sample_df)

    # Theta matrix aligned to unique_genotypes order, shape (n_geno, n_samples)
    unique_sim_indices = np.array([geno_to_sim_idx[g] for g in unique_genotypes])
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
        unique_genotypes, mut_growth_rate_shape, mut_growth_rate_scale,
        dk_geno_combine_fcn, rng,
    )
    phenotype_df["dk_geno"] = phenotype_df["genotype"].map(genotype_dk_geno_series)

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

    k_pre = _apply_growth_params(phenotype_df["condition_pre"].to_numpy(),
                                 theta, growth_params, activity_array=activity)
    phenotype_df["k_pre"] = k_pre + phenotype_df["dk_geno"].to_numpy()

    k_sel = _apply_growth_params(phenotype_df["condition_sel"].to_numpy(),
                                 theta, growth_params, activity_array=activity)
    phenotype_df["k_sel"] = k_sel + phenotype_df["dk_geno"].to_numpy()

    # ── Final column ordering ─────────────────────────────────────────────────

    final_columns = list(phenotype_df.columns)
    final_columns.remove("feature_id")
    final_columns.remove("theta")
    final_columns.append("theta")
    phenotype_df = phenotype_df.loc[:, final_columns]

    print("Done.", flush=True)

    return phenotype_df, genotype_theta_df
