"""
Stage 3 of the empirical-phenotype pipeline: resample fresh per-genotype
phenotypes from a Stage-2 ``PopulationModel`` and turn them into the override
dicts that ``thermo_to_growth`` / ``library_prediction`` inject.

Flow: draw one ``(dk_geno, theta_low, theta_high, log_hill_K, hill_n)`` per
library genotype from the fitted generating distribution, then hand those to
the existing ``build_theta_gc_override_hill_geno`` machinery to produce a
``theta_gc_override`` (theta at growth concentrations) and a
``theta_params_override`` (the effective Hill params written to
``parameters_df``), plus a ``dk_geno_override``.  These three dicts are exactly
the injection points ``thermo_to_growth`` already consumes, so no growth-path
changes are needed beyond the ``dk_geno_override`` hook.

Scope
-----
This targets ``hill_geno`` — the theta component whose per-genotype parameters
are independent (no mutation decomposition).  Drawing one i.i.d. phenotype per
genotype from the marginal population is exactly consistent with that
structure.  ``hill_mut`` (per-mutation deltas with epistasis) would instead
require resampling in mutation space and is a future extension.

wt handling
-----------
``wt`` is the reference genotype: its ``dk_geno`` is pinned to 0 (matching the
sim/inference convention) and its Hill curve defaults to the population mean
(back-transformed ``mu``) unless an explicit ``wt_ref`` is supplied.  All other
genotypes are fresh i.i.d. draws.
"""

import numpy as np
import pandas as pd

from tfscreen.genetics import standardize_genotypes
from tfscreen.simulate.sample_theta import _greedy_maximin
from tfscreen.simulate.binding_params import (
    build_theta_gc_override_hill_geno,
    build_binding_theta_from_params,
    read_binding_genotype_params,
    _hill_theta,
    _to_log_conc,
)

# Natural-space Hill columns, in the order the injection machinery expects.
_HILL_COLS = ["theta_low", "theta_high", "log_hill_K", "hill_n"]


def population_mean_params(model):
    """Return the back-transformed population-mean Hill params as a dict."""
    mean_natural = model._to_natural(model.mu)   # 1-row DataFrame
    return {c: float(mean_natural[c].iloc[0]) for c in _HILL_COLS}


def resample_phenotypes(model, genotypes, rng=None, wt_ref=None):
    """Draw one phenotype per genotype from a ``PopulationModel``.

    Parameters
    ----------
    model : PopulationModel
        Fitted Stage-2 generating distribution.
    genotypes : iterable of str
        Library genotypes to assign phenotypes to.  Duplicates are collapsed
        (first occurrence wins for ordering).
    rng : int, np.random.Generator, or None
        Seed / generator for the draws.
    wt_ref : dict or None
        Optional explicit Hill params for ``wt`` (``theta_low``, ``theta_high``,
        ``log_hill_K``, ``hill_n``).  Defaults to the population mean.

    Returns
    -------
    pandas.DataFrame
        One row per genotype: ``genotype`` plus ``dk_geno`` and the four Hill
        columns.  ``wt`` (if present) is first, with ``dk_geno == 0``.
    """
    genotypes = list(dict.fromkeys(genotypes))       # unique, order-preserving
    has_wt = "wt" in genotypes
    non_wt = [g for g in genotypes if g != "wt"]

    draws = model.sample(len(non_wt), rng=rng) if non_wt else \
        model.sample(0)
    draws = draws.reset_index(drop=True)
    draws.insert(0, "genotype", non_wt)

    frames = []
    if has_wt:
        wt_params = dict(wt_ref) if wt_ref is not None \
            else population_mean_params(model)
        wt_row = {"genotype": "wt", "dk_geno": 0.0}
        wt_row.update({c: float(wt_params[c]) for c in _HILL_COLS})
        frames.append(pd.DataFrame([wt_row]))
    frames.append(draws)

    out = pd.concat(frames, ignore_index=True)
    return out[["genotype", "dk_geno"] + _HILL_COLS]


def build_overrides(pheno_df, log_conc_growth, wt_params=None):
    """Turn resampled phenotypes into thermo_to_growth override dicts.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Output of :func:`resample_phenotypes` (``genotype``, ``dk_geno``, and
        the four Hill columns).
    log_conc_growth : array-like, shape (C,)
        Log growth concentrations, from ``sim_data.log_titrant_conc``.
    wt_params : dict or None
        Fallback Hill params for any NaN fields (required by
        ``build_theta_gc_override_hill_geno``).  Since resampled rows are fully
        specified this is never actually consulted; defaults to the first row's
        params so a valid dict is always passed.

    Returns
    -------
    theta_gc_override : dict[str, np.ndarray]
    theta_params_override : dict[str, dict[str, float]]
    dk_geno_override : dict[str, float]
    """
    params_dict = {
        row["genotype"]: {c: float(row[c]) for c in _HILL_COLS}
        for _, row in pheno_df.iterrows()
    }
    if wt_params is None:
        wt_params = next(iter(params_dict.values()))

    theta_gc_override, theta_params_override = build_theta_gc_override_hill_geno(
        params_dict, np.asarray(log_conc_growth, dtype=float), wt_params)

    dk_geno_override = {row["genotype"]: float(row["dk_geno"])
                        for _, row in pheno_df.iterrows()}

    return theta_gc_override, theta_params_override, dk_geno_override


def _resampled_binding_matrix(pheno_by_geno, genos, binding_concs):
    """Resampled binding-theta curves (rows=genos) at the binding concentrations."""
    log_concs = _to_log_conc(np.asarray(binding_concs, dtype=float))
    rows = []
    for g in genos:
        r = pheno_by_geno.loc[g]
        rows.append(_hill_theta(r["theta_low"], r["theta_high"],
                                r["log_hill_K"], r["hill_n"], log_concs))
    return np.asarray(rows)


def build_empirical_binding_theta(pheno_df, binding_cfg, spiked_names, rng):
    """Build the spiked ``binding_theta_df`` from resampled phenotypes.

    *Selection* (which spiked genotypes are measured) follows the
    ``spiked_binding`` block of ``binding_cfg`` — ``file`` (explicit list),
    ``random`` (``num`` at random), or ``stratified`` (greedy-maximin spread
    over the resampled binding curves).  The theta *values* come from each
    genotype's resampled Hill params in ``pheno_df``, keeping binding and
    growth derived from the same ground truth.  Returned rows are noise-free
    (``theta_true``); ``simulate_cli`` applies the assay noise downstream.

    Returns
    -------
    pandas.DataFrame or None
        ``genotype, titrant_name, titrant_conc, theta_true`` — or ``None`` when
        no ``spiked_binding`` is configured.
    """
    sb = binding_cfg.get("spiked_binding")
    if sb is None:
        return None

    titrant_name = binding_cfg["titrant_name"]
    binding_concs = binding_cfg["titrant_conc"]
    choose_by = sb["choose_by"]

    spiked_set = list(dict.fromkeys(standardize_genotypes(spiked_names)))
    pheno_by_geno = pheno_df.set_index("genotype")

    if choose_by not in ("stratified", "random"):
        # file: genotype list from the file's keys (params ignored — theta
        # comes from the resampled phenotypes); must be a subset of spiked.
        genos = list(standardize_genotypes(
            list(read_binding_genotype_params(choose_by).keys())))
        bad = set(genos) - set(spiked_set)
        if bad:
            raise ValueError(
                f"spiked_binding file genotypes must be spiked; not spiked: "
                f"{sorted(bad)}")
    else:
        num = sb.get("num")
        num = len(spiked_set) if num is None else int(num)
        num = min(num, len(spiked_set))
        if choose_by == "random":
            idx = rng.choice(len(spiked_set), size=num, replace=False)
            genos = [spiked_set[i] for i in sorted(idx)]
        else:  # stratified
            mat = _resampled_binding_matrix(pheno_by_geno, spiked_set,
                                            binding_concs)
            sel = _greedy_maximin(mat, num)
            genos = [spiked_set[i] for i in sel]

    params_dict = {g: {c: float(pheno_by_geno.loc[g, c]) for c in _HILL_COLS}
                   for g in genos if g in pheno_by_geno.index}
    if not params_dict:
        return None

    wt_params = next(iter(params_dict.values()))
    return build_binding_theta_from_params(
        params_dict, binding_concs, titrant_name, noise=0.0,
        rng=rng, wt_params=wt_params)


def make_empirical_overrides(model, genotypes, log_conc_growth,
                             rng=None, wt_ref=None):
    """Convenience: resample + build overrides in one call.

    Returns
    -------
    pheno_df : pandas.DataFrame
        The resampled per-genotype phenotypes (ground truth to save alongside).
    theta_gc_override, theta_params_override, dk_geno_override : dict
        Ready to pass to ``thermo_to_growth`` / ``library_prediction``.
    """
    pheno_df = resample_phenotypes(model, genotypes, rng=rng, wt_ref=wt_ref)
    wt_params = population_mean_params(model)
    theta_gc_override, theta_params_override, dk_geno_override = build_overrides(
        pheno_df, log_conc_growth, wt_params=wt_params)
    return (pheno_df, theta_gc_override, theta_params_override,
            dk_geno_override)
