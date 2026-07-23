"""
Per-genotype MLE fitting of the growth model against *real* experimental data.

This is a non-Bayesian, per-genotype inference engine for the same generative
growth model that ``tfmodel`` fits jointly by SVI/NUTS.  Given a processed
``ln_cfu`` DataFrame (the output of ``tfs-process-counts``) and a set of
*frozen* per-condition growth-calibration parameters (``k``, ``m`` from
``tfs-prefit-calibration``), it fits an independent, lightly regularized
nonlinear least-squares model to each genotype's ~O(100) observations:

    ln_cfu = ln_cfu0
             + (k_pre + dk_geno + m_pre * theta) * t_pre
             + (k_sel + dk_geno + m_sel * theta) * t_sel

where ``theta`` is a 4-parameter Hill curve in the titrant concentration
(``theta_low``, ``theta_high``, ``log_hill_K``, ``hill_n``) and activity is
asserted to be 1 (appropriate for a repressor: it blocks transcription or it
does not; leaky binding is absorbed into ``theta_low``).

It is exposed directly by ``tfs-fit-genotypes`` and is also Stage 1 of the
empirical-phenotype simulation pipeline (``simulate/empirical/``), whose
``fit_phenotypes`` re-exports this module.  In the simulation use, the purpose
is *not* genotype-specific truth — the per-genotype fits are confounded (see
the module notes) and, on bulk data, congression-attenuated — but to harvest a
realistic *joint distribution* of per-genotype phenotype parameters that a
later stage resamples.  For that, each fit returns both a point estimate **and**
its covariance, both in the transformed coordinate system in which the
downstream distribution is estimated and sampled.

Design / identifiability notes
------------------------------
* The fit is done **per genotype, calibration frozen** — there is no
  cross-genotype coupling and the condition ``k``/``m`` never re-float.  This
  is deliberate: a joint/global fit would let parameters soak up variance and
  re-open the k/dk_geno slide.
* ``dk_geno`` and the vertical placement of the theta curve (``theta_low``)
  are only softly separated.  What is robustly identified is the titration
  *shape* (``log_hill_K``, ``hill_n``) and *amplitude*
  (``theta_high - theta_low``).  The absolute level is separated only by the
  contrast between the two selective markers (different ``m_sel``); a weak
  prior on ``dk_geno`` (``dk_geno_prior_sd``) is applied as belt-and-suspenders.
* ``theta_low``/``theta_high`` are fit through a logit transform, so they are
  intrinsically confined to ``(0, 1)`` and cannot run off to the
  ``logit ~= +-16`` artefact that a raw-space fit produces.
* Per-tube starting-abundance variation is handled with a nuisance intercept
  per ``intercept_cols`` group (default: one ``ln_cfu0`` per ``replicate``).
  These intercepts are marginalized away for the downstream distribution.

Assumptions to confirm against real data (flagged intentionally)
----------------------------------------------------------------
1.  Titrant concentration is constant within a sample across the pre- and
    sel-phases, so a single ``theta`` enters both phase terms.
2.  ``condition_pre`` / ``condition_sel`` strings match the ``condition_rep``
    keys used by the calibration table verbatim (this is the same identity
    used on the fit side; see ``model_orchestrator._build_growth_tm``).
3.  The Hill functional form here mirrors the ``hill_geno`` theta component so
    that resampled parameters reproduce the fitted curves in Stage 3.
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit
import tqdm
import warnings
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

from tfscreen.util.io import read_dataframe
from tfscreen.util.dataframe import check_columns, get_scaled_cfu
from tfscreen.util import resolve_workers as _resolve_workers
from tfscreen.mle.fitters.least_squares import run_least_squares

# Numerical guards, matched to tfscreen.mle.curve_models.models so the fitted
# Hill is numerically identical to the empirical-curve library.
_EXP_CLIP = 700.0
_POWER_CLIP = 25.0

# Zero-concentration substitution, matched to simulate.binding_params so the
# Hill evaluated when *fitting* here is identical to the one used when the
# resampled parameters are *injected* in Stage 3 (and to hill_geno inference).
_ZERO_CONC_SENTINEL = 1e-20

# Logit-space bound on theta_low/theta_high (mirrors FitManager's stability
# bound): keeps theta strictly inside (expit(-16), expit(16)) so the fit
# cannot float-saturate to exactly 0 or 1 when the dk_geno/theta_low
# degeneracy pushes the vertical placement to an asymptote.
_LOGIT_BOUND = 16.0

# Phenotype parameters carried forward to the distribution-estimation stage,
# in their transformed (fit) coordinates.  The nuisance intercepts are *not*
# in this list.
PHENO_PARAMS_TRANSFORMED = [
    "dk_geno",
    "logit_theta_low",
    "logit_theta_high",
    "log_hill_K",
    "log_hill_n",
]

# Natural-space names, aligned index-for-index with PHENO_PARAMS_TRANSFORMED.
PHENO_PARAMS_NATURAL = [
    "dk_geno",
    "theta_low",
    "theta_high",
    "log_hill_K",
    "hill_n",
]

_Design = namedtuple(
    "_Design",
    ["conc", "t_pre", "t_sel", "k_pre", "m_pre", "k_sel", "m_sel",
     "intercept_onehot", "n_intercept", "prior_idx"],
)

# Per-genotype fit result.  ``est_t`` / ``cov_t`` are the *full* transformed
# parameter vector and covariance (intercepts first, then the five phenotype
# params in PHENO_PARAMS_TRANSFORMED order); ``pheno_slice`` indexes the
# phenotype block for the downstream stage.
GenotypeFit = namedtuple(
    "GenotypeFit",
    ["genotype", "titrant_name", "n_obs", "converged",
     "param_names_t", "est_t", "cov_t", "pheno_slice"],
)


def _hill_theta(conc, theta_low, theta_high, log_K, n):
    """Hill occupancy at ``conc`` (mirrors curve_models.models._hill).

    ``theta = theta_low + (theta_high - theta_low) * x^n / (x^n + K^n)`` with
    ``x = conc`` and ``K = exp(log_K)``.  For a repressor induced by titrant,
    ``theta_high < theta_low`` (occupancy falls as titrant rises).

    This is identical to ``hill_geno.run_model``'s
    ``sigmoid(n * (ln x - ln K))`` occupancy and to
    ``simulate.binding_params._hill_theta`` (the Stage-3 injection path); zero
    concentrations use the same ``_ZERO_CONC_SENTINEL`` substitution as the
    injection path so fitted and injected curves coincide.
    """
    conc = np.asarray(conc, dtype=float)
    x_safe = np.where(conc == 0.0, _ZERO_CONC_SENTINEL, conc)
    n_safe = np.clip(n, -_POWER_CLIP, _POWER_CLIP)
    ln_K_to_n = np.clip(n_safe * log_K, -_EXP_CLIP, _EXP_CLIP)
    ln_x_to_n = np.clip(n_safe * np.log(x_safe), -_EXP_CLIP, _EXP_CLIP)
    K_to_n = np.exp(ln_K_to_n)
    x_to_n = np.exp(ln_x_to_n)
    fx = x_to_n / (x_to_n + K_to_n)
    return theta_low + (theta_high - theta_low) * fx


def hill_theta_from_fit(fit, concs):
    """Evaluate a genotype's fitted Hill curve at ``concs`` (natural conc).

    Reads the theta-curve params out of a :class:`GenotypeFit`'s phenotype
    block and evaluates :func:`_hill_theta`.  Used by :func:`predict_theta` and
    by the congression de-attenuation stage.
    """
    pheno = np.asarray(fit.est_t)[fit.pheno_slice]
    theta_low = expit(pheno[1])
    theta_high = expit(pheno[2])
    log_K = pheno[3]
    n = np.exp(pheno[4])
    return _hill_theta(concs, theta_low, theta_high, log_K, n)


def _forward(p_t, design):
    """Predict ln_cfu (and prior pseudo-observations) from transformed params.

    Parameter layout in ``p_t``:
        [ln_cfu0_1 ... ln_cfu0_G, dk_geno, logit_theta_low,
         logit_theta_high, log_hill_K, log_hill_n]
    where ``G = design.n_intercept``.
    """
    n_int = design.n_intercept
    ln_cfu0_vec = p_t[:n_int]
    dk_geno = p_t[n_int + 0]
    theta_low = expit(p_t[n_int + 1])
    theta_high = expit(p_t[n_int + 2])
    log_K = p_t[n_int + 3]
    n = np.exp(p_t[n_int + 4])

    theta = _hill_theta(design.conc, theta_low, theta_high, log_K, n)
    rate_pre = design.k_pre + dk_geno + design.m_pre * theta
    rate_sel = design.k_sel + dk_geno + design.m_sel * theta

    ln_cfu0_row = design.intercept_onehot @ ln_cfu0_vec
    pred = ln_cfu0_row + rate_pre * design.t_pre + rate_sel * design.t_sel

    # Append the regularized parameter values as pseudo-observations so a
    # Gaussian prior enters as extra (Tikhonov) residuals in the same
    # least-squares problem.
    if design.prior_idx is not None:
        pred = np.concatenate([pred, p_t[design.prior_idx]])

    return pred


# prefit priors-CSV parameter names -> wide calibration columns (linear model).
# The prefit writes these via _csv_row_name = "growth.{component}.{field}".
_CALIB_PRIORS_MAP = {
    "growth.condition_growth.k_loc": "growth_k",
    "growth.condition_growth.m_loc": "growth_m",
}


def _pivot_priors_calibration(df):
    """Reshape a prefit priors CSV (long form) to wide (condition_rep, k, m)."""
    if "value" not in df.columns:
        raise ValueError("priors CSV lacks a 'value' column.")

    sub = df[df["parameter"].isin(list(_CALIB_PRIORS_MAP))].copy()
    if sub.empty:
        raise ValueError(
            "priors CSV has no 'growth.condition_growth.k_loc' / "
            "'growth.condition_growth.m_loc' rows; run tfs-prefit-calibration on "
            "a linear condition_growth model first (the empirical pipeline "
            "assumes the linear growth model).")

    if "condition_rep" not in sub.columns or sub["condition_rep"].isna().all():
        raise ValueError(
            "growth k/m priors are present but carry no per-condition "
            "'condition_rep' labels — this looks like a fresh configure. Run "
            "tfs-prefit-calibration to write the per-condition calibration first.")

    sub = sub.dropna(subset=["condition_rep"])
    sub["_field"] = sub["parameter"].map(_CALIB_PRIORS_MAP)
    wide = (sub.pivot_table(index="condition_rep", columns="_field",
                            values="value", aggfunc="first")
               .reset_index())
    wide.columns.name = None

    missing = {"growth_k", "growth_m"} - set(wide.columns)
    if missing:
        raise ValueError(
            f"priors CSV is missing calibration rows for: {sorted(missing)} "
            f"(expected both condition_growth_k_loc and condition_growth_m_loc).")

    wide["growth_k"] = wide["growth_k"].astype(float)
    wide["growth_m"] = wide["growth_m"].astype(float)
    return wide[["condition_rep", "growth_k", "growth_m"]]


def read_calibration(calib):
    """Normalize a calibration source to wide ``(condition_rep, growth_k, growth_m)``.

    Accepts, transparently:

    * the **priors CSV** written by ``tfs-prefit-calibration`` (long form: a
      ``parameter`` column with ``growth.condition_growth.k_loc`` /
      ``growth.condition_growth.m_loc`` rows carrying a ``condition_rep`` label), or
    * an already-wide table (columns ``condition_rep``, ``growth_k``,
      ``growth_m``),

    as either a path or a DataFrame.
    """
    df = read_dataframe(calib)

    if {"condition_rep", "growth_k", "growth_m"}.issubset(df.columns):
        out = df[["condition_rep", "growth_k", "growth_m"]].copy()
        out["growth_k"] = out["growth_k"].astype(float)
        out["growth_m"] = out["growth_m"].astype(float)
        return out

    if "parameter" in df.columns:
        return _pivot_priors_calibration(df)

    raise ValueError(
        "calibration must be either a wide table with columns "
        "[condition_rep, growth_k, growth_m] or a prefit priors CSV with "
        "'parameter' rows growth.condition_growth.k_loc / "
        "growth.condition_growth.m_loc.")


def _build_calib_lookup(calib):
    """Return (k_map, m_map): dicts condition_rep -> growth_k / growth_m.

    ``calib`` may be a mapping ``{condition_rep: {"growth_k": ..,
    "growth_m": ..}}``, or anything :func:`read_calibration` accepts (a path or
    DataFrame in either the prefit priors form or the wide form).
    """
    if isinstance(calib, dict):
        k_map, m_map = {}, {}
        for cond, d in calib.items():
            k_map[cond] = d["growth_k"]
            m_map[cond] = d["growth_m"]
        return k_map, m_map

    calib_df = read_calibration(calib)
    k_map = dict(zip(calib_df["condition_rep"], calib_df["growth_k"]))
    m_map = dict(zip(calib_df["condition_rep"], calib_df["growth_m"]))
    return k_map, m_map


def _lookup_km(conditions, k_map, m_map, which):
    """Vectorized (k, m) lookup with a fail-fast on unknown conditions."""
    conditions = np.asarray(conditions, dtype=object)
    missing = sorted({c for c in conditions if c not in k_map})
    if missing:
        raise ValueError(
            f"condition_{which} value(s) absent from the calibration table: "
            f"{missing}. The per-genotype fit requires frozen k/m for every "
            f"condition; run tfs-prefit-calibration first.")
    k = np.array([k_map[c] for c in conditions], dtype=float)
    m = np.array([m_map[c] for c in conditions], dtype=float)
    return k, m


def _initial_guess_transformed(sub, n_int):
    """Transformed-space initial guesses for one genotype group."""
    pos_conc = sub["titrant_conc"].to_numpy(dtype=float)
    pos_conc = pos_conc[pos_conc > 0]
    log_K0 = np.log(np.median(pos_conc)) if pos_conc.size else 0.0

    guess = np.empty(n_int + 5, dtype=float)
    guess[:n_int] = float(np.nanmedian(sub["ln_cfu"].to_numpy(dtype=float)))
    guess[n_int + 0] = 0.0                       # dk_geno
    guess[n_int + 1] = logit(0.9)                # theta_low  (repressed at 0)
    guess[n_int + 2] = logit(0.1)                # theta_high (induced)
    guess[n_int + 3] = log_K0                    # log_hill_K
    guess[n_int + 4] = 0.0                        # log_hill_n  (n = 1)
    return guess


def fit_one_genotype(sub, k_map, m_map, intercept_cols, dk_geno_prior=(0.0, 1.0)):
    """Fit the growth model to a single (genotype, titrant_name) group.

    Parameters
    ----------
    sub : pandas.DataFrame
        Rows for one genotype/titrant with columns ``ln_cfu``, ``ln_cfu_std``,
        ``t_pre``, ``t_sel``, ``titrant_conc``, ``condition_pre``,
        ``condition_sel`` and every column in ``intercept_cols``.
    k_map, m_map : dict
        ``condition_rep -> growth_k`` and ``-> growth_m`` (frozen calibration).
    intercept_cols : list of str
        Columns whose unique combinations each get their own nuisance
        ``ln_cfu0``.  Empty list -> a single shared intercept.
    dk_geno_prior : (loc, sd) or None
        Weak Gaussian prior on ``dk_geno`` (natural == transformed space).
        ``None`` disables regularization.

    Returns
    -------
    GenotypeFit
    """
    # Drop unusable observations (dead / zero-cfu rows carry NaN ln_cfu or a
    # non-finite / non-positive std that run_least_squares cannot weight).
    finite = (np.isfinite(sub["ln_cfu"].to_numpy(dtype=float))
              & np.isfinite(sub["ln_cfu_std"].to_numpy(dtype=float))
              & (sub["ln_cfu_std"].to_numpy(dtype=float) > 0))
    sub = sub.loc[finite]

    n_obs = len(sub)

    # Build the nuisance-intercept one-hot design.
    if intercept_cols:
        keys = sub[intercept_cols].astype(str).agg("|".join, axis=1)
        cats = pd.Categorical(keys)
        onehot = np.zeros((n_obs, len(cats.categories)), dtype=float)
        onehot[np.arange(n_obs), cats.codes] = 1.0
    else:
        onehot = np.ones((n_obs, 1), dtype=float)
    n_int = onehot.shape[1]

    param_names_t = ([f"ln_cfu0[{i}]" for i in range(n_int)]
                     + PHENO_PARAMS_TRANSFORMED)
    pheno_slice = slice(n_int, n_int + 5)

    # A group with fewer observations than parameters cannot be fit.
    n_params = n_int + 5
    if n_obs < n_params + 1:
        return GenotypeFit(
            genotype=sub["genotype"].iloc[0] if n_obs else None,
            titrant_name=sub["titrant_name"].iloc[0] if n_obs else None,
            n_obs=n_obs, converged=False, param_names_t=param_names_t,
            est_t=np.full(n_params, np.nan),
            cov_t=np.full((n_params, n_params), np.nan),
            pheno_slice=pheno_slice)

    k_pre, m_pre = _lookup_km(sub["condition_pre"], k_map, m_map, "pre")
    k_sel, m_sel = _lookup_km(sub["condition_sel"], k_map, m_map, "sel")

    # Assemble prior pseudo-observations (currently only dk_geno).
    if dk_geno_prior is not None:
        prior_idx = np.array([n_int + 0])
        prior_loc = np.array([dk_geno_prior[0]], dtype=float)
        prior_sd = np.array([dk_geno_prior[1]], dtype=float)
    else:
        prior_idx = None
        prior_loc = np.array([], dtype=float)
        prior_sd = np.array([], dtype=float)

    design = _Design(
        conc=sub["titrant_conc"].to_numpy(dtype=float),
        t_pre=sub["t_pre"].to_numpy(dtype=float),
        t_sel=sub["t_sel"].to_numpy(dtype=float),
        k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel,
        intercept_onehot=onehot, n_intercept=n_int, prior_idx=prior_idx)

    obs = np.concatenate([sub["ln_cfu"].to_numpy(dtype=float), prior_loc])
    obs_std = np.concatenate([sub["ln_cfu_std"].to_numpy(dtype=float), prior_sd])

    guess = _initial_guess_transformed(sub, n_int)

    # Bounds: intercepts and dk_geno free; theta logits clamped to keep theta
    # in (0, 1); log_hill_K to a wide but finite window; log_hill_n to the
    # component's power-clip range.
    lower = np.full(n_int + 5, -np.inf)
    upper = np.full(n_int + 5, np.inf)
    lower[n_int + 1:n_int + 3] = -_LOGIT_BOUND
    upper[n_int + 1:n_int + 3] = _LOGIT_BOUND
    lower[n_int + 3], upper[n_int + 3] = np.log(1e-12), np.log(1e6)
    lower[n_int + 4], upper[n_int + 4] = np.log(0.05), np.log(_POWER_CLIP)

    est_t, _std_t, cov_t, fit = run_least_squares(
        some_model=_forward, obs=obs, obs_std=obs_std,
        guesses=guess, lower_bounds=lower, upper_bounds=upper, args=(design,))

    converged = bool(getattr(fit, "success", False))

    return GenotypeFit(
        genotype=sub["genotype"].iloc[0],
        titrant_name=sub["titrant_name"].iloc[0],
        n_obs=n_obs, converged=converged, param_names_t=param_names_t,
        est_t=est_t, cov_t=cov_t, pheno_slice=pheno_slice)


def _natural_from_transformed(est_t, pheno_slice):
    """Back-transform the 5-element phenotype block to natural space."""
    dk, lo_t, hi_t, logK, logn = est_t[pheno_slice]
    return {
        "dk_geno": dk,
        "theta_low": expit(lo_t),
        "theta_high": expit(hi_t),
        "log_hill_K": logK,
        "hill_n": np.exp(logn),
    }


def fits_to_results_df(fits):
    """Assemble the per-genotype results table from a ``fits`` dict.

    ``fits`` is ``{(genotype, titrant_name): GenotypeFit}`` (as returned by
    :func:`fit_phenotypes` or produced by the congression de-attenuation
    stage).  Returns one row per fit with ``n_obs``, ``converged``, the five
    natural-space phenotype parameters, and their transformed-space
    estimate/std (``<param>_t``, ``<param>_t_std``).  Dict insertion order is
    preserved.

    Because this reads only ``est_t``/``cov_t``, it produces the same schema
    whether the fits are raw (Stage 1) or de-attenuated (Stage 1.5), which is
    how ``tfs-fit-genotypes`` emits the raw and corrected parameter tables.
    """
    rows = []
    for (geno, titr), gf in fits.items():
        natural = _natural_from_transformed(gf.est_t, gf.pheno_slice)
        # A near-singular least-squares fit can return a covariance with tiny
        # negative diagonal entries (non-PSD); clamp at 0 before sqrt so the
        # diagnostic std is 0 rather than NaN and no RuntimeWarning is raised.
        # Stage-2 repairs these covariances properly via _nearest_psd.
        var_t = np.clip(np.diag(gf.cov_t)[gf.pheno_slice], 0.0, None)
        std_t = np.sqrt(var_t)
        row = {"genotype": geno, "titrant_name": titr,
               "n_obs": gf.n_obs, "converged": gf.converged}
        row.update(natural)
        for name, e, s in zip(PHENO_PARAMS_TRANSFORMED,
                              gf.est_t[gf.pheno_slice], std_t):
            row[f"{name}_t"] = e
            row[f"{name}_t_std"] = s
        rows.append(row)

    if not rows:
        warnings.warn("fits_to_results_df received no fits (empty input).")

    return pd.DataFrame(rows)


def predict_theta(fits, growth_df, theta_col="theta"):
    """Long-form theta predictions from fitted Hill curves.

    Evaluates each genotype's fitted Hill curve on its titrant's
    sorted-unique concentration grid (taken from ``growth_df``).

    Parameters
    ----------
    fits : dict
        ``{(genotype, titrant_name): GenotypeFit}``.
    growth_df : pandas.DataFrame or str
        The ln_cfu data (supplies the per-titrant concentration grid).
    theta_col : str
        Name of the output theta column (default ``"theta"``).  Pass a
        distinct name to concatenate/merge raw and de-attenuated predictions.

    Returns
    -------
    pandas.DataFrame
        Columns ``[genotype, titrant_name, titrant_conc, <theta_col>]``, one
        row per (genotype, titrant_name, titrant_conc).  Non-finite fits emit
        NaN theta rather than being dropped.
    """
    growth_df = read_dataframe(growth_df)
    grids = {t: np.sort(sub["titrant_conc"].unique().astype(float))
             for t, sub in growth_df.groupby("titrant_name", observed=True)}

    rows = []
    for (geno, titr), fit in fits.items():
        concs = grids.get(titr)
        if concs is None:
            continue
        if np.all(np.isfinite(fit.est_t)):
            theta = hill_theta_from_fit(fit, concs)
        else:
            theta = np.full(concs.shape, np.nan)
        for c, th in zip(concs, theta):
            rows.append({"genotype": geno, "titrant_name": titr,
                         "titrant_conc": float(c), theta_col: float(th)})

    return pd.DataFrame(
        rows, columns=["genotype", "titrant_name", "titrant_conc", theta_col])


# Read-only per-worker state for the process pool, populated once per worker by
# ``_init_worker`` so the (small) calibration maps are not re-pickled per task.
_WORKER_STATE = {}


def _init_worker(k_map, m_map, intercept_cols, dk_geno_prior):
    _WORKER_STATE.update(k_map=k_map, m_map=m_map,
                         intercept_cols=intercept_cols,
                         dk_geno_prior=dk_geno_prior)


def _fit_one_task(sub):
    """Pool worker: fit one genotype group using the shared worker state."""
    return fit_one_genotype(sub, _WORKER_STATE["k_map"], _WORKER_STATE["m_map"],
                            _WORKER_STATE["intercept_cols"],
                            dk_geno_prior=_WORKER_STATE["dk_geno_prior"])


def fit_phenotypes(growth_df,
                   calib,
                   intercept_cols=("replicate",),
                   dk_geno_prior=(0.0, 1.0),
                   min_obs=None,
                   progress=True,
                   num_workers=1):
    """Fit the growth model to every genotype in a real ``ln_cfu`` DataFrame.

    Parameters
    ----------
    growth_df : pandas.DataFrame or str
        Processed ln_cfu data (``tfs-process-counts`` output) or a path to it.
        Required columns: ``genotype``, ``titrant_name``, ``titrant_conc``,
        ``condition_pre``, ``condition_sel``, ``t_pre``, ``t_sel``, ``ln_cfu``,
        ``ln_cfu_std``, plus every column in ``intercept_cols``.
    calib : pandas.DataFrame or dict
        Frozen per-condition growth calibration keyed by ``condition_rep``,
        with ``growth_k`` and ``growth_m`` (see ``_build_calib_lookup``).
    intercept_cols : sequence of str
        Columns whose unique combinations each get a nuisance ``ln_cfu0``.
        Default one intercept per ``replicate``; pass ``()`` for a single
        shared intercept.
    dk_geno_prior : (loc, sd) or None
        Weak Gaussian prior on ``dk_geno``.  ``None`` disables it.
    min_obs : int or None
        Skip genotype groups with fewer than this many usable observations.
    progress : bool
        Show a tqdm progress bar.
    num_workers : int
        Per-genotype fits are independent, so they can run in parallel over a
        process pool.  ``1`` (default) fits serially; ``-1`` uses
        ``os.cpu_count() - 1`` workers; ``N`` uses ``N`` workers.

    Returns
    -------
    results_df : pandas.DataFrame
        One row per (genotype, titrant_name): ``n_obs``, ``converged``, the
        five natural-space phenotype parameters, and their transformed-space
        estimate/std (``<param>_t``, ``<param>_t_std``) for the downstream
        distribution stage.
    fits : dict
        ``(genotype, titrant_name) -> GenotypeFit`` carrying the full
        transformed estimate and covariance (the Stage-2 deconvolution input).
    """
    growth_df = read_dataframe(growth_df)
    # Real tfs-process-counts output carries ln_cfu + ln_cfu_var; derive
    # ln_cfu / ln_cfu_std here exactly as the fit side (model_orchestrator) does.
    growth_df = get_scaled_cfu(growth_df, need_columns=["ln_cfu", "ln_cfu_std"])
    intercept_cols = list(intercept_cols)

    required = ["genotype", "titrant_name", "titrant_conc",
                "condition_pre", "condition_sel", "t_pre", "t_sel",
                "ln_cfu", "ln_cfu_std"] + intercept_cols
    check_columns(growth_df, required_columns=required)

    k_map, m_map = _build_calib_lookup(calib)

    workers = _resolve_workers(num_workers)

    # For the parallel path, drop the genotype Categorical: it carries the full
    # category list on every group, so pickling each sub-frame to a worker would
    # be O(num_genotype) per task (O(N^2) overall).  Plain strings pickle in
    # O(rows).  fit_one_genotype reads the genotype via ``.iloc[0]``, so a str
    # column behaves identically.
    if workers != 1 and str(growth_df["genotype"].dtype) == "category":
        growth_df = growth_df.copy()
        growth_df["genotype"] = growth_df["genotype"].astype(str)

    # Collect the per-genotype work items (each is an independent fit).
    keys, subs = [], []
    for key, sub in growth_df.groupby(["genotype", "titrant_name"],
                                      observed=True, sort=False):
        if min_obs is not None and len(sub) < min_obs:
            continue
        keys.append(key)
        subs.append(sub)

    if workers == 1:
        it = tqdm.tqdm(subs, desc="fitting genotypes") if progress else subs
        gfs = [fit_one_genotype(sub, k_map, m_map, intercept_cols,
                                dk_geno_prior=dk_geno_prior) for sub in it]
    else:
        chunksize = max(1, len(subs) // (workers * 8)) if subs else 1
        with ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker,
                initargs=(k_map, m_map, intercept_cols, dk_geno_prior)) as ex:
            mapped = ex.map(_fit_one_task, subs, chunksize=chunksize)
            if progress:
                mapped = tqdm.tqdm(mapped, total=len(subs),
                                   desc=f"fitting genotypes ({workers} workers)")
            gfs = list(mapped)

    fits = {key: gf for key, gf in zip(keys, gfs)}
    results_df = fits_to_results_df(fits)
    return results_df, fits
