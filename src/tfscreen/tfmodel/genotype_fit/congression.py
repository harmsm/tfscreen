"""
Congression de-attenuation of per-genotype theta curves (co-transformation).

A barcode observed in the bulk is really a cell that received a
zero-truncated-Poisson(lambda) number of plasmids.  Under **dominant-max
occupancy** — the tightest-bound operator in a co-transformed cell sets the
effective theta, regardless of marker — the theta a monoclonal Hill fit
recovers for genotype ``g`` is inflated toward the population's high-occupancy
envelope:

    theta_obs  =  E[ max(theta_g, M) ],   M = max of Poisson(lambda) co-residents

drawn from the population theta distribution.  This is exactly the inference's
own congression operator (``transformation._congression.update_thetas``, the
``E[max(x, M)]`` map with an empirical background CDF), and because a focal
barcode is size-biased into its cell, the co-resident count is Poisson at the
same ``lambda`` the simulator's ``transformation_poisson_lambda`` uses — so no
lambda conversion is needed.

We want ``theta_true`` such that pushing it through that forward map (with the
background built from ``theta_true`` itself) reproduces ``theta_obs``.  That is
a fixed point because the background *is* the corrected population, which
changes every pass.  Per titrant concentration is an independent 1-D
correction; the Hill refit afterward re-links the concentrations into a curve.

This is the standalone (``tfs-fit-genotypes``) and Stage-1.5 (empirical
simulation pipeline) de-attenuation layer; ``simulate/empirical/congression``
re-exports it.

Scope / assumptions
-------------------
* **Dominant-max occupancy only** (single ``E[max]`` operator; no min variant).
* Corrects only **bulk** genotypes; spiked genotypes are congression-free, so
  they are excluded from both the correction and the background CDF and pass
  through unchanged.
* Operates on the genotype × concentration theta *point estimates* (pooled over
  replicate/library by Stage 1); ``dk_geno`` is untouched (it is a growth
  parameter, not part of the theta curve).  Estimation-noise deconvolution
  stays in Stage 2 — this stage only removes the congression *bias*, and keeps
  each genotype's Stage-1 covariance (a deliberate approximation: the bias
  shift barely changes estimation precision).

This is the θ-level analogue of the simulator's growth-level congression; the
two agree to first order (exact at ``lambda -> 0``), and the spiked-only
distribution is the external check on the residual.
"""

import numpy as np
import jax.numpy as jnp
import pandas as pd

from tfscreen.util.io import read_dataframe
from tfscreen.mle.fitters.least_squares import run_least_squares
from tfscreen.tfmodel.genotype_fit.fit import (
    _hill_theta, hill_theta_from_fit, _POWER_CLIP, _LOGIT_BOUND,
)
from tfscreen.tfmodel.generative.components.transformation._congression import (
    update_thetas,
)
from scipy.special import expit  # noqa: F401  (kept for backward-compat imports)

# Positions within the 5-element pheno block (PHENO_PARAMS_TRANSFORMED) of the
# theta-curve params: logit_theta_low, logit_theta_high, log_hill_K, log_hill_n.
# Index 0 (dk_geno) is a growth parameter and is left untouched.
_THETA_PARAM_IDX = np.array([1, 2, 3, 4])

_THETA_EPS = 1e-6


def _theta_from_fit(fit, concs):
    """Backward-compatible alias for :func:`fit.hill_theta_from_fit`."""
    return hill_theta_from_fit(fit, concs)


def correct_theta_matrix(theta_obs, lam, gain=1.0, tol=1e-4, max_iter=50,
                         n_grid=256, return_history=False):
    """Fixed-point de-attenuation of an observed-theta matrix.

    Parameters
    ----------
    theta_obs : np.ndarray, shape (n_conc, n_geno)
        Observed theta per concentration (rows) and genotype (cols).  Each row
        is corrected against its own population CDF.
    lam : float
        Poisson co-resident rate (== the zero-truncated ``transformation_poisson_lambda``).
    gain, tol, max_iter, n_grid : see module notes.
    return_history : bool
        If True, also return the list of intermediate ``theta_true`` matrices,
        one per iteration (index 0 is the initial clamped ``theta_obs``, the
        last entry is the converged estimate).  Lets callers report the
        convergence trajectory per concentration.

    Returns
    -------
    theta_true : np.ndarray, shape (n_conc, n_geno)
    n_iter : int
    history : list of np.ndarray
        Only when ``return_history`` is True.
    """
    theta_obs = np.asarray(theta_obs, dtype=float)
    theta_true = np.clip(theta_obs.copy(), _THETA_EPS, 1.0 - _THETA_EPS)
    history = [theta_true.copy()]

    if lam is None or float(lam) <= 0:
        out = theta_obs.copy()
        return (out, 0, history) if return_history else (out, 0)

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        theta_pred = np.asarray(update_thetas(
            jnp.asarray(theta_true), (float(lam),), theta_dist="empirical",
            population_theta=jnp.asarray(theta_true), n_grid=n_grid))
        resid = theta_obs - theta_pred
        if np.nanmax(np.abs(resid)) < tol:
            break
        theta_true = np.clip(theta_true + gain * resid, _THETA_EPS, 1.0 - _THETA_EPS)
        history.append(theta_true.copy())

    return (theta_true, n_iter, history) if return_history else (theta_true, n_iter)


def _refit_hill_theta(concs, theta, guess_pheno):
    """Refit the 4 transformed theta-Hill params to a corrected theta curve.

    ``guess_pheno`` is the genotype's current pheno block (used to seed).
    Returns ``(logit_low, logit_high, log_K, log_n)``.
    """
    def model(p_t, x):
        low = expit(p_t[0])
        high = expit(p_t[1])
        log_K = p_t[2]
        n = np.exp(p_t[3])
        return _hill_theta(x, low, high, log_K, n)

    guess = np.asarray(guess_pheno)[_THETA_PARAM_IDX].astype(float)
    lower = np.array([-_LOGIT_BOUND, -_LOGIT_BOUND, np.log(1e-12), np.log(0.05)])
    upper = np.array([_LOGIT_BOUND, _LOGIT_BOUND, np.log(1e6), np.log(_POWER_CLIP)])
    est, _std, _cov, _fit = run_least_squares(
        some_model=model,
        obs=np.asarray(theta, dtype=float),
        obs_std=np.ones_like(theta, dtype=float),
        guesses=guess, lower_bounds=lower, upper_bounds=upper,
        args=(np.asarray(concs, dtype=float),))
    return est


def deattenuate_congression(fits, growth_df, lam, spiked=None,
                            gain=1.0, tol=1e-4, max_iter=50, n_grid=256,
                            return_theta_history=False):
    """Congression-correct the bulk theta curves in a Stage-1 ``fits`` dict.

    Parameters
    ----------
    fits : dict
        ``{(genotype, titrant_name): GenotypeFit}`` from ``fit_phenotypes``.
    growth_df : pandas.DataFrame or str
        The real ln_cfu data (supplies the per-titrant concentration grid the
        theta curves are corrected on).
    lam : float or None
        Zero-truncated Poisson congression rate.  ``None``/``<=0`` -> no-op.
    spiked : iterable of str or None
        Congression-free genotypes to exclude from correction and background.
    gain, tol, max_iter, n_grid : fixed-point controls.
    return_theta_history : bool
        If True, also return a long-form DataFrame of the fixed-point theta
        trajectory (columns ``genotype, titrant_name, titrant_conc, iter,
        theta``), for diagnosing the convergence.

    Returns
    -------
    corrected : dict
        A new fits dict: bulk genotypes' theta-Hill params de-attenuated (Hill
        refit; dk_geno and covariance unchanged); spiked genotypes untouched.
    history_df : pandas.DataFrame
        Only when ``return_theta_history`` is True.
    """
    empty_hist = pd.DataFrame(
        columns=["genotype", "titrant_name", "titrant_conc", "iter", "theta"])

    if lam is None or float(lam) <= 0:
        out = dict(fits)
        return (out, empty_hist) if return_theta_history else out

    spiked = set(spiked or [])
    growth_df = read_dataframe(growth_df)
    corrected = dict(fits)
    hist_rows = []

    for titrant_name, sub in growth_df.groupby("titrant_name", observed=True):
        concs = np.sort(sub["titrant_conc"].unique().astype(float))

        keys = [k for k in fits
                if k[1] == titrant_name and k[0] not in spiked
                and np.all(np.isfinite(fits[k].est_t))]
        if len(keys) < 2:
            continue   # need a population to build a background CDF

        # (n_conc, n_geno) observed theta.
        theta_obs = np.stack([hill_theta_from_fit(fits[k], concs) for k in keys],
                             axis=1)
        result = correct_theta_matrix(
            theta_obs, float(lam), gain=gain, tol=tol, max_iter=max_iter,
            n_grid=n_grid, return_history=return_theta_history)

        if return_theta_history:
            theta_true, _, history = result
            for it, mat in enumerate(history):
                for i_c, c in enumerate(concs):
                    for j, k in enumerate(keys):
                        hist_rows.append({
                            "genotype": k[0], "titrant_name": titrant_name,
                            "titrant_conc": float(c), "iter": it,
                            "theta": float(mat[i_c, j])})
        else:
            theta_true, _ = result

        for j, k in enumerate(keys):
            fit = fits[k]
            new_theta_t = _refit_hill_theta(
                concs, theta_true[:, j], fit.est_t[fit.pheno_slice])
            est_t = np.array(fit.est_t, dtype=float)
            pheno = est_t[fit.pheno_slice].copy()
            pheno[_THETA_PARAM_IDX] = new_theta_t
            est_t[fit.pheno_slice] = pheno
            corrected[k] = fit._replace(est_t=est_t)

    if return_theta_history:
        history_df = (pd.DataFrame(hist_rows, columns=empty_hist.columns)
                      if hist_rows else empty_hist)
        return corrected, history_df
    return corrected
