"""
Post-hoc assessment of fitted categorical-response curves.

Where ``cat_fit`` answers "which *shape* best explains this curve?" (AICc model
selection), this module answers the orthogonal magnitude question: "is the curve
distinguishable from zero, and where — and when it looks flat, do we *know* it is
flat or is the error just too large to tell?"

Two per-curve summaries drive downstream filtering:

- **omnibus test** (``assess_best_model``): a single Wald / Mahalanobis
  statistic ``W = yhat @ pinv(Sigma) @ yhat`` on the best model's predicted
  values at the observed x, using the full prediction covariance. It rewards
  both the magnitude of the deviation and how many points deviate, and it
  accounts for the strong correlation among predicted points (they share a
  handful of fit parameters). ``W ~ chi2`` with ``df = rank(Sigma)`` (the number
  of free parameters). This is the "distinguishable from zero" verdict; combine
  across curves with :func:`benjamini_hochberg`.

- **equivalence test** (:func:`classify_equiv`): a point is ``equiv_zero`` when
  its whole confidence interval falls inside a region of practical equivalence
  ``[-delta, delta]`` centered on zero. This is what separates "confidently
  flat" from "too noisy to tell" — a plain significance test cannot. ``delta``
  defaults to a multiple of the median predicted standard error
  (:func:`compute_delta`), i.e. a *detectability* threshold; pass an explicit
  ``delta`` for a fixed biologically-meaningful region instead.
"""

import numpy as np
from scipy.stats import chi2, norm

from tfscreen.mle import predict_with_error


def assess_best_model(model_func, params, cov_matrix, x, alpha=0.05):
    """
    Evaluate the best-fit model at the observed x and test it against zero.

    Parameters
    ----------
    model_func : callable
        The fitted model, signature ``model_func(params, x)``.
    params : np.ndarray
        Best-fit parameters.
    cov_matrix : np.ndarray
        Covariance matrix of the fitted parameters. May contain NaN (e.g. when
        the fit failed); in that case per-point errors and the omnibus test are
        returned as NaN.
    x : np.ndarray
        The observed independent-variable values (typically the ~8 titrant
        concentrations). Predictions and per-point tests are evaluated here.
    alpha : float, optional
        Two-sided significance level for the per-point ``sig_nonzero`` flag.
        Default 0.05.

    Returns
    -------
    per_point : dict
        Arrays of length ``len(x)``: ``x``, ``y_model`` (best-fit curve value),
        ``y_model_std`` (propagated fit uncertainty), ``z``, ``sig_nonzero``
        (bool), ``direction`` (-1/0/+1).
    rollup : dict
        Scalars: ``omnibus_W``, ``omnibus_df``, ``omnibus_p``, ``n_nonzero``,
        ``any_nonzero``.
    """
    x = np.asarray(x, dtype=float)

    y_est, y_std, y_cov = predict_with_error(
        model_func, params, cov_matrix, args=[x], full_cov=True
    )
    y_est = np.asarray(y_est, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    # Two-sided per-point z-test against zero. z_crit turns alpha into a
    # threshold on |z| (and, below, on the equivalence CI half-width).
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = y_est / y_std
    sig_nonzero = np.isfinite(z) & (np.abs(z) > z_crit)
    direction = np.where(sig_nonzero, np.sign(y_est), 0).astype(int)

    per_point = {
        "x": x,
        "y_model": y_est,
        "y_model_std": y_std,
        "z": z,
        "sig_nonzero": sig_nonzero,
        "direction": direction,
    }

    omnibus_W, omnibus_df, omnibus_p = _omnibus_chi2(y_est, y_cov)

    rollup = {
        "omnibus_W": omnibus_W,
        "omnibus_df": omnibus_df,
        "omnibus_p": omnibus_p,
        "n_nonzero": int(np.count_nonzero(sig_nonzero)),
        "any_nonzero": bool(np.any(sig_nonzero)),
    }

    return per_point, rollup


def _omnibus_chi2(y_est, y_cov):
    """
    Wald/Mahalanobis test that the predicted vector differs from zero.

    ``W = y_est @ pinv(y_cov) @ y_est`` with ``df = rank(y_cov)``. The predicted
    points live in a ``k``-dimensional space (``k`` = number of free params), so
    ``y_cov`` is rank-deficient and a pseudo-inverse with ``df = rank`` is the
    correct reduction. Returns ``(nan, 0, nan)`` when the covariance is not
    finite.
    """
    if not (np.all(np.isfinite(y_est)) and np.all(np.isfinite(y_cov))):
        return np.nan, 0, np.nan

    # Symmetrize to kill numerical asymmetry from the finite-difference Jacobian
    # before taking rank / pseudo-inverse.
    y_cov = 0.5 * (y_cov + y_cov.T)
    df = int(np.linalg.matrix_rank(y_cov))
    if df == 0:
        return np.nan, 0, np.nan

    pinv = np.linalg.pinv(y_cov)
    W = float(y_est @ pinv @ y_est)
    if not np.isfinite(W) or W < 0:
        return np.nan, df, np.nan

    p = float(chi2.sf(W, df))
    return W, df, p


def compute_delta(pred_std, delta_c=2.0):
    """
    Default region-of-practical-equivalence half-width from prediction error.

    ``delta = delta_c * median(pred_std)`` over all finite predicted standard
    errors. This ties "practically zero" to the typical *detectability* of the
    experiment rather than to any biological effect size. Returns NaN if no
    finite values are present.

    Parameters
    ----------
    pred_std : array-like
        Predicted per-point standard errors, pooled across all groups.
    delta_c : float, optional
        Multiplier on the median. Default 2.0.
    """
    pred_std = np.asarray(pred_std, dtype=float)
    finite = pred_std[np.isfinite(pred_std)]
    if finite.size == 0:
        return np.nan
    return float(delta_c * np.median(finite))


def classify_equiv(y_est, y_std, delta, alpha=0.05):
    """
    Flag points whose whole CI lies inside the equivalence region [-delta, delta].

    A point is ``equiv_zero`` when ``|y_est| + z_crit * y_std <= delta`` — i.e.
    the two-sided ``(1 - alpha)`` confidence interval is entirely within the
    region of practical equivalence around zero. NaN std or NaN/invalid delta
    yield False.

    Parameters
    ----------
    y_est, y_std : array-like
        Per-point estimate and standard error.
    delta : float
        Equivalence half-width (see :func:`compute_delta`).
    alpha : float, optional
        Matches the confidence level used elsewhere. Default 0.05.

    Returns
    -------
    np.ndarray of bool
    """
    y_est = np.asarray(y_est, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    if not np.isfinite(delta):
        return np.zeros(y_est.shape, dtype=bool)

    z_crit = norm.ppf(1.0 - alpha / 2.0)
    ci_upper = np.abs(y_est) + z_crit * y_std
    with np.errstate(invalid="ignore"):
        equiv = np.isfinite(ci_upper) & (ci_upper <= delta)
    return equiv


def benjamini_hochberg(pvals):
    """
    Benjamini-Hochberg FDR-adjusted q-values.

    NaN p-values (failed / unassessable curves) pass through as NaN and are
    excluded from the ranking. The returned q-values are aligned to the input
    order and enforced monotone-nondecreasing in p, clipped to [0, 1].

    Parameters
    ----------
    pvals : array-like
        Raw p-values, one per test (curve).

    Returns
    -------
    np.ndarray of float
        Adjusted q-values, same length/order as ``pvals``.
    """
    pvals = np.asarray(pvals, dtype=float)
    q = np.full(pvals.shape, np.nan)

    finite_idx = np.flatnonzero(np.isfinite(pvals))
    m = finite_idx.size
    if m == 0:
        return q

    p = pvals[finite_idx]
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)

    q_sorted = p[order] * m / ranks
    # Enforce monotonicity from the largest p downward (standard BH step-up).
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_finite = np.empty(m)
    q_finite[order] = q_sorted
    q[finite_idx] = q_finite
    return q
