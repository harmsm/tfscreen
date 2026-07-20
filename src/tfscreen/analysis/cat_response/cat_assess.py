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
  ``[-rope_cutoff, rope_cutoff]`` centered on zero. This is what separates
  "confidently flat" from "too noisy to tell" (drives ``confident_zero``). The
  auto ``rope_cutoff`` (:func:`compute_rope`, ``rope_multiplier * median(y_std)``)
  is a *detectability* threshold and rarely lets a whole CI fit inside it -- pass
  an explicit ``rope_cutoff`` (a biologically-meaningful region) to make
  ``confident_zero`` fire.
"""

import numpy as np
from scipy.stats import chi2, norm

from tfscreen.mle import predict_with_error

# Minimum number of usable (finite, nonzero) residuals for the runs test to be
# computed. Below this the test is uninformative and returns NaN.
_MIN_RUNS_N = 4


def assess_best_model(model_func, params, cov_matrix, x, y_obs, y_std,
                      alpha=0.05):
    """
    Grade the best-fit curve against zero -- data-driven, with the model test
    reported alongside.

    The "distinguishable from zero" decision uses the **observed** points
    (``y_obs`` +/- ``y_std``), not the fitted curve: a flexible model fit to
    noisy data reports an overconfident curve, so its propagated error can call a
    curve "nonzero" even when every observed error bar overlaps zero. The
    per-point z-test, the ``nonzero`` portmanteau chi-square, and the equivalence
    test all read the observed errors. The fitted curve (``y_model`` /
    ``y_model_std``) is still returned for plotting, and the model-based omnibus
    (``omnibus_*``) is still computed but is reported-only (it gates nothing).

    Parameters
    ----------
    model_func : callable
        The fitted model, signature ``model_func(params, x)``.
    params : np.ndarray
        Best-fit parameters.
    cov_matrix : np.ndarray
        Covariance matrix of the fitted parameters. May contain NaN (failed
        fit); the fitted curve and model omnibus are then NaN, but the
        data-based tests still run on ``y_obs``/``y_std``.
    x : np.ndarray
        Assessment grid (the unique observed x).
    y_obs, y_std : np.ndarray
        Observed values and their standard errors on the ``x`` grid.
    alpha : float, optional
        Two-sided significance level for the per-point ``sig_nonzero`` flag.
        Default 0.05.

    Returns
    -------
    per_point : dict
        Arrays of length ``len(x)``: ``x``, ``y_model`` (fitted curve value),
        ``y_model_std`` (propagated fit uncertainty), ``z`` (= y_obs/y_std),
        ``sig_nonzero`` (bool). (``direction`` is derivable as ``sign(y_obs)``.)
    rollup : dict
        Scalars: data-based ``nonzero_chi2``/``nonzero_df``/``nonzero_p``,
        model-based ``omnibus_W``/``omnibus_df``/``omnibus_p`` (reported-only),
        ``n_nonzero``, ``any_nonzero``.
    """
    x = np.asarray(x, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    # Fitted curve (for plotting) + its full covariance (for the model omnibus).
    y_model, y_model_std, y_cov = predict_with_error(
        model_func, params, cov_matrix, args=[x], full_cov=True
    )
    y_model = np.asarray(y_model, dtype=float)
    y_model_std = np.asarray(y_model_std, dtype=float)

    # Per-point z-test against zero on the OBSERVED data.
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = y_obs / y_std
    sig_nonzero = np.isfinite(z) & (np.abs(z) > z_crit)

    per_point = {
        "x": x,
        "y_model": y_model,
        "y_model_std": y_model_std,
        "z": z,
        "sig_nonzero": sig_nonzero,
    }

    # Data-based portmanteau chi-square vs the zero line (drives fittable).
    nonzero_chi2, nonzero_df, nonzero_p = _nonzero_chi2(y_obs, y_std)
    # Model-based omnibus on the fitted curve (reported only).
    omnibus_W, omnibus_df, omnibus_p = _omnibus_chi2(y_model, y_cov)

    rollup = {
        "nonzero_chi2": nonzero_chi2,
        "nonzero_df": nonzero_df,
        "nonzero_p": nonzero_p,
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


def _nonzero_chi2(y_obs, y_std):
    """
    Model-free test that the observed curve differs from the zero line.

    Weighted portmanteau ``chi2 = sum((y_obs / y_std) ** 2) ~ chi2(n)`` under the
    null that every point is zero. Uses the *observed* error bars, so a curve
    whose CIs all overlap zero is not called nonzero no matter how confidently a
    flexible model fits it. Returns ``(chi2, df, p)``; ``(nan, 0, nan)`` when no
    point has a finite value and positive std.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    m = np.isfinite(y_obs) & np.isfinite(y_std) & (y_std > 0)
    df = int(np.count_nonzero(m))
    if df == 0:
        return np.nan, 0, np.nan
    stat = float(np.sum((y_obs[m] / y_std[m]) ** 2))
    return stat, df, float(chi2.sf(stat, df))


def residual_runs_p(resid):
    """
    Wald-Wolfowitz runs-test p-value for systematic structure in residuals.

    Given residuals ordered along the independent variable, tests whether the
    sign sequence shows *positive clustering* -- fewer runs than random -- which
    is the fingerprint of a systematically wrong shape (the residuals sit on one
    side, then the other). This is a **one-sided, lower-tail** test: it flags
    same-sign clustering but not over-dispersion (alternating residuals are not
    a shape error), which also gives it usable power at the small n typical
    here. It uses only the residual *signs*, so it is robust to the scale of the
    ``y_std`` used for weighting. This is the primary adequacy check for shape
    selection.

    Parameters
    ----------
    resid : array-like
        Residuals in independent-variable order (e.g. weighted residuals sorted
        by x). Non-finite and exactly-zero residuals are dropped.

    Returns
    -------
    float
        Lower-tail p-value under the null of random sign order; small when the
        residuals cluster by sign (systematic misfit). NaN when it cannot be
        computed: fewer than ``_MIN_RUNS_N`` usable residuals, or all residuals
        share one sign (itself a sign of a biased fit; callers treat NaN as "not
        adequate" unless *no* model could be assessed).
    """
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid) & (resid != 0.0)]
    n = resid.size
    if n < _MIN_RUNS_N:
        return np.nan

    signs = resid > 0
    n_pos = int(np.count_nonzero(signs))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan

    runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1]))
    mu = 1.0 + 2.0 * n_pos * n_neg / n
    var = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)
           / (n ** 2 * (n - 1.0)))
    if var <= 0:
        return np.nan

    z = (runs - mu) / np.sqrt(var)
    return float(norm.cdf(z))


def residual_autocorr(resid):
    """
    Lag-1 autocorrelation of residuals ordered along the independent variable.

    A *weighted*, magnitude-aware structure detector (pass the standardized
    residuals ``(y - yfit) / y_std``): smooth systematic misfit makes consecutive
    residuals track each other, so a bounded flat/constant fit to a real curve
    shows strong positive autocorrelation. Unlike the sign-based runs test, this
    is not washed out by many near-baseline points, so it catches structure the
    runs test misses on noisy, heteroscedastic (logit) data.

    Parameters
    ----------
    resid : array-like
        Residuals in independent-variable order. Non-finite values are dropped.

    Returns
    -------
    (autocorr, autocorr_p) : (float, float)
        ``autocorr`` is the Durbin-Watson lag-1 autocorrelation estimate
        ``1 - DW/2`` (~0 = no structure, ->1 = smooth positive autocorrelation).
        ``autocorr_p`` is a one-sided (positive-autocorrelation) p-value from the
        normal approximation ``DW ~ N(2, 4/n)``; small = systematic structure.
        Both NaN when fewer than ``_MIN_RUNS_N`` residuals or all-zero.
    """
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    n = resid.size
    if n < _MIN_RUNS_N:
        return np.nan, np.nan

    ss = float(np.sum(resid ** 2))
    if ss <= 0.0:
        return np.nan, np.nan

    dw = float(np.sum(np.diff(resid) ** 2) / ss)
    autocorr = 1.0 - dw / 2.0
    z = (dw - 2.0) / (2.0 / np.sqrt(n))
    autocorr_p = float(norm.cdf(z))     # lower tail: DW < 2 -> positive autocorr
    return autocorr, autocorr_p


def goodness_of_fit_p(chi2_w, n, k):
    """
    Lack-of-fit p-value from the weighted chi-square of a fit.

    Under a correct model with calibrated ``y_std`` the weighted residual sum of
    squares ~ chi2(n - k), so ``p = chi2.sf(chi2_w, n - k)`` is an *absolute*
    adequacy check complementary to AICc's relative one. Reported alongside
    selection but -- unlike the runs test -- not used to gate it, since it
    depends on the ``y_std`` scale. NaN when ``n - k <= 0``.

    Parameters
    ----------
    chi2_w : float
        Weighted residual sum of squares, ``sum(((y - yfit) / y_std) ** 2)``.
    n, k : int
        Number of points and number of fitted parameters.
    """
    df = n - k
    if df <= 0:
        return np.nan
    return float(chi2.sf(chi2_w, df))


def compute_rope(pred_std, rope_multiplier=2.0):
    """
    Auto region-of-practical-equivalence (ROPE) half-width from the error bars.

    ``rope_cutoff = rope_multiplier * median(pred_std)`` over all finite standard
    errors. This ties "practically zero" to the typical *detectability* of the
    experiment rather than to any biological effect size -- and because it scales
    with the noise it rarely lets a whole CI fit inside, so ``confident_zero``
    seldom fires under the auto value; pass an explicit ``rope_cutoff`` for a
    biologically-meaningful region. Returns NaN if no finite values are present.

    Parameters
    ----------
    pred_std : array-like
        Per-point standard errors, pooled across all groups.
    rope_multiplier : float, optional
        Multiplier on the median. Default 2.0.
    """
    pred_std = np.asarray(pred_std, dtype=float)
    finite = pred_std[np.isfinite(pred_std)]
    if finite.size == 0:
        return np.nan
    return float(rope_multiplier * np.median(finite))


def classify_equiv(y_est, y_std, rope_cutoff, alpha=0.05):
    """
    Flag points whose whole CI lies inside the ROPE [-rope_cutoff, rope_cutoff].

    A point is ``equiv_zero`` when ``|y_est| + z_crit * y_std <= rope_cutoff`` --
    i.e. the two-sided ``(1 - alpha)`` confidence interval is entirely within the
    region of practical equivalence around zero. NaN std or NaN/invalid
    ``rope_cutoff`` yield False.

    Parameters
    ----------
    y_est, y_std : array-like
        Per-point estimate and standard error.
    rope_cutoff : float
        ROPE half-width (see :func:`compute_rope`).
    alpha : float, optional
        Matches the confidence level used elsewhere. Default 0.05.

    Returns
    -------
    np.ndarray of bool
    """
    y_est = np.asarray(y_est, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    if not np.isfinite(rope_cutoff):
        return np.zeros(y_est.shape, dtype=bool)

    z_crit = norm.ppf(1.0 - alpha / 2.0)
    ci_upper = np.abs(y_est) + z_crit * y_std
    with np.errstate(invalid="ignore"):
        equiv = np.isfinite(ci_upper) & (ci_upper <= rope_cutoff)
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
