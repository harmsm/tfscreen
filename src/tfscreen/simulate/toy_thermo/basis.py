"""
Basis-curve decomposition of logit-scale epistasis for the four-state model.

Why this exists
---------------
For any genotype the observable satisfies, *exactly*,

    logit(theta)  =  ln K_conf + ln K_dna + ln[L_free]
                  =  -g_HD + ln[L_free] + const,

and free [L] is strictly monotone-decreasing in effector (mass balance). So a
*single* genotype's logit(theta) curve can never peak. All peak structure lives
in the epistasis second difference

    eps(E) = logit(theta_AB) - logit(theta_A) - logit(theta_B) + logit(theta_wt).

Writing g_k^geno = g_k^wt + (sum of single ddG) + (pair epistasis e_k), the g_HD
term contributes only a flat offset -e_HD, and the partition function
Z = e^-g_L + e^-g_H + E e^-g_LE contributes, to second order in the ddGs,

    eps(E) ~  - Cov_p(delta_A, delta_B)  -  sum_k p_k(E) e_k  -  e_HD

where p_k(E) is the wild-type free-protein occupancy of state k in {L, H, LE}
and the covariance is over that ensemble. Using the identity
Cov_p(a,b) = sum_{j<k} p_j p_k (a_j - a_k)(b_j - b_k), the *additive* epistasis
is a linear combination of just three basis curves -- the pairwise occupancy
products p_L p_H, p_L p_LE, p_H p_LE:

    eps_add(E) = - sum_{j<k in {L,H,LE}} (dA_j - dA_k)(dB_j - dB_k) p_j(E) p_k(E).

The two products that involve LE (p_L p_LE, p_H p_LE) are zero at both ends of
the titration and peak at the ligand transition -- they are the *only* source of
peak-shaped epistasis. p_L p_H is a monotone step; the within-state (direct)
terms -sum_k p_k e_k are monotone steps (e_HD is flat). This module exposes the
occupancies, the basis curves, the coefficient bookkeeping, and both the
second-order-predicted and exact epistasis so the approximation can be checked.

All functions are NumPy/SciPy/pandas only (no plotting dependency at import).
"""

import numpy as np

from .core import solve_species
from .genotypes import STATES, parse_genotype  # noqa: F401  (re-export convenience)

# The free-protein ensemble the partition function sums over. HD (DNA-bound) is
# deliberately excluded: it enters logit(theta) only through the flat -g_HD term.
FREE_STATES = ("L", "H", "LE")
_STATE_PAIRS = (("L", "H"), ("L", "LE"), ("H", "LE"))


def _as_ddg(ddg):
    """Normalize a per-state ddG dict to a full {state: float} over STATES."""
    ddg = ddg or {}
    bad = set(ddg) - set(STATES)
    if bad:
        raise ValueError(f"unknown state(s) {sorted(bad)}; valid: {STATES}")
    return {s: float(ddg.get(s, 0.0)) for s in STATES}


def free_ensemble(model, effector, genotype="wt", effects=None, ddg=None):
    """
    Free-protein occupancies p_L, p_H, p_LE over an effector grid.

    Occupancies are the *exact* solved species concentrations (so effector
    depletion is included), renormalized over the three free states L, H, LE
    (i.e. excluding the DNA-bound HD population). Genotype is specified exactly
    as for ``ThermoModel.observable`` -- a ``ddg`` dict, or an ``effects``
    catalog plus a ``genotype`` string.

    Returns
    -------
    dict of numpy.ndarray
        Keys ``"L"``, ``"H"``, ``"LE"``; each a 1-D array over the effector
        grid, summing to 1 at every concentration.
    """
    Kc, Kd, Ke = model.genotype_ks(genotype, effects, ddg)
    E = np.atleast_1d(np.asarray(effector, dtype=float))
    p = {s: np.empty(E.shape) for s in FREE_STATES}
    for i, e in enumerate(E):
        sp = solve_species(Kc, Kd, Ke, model.protein_total, model.dna_total, e)
        tot = sp["L"] + sp["H"] + sp["LE"]
        for s in FREE_STATES:
            p[s][i] = sp[s] / tot
    return p


def basis_curves(model, effector, genotype="wt", effects=None, ddg=None):
    """
    The three occupancy products that span additive logit epistasis.

    Returns a dict with the occupancies (``"L"``, ``"H"``, ``"LE"``) and the
    three basis curves keyed by frozenset pairs and by readable aliases:
    ``"L*H"``, ``"L*LE"``, ``"H*LE"``. ``"L*LE"`` and ``"H*LE"`` are the
    peak-shaped (turn-on-then-off) basis functions; ``"L*H"`` is the monotone
    one.
    """
    p = free_ensemble(model, effector, genotype, effects, ddg)
    out = dict(p)
    for a, b in _STATE_PAIRS:
        out[f"{a}*{b}"] = p[a] * p[b]
    return out


def epistasis_coeffs(ddg_A, ddg_B, epi=None):
    """
    Coefficients of the basis / direct decomposition for a double mutant.

    Parameters
    ----------
    ddg_A, ddg_B : dict
        Per-state ddG of the two single mutations (additive part).
    epi : dict, optional
        Per-state within-state pair epistasis applied only to the double.

    Returns
    -------
    dict
        ``"cross"``  : {(j, k): coeff} for the three ``{L,H,LE}`` products; the
                       additive epistasis is ``sum coeff * p_j * p_k``.
        ``"direct"`` : {state: coeff} for L/H/LE; contributes ``-coeff * p_k``.
        ``"offset"`` : scalar flat contribution from ``e_HD`` (``-e_HD``).

    Notes
    -----
    ``cross[(j,k)] = -(dA_j - dA_k)(dB_j - dB_k)``. A nonzero coefficient on a
    ``*_LE`` pair is the necessary condition for a peak (given that apo state is
    populated). The sign of that coefficient is the sign of the peak.
    """
    dA, dB = _as_ddg(ddg_A), _as_ddg(ddg_B)
    e = _as_ddg(epi)
    cross = {}
    for j, k in _STATE_PAIRS:
        cross[(j, k)] = -(dA[j] - dA[k]) * (dB[j] - dB[k])
    direct = {s: e[s] for s in FREE_STATES}
    return {"cross": cross, "direct": direct, "offset": -e["HD"]}


def predict_epistasis(model, effector, ddg_A, ddg_B, epi=None):
    """
    Second-order-predicted logit epistasis and its term-by-term breakdown.

    Returns
    -------
    dict
        ``"total"``  : the predicted eps(E).
        ``"cross"``  : {(j,k): coeff * p_j * p_k} per basis curve.
        ``"direct"`` : {state: -coeff * p_k} per within-state channel.
        ``"offset"`` : scalar array (broadcast ``-e_HD``).

    This is the analytic approximation (``-Cov_p`` plus occupancy-weighted
    direct terms). Compare against :func:`exact_epistasis` to see where the
    second-order expansion breaks down (typically for |ddG| >~ 1-2 kT).
    """
    coeffs = epistasis_coeffs(ddg_A, ddg_B, epi)
    p = free_ensemble(model, effector, "wt")
    E = np.atleast_1d(np.asarray(effector, dtype=float))

    cross = {(j, k): coeffs["cross"][(j, k)] * p[j] * p[k]
             for j, k in _STATE_PAIRS}
    direct = {s: -coeffs["direct"][s] * p[s] for s in FREE_STATES}
    offset = np.full(E.shape, coeffs["offset"])

    total = offset.copy()
    for v in cross.values():
        total = total + v
    for v in direct.values():
        total = total + v
    return {"total": total, "cross": cross, "direct": direct, "offset": offset}


def _logit_theta(model, effector, ddg):
    th = np.atleast_1d(model.observable(effector, ddg=ddg))
    th = np.clip(th, 1e-15, 1.0 - 1e-15)
    return np.log(th / (1.0 - th))


def exact_epistasis(model, effector, ddg_A, ddg_B, epi=None):
    """
    Exact logit-scale epistasis (11-10)-(01-00) from the full solver.

    ``ddg_A``/``ddg_B`` are the two singles; ``epi`` (per-state) is applied only
    to the double. No second-order approximation -- this is the ground truth the
    basis prediction is meant to reproduce.
    """
    dA, dB, e = _as_ddg(ddg_A), _as_ddg(ddg_B), _as_ddg(epi)
    dAB = {s: dA[s] + dB[s] + e[s] for s in STATES}
    l00 = _logit_theta(model, effector, {})
    l10 = _logit_theta(model, effector, dA)
    l01 = _logit_theta(model, effector, dB)
    l11 = _logit_theta(model, effector, dAB)
    return (l11 - l10) - (l01 - l00)


# --------------------------------------------------------------------------- #
# Measurement range / logit error propagation
# --------------------------------------------------------------------------- #

def resolvable_logit(eps=0.01):
    """
    Half-width of the usable logit band given a theta resolution floor ``eps``.

    If theta can only be resolved to within ``eps`` of 0 or 1, then logit(theta)
    is only trustworthy inside ``[-L*, +L*]`` with ``L* = ln((1-eps)/eps)``
    (``~4.595`` for ``eps=0.01``). Outside that band one tail of the confidence
    interval runs to +/-inf (see :func:`logit_ci`).
    """
    eps = float(eps)
    return float(np.log((1.0 - eps) / eps))


def logit_ci(theta, sigma, z=1.0):
    """
    Asymmetric (censored) logit confidence interval from ``theta +/- z*sigma``.

    Propagates a symmetric measurement error in *theta* space through the logit
    by transforming the interval endpoints, not by linearizing. As a theta band
    touches 0 or 1 the corresponding logit tail diverges, so this returns
    ``+/-inf`` there -- the one-sided blow-up that the delta-method scale
    ``sigma/(theta*(1-theta))`` cannot represent.

    Parameters
    ----------
    theta : array_like
        Observed occupancy/observable in (0, 1).
    sigma : array_like
        Measurement standard deviation in theta space (broadcast to ``theta``).
    z : float
        Half-width of the interval in sigmas (e.g. 1.0 or 1.96).

    Returns
    -------
    (center, lower, upper) : tuple of numpy.ndarray
        ``center`` = logit(theta); ``lower``/``upper`` are the logit-transformed
        ``theta -/+ z*sigma`` endpoints, with ``-inf``/``+inf`` where the theta
        band reaches 0/1.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), theta.shape)
    tiny = 1e-15

    def _logit(x):
        return np.log(x / (1.0 - x))

    center = _logit(np.clip(theta, tiny, 1.0 - tiny))
    hi_th = theta + z * sigma
    lo_th = theta - z * sigma
    upper = np.where(hi_th < 1.0, _logit(np.clip(hi_th, tiny, 1.0 - tiny)), np.inf)
    lower = np.where(lo_th > 0.0, _logit(np.clip(lo_th, tiny, 1.0 - tiny)), -np.inf)
    return center, lower, upper


def measurement_window(model, effector, genotypes, eps=0.01):
    """
    Effector range over which each genotype -- and the whole set jointly -- is
    resolvable (theta in ``[eps, 1-eps]``).

    Because theta is monotone in effector, each resolvable set is a contiguous
    interval, reported as ``(E_lo, E_hi)`` on the supplied grid (so pass a fine
    ``effector`` grid). Epistasis is a second difference, so it is measurable only
    where **all** genotypes are simultaneously resolvable -- the ``"joint"`` key
    holds that intersection.

    Parameters
    ----------
    genotypes : dict
        ``{label: ddg_dict}``. Use e.g. ``{"wt": {}, "A": ddg_A, "B": ddg_B,
        "AB": {...}}`` for an epistasis quartet.

    Returns
    -------
    dict
        ``label -> (E_lo, E_hi)`` per genotype (``None`` if never resolvable on
        the grid), plus ``"joint" -> (E_lo, E_hi)`` (``None`` if the intersection
        is empty or any genotype is unresolvable).
    """
    E = np.asarray(effector, dtype=float)
    out = {}
    for label, ddg in genotypes.items():
        th = np.atleast_1d(model.observable(E, ddg=ddg))
        ok = (th >= eps) & (th <= 1.0 - eps)
        out[label] = (float(E[ok].min()), float(E[ok].max())) if ok.any() else None

    windows = list(out.values())
    if windows and all(w is not None for w in windows):
        lo = max(w[0] for w in windows)
        hi = min(w[1] for w in windows)
        out["joint"] = (lo, hi) if lo <= hi else None
    else:
        out["joint"] = None
    return out


def plot_measurement_window(model, effector, ddg_A, ddg_B, eps=0.01, ax=None):
    """
    Plot the epistasis quartet's logit(theta) against the resolvable band.

    Shades the usable band ``[-L*, +L*]`` (from ``eps``), draws each of wt/A/B/AB
    as ``logit(theta)`` (solid where resolvable, dotted where censored), marks the
    epistasis peak location, and shades the joint measurement window. Makes it
    visible whether the peak falls where all four genotypes can actually be
    measured. Requires matplotlib (lazy import). Returns the Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    E = np.asarray(effector, dtype=float)
    Lstar = resolvable_logit(eps)

    ax.axhspan(-Lstar, Lstar, color="#68d391", alpha=0.15, zorder=0,
               label=f"resolvable band  (theta in [{eps}, {1-eps}])")
    quartet = {"wt": {}, "A": _as_ddg(ddg_A), "B": _as_ddg(ddg_B),
               "AB": {s: _as_ddg(ddg_A)[s] + _as_ddg(ddg_B)[s] for s in STATES}}
    colors = {"wt": "#2b6cb0", "A": "#dd6b20", "B": "#38a169", "AB": "#805ad5"}
    for label, ddg in quartet.items():
        lt = _logit_theta(model, E, ddg)
        inband = np.abs(lt) <= Lstar
        ax.semilogx(E, np.where(inband, lt, np.nan), "-", color=colors[label],
                    lw=1.8, label=label)
        ax.semilogx(E, np.where(~inband, lt, np.nan), ":", color=colors[label],
                    lw=1.2, alpha=0.7)

    ep = exact_epistasis(model, E, ddg_A, ddg_B)
    peak_E = E[int(np.argmax(np.abs(ep)))]
    ax.axvline(peak_E, color="0.35", ls="--", lw=1, label="epistasis peak")

    win = measurement_window(model, E, quartet, eps=eps)["joint"]
    if win is not None:
        ax.axvspan(win[0], win[1], color="0.6", alpha=0.12, zorder=0)
        captured = win[0] <= peak_E <= win[1]
    else:
        captured = False
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("total effector")
    ax.set_ylabel("logit(theta)")
    ax.set_title(f"peak {'INSIDE' if captured else 'OUTSIDE'} joint window")
    ax.legend(fontsize=7, loc="upper right")
    return ax


def plot_basis(model, effector, ax=None):
    """
    Plot the wt occupancies and the three basis curves for a backdrop.

    Left of the shared axis: p_L, p_H, p_LE. Overlaid: the products p_L p_H
    (monotone), p_L p_LE and p_H p_LE (peak-shaped). Requires matplotlib; import
    is lazy so the module has no plotting dependency at import time. Returns the
    Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))
    E = np.asarray(effector, dtype=float)
    b = basis_curves(model, E)
    for s, c in zip(FREE_STATES, ("#f6ad55", "#63b3ed", "#dd6b20")):
        ax.semilogx(E, b[s], "-", lw=1, color=c, alpha=0.5, label=f"p_{s}")
    styles = {"L*H": ("#718096", "--"), "L*LE": ("#2f855a", "-"),
              "H*LE": ("#805ad5", "-")}
    for key, (c, ls) in styles.items():
        ax.semilogx(E, b[key], ls, lw=2, color=c, label=key)
    ax.set_xlabel("total effector")
    ax.set_ylabel("occupancy / product")
    ax.set_title("Ensemble occupancies and epistasis basis curves")
    ax.legend(fontsize=8, ncol=2)
    return ax


def plot_epistasis_decomposition(model, effector, ddg_A, ddg_B, epi=None,
                                 ax=None):
    """
    Overlay exact epistasis, the 2nd-order prediction, and its contributions.

    Solid black: exact eps(E). Dashed: the predicted total. Thin coloured lines:
    each nonzero basis / direct / offset contribution. Makes it obvious which
    term produces a peak and how far the second-order prediction drifts from the
    truth. Requires matplotlib (lazy import). Returns the Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))
    E = np.asarray(effector, dtype=float)
    exact = exact_epistasis(model, E, ddg_A, ddg_B, epi)
    parts = predict_epistasis(model, E, ddg_A, ddg_B, epi)

    ax.axhline(0, color="0.6", lw=1)
    ax.semilogx(E, exact, "-", color="black", lw=2.5, label="exact")
    ax.semilogx(E, parts["total"], "--", color="#c53030", lw=1.5,
                label="2nd-order total")
    for (j, k), v in parts["cross"].items():
        if np.any(v):
            ax.semilogx(E, v, "-", lw=1, alpha=0.7, label=f"cross {j}*{k}")
    for s, v in parts["direct"].items():
        if np.any(v):
            ax.semilogx(E, v, ":", lw=1, alpha=0.7, label=f"direct {s}")
    if np.any(parts["offset"]):
        ax.semilogx(E, parts["offset"], ":", lw=1, alpha=0.7, label="offset e_HD")
    ax.set_xlabel("total effector")
    ax.set_ylabel("logit-scale epistasis")
    ax.set_title(f"exact shape: {classify_shape(exact)}")
    ax.legend(fontsize=8)
    return ax


def classify_shape(eps, tol=1e-6, peak_ratio=1.3, edge=3):
    """
    Coarse label for an epistasis curve: 'flat', 'step', or 'peak'.

    A quick, dependency-free classifier (not a substitute for ``cat_response``):
    'flat' if the amplitude is below ``tol``; 'peak' if the extremum is interior
    (at least ``edge`` points from either end) and exceeds ``peak_ratio`` times
    the larger endpoint magnitude; otherwise 'step'.
    """
    eps = np.asarray(eps, dtype=float)
    amp = np.max(np.abs(eps))
    if amp < tol:
        return "flat"
    i = int(np.argmax(np.abs(eps)))
    interior = edge <= i <= len(eps) - 1 - edge
    ends = max(abs(eps[0]), abs(eps[-1]))
    if interior and abs(eps[i]) > peak_ratio * ends:
        return "peak"
    return "step"
