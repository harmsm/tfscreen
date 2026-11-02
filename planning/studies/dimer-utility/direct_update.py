"""
Direct (type-II) update of the held thermo.dimer scales from one Laplace.

EM moves a scale slowly when the posteriors lean on the prior, so this
maximizes the Laplace-approximate marginal likelihood in one step instead.
For each mutation, the Laplace at held prior covariance C0 gives the
posterior mean r̂ (about the prior mean) and covariance V of its shifts.
Removing the prior from the posterior precision leaves the likelihood's
Gaussian approximation, precision Λ = V⁻¹ − C0⁻¹ and information b = V⁻¹ r̂
(Λ's eigenvalues are kept only where they clear the Monte Carlo error of
V estimated from N draws, NOISE_Z · sqrt(2 / N) in units whitened by C0;
otherwise the clipped noise of a term the data do not see reads as
information that it is near 0 and pulls its scale down). Then,
up to a constant,

    log Z(C) = −½ log|C| − ½ log|Λ + C⁻¹| + ½ bᵀ (Λ + C⁻¹)⁻¹ b

summed over mutations, plus a LogNormal(HYPER_LOC, HYPER_SCALE) prior on
each scale (the structural prior's own σ_s prior, median 1 kT), and
maximized over the scales. Without that prior a scale the data barely see
has a nearly flat marginal likelihood whose noise sends it to 0; with it
the scale stays near the prior median. PK terms are
independent, so each is a 1-D search; PddG's six σ_s share the 6 × 6
covariance P diag(σ²) Pᵀ. Per-mutation blocks only: the Laplace's coupling
across mutations and to the shared parameters is ignored.

    ~/miniconda3/bin/python direct_update.py <posterior.h5> <arm> <held json>
"""

import json
import sys

import h5py
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from tfscreen.tfmodel.generative.components.theta.thermo.dimer import PddG


HYPER_LOC, HYPER_SCALE = 0.0, 1.2      # log kT


def _log_hyper(u):
    """LogNormal log density of σ = exp(u), in u (including the Jacobian)."""
    u = np.atleast_1d(u)
    return float(np.sum(-0.5 * ((u - HYPER_LOC) / HYPER_SCALE) ** 2))


NOISE_Z = 3.0


def likelihood_blocks(draws, c0):
    """(Λ, b) per mutation from draws (N, M, C) and held prior cov (C, C)."""
    n = draws.shape[0]
    r = draws.mean(axis=0)                                     # (M, C)
    x = draws - r
    V = np.einsum("nmi,nmj->mij", x, x) / (n - 1)
    Vi = np.linalg.inv(V)
    # whiten by C0: W = C0^1/2 V⁻¹ C0^1/2 = I + Λ̃, eigenvalues e >= 1 where
    # the likelihood informs; keep e - 1 only beyond the sampling error.
    w0, u0 = np.linalg.eigh(c0)
    h = (u0 * np.sqrt(w0)) @ u0.T                              # C0^1/2
    hi = (u0 / np.sqrt(w0)) @ u0.T                             # C0^-1/2
    W = h @ Vi @ h
    e, u = np.linalg.eigh(0.5 * (W + np.swapaxes(W, -1, -2)))
    keep = (e - 1.0) > NOISE_Z * np.sqrt(2.0 / (n - 1)) * e
    lt = np.where(keep, e - 1.0, 0.0)
    lam = hi @ ((u * lt[..., None, :]) @ np.swapaxes(u, -1, -2)) @ hi
    # b = Λ y lies in Λ's range: project V⁻¹ r̂ onto the kept directions
    # (in a direction without information it is Monte Carlo noise).
    bw = np.einsum("ij,mj->mi", h, np.einsum("mij,mj->mi", Vi, r))
    proj = (u * keep[..., None, :]) @ np.swapaxes(u, -1, -2)
    b = np.einsum("ij,mj->mi", hi, np.einsum("mij,mj->mi", proj, bw))
    return lam, b


def log_z(lam, b, c):
    ci = np.linalg.inv(c)
    a = lam + ci[None]
    _, ld_a = np.linalg.slogdet(a)
    _, ld_c = np.linalg.slogdet(c)
    q = np.einsum("mi,mi->m", b, np.linalg.solve(a, b[..., None])[..., 0])
    return float(np.sum(-0.5 * ld_c - 0.5 * ld_a + 0.5 * q))


def scalar_scale(d, sigma0):
    """1-D: draws (N, M) of one term's shifts at held σ0."""
    lam, b = likelihood_blocks(d[..., None], np.array([[sigma0 ** 2]]))
    f = lambda u: log_z(lam, b, np.array([[np.exp(2 * u)]]))   # noqa: E731
    res = minimize_scalar(lambda u: -f(u) - _log_hyper(u),
                          bounds=(np.log(0.01), np.log(20.0)),
                          method="bounded")
    return float(np.exp(res.x)), f(res.x) - f(np.log(sigma0))


def state_scales(dlnK, mu, sigma0):
    """PddG σ_s: draws (N, M, 6) of Δln K, prior means (M, S), held σ0 (S,)."""
    P = PddG.DLNK_PROJ
    resid = dlnK - mu @ P.T
    c = lambda s: P @ np.diag(s ** 2) @ P.T                    # noqa: E731
    lam, b = likelihood_blocks(resid, c(np.asarray(sigma0)))
    f = lambda u: log_z(lam, b, c(np.exp(u)))                  # noqa: E731
    res = minimize(lambda u: -f(u) - _log_hyper(u),
                   np.log(sigma0), method="L-BFGS-B",
                   bounds=[(np.log(0.01), np.log(20.0))] * len(sigma0))
    return np.exp(res.x), f(res.x) - f(np.log(sigma0))


def update(h5, arm, held, terms):
    out, gain = {}, {}
    with h5py.File(h5) as f:
        for t in terms:
            d = np.asarray(f[f"theta_d_{t}"]).reshape(
                f[f"theta_d_{t}"].shape[0], -1)
            out[t], gain[t] = scalar_scale(d, held[t])
        if arm == "dimerPddG":
            names = list(PddG.STRUCTURE_NAMES)
            s, gain["states"] = state_scales(np.asarray(f["theta_dlnK"]),
                             np.asarray(f["theta_ddG_prior_mean"])[0],
                             np.array([held[n] for n in names]))
            out.update(dict(zip(names, map(float, s))))
    return out, gain


if __name__ == "__main__":
    h5, arm, held = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    terms = [k for k in held if k.startswith("ln_K_")]
    out, gain = update(h5, arm, held, terms)
    for k, v in out.items():
        print(f"{k:12s} held {held[k]:.2f} -> {v:.2f}")
    print("log Z gain over held:", {k: round(v, 1) for k, v in gain.items()})
