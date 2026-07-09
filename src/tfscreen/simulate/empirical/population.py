"""
Stage 2 of the empirical-phenotype pipeline: estimate the *generating*
distribution of per-genotype phenotype parameters from the Stage-1 fits.

Each Stage-1 fit gives a point estimate ``y_i`` and a sampling covariance
``S_i`` (both in the transformed coordinate system:
``[dk_geno, logit_theta_low, logit_theta_high, log_hill_K, log_hill_n]``).
The spread of the point estimates is inflated by estimation noise:

    Cov(y_i)  ==  Sigma_pop  +  mean_i(S_i)

so naively fitting a distribution to the point estimates (a KDE/GMM on
``y_i``) would bake that inflation in and produce a library more heterogeneous
than reality.  This module instead fits a measurement-error (random-effects)
model

    z_i ~ Normal(mu, Sigma_pop)          # true phenotype (what we want)
    y_i ~ Normal(z_i, S_i)               # Stage-1 estimate, known S_i

by maximum likelihood (EM), recovering the **deconvolved** population
covariance ``Sigma_pop`` with estimation noise removed.  ``Sigma_pop`` is the
distribution Stage 3 resamples fresh genotypes from.

Scope
-----
This stage removes *estimation* noise only.  It does **not** correct the
congression attenuation present in bulk (in-library) fits — that is the
deferred iteration-to-consistency layer.  To validate/anchor the result,
fit the population on the congression-free spiked subset separately (pass a
``subset`` mask) and compare to the bulk fit.

The population is modeled as a single multivariate Normal in transformed
space.  A Gaussian-mixture extension (for multimodal libraries, e.g. a
functional mode plus a dead mode) is a natural drop-in: run one
measurement-error EM per mixture component with responsibilities.  Left as a
future extension so v1 stays simple and testable.
"""

import json
import os
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass

from tfscreen.simulate.empirical.fit_phenotypes import (
    PHENO_PARAMS_TRANSFORMED,
    PHENO_PARAMS_NATURAL,
    _LOGIT_BOUND,
)
from scipy.special import expit

_JITTER = 1e-9          # added to matrices before inversion
_EIG_FLOOR = 1e-8       # floor on eigenvalues of S_i and projected Sigma


def _nearest_psd(A, floor=_EIG_FLOOR):
    """Symmetrize and clip eigenvalues to ``floor`` (nearest PSD matrix)."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, floor, None)
    return (V * w) @ V.T


def _batched_inv(mats):
    """Invert a stack of matrices with a small diagonal jitter."""
    D = mats.shape[-1]
    return np.linalg.inv(mats + _JITTER * np.eye(D))


def _pheno_block(fit):
    """Extract the (estimate, covariance) phenotype block from a GenotypeFit."""
    est = np.asarray(fit.est_t)[fit.pheno_slice]
    cov = np.asarray(fit.cov_t)[fit.pheno_slice, fit.pheno_slice]
    return est, cov


def _collect_estimates(fits, subset=None, require_converged=True,
                       drop_railed=True):
    """Gather transformed phenotype estimates + covariances from Stage-1 fits.

    Parameters
    ----------
    fits : dict
        ``{key: GenotypeFit}`` from ``fit_phenotypes``.
    subset : set/list or None
        If given, keep only fits whose key is in ``subset`` (e.g. spiked-only).
    require_converged : bool
        Drop fits whose optimizer did not report success.
    drop_railed : bool
        Drop fits whose ``logit_theta_low``/``logit_theta_high`` sit on the
        ``+-_LOGIT_BOUND`` clamp (unidentified vertical placement).

    Returns
    -------
    Y : (N, D) float
    S : (N, D, D) float   (each PSD-floored)
    keys : list          (kept keys, aligned with Y/S rows)
    """
    Y, S, keys = [], [], []
    for key, fit in fits.items():
        if subset is not None and key not in subset:
            continue
        if require_converged and not fit.converged:
            continue

        est, cov = _pheno_block(fit)
        if not (np.all(np.isfinite(est)) and np.all(np.isfinite(cov))):
            continue

        if drop_railed:
            # indices 1, 2 are logit_theta_low / logit_theta_high
            if np.any(np.abs(est[[1, 2]]) >= _LOGIT_BOUND - 1e-6):
                continue

        Y.append(est)
        S.append(_nearest_psd(cov))
        keys.append(key)

    if not Y:
        raise ValueError(
            "No usable fits after filtering (converged / finite / not railed). "
            "Relax the filters or check the Stage-1 output.")

    return np.asarray(Y), np.asarray(S), keys


def _moment_init(Y, S):
    """Method-of-moments start: mu = mean(Y), Sigma = Cov(Y) - mean(S)."""
    mu = Y.mean(axis=0)
    sigma = np.cov(Y, rowvar=False) - S.mean(axis=0)
    return mu, _nearest_psd(np.atleast_2d(sigma))


def _marginal_loglik(Y, S, mu, sigma):
    """Sum_i log Normal(Y_i | mu, Sigma + S_i)."""
    D = Y.shape[1]
    C = sigma[None] + S                        # (N, D, D)
    r = Y - mu                                 # (N, D)
    sign, logdet = np.linalg.slogdet(C)
    sol = np.linalg.solve(C, r[..., None])[..., 0]
    quad = np.einsum("ni,ni->n", r, sol)
    return float(np.sum(-0.5 * (D * np.log(2 * np.pi) + logdet + quad)))


def _em_measurement_error(Y, S, max_iter=500, tol=1e-8):
    """EM for z_i ~ N(mu, Sigma), y_i ~ N(z_i, S_i) with known S_i.

    Returns (mu, Sigma, n_iter, loglik).
    """
    mu, sigma = _moment_init(Y, S)
    S_inv = _batched_inv(S)                    # (N, D, D), fixed across iters
    ll_prev = -np.inf

    for it in range(1, max_iter + 1):
        sigma_inv = np.linalg.inv(sigma + _JITTER * np.eye(sigma.shape[0]))

        # E-step: posterior of each z_i given y_i.
        prec = sigma_inv[None] + S_inv         # (N, D, D)
        V = _batched_inv(prec)                  # posterior covariances
        rhs = (sigma_inv @ mu)[None] + np.einsum("nij,nj->ni", S_inv, Y)
        m = np.einsum("nij,nj->ni", V, rhs)     # posterior means (N, D)

        # M-step.
        mu = m.mean(axis=0)
        diff = m - mu
        sigma = V.mean(axis=0) + np.einsum("ni,nj->ij", diff, diff) / len(Y)
        sigma = _nearest_psd(sigma)

        ll = _marginal_loglik(Y, S, mu, sigma)
        if np.abs(ll - ll_prev) < tol * (1 + np.abs(ll_prev)):
            return mu, sigma, it, ll
        ll_prev = ll

    warnings.warn(f"measurement-error EM did not converge in {max_iter} "
                  f"iterations (last loglik={ll_prev:.4g}).")
    return mu, sigma, max_iter, ll_prev


@dataclass
class PopulationModel:
    """A fitted generating distribution over transformed phenotype params."""
    param_names_t: list        # transformed-space names
    param_names_natural: list  # natural-space names (aligned index-for-index)
    mu: np.ndarray             # (D,)
    cov: np.ndarray            # (D, D)
    n_used: int
    loglik: float = np.nan
    n_iter: int = 0
    wt_ref: dict = None        # actual wt Stage-1 Hill params (theta_low, ...)

    def sample_transformed(self, n, rng=None):
        """Draw ``n`` phenotype vectors in transformed space, shape (n, D)."""
        rng = np.random.default_rng(rng)
        return rng.multivariate_normal(self.mu, self.cov, size=n)

    def sample(self, n, rng=None):
        """Draw ``n`` phenotypes as a natural-space DataFrame.

        Columns: ``dk_geno``, ``theta_low``, ``theta_high``, ``log_hill_K``,
        ``hill_n`` — exactly the per-genotype quantities Stage 3 injects.
        """
        Z = self.sample_transformed(n, rng=rng)
        return self._to_natural(Z)

    def _to_natural(self, Z):
        """Back-transform a (n, D) transformed array to a natural DataFrame."""
        Z = np.atleast_2d(Z)
        out = {
            "dk_geno":    Z[:, 0],
            "theta_low":  expit(Z[:, 1]),
            "theta_high": expit(Z[:, 2]),
            "log_hill_K": Z[:, 3],
            "hill_n":     np.exp(Z[:, 4]),
        }
        return pd.DataFrame(out, columns=PHENO_PARAMS_NATURAL)

    def save(self, path):
        """Persist as a single, self-contained, human-readable JSON file.

        The whole model is the generating distribution: a small multivariate
        Normal (``mu``, ``cov``) over the transformed per-genotype phenotype
        parameters, plus the parameter names and wt's reference phenotype.  It
        is written as one ``.json`` file (``.json`` appended if absent) that you
        can open and read.  Returns the path actually written.
        """
        path = str(path)
        if not path.endswith(".json"):
            path = path + ".json"
        payload = {
            "_description": (
                "tfscreen empirical phenotype generating distribution: a "
                "multivariate Normal over transformed per-genotype phenotype "
                "parameters. Produced by tfs-build-empirical; consumed "
                "by tfs-simulate via 'phenotype_source: empirical'."),
            "param_names_transformed": list(self.param_names_t),
            "param_names_natural": list(self.param_names_natural),
            "mu": np.asarray(self.mu).tolist(),
            "cov": np.asarray(self.cov).tolist(),
            "wt_ref": self.wt_ref,
            "n_genotypes_used": int(self.n_used),
            "loglik": float(self.loglik),
            "em_iters": int(self.n_iter),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    @classmethod
    def load(cls, path):
        path = str(path)
        if not os.path.exists(path) and os.path.exists(path + ".json"):
            path = path + ".json"
        with open(path) as f:
            d = json.load(f)
        return cls(param_names_t=d["param_names_transformed"],
                   param_names_natural=d["param_names_natural"],
                   mu=np.asarray(d["mu"], dtype=float),
                   cov=np.asarray(d["cov"], dtype=float),
                   n_used=int(d["n_genotypes_used"]),
                   loglik=float(d.get("loglik", np.nan)),
                   n_iter=int(d.get("em_iters", 0)),
                   wt_ref=d.get("wt_ref"))


def fit_population(fits,
                   subset=None,
                   require_converged=True,
                   drop_railed=True,
                   max_iter=500,
                   tol=1e-8):
    """Estimate the deconvolved generating distribution from Stage-1 fits.

    Parameters
    ----------
    fits : dict
        ``{key: GenotypeFit}`` from ``fit_phenotypes``.
    subset : set/list or None
        Restrict to these keys (e.g. spiked-only for the congression-free
        validation fit).
    require_converged, drop_railed : bool
        Fit-quality filters (see ``_collect_estimates``).
    max_iter, tol : int, float
        EM controls.

    Returns
    -------
    PopulationModel
    """
    Y, S, keys = _collect_estimates(
        fits, subset=subset, require_converged=require_converged,
        drop_railed=drop_railed)

    D = Y.shape[1]
    if len(Y) < D + 1:
        raise ValueError(
            f"Only {len(Y)} usable fits for a {D}-dimensional population; "
            f"need at least {D + 1}. Loosen filters or gather more genotypes.")

    mu, sigma, n_iter, ll = _em_measurement_error(Y, S, max_iter=max_iter,
                                                  tol=tol)

    return PopulationModel(
        param_names_t=list(PHENO_PARAMS_TRANSFORMED),
        param_names_natural=list(PHENO_PARAMS_NATURAL),
        mu=mu, cov=sigma, n_used=len(Y), loglik=ll, n_iter=n_iter)
