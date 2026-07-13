"""Tests for the Stage-2 measurement-error deconvolution."""

import numpy as np
import pandas as pd
import pytest

from tfscreen.simulate.empirical.fit_phenotypes import (
    GenotypeFit,
    PHENO_PARAMS_NATURAL,
    _LOGIT_BOUND,
)
from tfscreen.simulate.empirical.population import (
    fit_population,
    PopulationModel,
    _collect_estimates,
    _batched_inv,
)

D = 5
_PLACEHOLDER_NAMES = ["ln_cfu0[0]"] + ["p"] * D


def _make_fit(z, S, converged=True):
    """Wrap a transformed phenotype vector + covariance as a GenotypeFit.

    One nuisance intercept is prepended so ``pheno_slice`` is slice(1, 6),
    mirroring the real Stage-1 layout.
    """
    est_t = np.concatenate([[5.0], np.asarray(z, dtype=float)])
    cov_t = np.zeros((D + 1, D + 1))
    cov_t[0, 0] = 0.01
    cov_t[1:, 1:] = S
    return GenotypeFit(
        genotype="g", titrant_name="iptg", n_obs=100, converged=converged,
        param_names_t=_PLACEHOLDER_NAMES, est_t=est_t, cov_t=cov_t,
        pheno_slice=slice(1, D + 1))


def _random_psd(rng, scale=0.3):
    A = rng.normal(size=(D, D)) * scale
    return A @ A.T + 0.05 * np.eye(D)


def _make_population(n, mu, sigma, meas_scale, seed=0):
    """Simulate n genotypes: true z ~ N(mu, sigma), estimate y ~ N(z, S_i)."""
    rng = np.random.default_rng(seed)
    Z = rng.multivariate_normal(mu, sigma, size=n)      # true phenotypes
    fits = {}
    S_list = []
    for i in range(n):
        # Heteroscedastic diagonal measurement covariance.
        s_diag = meas_scale * (0.5 + rng.random(D))
        S = np.diag(s_diag)
        y = Z[i] + rng.multivariate_normal(np.zeros(D), S)
        fits[(f"g{i}", "iptg")] = _make_fit(y, S)
        S_list.append(S)
    return fits, np.array(S_list), Z


# True generating distribution used across recovery tests.
_MU = np.array([0.0, 1.5, -1.5, np.log(0.01), 0.0])
_SIGMA = _random_psd(np.random.default_rng(0), scale=0.35)


def test_deconvolution_removes_estimation_inflation():
    """Sigma_hat must recover the true pop cov, beating naive Cov(Y)."""
    fits, S, Z = _make_population(4000, _MU, _SIGMA, meas_scale=0.25, seed=1)
    model = fit_population(fits, drop_railed=False)

    # Reconstruct the naive point-estimate covariance for comparison.
    Y = np.array([f.est_t[f.pheno_slice] for f in fits.values()])
    naive_cov = np.cov(Y, rowvar=False)

    def rel(A):
        return np.linalg.norm(A - _SIGMA) / np.linalg.norm(_SIGMA)

    # Deconvolved estimate is close to truth...
    assert rel(model.cov) < 0.15
    # ...and materially closer than the (inflated) naive covariance.
    assert rel(model.cov) < 0.5 * rel(naive_cov)
    # The inflation is real: naive diagonal exceeds the true diagonal.
    assert np.all(np.diag(naive_cov) > np.diag(_SIGMA))
    # Mean is recovered.
    assert model.mu == pytest.approx(_MU, abs=0.05)


def test_sampling_matches_fitted_distribution():
    fits, _, _ = _make_population(3000, _MU, _SIGMA, meas_scale=0.2, seed=2)
    model = fit_population(fits, drop_railed=False)

    draws = model.sample(20000, rng=7)
    assert list(draws.columns) == PHENO_PARAMS_NATURAL
    assert (draws["theta_low"] > 0).all() and (draws["theta_low"] < 1).all()
    assert (draws["theta_high"] > 0).all() and (draws["theta_high"] < 1).all()
    assert (draws["hill_n"] > 0).all()

    # Sample covariance of transformed draws ~ model.cov.
    Zt = model.sample_transformed(50000, rng=11)
    assert np.allclose(np.cov(Zt, rowvar=False), model.cov, atol=0.05)


def test_filters_drop_bad_fits():
    rng = np.random.default_rng(3)
    good = {(f"g{i}", "iptg"): _make_fit(rng.normal(size=D) * 0.3 + _MU,
                                         0.1 * np.eye(D))
            for i in range(30)}

    # Non-converged.
    good[("bad_conv", "iptg")] = _make_fit(_MU, 0.1 * np.eye(D),
                                           converged=False)
    # Railed theta logit.
    railed = _MU.copy()
    railed[1] = _LOGIT_BOUND
    good[("bad_railed", "iptg")] = _make_fit(railed, 0.1 * np.eye(D))
    # Non-finite covariance.
    good[("bad_nan", "iptg")] = _make_fit(_MU, np.full((D, D), np.nan))

    Y, S, keys = _collect_estimates(good)
    assert len(keys) == 30
    assert "bad_conv" not in [k[0] for k in keys]
    assert "bad_railed" not in [k[0] for k in keys]
    assert "bad_nan" not in [k[0] for k in keys]


def test_subset_restricts_fit():
    fits, _, _ = _make_population(200, _MU, _SIGMA, meas_scale=0.2, seed=4)
    subset = list(fits.keys())[:50]
    model = fit_population(fits, subset=set(subset), drop_railed=False)
    assert model.n_used == 50


def test_too_few_fits_raises():
    fits, _, _ = _make_population(3, _MU, _SIGMA, meas_scale=0.2, seed=5)
    with pytest.raises(ValueError, match="usable fits"):
        fit_population(fits, drop_railed=False)


def test_save_load_roundtrip(tmp_path):
    fits, _, _ = _make_population(500, _MU, _SIGMA, meas_scale=0.2, seed=6)
    model = fit_population(fits, drop_railed=False)

    path = tmp_path / "pop"
    model.save(str(path))
    loaded = PopulationModel.load(str(path))

    assert np.allclose(loaded.mu, model.mu)
    assert np.allclose(loaded.cov, model.cov)
    assert loaded.param_names_natural == model.param_names_natural
    assert loaded.n_used == model.n_used


def test_batched_inv_handles_singular_and_ill_conditioned():
    """_batched_inv must not raise on singular / ill-conditioned matrices."""
    rng = np.random.default_rng(0)

    # A well-conditioned PSD matrix: robust inverse ~ true inverse.
    A = _random_psd(rng)
    inv = _batched_inv(A)
    assert np.allclose(inv @ A, np.eye(D), atol=1e-6)

    # A stack containing a genuinely singular matrix (a rank-1 outer product)
    # and a wildly ill-conditioned one (eigenvalues spanning 1e12 .. 1e-8, the
    # case that makes np.linalg.inv report "Singular matrix"). The robust
    # inverse must produce a finite result rather than raising.
    v = rng.normal(size=D)
    singular = np.outer(v, v)                      # rank 1

    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    eigs = np.array([1e12, 1e6, 1.0, 1e-4, 1e-8])
    ill = (Q * eigs) @ Q.T

    out = _batched_inv(np.stack([singular, ill]))  # robust inv copes
    assert np.all(np.isfinite(out))
    # Precision is bounded by the eigenvalue floor (1 / 1e-8 = 1e8).
    assert np.max(np.abs(out)) <= 1e8 * (1 + 1e-6)


def test_fit_population_survives_singular_covariance():
    """A single unidentified (huge-variance) genotype must not crash the EM."""
    fits, _, _ = _make_population(300, _MU, _SIGMA, meas_scale=0.2, seed=9)

    # Inject a genotype whose Stage-1 covariance is effectively singular: one
    # parameter is unconstrained (variance 1e12), mirroring a real failed fit.
    bad_S = 0.1 * np.eye(D)
    bad_S[0, 0] = 1e12
    fits[("unidentified", "iptg")] = _make_fit(_MU, bad_S)

    model = fit_population(fits, drop_railed=False)   # must not raise
    assert np.all(np.isfinite(model.mu))
    assert np.all(np.isfinite(model.cov))


def test_wt_ref_roundtrip(tmp_path):
    fits, _, _ = _make_population(300, _MU, _SIGMA, meas_scale=0.2, seed=8)
    model = fit_population(fits, drop_railed=False)
    model.wt_ref = {"theta_low": 0.99, "theta_high": 0.01,
                    "log_hill_K": np.log(0.017), "hill_n": 2.0}

    path = tmp_path / "pop"
    model.save(str(path))
    loaded = PopulationModel.load(str(path))

    assert loaded.wt_ref == model.wt_ref
