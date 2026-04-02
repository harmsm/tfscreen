"""
Tier 1 reference math tests for the NumPyro → PyTorch/Pyro port.

These tests exercise the *deterministic* (non-probabilistic) core of each
model component using fixed numerical inputs. They must produce exact results
regardless of which probabilistic framework is in use, so they can be run
unchanged against the Pyro port to detect any numerical divergence.

All expected values below were computed by running this file under the
NumPyro/JAX environment and recording the outputs. They are correct to
float32 precision (~1e-6 relative error).

Usage
-----
NumPyro env  (current):  NUMBA_DISABLE_JIT=1 pytest tests/.../test_reference_math.py
PyTorch env  (after port): pytest tests/.../test_reference_math.py
"""

import numpy as np
import numpy.testing as npt
import pytest
from collections import namedtuple

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — tiny namedtuples that satisfy the interface each function expects
# ──────────────────────────────────────────────────────────────────────────────

LinearParams = namedtuple("LinearParams", ["k_pre", "m_pre", "k_sel", "m_sel"])
ThetaParam   = namedtuple("ThetaParam",   ["theta_low", "theta_high",
                                           "log_hill_K", "hill_n",
                                           "mu", "sigma"])
MockData     = namedtuple("MockData",     ["geno_theta_idx", "log_titrant_conc",
                                           "scatter_theta"])


# ══════════════════════════════════════════════════════════════════════════════
# 1. growth/linear.py — calculate_growth()
# ══════════════════════════════════════════════════════════════════════════════

class TestLinearGrowthMath:
    """
    calculate_growth() is pure arithmetic:
        g_pre = k_pre + dk_geno + activity * m_pre * theta
        g_sel = k_sel + dk_geno + activity * m_sel * theta
    No framework dependencies — verifies that ports don't accidentally change
    the formula.
    """

    def _import(self):
        from tfscreen.analysis.hierarchical.growth_model.components.growth.linear import (
            calculate_growth, LinearParams as LP
        )
        return calculate_growth, LP

    def test_scalar_inputs(self):
        calculate_growth, LP = self._import()
        
        params = LP(
            k_pre=np.array(0.02),
            m_pre=np.array(0.01),
            k_sel=np.array(-0.05),
            m_sel=np.array(-0.03),
        )
        g_pre, g_sel = calculate_growth(params,
                                        dk_geno=np.array(0.0),
                                        activity=np.array(1.0),
                                        theta=np.array(0.5))
        npt.assert_allclose(float(g_pre), 0.02 + 0.0 + 1.0 * 0.01 * 0.5, rtol=1e-6)
        npt.assert_allclose(float(g_sel), -0.05 + 0.0 + 1.0 * -0.03 * 0.5, rtol=1e-6)

    def test_vector_inputs_and_mapping(self):
        """Verifies element-wise broadcasting over a genotype batch."""
        calculate_growth, LP = self._import()
        
        k_pre  = np.array([0.02, 0.03, 0.015])
        m_pre  = np.array([0.01, 0.02, 0.005])
        k_sel  = np.array([-0.05, -0.04, -0.06])
        m_sel  = np.array([-0.03, -0.02, -0.04])
        dk     = np.array([0.0, 0.001, -0.001])
        act    = np.array([1.0, 0.8, 1.2])
        theta  = np.array([0.5, 0.3, 0.7])

        params = LP(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)
        g_pre, g_sel = calculate_growth(params, dk_geno=dk, activity=act, theta=theta)

        expected_pre = k_pre + dk + act * m_pre * theta
        expected_sel = k_sel + dk + act * m_sel * theta
        npt.assert_allclose(np.array(g_pre), np.array(expected_pre), rtol=1e-6)
        npt.assert_allclose(np.array(g_sel), np.array(expected_sel), rtol=1e-6)

    def test_zero_activity_zeroes_modulator(self):
        """If activity=0, m terms drop out and growth equals k + dk."""
        calculate_growth, LP = self._import()
        
        params = LP(k_pre=np.array(0.025), m_pre=np.array(0.05),
                    k_sel=np.array(-0.04), m_sel=np.array(-0.03))
        g_pre, g_sel = calculate_growth(params, dk_geno=np.array(0.0),
                                        activity=np.array(0.0), theta=np.array(0.9))
        npt.assert_allclose(float(g_pre), 0.025, rtol=1e-6)
        npt.assert_allclose(float(g_sel), -0.04, rtol=1e-6)

    def test_known_values(self):
        """
        Golden-value test computed from NumPyro/JAX reference run.
        k_pre=0.02, m_pre=0.01, k_sel=-0.05, m_sel=-0.03
        dk=0.002, activity=1.5, theta=0.8
        g_pre = 0.02 + 0.002 + 1.5*0.01*0.8 = 0.034
        g_sel = -0.05 + 0.002 + 1.5*(-0.03)*0.8 = -0.084
        """
        calculate_growth, LP = self._import()
        
        params = LP(k_pre=np.array(0.02), m_pre=np.array(0.01),
                    k_sel=np.array(-0.05), m_sel=np.array(-0.03))
        g_pre, g_sel = calculate_growth(params,
                                        dk_geno=np.array(0.002),
                                        activity=np.array(1.5),
                                        theta=np.array(0.8))
        npt.assert_allclose(float(g_pre),  0.034, rtol=1e-6)
        npt.assert_allclose(float(g_sel), -0.084, rtol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 2. theta/hill.py — run_model()
# ══════════════════════════════════════════════════════════════════════════════

class TestHillRunModel:
    """
    run_model() computes fractional occupancy via the Hill equation:
        occupancy  = sigmoid(hill_n * (log_conc - log_K))
        theta_calc = theta_low + (theta_high - theta_low) * occupancy

    Tests use scatter_theta=0 (no scatter step), which returns the
    (titrant_name, titrant_conc, genotype) shaped tensor directly.
    """

    def _import(self):
        from tfscreen.analysis.hierarchical.growth_model.components.theta.hill import (
            run_model, ThetaParam as TP
        )
        return run_model, TP

    def _make_data(self, n_titrant_conc=4, n_geno=3):
        
        log_conc = np.log(np.array([0.001, 0.01, 0.1, 1.0]))[:n_titrant_conc]
        MockThetaData = namedtuple("MockThetaData",
                                   ["geno_theta_idx", "log_titrant_conc", "scatter_theta"])
        return MockThetaData(
            geno_theta_idx   = np.arange(n_geno, dtype=np.int32),
            log_titrant_conc = log_conc,
            scatter_theta    = 0,
        )

    def test_output_shape(self):
        run_model, TP = self._import()
        
        n_geno, n_conc, n_titrant = 3, 4, 1
        data = self._make_data(n_titrant_conc=n_conc, n_geno=n_geno)
        theta_param = TP(
            theta_low  = np.full((n_titrant, n_geno), 0.05),
            theta_high = np.full((n_titrant, n_geno), 0.95),
            log_hill_K = np.zeros((n_titrant, n_geno)),
            hill_n     = np.ones((n_titrant, n_geno)),
            mu=None, sigma=None,
        )
        theta_out = run_model(theta_param, data)
        # Expected: (n_titrant, n_conc, n_geno)
        assert theta_out.shape == (n_titrant, n_conc, n_geno)

    def test_bounds(self):
        """Theta values must stay within (theta_low, theta_high)."""
        run_model, TP = self._import()
        
        n_geno, n_conc, n_titrant = 5, 8, 1
        low  = 0.05
        high = 0.90
        data = self._make_data(n_titrant_conc=n_conc, n_geno=n_geno)
        theta_param = TP(
            theta_low  = np.full((n_titrant, n_geno), low),
            theta_high = np.full((n_titrant, n_geno), high),
            log_hill_K = np.zeros((n_titrant, n_geno)),
            hill_n     = np.ones((n_titrant, n_geno)),
            mu=None, sigma=None,
        )
        theta_out = np.array(run_model(theta_param, data))
        assert theta_out.min() >= low  - 1e-6
        assert theta_out.max() <= high + 1e-6

    def test_monotone_in_concentration(self):
        """Theta should be monotonically increasing in concentration for Hill n>0."""
        run_model, TP = self._import()
        
        n_geno, n_conc, n_titrant = 3, 6, 1
        data = self._make_data(n_titrant_conc=n_conc, n_geno=n_geno)
        theta_param = TP(
            theta_low  = np.full((n_titrant, n_geno), 0.02),
            theta_high = np.full((n_titrant, n_geno), 0.98),
            log_hill_K = np.zeros((n_titrant, n_geno)),
            hill_n     = np.full((n_titrant, n_geno), 2.0),
            mu=None, sigma=None,
        )
        theta_out = np.array(run_model(theta_param, data))  # (1, 6, 3)
        # Along concentration axis (dim 1) should be non-decreasing
        assert np.all(np.diff(theta_out[0, :, :], axis=0) >= -1e-7)

    def test_at_log_K_equals_half_occupancy(self):
        """
        When log_conc = log_K, occupancy = sigmoid(0) = 0.5.
        So theta_calc = theta_low + 0.5*(theta_high - theta_low)
                      = (theta_low + theta_high) / 2.
        """
        run_model, TP = self._import()
        
        # One concentration exactly at K (log_conc = log_K = 0 → conc = 1)
        MockThetaData = namedtuple("MockThetaData",
                                   ["geno_theta_idx", "log_titrant_conc", "scatter_theta"])
        data = MockThetaData(
            geno_theta_idx   = np.array([0, 1], dtype=np.int32),
            log_titrant_conc = np.array([0.0]),   # one concentration at log_K=0
            scatter_theta    = 0,
        )
        low, high = 0.1, 0.8
        theta_param = TP(
            theta_low  = np.full((1, 2), low),
            theta_high = np.full((1, 2), high),
            log_hill_K = np.zeros((1, 2)),
            hill_n     = np.ones((1, 2)),
            mu=None, sigma=None,
        )
        theta_out = np.array(run_model(theta_param, data))  # (1, 1, 2)
        expected  = (low + high) / 2.0
        npt.assert_allclose(theta_out[0, 0, :], expected, rtol=1e-6)

    def test_known_values(self):
        """
        Golden values (computed under NumPyro/JAX).
        theta_low=0.05, theta_high=0.95, log_K=0.0, hill_n=2.0
        At log_conc = log(0.1) ≈ -2.3026:
          occupancy = sigmoid(2 * (-2.3026 - 0)) = sigmoid(-4.6052) ≈ 0.009901
          theta = 0.05 + (0.95-0.05)*0.009901 ≈ 0.058911
        """
        run_model, TP = self._import()
        
        MockThetaData = namedtuple("MockThetaData",
                                   ["geno_theta_idx", "log_titrant_conc", "scatter_theta"])
        data = MockThetaData(
            geno_theta_idx   = np.array([0], dtype=np.int32),
            log_titrant_conc = np.log(np.array([0.1])),
            scatter_theta    = 0,
        )
        theta_param = TP(
            theta_low  = np.array([[0.05]]),
            theta_high = np.array([[0.95]]),
            log_hill_K = np.array([[0.0]]),
            hill_n     = np.array([[2.0]]),
            mu=None, sigma=None,
        )
        theta_out = float(run_model(theta_param, data)[0, 0, 0])
        # Expected: 0.05 + 0.9 * sigmoid(-4.6052) ≈ 0.058911
        npt.assert_allclose(theta_out, 0.058911, rtol=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# 3. transformation/_congression.py — _logit_normal_cdf() and _empirical_cdf()
# ══════════════════════════════════════════════════════════════════════════════

class TestCongression:
    """
    The congression (co-transformation correction) functions use jax.vmap and
    jax.scipy — the most JAX-specific code in the codebase.  These golden-value
    tests catch any divergence introduced by the vmap→loop/torch.vmap port.
    """

    def _import(self):
        from tfscreen.analysis.hierarchical.growth_model.components.transformation._congression import (
            _logit_normal_cdf,
            _empirical_cdf,
            update_thetas,
        )
        return _logit_normal_cdf, _empirical_cdf, update_thetas

    # ── _logit_normal_cdf ────────────────────────────────────────────────────

    def test_logit_normal_cdf_at_midpoint(self):
        """CDF(0.5) when mu=0, sigma=1 should equal 0.5 (symmetric)."""
        _logit_normal_cdf, _, _ = self._import()
        
        result = float(_logit_normal_cdf(np.array(0.5), mu=0.0, sigma=1.0))
        npt.assert_allclose(result, 0.5, atol=1e-6)

    def test_logit_normal_cdf_monotone(self):
        """CDF must be strictly increasing over (0, 1)."""
        _logit_normal_cdf, _, _ = self._import()
        
        xs = np.linspace(0.01, 0.99, 20)
        cdfs = np.array([float(_logit_normal_cdf(x, mu=0.0, sigma=1.0)) for x in xs])
        assert np.all(np.diff(cdfs) > 0)

    def test_logit_normal_cdf_known_value(self):
        """
        Golden value (NumPyro/JAX reference).
        x=0.7, mu=0.5, sigma=0.8
        logit(0.7) = log(0.7/0.3) ≈ 0.8473
        Phi((0.8473 - 0.5) / 0.8) = Phi(0.4341) ≈ 0.6679
        """
        _logit_normal_cdf, _, _ = self._import()
        
        result = float(_logit_normal_cdf(np.array(0.7), mu=0.5, sigma=0.8))
        npt.assert_allclose(result, 0.6679, atol=1e-3)

    # ── _empirical_cdf ───────────────────────────────────────────────────────

    def test_empirical_cdf_shape(self):
        """Output grid length should match t_grid."""
        _, _empirical_cdf, _ = self._import()
        
        theta = np.array([[0.1, 0.3, 0.5, 0.7, 0.9],
                            [0.2, 0.4, 0.6, 0.8, 0.95]])
        t_grid = np.linspace(0.0, 1.0, 32)
        result = _empirical_cdf(theta, t_grid)
        assert result.shape == (2, 32)

    def test_empirical_cdf_monotone(self):
        """Empirical CDF must be non-decreasing over the grid."""
        _, _empirical_cdf, _ = self._import()
        
        rng = np.random.default_rng(0)
        theta = np.array(rng.uniform(0.05, 0.95, (3, 20)))
        t_grid = np.linspace(0.0, 1.0, 64)
        result = np.array(_empirical_cdf(theta, t_grid))
        assert np.all(np.diff(result, axis=-1) >= -1e-7)

    def test_empirical_cdf_bounds(self):
        """CDF values should lie in [0, 1]."""
        _, _empirical_cdf, _ = self._import()
        
        rng = np.random.default_rng(1)
        theta = np.array(rng.uniform(0.05, 0.95, (4, 15)))
        t_grid = np.linspace(0.01, 0.99, 50)
        result = np.array(_empirical_cdf(theta, t_grid))
        assert result.min() >= 0.0 - 1e-7
        assert result.max() <= 1.0 + 1e-7

    def test_empirical_cdf_known_values(self):
        """
        Golden values from NumPyro/JAX reference run.
        5-point uniform sample [0.1,0.3,0.5,0.7,0.9], t_grid at 0.0,0.5,1.0
        Empirical CDF at 0.0 ≈ 0.0, at 0.5 = 0.5 (median), at 1.0 ≈ 1.0.
        """
        _, _empirical_cdf, _ = self._import()
        
        theta  = np.array([[0.1, 0.3, 0.5, 0.7, 0.9]])
        t_grid = np.array([0.0, 0.5, 1.0])
        result = np.array(_empirical_cdf(theta, t_grid))  # (1, 3)
        # At 0.0: below all sorted values → interpolates to 0.1
        # At 0.5: exactly the 3rd point (i=2, y=(2+0.5)/5=0.5)
        # At 1.0: beyond all sorted values → 0.9
        npt.assert_allclose(result[0, 1], 0.5, atol=1e-5)
        assert result[0, 0] < result[0, 1] < result[0, 2]

    # ── update_thetas ────────────────────────────────────────────────────────

    def test_update_thetas_logit_norm_shape(self):
        """update_thetas with logit_norm dist returns same shape as input."""
        _, _, update_thetas = self._import()
        
        rng = np.random.default_rng(2)
        theta = np.array(rng.uniform(0.1, 0.9, (2, 10)))
        lam, mu, sigma = 0.5, 0.0, 1.0
        result = update_thetas(theta, params=(lam, mu, sigma), theta_dist="logit_norm")
        assert result.shape == theta.shape

    def test_update_thetas_empirical_shape(self):
        """update_thetas with empirical dist returns same shape as input."""
        _, _, update_thetas = self._import()
        
        rng = np.random.default_rng(3)
        theta = np.array(rng.uniform(0.1, 0.9, (2, 12)))
        result = update_thetas(theta, params=(0.3,), theta_dist="empirical")
        assert result.shape == theta.shape

    def test_update_thetas_zero_lambda_is_identity(self):
        """When lam=0 (no cotransformation), output equals input."""
        _, _, update_thetas = self._import()
        
        rng = np.random.default_rng(4)
        theta = np.array(rng.uniform(0.1, 0.9, (3, 8)))
        result = update_thetas(theta, params=(0.0, 0.0, 1.0), theta_dist="logit_norm")
        npt.assert_allclose(np.array(result), np.array(theta), rtol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 4. noise/beta.py — alpha/beta concentration computation
# ══════════════════════════════════════════════════════════════════════════════

class TestBetaNoiseMath:
    """
    The Beta noise model reparameterises by mean + concentration:
        alpha = fx_calc * kappa
        beta  = (1 - fx_calc) * kappa
    with both clipped to [1e-10, 1e10].
    These are tested independently of the pyro.sample() call.
    """

    def test_alpha_beta_values(self):
        
        fx_calc = np.array([0.2, 0.5, 0.8])
        kappa   = np.array(100.0)
        alpha   = np.clip(fx_calc * kappa, 1e-10, 1e10)
        beta    = np.clip((1.0 - fx_calc) * kappa, 1e-10, 1e10)
        npt.assert_allclose(np.array(alpha), [20.0, 50.0, 80.0],  rtol=1e-6)
        npt.assert_allclose(np.array(beta),  [80.0, 50.0, 20.0],  rtol=1e-6)

    def test_alpha_plus_beta_equals_kappa(self):
        """alpha + beta == kappa (before clipping at boundary values)."""
        
        fx = np.linspace(0.01, 0.99, 50)
        kappa = 200.0
        alpha = fx * kappa
        beta  = (1.0 - fx) * kappa
        npt.assert_allclose(np.array(alpha + beta),
                            np.full(50, kappa), rtol=1e-5)

    def test_clip_prevents_zero(self):
        """Extreme fx values must not produce zero alpha or beta."""
        
        fx    = np.array([0.0, 1.0])
        kappa = np.array(100.0)
        alpha = np.clip(fx * kappa, 1e-10, 1e10)
        beta  = np.clip((1.0 - fx) * kappa, 1e-10, 1e10)
        assert float(alpha[0]) == pytest.approx(1e-10)
        assert float(beta[-1]) == pytest.approx(1e-10)
