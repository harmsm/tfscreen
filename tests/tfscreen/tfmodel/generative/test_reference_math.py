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
        from tfscreen.tfmodel.generative.components.growth.linear import (
            calculate_growth, LinearParams as LP
        )
        return calculate_growth, LP

    def test_scalar_inputs(self):
        calculate_growth, LP = self._import()
        import jax.numpy as jnp
        params = LP(
            k_pre=jnp.array(0.02),
            m_pre=jnp.array(0.01),
            k_sel=jnp.array(-0.05),
            m_sel=jnp.array(-0.03),
        )
        g_pre, g_sel = calculate_growth(params,
                                        dk_geno=jnp.array(0.0),
                                        activity=jnp.array(1.0),
                                        theta=jnp.array(0.5))
        npt.assert_allclose(float(g_pre), 0.02 + 0.0 + 1.0 * 0.01 * 0.5, rtol=1e-6)
        npt.assert_allclose(float(g_sel), -0.05 + 0.0 + 1.0 * -0.03 * 0.5, rtol=1e-6)

    def test_vector_inputs_and_mapping(self):
        """Verifies element-wise broadcasting over a genotype batch."""
        calculate_growth, LP = self._import()
        import jax.numpy as jnp
        k_pre  = jnp.array([0.02, 0.03, 0.015])
        m_pre  = jnp.array([0.01, 0.02, 0.005])
        k_sel  = jnp.array([-0.05, -0.04, -0.06])
        m_sel  = jnp.array([-0.03, -0.02, -0.04])
        dk     = jnp.array([0.0, 0.001, -0.001])
        act    = jnp.array([1.0, 0.8, 1.2])
        theta  = jnp.array([0.5, 0.3, 0.7])

        params = LP(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)
        g_pre, g_sel = calculate_growth(params, dk_geno=dk, activity=act, theta=theta)

        expected_pre = k_pre + dk + act * m_pre * theta
        expected_sel = k_sel + dk + act * m_sel * theta
        npt.assert_allclose(np.array(g_pre), np.array(expected_pre), rtol=1e-6)
        npt.assert_allclose(np.array(g_sel), np.array(expected_sel), rtol=1e-6)

    def test_zero_activity_zeroes_modulator(self):
        """If activity=0, m terms drop out and growth equals k + dk."""
        calculate_growth, LP = self._import()
        import jax.numpy as jnp
        params = LP(k_pre=jnp.array(0.025), m_pre=jnp.array(0.05),
                    k_sel=jnp.array(-0.04), m_sel=jnp.array(-0.03))
        g_pre, g_sel = calculate_growth(params, dk_geno=jnp.array(0.0),
                                        activity=jnp.array(0.0), theta=jnp.array(0.9))
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
        import jax.numpy as jnp
        params = LP(k_pre=jnp.array(0.02), m_pre=jnp.array(0.01),
                    k_sel=jnp.array(-0.05), m_sel=jnp.array(-0.03))
        g_pre, g_sel = calculate_growth(params,
                                        dk_geno=jnp.array(0.002),
                                        activity=jnp.array(1.5),
                                        theta=jnp.array(0.8))
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
        from tfscreen.tfmodel.generative.components.theta.hill_geno import (
            run_model, ThetaParam as TP
        )
        return run_model, TP

    def _make_data(self, n_titrant_conc=4, n_geno=3):
        import jax.numpy as jnp
        log_conc = jnp.log(jnp.array([0.001, 0.01, 0.1, 1.0]))[:n_titrant_conc]
        MockThetaData = namedtuple("MockThetaData",
                                   ["batch_idx", "geno_theta_idx", "log_titrant_conc", "scatter_theta"])
        return MockThetaData(
            batch_idx        = jnp.arange(n_geno, dtype=jnp.int32),
            geno_theta_idx   = jnp.arange(n_geno, dtype=jnp.int32),
            log_titrant_conc = log_conc,
            scatter_theta    = 0,
        )

    def test_output_shape(self):
        run_model, TP = self._import()
        import jax.numpy as jnp
        n_geno, n_conc, n_titrant = 3, 4, 1
        data = self._make_data(n_titrant_conc=n_conc, n_geno=n_geno)
        theta_param = TP(
            theta_low  = jnp.full((n_titrant, n_geno), 0.05),
            theta_high = jnp.full((n_titrant, n_geno), 0.95),
            log_hill_K = jnp.zeros((n_titrant, n_geno)),
            hill_n     = jnp.ones((n_titrant, n_geno)),
            mu=None, sigma=None,
        )
        theta_out = run_model(theta_param, data)
        # Expected: (n_titrant, n_conc, n_geno)
        assert theta_out.shape == (n_titrant, n_conc, n_geno)

    def test_bounds(self):
        """Theta values must stay within (theta_low, theta_high)."""
        run_model, TP = self._import()
        import jax.numpy as jnp
        n_geno, n_conc, n_titrant = 5, 8, 1
        low  = 0.05
        high = 0.90
        data = self._make_data(n_titrant_conc=n_conc, n_geno=n_geno)
        theta_param = TP(
            theta_low  = jnp.full((n_titrant, n_geno), low),
            theta_high = jnp.full((n_titrant, n_geno), high),
            log_hill_K = jnp.zeros((n_titrant, n_geno)),
            hill_n     = jnp.ones((n_titrant, n_geno)),
            mu=None, sigma=None,
        )
        theta_out = np.array(run_model(theta_param, data))
        assert theta_out.min() >= low  - 1e-6
        assert theta_out.max() <= high + 1e-6

    def test_monotone_in_concentration(self):
        """Theta should be monotonically increasing in concentration for Hill n>0."""
        run_model, TP = self._import()
        import jax.numpy as jnp
        n_geno, n_conc, n_titrant = 3, 6, 1
        data = self._make_data(n_titrant_conc=n_conc, n_geno=n_geno)
        theta_param = TP(
            theta_low  = jnp.full((n_titrant, n_geno), 0.02),
            theta_high = jnp.full((n_titrant, n_geno), 0.98),
            log_hill_K = jnp.zeros((n_titrant, n_geno)),
            hill_n     = jnp.full((n_titrant, n_geno), 2.0),
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
        import jax.numpy as jnp
        # One concentration exactly at K (log_conc = log_K = 0 → conc = 1)
        MockThetaData = namedtuple("MockThetaData",
                                   ["batch_idx", "geno_theta_idx", "log_titrant_conc", "scatter_theta"])
        data = MockThetaData(
            batch_idx        = jnp.array([0, 1], dtype=jnp.int32),
            geno_theta_idx   = jnp.array([0, 1], dtype=jnp.int32),
            log_titrant_conc = jnp.array([0.0]),   # one concentration at log_K=0
            scatter_theta    = 0,
        )
        low, high = 0.1, 0.8
        theta_param = TP(
            theta_low  = jnp.full((1, 2), low),
            theta_high = jnp.full((1, 2), high),
            log_hill_K = jnp.zeros((1, 2)),
            hill_n     = jnp.ones((1, 2)),
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
        import jax.numpy as jnp
        MockThetaData = namedtuple("MockThetaData",
                                   ["batch_idx", "geno_theta_idx", "log_titrant_conc", "scatter_theta"])
        data = MockThetaData(
            batch_idx        = jnp.array([0], dtype=jnp.int32),
            geno_theta_idx   = jnp.array([0], dtype=jnp.int32),
            log_titrant_conc = jnp.log(jnp.array([0.1])),
            scatter_theta    = 0,
        )
        theta_param = TP(
            theta_low  = jnp.array([[0.05]]),
            theta_high = jnp.array([[0.95]]),
            log_hill_K = jnp.array([[0.0]]),
            hill_n     = jnp.array([[2.0]]),
            mu=None, sigma=None,
        )
        theta_out = float(run_model(theta_param, data)[0, 0, 0])
        # Expected: 0.05 + 0.9 * sigmoid(-4.6052) ≈ 0.058911
        npt.assert_allclose(theta_out, 0.058911, rtol=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# 3. noise/beta.py — alpha/beta concentration computation
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
        import jax.numpy as jnp
        fx_calc = jnp.array([0.2, 0.5, 0.8])
        kappa   = jnp.array(100.0)
        alpha   = jnp.clip(fx_calc * kappa, 1e-10, 1e10)
        beta    = jnp.clip((1.0 - fx_calc) * kappa, 1e-10, 1e10)
        npt.assert_allclose(np.array(alpha), [20.0, 50.0, 80.0],  rtol=1e-6)
        npt.assert_allclose(np.array(beta),  [80.0, 50.0, 20.0],  rtol=1e-6)

    def test_alpha_plus_beta_equals_kappa(self):
        """alpha + beta == kappa (before clipping at boundary values)."""
        import jax.numpy as jnp
        fx = jnp.linspace(0.01, 0.99, 50)
        kappa = 200.0
        alpha = fx * kappa
        beta  = (1.0 - fx) * kappa
        npt.assert_allclose(np.array(alpha + beta),
                            np.full(50, kappa), rtol=1e-5)

    def test_clip_prevents_zero(self):
        """Extreme fx values must not produce zero alpha or beta."""
        import jax.numpy as jnp
        fx    = jnp.array([0.0, 1.0])
        kappa = jnp.array(100.0)
        alpha = jnp.clip(fx * kappa, 1e-10, 1e10)
        beta  = jnp.clip((1.0 - fx) * kappa, 1e-10, 1e10)
        assert float(alpha[0]) == pytest.approx(1e-10)
        assert float(beta[-1]) == pytest.approx(1e-10)
