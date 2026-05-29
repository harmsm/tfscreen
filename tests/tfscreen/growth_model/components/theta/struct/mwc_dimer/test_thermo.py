"""
Tests for struct/mwc_dimer/thermo.py.

Covers: _compute_theta, _population_moments, run_model, get_population_moments.
The Newton-solver for the free-effector cubic is exercised implicitly through
_compute_theta.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from collections import namedtuple

from tfscreen.growth_model.components.theta.struct.mwc_dimer.thermo import (
    ThetaParam,
    _compute_theta,
    _population_moments,
    run_model,
    get_population_moments,
)

# ---------------------------------------------------------------------------
# Helpers — analytical apo formula
# ---------------------------------------------------------------------------

def _apo_theta(ln_K_h_l, ln_K_h_o, ln_K_l_o, tf_total):
    """
    Analytic theta at zero effector.

    At e_total = 0:  e_free = 0,  P_H = P_L = 1,  F = 1 + K_h_l
    h_free = r / (1 + K_h_l)
    W      = h_free * (K_h_o + K_h_l * K_l_o)
    theta  = W / (1 + W)
    """
    r     = tf_total / 2.0
    K_h_l = np.exp(ln_K_h_l)
    K_h_o = np.exp(ln_K_h_o)
    K_l_o = np.exp(ln_K_l_o)
    h_free = r / (1.0 + K_h_l)
    W = h_free * (K_h_o + K_h_l * K_l_o)
    return W / (1.0 + W)


# ---------------------------------------------------------------------------
# Mock data (only fields used by thermo functions)
# ---------------------------------------------------------------------------

MockData = namedtuple("MockData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "titrant_conc",
    "geno_theta_idx",
    "scatter_theta",
])

_CONC = np.array([0.0, 1e-5, 1e-4, 1e-3])


@pytest.fixture
def mock_data():
    G = 4
    return MockData(
        num_titrant_name=2,
        num_titrant_conc=len(_CONC),
        num_genotype=G,
        titrant_conc=jnp.array(_CONC, dtype=jnp.float32),
        geno_theta_idx=jnp.arange(G, dtype=jnp.int32),
        scatter_theta=0,
    )


def _wt_theta_param(T=2, G=4, C=4):
    """ThetaParam at WT lac repressor values (Sochor 2014, Table 1).

    K_h_l = K_RR* = 6.3  →  ln = 1.84  (L-state dominant, ~86% without IPTG)
    K_h_o = K_RO = 4.2e8 M⁻¹  →  ln = 19.9
    K_l_o = K_R*O ≈ 0  →  ln = -2.3  (L-state barely binds DNA)
    K_h_e = K_RE = 5.6e4 M⁻¹  →  ln = 10.9
    K_l_e = K_R*E = 7.6e5 M⁻¹  →  ln = 13.5  (IPTG binds L ~14× tighter)

    With these values: theta_apo ≈ 0.95; theta saturates to near 0 at mM IPTG.
    """
    return ThetaParam(
        ln_K_h_l=jnp.full(G, 1.84),
        ln_K_h_o=jnp.full(G, 19.9),
        ln_K_h_e=jnp.full((T, G), 10.9),
        ln_K_l_o=jnp.full(G, -2.3),
        ln_K_l_e=jnp.full((T, G), 13.5),
        tf_total=6.5e-7,
        op_total=2.5e-8,
        mu=jnp.zeros((T, C, 1)),
        sigma=jnp.ones((T, C, 1)),
        conc_unit_scale=1.0,   # test concentrations are already in M
    )


# ---------------------------------------------------------------------------
# _compute_theta
# ---------------------------------------------------------------------------

class TestComputeTheta:

    def test_output_shape(self):
        T, G = 2, 4
        conc = jnp.array(_CONC)
        theta = _compute_theta(
            jnp.full(G, -7.4), jnp.full(G, 23.0), jnp.full((T, G), 11.5),
            jnp.full(G, 13.8), jnp.full((T, G), 15.0),
            conc, 6.5e-7, 2.5e-8,
        )
        assert theta.shape == (T, len(_CONC), G)

    def test_values_in_0_1(self):
        T, G = 1, 3
        conc = jnp.linspace(0.0, 1e-3, 8)
        theta = _compute_theta(
            jnp.full(G, -7.4), jnp.full(G, 23.0), jnp.full((T, G), 11.5),
            jnp.full(G, 13.8), jnp.full((T, G), 15.0),
            conc, 6.5e-7, 2.5e-8,
        )
        assert jnp.all(theta >= 0)
        assert jnp.all(theta <= 1)

    def test_all_finite(self):
        T, G = 2, 3
        conc = jnp.linspace(0.0, 1e-2, 6)
        theta = _compute_theta(
            jnp.linspace(-10.0, -5.0, G), jnp.linspace(18.0, 25.0, G),
            jnp.full((T, G), 11.5),
            jnp.linspace(10.0, 16.0, G), jnp.full((T, G), 15.0),
            conc, 6.5e-7, 2.5e-8,
        )
        assert jnp.all(jnp.isfinite(theta))

    def test_monotone_decreasing_in_effector(self):
        """Higher IPTG → more L-state → lower DNA occupancy.

        Uses Sochor 2014 WT values (K_h_l = 6.3, K_l_o ≈ 0) where IPTG
        produces a large, physically meaningful decrease in theta.
        """
        G = 2
        conc = jnp.array([0.0, 1e-5, 1e-4, 1e-3, 1e-2])
        theta = _compute_theta(
            jnp.full(G, 1.84), jnp.full(G, 19.9), jnp.full((1, G), 10.9),
            jnp.full(G, -2.3), jnp.full((1, G), 13.5),
            conc, 6.5e-7, 2.5e-8,
        )
        for c in range(len(conc) - 1):
            assert jnp.all(theta[:, c, :] >= theta[:, c + 1, :])

    def test_apo_matches_analytic_formula(self):
        """At e_total=0, theta equals the analytical apo formula."""
        G = 3
        # Use simple numbers so both branches are exercised
        ln_K_h_l_vals = np.array([0.0, -1.0, 1.0])
        ln_K_h_o_vals = np.array([2.0,  3.0,  2.5])
        ln_K_l_o_vals = np.array([1.0,  1.5,  0.5])
        tf_total = 10.0

        theta = _compute_theta(
            jnp.array(ln_K_h_l_vals), jnp.array(ln_K_h_o_vals),
            jnp.full((1, G), -100.0),   # K_h_e negligible
            jnp.array(ln_K_l_o_vals), jnp.full((1, G), -100.0),
            jnp.array([0.0, 1e-8]),     # [0] is apo; [1] ≈ apo (tiny conc)
            tf_total, 0.0,
        )                                # (1, 2, G)

        for g in range(G):
            expected = _apo_theta(ln_K_h_l_vals[g], ln_K_h_o_vals[g],
                                  ln_K_l_o_vals[g], tf_total)
            assert theta[0, 0, g] == pytest.approx(float(expected), rel=1e-3)

    def test_higher_K_h_o_gives_higher_theta_at_zero_effector(self):
        """Stronger H-state DNA binding → higher apo occupancy."""
        G = 2
        conc = jnp.array([0.0])
        theta = _compute_theta(
            jnp.full(G, 1.84),
            jnp.array([17.0, 21.0]),   # G=0 weaker, G=1 stronger
            jnp.full((1, G), 10.9),
            jnp.full(G, -2.3), jnp.full((1, G), 13.5),
            conc, 6.5e-7, 2.5e-8,
        )
        assert theta[0, 0, 1] > theta[0, 0, 0]

    def test_higher_K_h_l_gives_lower_theta_at_zero_effector(self):
        """More L-state (larger K_h_l) → lower apo occupancy when K_l_o << K_h_o.

        Uses near-zero K_l_o (matching Sochor 2014: L-state barely binds DNA)
        so the L-state contribution to W is negligible.
        """
        G = 2
        conc = jnp.array([0.0])
        theta = _compute_theta(
            jnp.array([-2.0, 2.0]),  # G=0 H-favored, G=1 L-favored
            jnp.full(G, 19.9), jnp.full((1, G), 10.9),
            jnp.full(G, -2.3), jnp.full((1, G), 13.5),
            conc, 6.5e-7, 2.5e-8,
        )
        assert theta[0, 0, 0] > theta[0, 0, 1]

    def test_titrant_dim_broadcasts_independently(self):
        """Different K_h_e/K_l_e per titrant produce different IPTG dose-responses.

        When K_h_e ≈ 0 (effector barely binds H-state) but K_l_e is large,
        IPTG exclusively stabilises the L-state → large drop in theta.
        When K_h_e is equally large, IPTG also stabilises H, partially cancelling
        the L-shift → smaller drop.
        """
        G = 1
        conc = jnp.array([0.0, 1e-3])
        # T=0: negligible H-state binding (K_h_e ≈ 0), strong L-state binding
        #      → IPTG strongly favours L → large theta drop
        # T=1: strong H-state binding too → IPTG less effective at lowering theta
        ln_K_h_e = jnp.array([[-20.0, -20.0],   # T=0: K_h_e ≈ 0
                               [ 15.0,  15.0]])  # T=1: K_h_e = K_l_e (no net shift)
        ln_K_l_e = jnp.array([[15.0, 15.0],
                               [15.0, 15.0]])
        theta = _compute_theta(
            jnp.full(G, 1.84), jnp.full(G, 19.9), ln_K_h_e,
            jnp.full(G, -2.3), ln_K_l_e,
            conc, 6.5e-7, 2.5e-8,
        )
        # T=0: large drop from apo→saturated IPTG
        drop_t0 = theta[0, 0, 0] - theta[0, 1, 0]
        # T=1: smaller drop (effector also stabilises H, countering L shift)
        drop_t1 = theta[1, 0, 0] - theta[1, 1, 0]
        assert float(drop_t0) > float(drop_t1), (
            f"T=0 drop ({drop_t0:.4f}) should exceed T=1 drop ({drop_t1:.4f})"
        )

    def test_wt_params_give_high_apo_theta(self):
        """WT lac repressor should have high apo occupancy (≥ 0.85).

        Uses Sochor 2014 values: K_h_l = 6.3, K_h_o = 4.2e8 M⁻¹,
        K_l_o ≈ 0.1 M⁻¹.  The operator-depletion approximation gives
        theta_apo ≈ 0.95 with tf_total = 650 nM.
        """
        G = 1
        theta = _compute_theta(
            jnp.full(G, 1.84), jnp.full(G, 19.9), jnp.full((1, G), 10.9),
            jnp.full(G, -2.3), jnp.full((1, G), 13.5),
            jnp.array([0.0]), 6.5e-7, 2.5e-8,
        )
        assert float(theta[0, 0, 0]) >= 0.85

    def test_e_free_bounded_by_e_total(self):
        """Newton solution should never produce negative theta or theta > 1,
        even with extreme effector concentrations.  Uses Sochor 2014 WT values."""
        G = 2
        conc = jnp.array([0.0, 1e-6, 1e-4, 1e-2, 1.0])   # up to 1 M IPTG
        theta = _compute_theta(
            jnp.full(G, 1.84), jnp.full(G, 19.9), jnp.full((1, G), 10.9),
            jnp.full(G, -2.3), jnp.full((1, G), 13.5),
            conc, 6.5e-7, 2.5e-8,
        )
        assert jnp.all(theta >= 0)
        assert jnp.all(theta <= 1)
        assert jnp.all(jnp.isfinite(theta))


# ---------------------------------------------------------------------------
# _population_moments
# ---------------------------------------------------------------------------

class TestPopulationMoments:

    def test_output_shapes(self, mock_data):
        T, C, G = 2, len(_CONC), 4
        theta = jnp.full((T, C, G), 0.7)
        mu, sigma = _population_moments(theta, mock_data)
        assert mu.shape == (T, C, 1)
        assert sigma.shape == (T, C, 1)

    def test_sigma_zero_when_uniform(self, mock_data):
        T, C, G = 2, len(_CONC), 4
        theta = jnp.full((T, C, G), 0.5)
        _, sigma = _population_moments(theta, mock_data)
        assert jnp.allclose(sigma, 0.0, atol=1e-6)

    def test_sigma_positive_when_varying(self, mock_data):
        T, C, G = 2, len(_CONC), 4
        theta = jnp.linspace(0.1, 0.9, T * C * G).reshape(T, C, G)
        _, sigma = _population_moments(theta, mock_data)
        assert jnp.all(sigma > 0)

    def test_mu_finite(self, mock_data):
        T, C, G = 2, len(_CONC), 4
        theta = jnp.full((T, C, G), 0.8)
        mu, _ = _population_moments(theta, mock_data)
        assert jnp.all(jnp.isfinite(mu))

    def test_data_arg_unused(self, mock_data):
        """_population_moments does not use fields from data."""
        T, C, G = 2, len(_CONC), 4
        theta = jnp.linspace(0.2, 0.8, T * C * G).reshape(T, C, G)
        mu1, sigma1 = _population_moments(theta, mock_data)
        mu2, sigma2 = _population_moments(theta, None)
        assert jnp.allclose(mu1, mu2)
        assert jnp.allclose(sigma1, sigma2)


# ---------------------------------------------------------------------------
# run_model
# ---------------------------------------------------------------------------

class TestRunModel:

    def test_scatter_theta_0_shape(self, mock_data):
        tp = _wt_theta_param(T=2, G=4, C=len(_CONC))
        result = run_model(tp, mock_data)
        assert result.shape == (2, len(_CONC), 4)

    def test_scatter_theta_1_shape(self, mock_data):
        tp = _wt_theta_param(T=2, G=4, C=len(_CONC))
        data = mock_data._replace(scatter_theta=1)
        result = run_model(tp, data)
        assert result.shape == (1, 1, 1, 1, 2, len(_CONC), 4)

    def test_values_in_0_1(self, mock_data):
        tp = _wt_theta_param(T=2, G=4, C=len(_CONC))
        result = run_model(tp, mock_data)
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)

    def test_all_finite(self, mock_data):
        tp = _wt_theta_param(T=2, G=4, C=len(_CONC))
        result = run_model(tp, mock_data)
        assert jnp.all(jnp.isfinite(result))

    def test_geno_theta_idx_selects_genotypes(self):
        """geno_theta_idx should index into the G dimension correctly."""
        G = 4
        # Build distinct K_h_o per genotype so theta differs
        T, C = 1, 2
        ln_K_h_o = jnp.array([18.0, 20.0, 22.0, 24.0])
        tp = ThetaParam(
            ln_K_h_l=jnp.full(G, -7.4),
            ln_K_h_o=ln_K_h_o,
            ln_K_h_e=jnp.full((T, G), 11.5),
            ln_K_l_o=jnp.full(G, 13.8),
            ln_K_l_e=jnp.full((T, G), 15.0),
            tf_total=6.5e-7, op_total=2.5e-8,
            mu=jnp.zeros((T, C, 1)), sigma=jnp.ones((T, C, 1)),
            conc_unit_scale=1.0,
        )
        # Select genotypes [3, 0] (reversed)
        MockSmall = namedtuple("MockSmall", ["titrant_conc", "geno_theta_idx", "scatter_theta"])
        data = MockSmall(jnp.array([0.0, 1e-4]), jnp.array([3, 0], dtype=jnp.int32), 0)
        result = run_model(tp, data)              # (T, C, 2)
        # Compute all-genotype theta for reference
        theta_all = _compute_theta(
            tp.ln_K_h_l, tp.ln_K_h_o, tp.ln_K_h_e,
            tp.ln_K_l_o, tp.ln_K_l_e,
            data.titrant_conc, tp.tf_total, tp.op_total,
        )                                          # (T, C, G)
        assert jnp.allclose(result[:, :, 0], theta_all[:, :, 3], atol=1e-6)
        assert jnp.allclose(result[:, :, 1], theta_all[:, :, 0], atol=1e-6)

    def test_apo_matches_analytic_formula(self):
        """run_model at e=0 matches the closed-form apo theta."""
        G = 1
        ln_K_h_l_val, ln_K_h_o_val, ln_K_l_o_val = 0.0, 2.0, 1.0
        tf_total = 10.0
        tp = ThetaParam(
            ln_K_h_l=jnp.full(G, ln_K_h_l_val),
            ln_K_h_o=jnp.full(G, ln_K_h_o_val),
            ln_K_h_e=jnp.full((1, G), -100.0),
            ln_K_l_o=jnp.full(G, ln_K_l_o_val),
            ln_K_l_e=jnp.full((1, G), -100.0),
            tf_total=tf_total, op_total=0.0,
            mu=jnp.zeros((1, 2, 1)), sigma=jnp.ones((1, 2, 1)),
            conc_unit_scale=1.0,
        )
        MockSmall = namedtuple("MockSmall", ["titrant_conc", "geno_theta_idx", "scatter_theta"])
        data = MockSmall(jnp.array([0.0, 1e-8]), jnp.array([0], dtype=jnp.int32), 0)
        result = run_model(tp, data)    # (1, 2, 1)

        expected = _apo_theta(ln_K_h_l_val, ln_K_h_o_val, ln_K_l_o_val, tf_total)
        assert float(result[0, 0, 0]) == pytest.approx(expected, rel=1e-3)

    def test_higher_effector_lower_theta(self, mock_data):
        """run_model theta is monotone non-increasing in effector concentration."""
        tp = _wt_theta_param(T=2, G=4, C=len(_CONC))
        result = run_model(tp, mock_data)
        for c in range(len(_CONC) - 1):
            assert jnp.all(result[:, c, :] >= result[:, c + 1, :])


# ---------------------------------------------------------------------------
# get_population_moments
# ---------------------------------------------------------------------------

def test_get_population_moments_returns_stored_values(mock_data):
    """get_population_moments is a passthrough from ThetaParam.mu/sigma."""
    tp = _wt_theta_param(T=2, G=4, C=len(_CONC))
    mu, sigma = get_population_moments(tp, mock_data)
    assert jnp.allclose(mu,    tp.mu)
    assert jnp.allclose(sigma, tp.sigma)
