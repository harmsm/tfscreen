import pytest
import numpy as np
import jax.numpy as jnp
from collections import namedtuple

from tfscreen.tfmodel.components.theta.struct.lac_dimer_unfolded.thermo import (
    ThetaParam,
    _solve_free_effector,
    _compute_theta,
    _population_moments,
    run_model,
    get_population_moments,
)

# ---------------------------------------------------------------------------
# Minimal mock data (only fields used by thermo functions)
# ---------------------------------------------------------------------------

MockData = namedtuple("MockData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "titrant_conc",
    "geno_theta_idx",
    "scatter_theta",
])

_CONC = np.array([0.0, 100.0, 1000.0])


@pytest.fixture
def mock_data():
    return MockData(
        num_titrant_name=2,
        num_titrant_conc=3,
        num_genotype=4,
        titrant_conc=jnp.array(_CONC, dtype=jnp.float32),
        geno_theta_idx=jnp.arange(4, dtype=jnp.int32),
        scatter_theta=0,
    )


@pytest.fixture
def theta_param(mock_data):
    T, G = mock_data.num_titrant_name, mock_data.num_genotype
    C = mock_data.num_titrant_conc
    return ThetaParam(
        ln_K_op=jnp.full(G, 2.3),
        ln_K_HL=jnp.full(G, -9.0),
        ln_K_E=jnp.full((T, G), -4.0),
        ln_K_U=jnp.full(G, -12.0),
        tf_total=650.0,
        op_total=25.0,
        mu=jnp.zeros((T, C, 1)),
        sigma=jnp.ones((T, C, 1)),
    )


# ---------------------------------------------------------------------------
# _solve_free_effector
# ---------------------------------------------------------------------------

class TestSolveFreeEffector:

    def test_zero_effector_returns_zero(self):
        e_total = jnp.zeros((1, 3, 1))
        a  = jnp.ones((2, 1, 4)) * 1e-4
        # Z0 includes K_U term: 1 + K_op*op + K_HL + K_U
        Z0 = jnp.ones((1, 1, 4)) * 252.0
        e_free = _solve_free_effector(e_total, tf_total=650.0, a=a, Z0=Z0)
        assert jnp.allclose(e_free, 0.0, atol=1e-6)

    def test_e_free_leq_e_total(self):
        e_total = jnp.array([0.0, 100.0, 1000.0])[None, :, None]
        a  = jnp.ones((1, 1, 1)) * 1e-4
        Z0 = jnp.ones((1, 1, 1)) * 252.0
        e_free = _solve_free_effector(e_total, tf_total=650.0, a=a, Z0=Z0)
        assert jnp.all(e_free >= 0)
        assert jnp.all(e_free <= e_total + 1e-9)

    def test_output_shape(self):
        T, C, G = 2, 3, 4
        e_total = jnp.ones((1, C, 1)) * 500.0
        a  = jnp.ones((T, 1, G)) * 1e-5
        Z0 = jnp.ones((1, 1, G)) * 252.0
        e_free = _solve_free_effector(e_total, tf_total=650.0, a=a, Z0=Z0)
        assert e_free.shape == (T, C, G)

    def test_finite_values(self):
        e_total = jnp.array([0.0, 100.0, 1000.0])[None, :, None]
        a  = jnp.ones((1, 1, 1)) * 1e-4
        Z0 = jnp.ones((1, 1, 1)) * 252.0
        e_free = _solve_free_effector(e_total, tf_total=650.0, a=a, Z0=Z0)
        assert jnp.all(jnp.isfinite(e_free))

    def test_high_affinity_depletes_effector(self):
        """Very high K_HL·K_E should deplete free effector significantly."""
        e_total = jnp.array([1000.0])[None, :, None]
        a_high = jnp.ones((1, 1, 1)) * 10.0    # high affinity
        a_low  = jnp.ones((1, 1, 1)) * 1e-10   # negligible affinity
        Z0 = jnp.ones((1, 1, 1)) * 252.0
        e_free_high = _solve_free_effector(e_total, 650.0, a_high, Z0)
        e_free_low  = _solve_free_effector(e_total, 650.0, a_low,  Z0)
        assert jnp.all(e_free_high < e_free_low)

    def test_larger_Z0_reduces_depletion(self):
        """Larger Z0 (e.g. large K_U) means less TF sequestering effector → less depletion."""
        e_total = jnp.array([500.0])[None, :, None]
        a  = jnp.ones((1, 1, 1)) * 1e-3
        Z0_small = jnp.ones((1, 1, 1)) * 10.0
        Z0_large = jnp.ones((1, 1, 1)) * 1e6   # dominated by K_U
        e_free_small = _solve_free_effector(e_total, 650.0, a, Z0_small)
        e_free_large = _solve_free_effector(e_total, 650.0, a, Z0_large)
        # More protein in unfolded state → less L state available → less effector bound
        assert jnp.all(e_free_large >= e_free_small)


# ---------------------------------------------------------------------------
# _compute_theta
# ---------------------------------------------------------------------------

class TestComputeTheta:

    def test_output_shape(self):
        T, G = 2, 4
        conc = jnp.array([0.0, 100.0, 1000.0])
        theta = _compute_theta(jnp.zeros(G), jnp.full(G, -9.0),
                               jnp.full((T, G), -4.0), jnp.full(G, -12.0),
                               conc, 650.0, 25.0)
        assert theta.shape == (T, 3, G)

    def test_values_in_0_1(self):
        G = 4
        conc = jnp.array([0.0, 50.0, 500.0, 5000.0])
        theta = _compute_theta(jnp.zeros(G), jnp.full(G, -9.0),
                               jnp.full((1, G), -4.0), jnp.full(G, -12.0),
                               conc, 650.0, 25.0)
        assert jnp.all(theta >= 0)
        assert jnp.all(theta <= 1)

    def test_all_finite(self):
        G = 4
        conc = jnp.linspace(0, 1000, 6)
        theta = _compute_theta(jnp.linspace(0.0, 4.0, G),
                               jnp.linspace(-12.0, -6.0, G),
                               jnp.full((2, G), -5.0), jnp.full(G, -12.0),
                               conc, 650.0, 25.0)
        assert jnp.all(jnp.isfinite(theta))

    def test_theta_monotone_decreasing_in_effector(self):
        """Higher effector → lower theta (effector competes with operator binding)."""
        G = 2
        conc = jnp.array([0.0, 100.0, 1000.0])
        theta = _compute_theta(jnp.full(G, 2.3), jnp.full(G, -9.0),
                               jnp.full((1, G), -4.0), jnp.full(G, -12.0),
                               conc, 650.0, 25.0)
        assert jnp.all(theta[:, 0, :] >= theta[:, 1, :])
        assert jnp.all(theta[:, 1, :] >= theta[:, 2, :])

    def test_zero_effector_matches_apo_formula(self):
        """At e_total=0, theta = K_op*[op] / (1 + K_op*[op] + K_HL + K_U)."""
        G = 2
        ln_K_op = jnp.array([2.3, 2.0])
        ln_K_HL = jnp.array([-9.0, -8.0])
        ln_K_U  = jnp.array([-12.0, -10.0])
        conc = jnp.array([0.0, 1.0])
        theta = _compute_theta(ln_K_op, ln_K_HL, jnp.full((1, G), -4.0), ln_K_U,
                               conc, 650.0, 25.0)
        K_op = np.exp(np.array([2.3, 2.0]))
        K_HL = np.exp(np.array([-9.0, -8.0]))
        K_U  = np.exp(np.array([-12.0, -10.0]))
        expected = K_op * 25.0 / (1.0 + K_op * 25.0 + K_HL + K_U)
        assert jnp.allclose(theta[0, 0, :], jnp.array(expected), atol=1e-5)

    def test_higher_K_op_gives_higher_theta(self):
        """Stronger operator binding → higher occupancy at fixed effector."""
        G = 2
        conc = jnp.array([0.0])
        ln_K_op = jnp.array([1.0, 3.0])   # G=0 weaker, G=1 stronger
        ln_K_HL = jnp.full(G, -9.0)
        theta = _compute_theta(ln_K_op, ln_K_HL, jnp.full((1, G), -4.0), jnp.full(G, -12.0),
                               conc, 650.0, 25.0)
        assert theta[0, 0, 1] > theta[0, 0, 0]

    def test_large_K_U_collapses_theta_to_zero(self):
        """Very large K_U (ln_K_U=20) means unfolded state dominates → theta << 0.01."""
        G = 2
        conc = jnp.array([0.0, 100.0])
        ln_K_U_large = jnp.full(G, 20.0)   # K_U ≈ 5e8, completely dominates Z
        theta = _compute_theta(jnp.full(G, 2.3), jnp.full(G, -9.0),
                               jnp.full((1, G), -4.0), ln_K_U_large,
                               conc, 650.0, 25.0)
        assert jnp.all(theta < 0.01)

    def test_small_K_U_matches_lac_dimer_formula(self):
        """When K_U → 0, the five-state model reduces to four-state lac_dimer."""
        G = 2
        conc = jnp.array([0.0])
        ln_K_op = jnp.array([2.3, 2.0])
        ln_K_HL = jnp.array([-9.0, -8.0])
        ln_K_U_tiny = jnp.full(G, -50.0)   # K_U ≈ 2e-22, negligible
        theta = _compute_theta(ln_K_op, ln_K_HL, jnp.full((1, G), -4.0), ln_K_U_tiny,
                               conc, 650.0, 25.0)
        K_op = np.exp(np.array([2.3, 2.0]))
        K_HL = np.exp(np.array([-9.0, -8.0]))
        # Four-state apo formula: theta = K_op*[op] / (1 + K_op*[op] + K_HL)
        expected_4state = K_op * 25.0 / (1.0 + K_op * 25.0 + K_HL)
        assert jnp.allclose(theta[0, 0, :], jnp.array(expected_4state), atol=1e-4)

    def test_K_U_lowers_theta_vs_no_K_U(self):
        """Adding K_U > 0 to the partition function reduces theta relative to K_U=0."""
        G = 2
        conc = jnp.array([0.0])
        ln_K_op = jnp.full(G, 2.3)
        ln_K_HL = jnp.full(G, -9.0)
        ln_K_E  = jnp.full((1, G), -4.0)
        theta_no_U = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, jnp.full(G, -50.0),
                                    conc, 650.0, 25.0)
        theta_with_U = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, jnp.full(G, 0.0),
                                      conc, 650.0, 25.0)
        assert jnp.all(theta_with_U < theta_no_U)


# ---------------------------------------------------------------------------
# _population_moments
# ---------------------------------------------------------------------------

class TestPopulationMoments:

    def test_output_shape(self, mock_data):
        T, C, G = 2, 3, 4
        theta = jnp.full((T, C, G), 0.7)
        mu, sigma = _population_moments(theta, mock_data)
        assert mu.shape == (T, C, 1)
        assert sigma.shape == (T, C, 1)

    def test_sigma_zero_when_uniform(self, mock_data):
        T, C, G = 2, 3, 4
        theta = jnp.full((T, C, G), 0.5)
        _, sigma = _population_moments(theta, mock_data)
        assert jnp.allclose(sigma, 0.0, atol=1e-6)

    def test_sigma_positive_when_varying(self, mock_data):
        T, C, G = 2, 3, 4
        theta = jnp.linspace(0.1, 0.9, T * C * G).reshape(T, C, G)
        _, sigma = _population_moments(theta, mock_data)
        assert jnp.all(sigma > 0)

    def test_mu_finite(self, mock_data):
        T, C, G = 2, 3, 4
        theta = jnp.full((T, C, G), 0.8)
        mu, _ = _population_moments(theta, mock_data)
        assert jnp.all(jnp.isfinite(mu))

    def test_data_arg_unused(self, mock_data):
        """_population_moments ignores data; two different data objects give same result."""
        T, C, G = 2, 3, 4
        theta = jnp.linspace(0.2, 0.8, T * C * G).reshape(T, C, G)
        mu1, sigma1 = _population_moments(theta, mock_data)
        mu2, sigma2 = _population_moments(theta, None)
        assert jnp.allclose(mu1, mu2)
        assert jnp.allclose(sigma1, sigma2)


# ---------------------------------------------------------------------------
# run_model
# ---------------------------------------------------------------------------

class TestRunModel:

    def test_scatter_theta_0_shape(self, theta_param, mock_data):
        result = run_model(theta_param, mock_data)
        T, C, G = 2, 3, 4
        assert result.shape == (T, C, G)

    def test_scatter_theta_1_shape(self, theta_param, mock_data):
        data = mock_data._replace(scatter_theta=1)
        result = run_model(theta_param, data)
        T, C, G = 2, 3, 4
        assert result.shape == (1, 1, 1, 1, T, C, G)

    def test_values_in_0_1(self, theta_param, mock_data):
        result = run_model(theta_param, mock_data)
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)

    def test_all_finite(self, theta_param, mock_data):
        result = run_model(theta_param, mock_data)
        assert jnp.all(jnp.isfinite(result))

    def test_zero_effector_matches_apo_formula(self, mock_data):
        """At zero effector: theta = K_op*[op] / (1 + K_op*[op] + K_HL + K_U)."""
        G = mock_data.num_genotype
        T = mock_data.num_titrant_name
        ln_K_op_val, ln_K_HL_val, ln_K_U_val = 2.3, -9.0, -12.0
        tp = ThetaParam(
            ln_K_op=jnp.full(G, ln_K_op_val),
            ln_K_HL=jnp.full(G, ln_K_HL_val),
            ln_K_E=jnp.full((T, G), -4.0),
            ln_K_U=jnp.full(G, ln_K_U_val),
            tf_total=650.0,
            op_total=25.0,
            mu=jnp.zeros((T, 3, 1)),
            sigma=jnp.ones((T, 3, 1)),
        )
        data = mock_data._replace(titrant_conc=jnp.array([0.0, 1.0, 2.0]))
        result = run_model(tp, data)
        K_op = np.exp(ln_K_op_val)
        K_HL = np.exp(ln_K_HL_val)
        K_U  = np.exp(ln_K_U_val)
        expected = K_op * 25.0 / (1.0 + K_op * 25.0 + K_HL + K_U)
        assert jnp.allclose(result[:, 0, :], expected, atol=1e-5)

    def test_uses_data_titrant_conc(self, theta_param):
        """run_model recomputes theta from data.titrant_conc, not stored moments."""
        MockDataSmall = namedtuple("MockDataSmall",
                                   ["titrant_conc", "geno_theta_idx", "scatter_theta"])
        G = 4
        data_low  = MockDataSmall(jnp.array([0.0, 10.0]),
                                  jnp.arange(G, dtype=jnp.int32), 0)
        data_high = MockDataSmall(jnp.array([500.0, 1000.0]),
                                  jnp.arange(G, dtype=jnp.int32), 0)
        res_low  = run_model(theta_param, data_low)
        res_high = run_model(theta_param, data_high)
        assert jnp.all(res_low >= res_high)

    def test_large_K_U_reduces_theta(self, mock_data):
        """ThetaParam with large ln_K_U gives lower theta than one with small ln_K_U."""
        G = mock_data.num_genotype
        T = mock_data.num_titrant_name
        C = mock_data.num_titrant_conc
        common = dict(
            ln_K_op=jnp.full(G, 2.3),
            ln_K_HL=jnp.full(G, -9.0),
            ln_K_E=jnp.full((T, G), -4.0),
            tf_total=650.0,
            op_total=25.0,
            mu=jnp.zeros((T, C, 1)),
            sigma=jnp.ones((T, C, 1)),
        )
        tp_small_U = ThetaParam(**common, ln_K_U=jnp.full(G, -50.0))
        tp_large_U = ThetaParam(**common, ln_K_U=jnp.full(G, 5.0))
        res_small = run_model(tp_small_U, mock_data)
        res_large = run_model(tp_large_U, mock_data)
        assert jnp.all(res_large < res_small)


# ---------------------------------------------------------------------------
# get_population_moments
# ---------------------------------------------------------------------------

def test_get_population_moments_passthrough(theta_param, mock_data):
    mu, sigma = get_population_moments(theta_param, mock_data)
    assert jnp.allclose(mu, theta_param.mu)
    assert jnp.allclose(sigma, theta_param.sigma)
