import pytest
import numpy as np
from scipy.integrate import quad

from tfscreen.tfmodel.generative.components.growth_transition._baranyi import (
    compute_growth as _baranyi_compute_growth,
)
from tfscreen.tfmodel.generative.components.growth_transition.two_pop import (
    _compute_growth as _two_pop_compute_growth,
)
from tfscreen.simulate.growth.transition_linkage import (
    InstantTransition,
    MemoryTransition,
    BaranyiTransition,
    TwoPopTransition,
    TRANSITION_REGISTRY,
    get_transition_model,
)


class TestInstantTransition:

    def test_formula(self):
        m = InstantTransition()
        result = m.compute_kt(0.1, 0.5, 30.0, 100.0)
        np.testing.assert_allclose(result, 0.1 * 30.0 + 0.5 * 100.0)

    def test_vectorized(self):
        m = InstantTransition()
        k_pre = np.array([0.1, 0.2])
        k_sel = np.array([0.5, 0.3])
        t_pre = np.array([30.0, 40.0])
        t_sel = np.array([100.0, 80.0])
        result = m.compute_kt(k_pre, k_sel, t_pre, t_sel)
        np.testing.assert_allclose(result, k_pre * t_pre + k_sel * t_sel)

    def test_theta_ignored(self):
        m = InstantTransition()
        r1 = m.compute_kt(0.1, 0.5, 30.0, 100.0, theta=0.0)
        r2 = m.compute_kt(0.1, 0.5, 30.0, 100.0, theta=1.0)
        np.testing.assert_allclose(r1, r2)


class TestMemoryTransition:

    def test_lag_not_expired(self):
        """When tau > t_sel, bacteria grow at k_pre throughout selection."""
        m = MemoryTransition()
        # tau = 200 + 1/(0.5 + 1) = 200.667 > 100
        result = m.compute_kt(0.1, 0.5, 30.0, 100.0, theta=0.5,
                               tau0=200.0, k1=1.0, k2=1.0)
        np.testing.assert_allclose(result, 0.1 * 30.0 + 0.1 * 100.0)

    def test_lag_expired(self):
        """When tau < t_sel, transition happens partway through selection."""
        tau0, k1, k2, theta = 50.0, 1.0, 1.0, 0.5
        tau = tau0 + k1 / (theta + k2)   # ~50.667
        k_pre, k_sel = 0.1, 0.5
        m = MemoryTransition()
        result = m.compute_kt(k_pre, k_sel, 30.0, 100.0, theta=theta,
                               tau0=tau0, k1=k1, k2=k2)
        expected = k_pre * 30.0 + k_pre * tau + k_sel * (100.0 - tau)
        np.testing.assert_allclose(result, expected)

    def test_tau_equals_t_sel(self):
        """tau == t_sel: both branches converge to the same value."""
        k_pre, k_sel, t_pre, t_sel = 0.1, 0.5, 30.0, 100.0
        theta, k1, k2 = 0.5, 1.0, 1.0
        tau0 = t_sel - k1 / (theta + k2)   # tau will equal exactly t_sel
        tau = tau0 + k1 / (theta + k2)
        expected = k_pre * t_pre + k_pre * tau   # k_sel * 0 = 0
        m = MemoryTransition()
        result = m.compute_kt(k_pre, k_sel, t_pre, t_sel, theta=theta,
                               tau0=tau0, k1=k1, k2=k2)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_higher_theta_shorter_tau_more_growth(self):
        """Higher theta (k1>0, k2>0) → shorter tau → more time at k_sel (>k_pre)."""
        m = MemoryTransition()
        kt_lo = m.compute_kt(0.1, 0.5, 30.0, 200.0, theta=0.1,
                              tau0=50.0, k1=10.0, k2=1.0)
        kt_hi = m.compute_kt(0.1, 0.5, 30.0, 200.0, theta=0.9,
                              tau0=50.0, k1=10.0, k2=1.0)
        assert kt_hi > kt_lo

    def test_memory_le_instant_when_k_sel_gt_k_pre(self):
        """Lag delays the switch to faster k_sel so total growth is lower."""
        m = MemoryTransition()
        instant = InstantTransition()
        kt_inst = instant.compute_kt(0.1, 0.5, 30.0, 100.0)
        kt_mem  = m.compute_kt(0.1, 0.5, 30.0, 100.0, theta=0.5,
                                tau0=50.0, k1=1.0, k2=1.0)
        assert kt_mem <= kt_inst

    def test_vectorized(self):
        m = MemoryTransition()
        theta = np.array([0.0, 1.0])
        result = m.compute_kt(0.1, 0.5, 30.0, 200.0, theta=theta,
                               tau0=50.0, k1=10.0, k2=1.0)
        assert result.shape == (2,)
        assert result[1] > result[0]   # higher theta → shorter lag → more growth


class TestBaranyiTransition:

    def test_formula_matches_numerical_integral(self):
        """Validate closed-form against numerical integration of the sigmoid."""
        k_pre, k_sel = 0.1, 0.5
        t_pre, t_sel = 30.0, 100.0
        tau_lag, k_sharp = 50.0, 0.2

        def integrand(t):
            return 1.0 / (1.0 + np.exp(-k_sharp * (t - tau_lag)))

        integral, _ = quad(integrand, 0, t_sel)
        expected = k_pre * t_pre + k_pre * t_sel + (k_sel - k_pre) * integral

        m = BaranyiTransition()
        result = m.compute_kt(k_pre, k_sel, t_pre, t_sel,
                               tau_lag=tau_lag, k_sharp=k_sharp)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_matches_inference_formula_exactly(self):
        """Verify bit-for-bit equivalence with _baranyi.compute_growth formula."""
        k_pre, k_sel = 0.1, 0.5
        t_pre, t_sel = 30.0, 100.0
        tau, k = 50.0, 0.2

        term1 = np.logaddexp(0.0, k * (t_sel - tau))
        term0 = np.logaddexp(0.0, -k * tau)
        integrated_sigmoid = (term1 - term0) / k
        expected = k_pre * t_pre + k_pre * t_sel + (k_sel - k_pre) * integrated_sigmoid

        m = BaranyiTransition()
        result = m.compute_kt(k_pre, k_sel, t_pre, t_sel,
                               tau_lag=tau, k_sharp=k)
        np.testing.assert_allclose(result, expected)

    def test_large_k_sharp_zero_tau_approaches_instant(self):
        """Infinitely sharp transition at tau=0 equals instant switching."""
        k_pre, k_sel = 0.1, 0.5
        t_pre, t_sel = 30.0, 100.0

        m = BaranyiTransition()
        result = m.compute_kt(k_pre, k_sel, t_pre, t_sel,
                               tau_lag=0.0, k_sharp=1e6)
        expected = k_pre * t_pre + k_sel * t_sel
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_very_large_tau_approaches_all_pre(self):
        """Transition far in the future → essentially no transition."""
        k_pre, k_sel = 0.1, 0.5
        t_pre, t_sel = 30.0, 100.0

        m = BaranyiTransition()
        result = m.compute_kt(k_pre, k_sel, t_pre, t_sel,
                               tau_lag=1e6, k_sharp=1.0)
        expected = k_pre * t_pre + k_pre * t_sel
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_theta_ignored(self):
        m = BaranyiTransition()
        r1 = m.compute_kt(0.1, 0.5, 30.0, 100.0, theta=0.0,
                           tau_lag=50.0, k_sharp=0.5)
        r2 = m.compute_kt(0.1, 0.5, 30.0, 100.0, theta=1.0,
                           tau_lag=50.0, k_sharp=0.5)
        np.testing.assert_allclose(r1, r2)

    def test_vectorized(self):
        m = BaranyiTransition()
        k_pre = np.array([0.1, 0.2])
        k_sel = np.array([0.5, 0.3])
        t_pre = np.array([30.0, 40.0])
        t_sel = np.array([100.0, 80.0])
        result = m.compute_kt(k_pre, k_sel, t_pre, t_sel,
                               tau_lag=50.0, k_sharp=0.2)
        assert result.shape == (2,)


class TestTwoPopTransition:

    def test_formula_matches_inference(self):
        """Verify formula matches two_pop._compute_growth exactly."""
        k_pre, k_sel = 0.1, 0.05   # k_pre > k_sel → D > 0
        t_pre, t_sel = 30.0, 100.0
        k_trans = 0.001

        # Direct formula from two_pop.py (translated to numpy)
        D = k_pre - k_sel - k_trans
        a = k_sel * t_sel
        b = (k_pre - k_trans) * t_sel
        m_val = max(a, b)
        scaled_num = k_trans * np.exp(a - m_val) + D * np.exp(b - m_val)
        expected = k_pre * t_pre + m_val + np.log(scaled_num) - np.log(D)

        m = TwoPopTransition()
        result = m.compute_kt(k_pre, k_sel, t_pre, t_sel, k_trans=k_trans)
        np.testing.assert_allclose(result, expected)

    def test_fallback_when_d_le_zero(self):
        """When D <= 0 (k_sel >= k_pre - k_trans), fall back to k_pre growth."""
        k_pre, k_sel = 0.1, 0.2    # k_sel > k_pre → D < 0
        k_trans = 0.001
        t_pre, t_sel = 30.0, 100.0

        m = TwoPopTransition()
        result = m.compute_kt(k_pre, k_sel, t_pre, t_sel, k_trans=k_trans)
        expected = k_pre * t_pre + k_pre * t_sel
        np.testing.assert_allclose(result, expected)

    def test_small_k_trans_approaches_pre_growth(self):
        """k_trans → 0 limit: no transition, bacteria grow at k_pre."""
        k_pre, k_sel = 0.1, 0.05
        t_pre, t_sel = 30.0, 100.0

        m = TwoPopTransition()
        result = m.compute_kt(k_pre, k_sel, t_pre, t_sel, k_trans=1e-10)
        expected = k_pre * t_pre + k_pre * t_sel
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_theta_ignored(self):
        m = TwoPopTransition()
        r1 = m.compute_kt(0.1, 0.05, 30.0, 100.0, theta=0.0, k_trans=0.001)
        r2 = m.compute_kt(0.1, 0.05, 30.0, 100.0, theta=1.0, k_trans=0.001)
        np.testing.assert_allclose(r1, r2)

    def test_vectorized(self):
        m = TwoPopTransition()
        k_pre = np.array([0.1, 0.15])
        k_sel = np.array([0.05, 0.06])
        t_pre = np.array([30.0, 30.0])
        t_sel = np.array([100.0, 100.0])
        result = m.compute_kt(k_pre, k_sel, t_pre, t_sel, k_trans=0.001)
        assert result.shape == (2,)


class TestGetTransitionModel:

    @pytest.mark.parametrize("name", list(TRANSITION_REGISTRY))
    def test_known_models_returned(self, name):
        m = get_transition_model(name)
        assert m is not None
        assert hasattr(m, "compute_kt")

    def test_instant_returned(self):
        assert isinstance(get_transition_model("instant"), InstantTransition)

    def test_memory_returned(self):
        assert isinstance(get_transition_model("memory"), MemoryTransition)

    def test_baranyi_returned(self):
        assert isinstance(get_transition_model("baranyi"), BaranyiTransition)

    def test_two_pop_returned(self):
        assert isinstance(get_transition_model("two_pop"), TwoPopTransition)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown transition model"):
            get_transition_model("not_a_real_model")


# ---------------------------------------------------------------------------
# Cross-validation: simulate wrappers must agree with tfmodel implementations
# ---------------------------------------------------------------------------

_SCALAR_CASES = [
    dict(k_pre=0.1,  k_sel=0.05,  t_pre=30.0,  t_sel=100.0),
    dict(k_pre=0.05, k_sel=0.1,   t_pre=10.0,  t_sel=200.0),   # k_sel > k_pre
    dict(k_pre=0.2,  k_sel=0.001, t_pre=0.0,   t_sel=60.0),
]

_VECTOR_CASES = [
    dict(k_pre=np.array([0.1, 0.2]),
         k_sel=np.array([0.05, 0.06]),
         t_pre=np.array([30.0, 40.0]),
         t_sel=np.array([100.0, 80.0])),
]


class TestBaranyiAgreementWithTFModel:
    """BaranyiTransition.compute_kt must return the same values as _baranyi.compute_growth."""

    @pytest.mark.parametrize("case", _SCALAR_CASES)
    def test_scalar_agreement(self, case):
        tau_lag, k_sharp = 50.0, 0.2
        sim = BaranyiTransition().compute_kt(
            case["k_pre"], case["k_sel"], case["t_pre"], case["t_sel"],
            tau_lag=tau_lag, k_sharp=k_sharp,
        )
        ref = float(_baranyi_compute_growth(
            case["k_pre"], case["k_sel"], case["t_pre"], case["t_sel"],
            tau=tau_lag, k=k_sharp,
        ))
        np.testing.assert_allclose(sim, ref, rtol=1e-6)

    @pytest.mark.parametrize("case", _VECTOR_CASES)
    def test_vector_agreement(self, case):
        tau_lag, k_sharp = 50.0, 0.2
        sim = BaranyiTransition().compute_kt(
            case["k_pre"], case["k_sel"], case["t_pre"], case["t_sel"],
            tau_lag=tau_lag, k_sharp=k_sharp,
        )
        ref = np.array(_baranyi_compute_growth(
            case["k_pre"], case["k_sel"], case["t_pre"], case["t_sel"],
            tau=tau_lag, k=k_sharp,
        ))
        np.testing.assert_allclose(sim, ref, rtol=1e-6)

    def test_returns_numpy_array(self):
        result = BaranyiTransition().compute_kt(0.1, 0.05, 30.0, 100.0,
                                                tau_lag=50.0, k_sharp=0.2)
        assert isinstance(result, np.ndarray)


class TestTwoPopAgreementWithTFModel:
    """TwoPopTransition.compute_kt must return the same values as two_pop._compute_growth."""

    @pytest.mark.parametrize("case", _SCALAR_CASES)
    def test_scalar_agreement(self, case):
        k_trans = 0.001
        sim = TwoPopTransition().compute_kt(
            case["k_pre"], case["k_sel"], case["t_pre"], case["t_sel"],
            k_trans=k_trans,
        )
        ref = float(_two_pop_compute_growth(
            case["k_pre"], case["k_sel"], case["t_pre"], case["t_sel"],
            k_trans=k_trans,
        ))
        np.testing.assert_allclose(sim, ref, rtol=1e-6)

    @pytest.mark.parametrize("case", _VECTOR_CASES)
    def test_vector_agreement(self, case):
        k_trans = 0.001
        sim = TwoPopTransition().compute_kt(
            case["k_pre"], case["k_sel"], case["t_pre"], case["t_sel"],
            k_trans=k_trans,
        )
        ref = np.array(_two_pop_compute_growth(
            case["k_pre"], case["k_sel"], case["t_pre"], case["t_sel"],
            k_trans=k_trans,
        ))
        np.testing.assert_allclose(sim, ref, rtol=1e-6)

    def test_returns_numpy_array(self):
        result = TwoPopTransition().compute_kt(0.1, 0.05, 30.0, 100.0, k_trans=0.001)
        assert isinstance(result, np.ndarray)
