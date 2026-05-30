import pytest
import numpy as np

from tfscreen.simulate.growth.growth_linkage import (
    LinearGrowth,
    PowerGrowth,
    SaturationGrowth,
    MODEL_REGISTRY,
    get_growth_model,
)


THETA = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


class TestLinearGrowth:

    def test_formula(self):
        m = LinearGrowth()
        result = m.predict(THETA, b=0.025, m=-0.01)
        np.testing.assert_allclose(result, 0.025 + (-0.01) * THETA)

    def test_scalar_input(self):
        m = LinearGrowth()
        assert np.isclose(m.predict(0.5, b=0.0, m=1.0), 0.5)

    def test_zero_slope_is_constant(self):
        m = LinearGrowth()
        result = m.predict(THETA, b=0.025, m=0.0)
        np.testing.assert_allclose(result, 0.025)

    def test_theta_zero_returns_b(self):
        m = LinearGrowth()
        assert np.isclose(m.predict(0.0, b=0.03, m=-0.02), 0.03)

    def test_theta_one_returns_b_plus_m(self):
        m = LinearGrowth()
        assert np.isclose(m.predict(1.0, b=0.03, m=-0.02), 0.01)

    def test_activity_default_one_unchanged(self):
        m = LinearGrowth()
        np.testing.assert_allclose(
            m.predict(THETA, b=0.025, m=-0.01),
            m.predict(THETA, b=0.025, m=-0.01, activity=1.0),
        )

    def test_activity_scales_theta_term(self):
        m = LinearGrowth()
        result = m.predict(THETA, b=0.0, m=1.0, activity=2.0)
        np.testing.assert_allclose(result, 2.0 * THETA)

    def test_activity_zero_returns_b(self):
        m = LinearGrowth()
        result = m.predict(THETA, b=0.05, m=-0.01, activity=0.0)
        np.testing.assert_allclose(result, 0.05)

    def test_activity_vectorized(self):
        m = LinearGrowth()
        activity = np.array([0.5, 1.0, 2.0, 0.5, 1.0])
        result = m.predict(THETA, b=0.0, m=1.0, activity=activity)
        np.testing.assert_allclose(result, activity * THETA)


class TestPowerGrowth:

    def test_formula(self):
        m = PowerGrowth()
        result = m.predict(THETA, b=0.025, a=-0.01, n=2.0)
        np.testing.assert_allclose(result, 0.025 + (-0.01) * (THETA ** 2.0))

    def test_n_equals_one_matches_linear(self):
        power = PowerGrowth()
        linear = LinearGrowth()
        np.testing.assert_allclose(
            power.predict(THETA, b=0.025, a=-0.01, n=1.0),
            linear.predict(THETA, b=0.025, m=-0.01),
        )

    def test_theta_zero_returns_b(self):
        m = PowerGrowth()
        assert np.isclose(m.predict(0.0, b=0.03, a=-0.02, n=3.0), 0.03)

    def test_scalar_input(self):
        m = PowerGrowth()
        assert np.isclose(m.predict(2.0, b=0.0, a=1.0, n=3.0), 8.0)

    def test_activity_default_one_unchanged(self):
        m = PowerGrowth()
        np.testing.assert_allclose(
            m.predict(THETA, b=0.025, a=-0.01, n=2.0),
            m.predict(THETA, b=0.025, a=-0.01, n=2.0, activity=1.0),
        )

    def test_activity_scales_theta_term(self):
        m = PowerGrowth()
        result = m.predict(THETA, b=0.0, a=1.0, n=2.0, activity=3.0)
        np.testing.assert_allclose(result, 3.0 * THETA ** 2.0)

    def test_activity_zero_returns_b(self):
        m = PowerGrowth()
        result = m.predict(THETA, b=0.05, a=-0.01, n=2.0, activity=0.0)
        np.testing.assert_allclose(result, 0.05)


class TestSaturationGrowth:

    def test_formula(self):
        m = SaturationGrowth()
        result = m.predict(THETA, kmin=0.01, kmax=0.05)
        expected = 0.01 + (0.05 - 0.01) * THETA / (1.0 + THETA)
        np.testing.assert_allclose(result, expected)

    def test_theta_zero_returns_kmin(self):
        m = SaturationGrowth()
        assert np.isclose(m.predict(0.0, kmin=0.01, kmax=0.05), 0.01)

    def test_large_theta_approaches_kmax(self):
        m = SaturationGrowth()
        result = m.predict(1e8, kmin=0.01, kmax=0.05)
        np.testing.assert_allclose(result, 0.05, rtol=1e-5)

    def test_monotone_increasing_when_kmax_gt_kmin(self):
        m = SaturationGrowth()
        result = m.predict(THETA, kmin=0.01, kmax=0.05)
        assert np.all(np.diff(result) > 0)

    def test_scalar_input(self):
        m = SaturationGrowth()
        result = m.predict(1.0, kmin=0.0, kmax=1.0)
        np.testing.assert_allclose(result, 0.5)

    def test_activity_default_one_unchanged(self):
        m = SaturationGrowth()
        np.testing.assert_allclose(
            m.predict(THETA, kmin=0.01, kmax=0.05),
            m.predict(THETA, kmin=0.01, kmax=0.05, activity=1.0),
        )

    def test_activity_scales_theta_term(self):
        m = SaturationGrowth()
        result_half = m.predict(THETA, kmin=0.0, kmax=1.0, activity=0.5)
        result_full = m.predict(THETA, kmin=0.0, kmax=1.0, activity=1.0)
        np.testing.assert_allclose(result_half, 0.5 * result_full)

    def test_activity_zero_returns_kmin(self):
        m = SaturationGrowth()
        result = m.predict(THETA, kmin=0.03, kmax=0.10, activity=0.0)
        np.testing.assert_allclose(result, 0.03)


class TestGetTFModel:

    @pytest.mark.parametrize("name", list(MODEL_REGISTRY))
    def test_known_models_returned(self, name):
        m = get_growth_model(name)
        assert m is not None
        assert hasattr(m, "predict")

    def test_linear_returned(self):
        assert isinstance(get_growth_model("linear"), LinearGrowth)

    def test_power_returned(self):
        assert isinstance(get_growth_model("power"), PowerGrowth)

    def test_saturation_returned(self):
        assert isinstance(get_growth_model("saturation"), SaturationGrowth)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown growth model"):
            get_growth_model("not_a_real_model")
