"""Tests for simulate/growth/growth_linkage.py — numpy growth model implementations."""
import pytest
import numpy as np

from tfscreen.simulate.growth.growth_linkage import (
    LinearGrowth,
    PowerGrowth,
    SaturationGrowth,
    MODEL_REGISTRY,
    get_growth_model,
)


class TestLinearGrowth:

    def test_formula_scalar(self):
        m = LinearGrowth()
        result = m.predict(theta=0.5, b=0.025, m=-0.01)
        np.testing.assert_allclose(result, 0.025 + (-0.01) * 0.5)

    def test_formula_with_activity(self):
        m = LinearGrowth()
        result = m.predict(theta=0.5, b=0.025, m=-0.01, activity=2.0)
        np.testing.assert_allclose(result, 0.025 + 2.0 * (-0.01) * 0.5)

    def test_zero_theta_returns_baseline(self):
        m = LinearGrowth()
        np.testing.assert_allclose(m.predict(theta=0.0, b=0.03, m=-0.01), 0.03)

    def test_unit_theta_unit_activity(self):
        m = LinearGrowth()
        b, slope = 0.025, -0.01
        np.testing.assert_allclose(m.predict(theta=1.0, b=b, m=slope), b + slope)

    def test_vectorized_theta(self):
        m = LinearGrowth()
        theta = np.array([0.0, 0.5, 1.0])
        result = m.predict(theta=theta, b=0.025, m=-0.01)
        np.testing.assert_allclose(result, 0.025 + (-0.01) * theta)

    def test_vectorized_activity(self):
        m = LinearGrowth()
        activity = np.array([0.5, 1.0, 2.0])
        result = m.predict(theta=0.5, b=0.0, m=1.0, activity=activity)
        np.testing.assert_allclose(result, 0.5 * activity)

    def test_default_activity_is_one(self):
        m = LinearGrowth()
        with_one = m.predict(theta=0.4, b=0.01, m=0.05, activity=1.0)
        default  = m.predict(theta=0.4, b=0.01, m=0.05)
        np.testing.assert_allclose(with_one, default)


class TestPowerGrowth:

    def test_formula_scalar(self):
        m = PowerGrowth()
        theta, b, a, n = 0.5, 0.001, 0.04, 2.0
        result = m.predict(theta=theta, b=b, a=a, n=n)
        np.testing.assert_allclose(result, b + a * theta**n)

    def test_formula_with_activity(self):
        m = PowerGrowth()
        theta, b, a, n, activity = 0.5, 0.001, 0.04, 2.0, 1.5
        result = m.predict(theta=theta, b=b, a=a, n=n, activity=activity)
        np.testing.assert_allclose(result, b + activity * a * theta**n)

    def test_n_one_equals_linear(self):
        """Power growth with n=1 must match LinearGrowth (m=a)."""
        theta = np.linspace(0, 1, 11)
        b, a = 0.025, -0.01
        power = PowerGrowth().predict(theta=theta, b=b, a=a, n=1.0)
        linear = LinearGrowth().predict(theta=theta, b=b, m=a)
        np.testing.assert_allclose(power, linear, rtol=1e-12)

    def test_zero_theta_returns_baseline(self):
        m = PowerGrowth()
        np.testing.assert_allclose(m.predict(theta=0.0, b=0.01, a=0.5, n=3.0), 0.01)

    def test_vectorized_theta(self):
        m = PowerGrowth()
        theta = np.array([0.0, 0.5, 1.0])
        result = m.predict(theta=theta, b=0.001, a=0.04, n=2.0)
        np.testing.assert_allclose(result, 0.001 + 0.04 * theta**2)

    def test_vectorized_activity(self):
        m = PowerGrowth()
        activity = np.array([0.5, 1.0, 2.0])
        result = m.predict(theta=0.5, b=0.0, a=1.0, n=2.0, activity=activity)
        np.testing.assert_allclose(result, activity * 0.5**2)

    def test_default_activity_is_one(self):
        m = PowerGrowth()
        with_one = m.predict(theta=0.7, b=0.01, a=0.05, n=3.0, activity=1.0)
        default  = m.predict(theta=0.7, b=0.01, a=0.05, n=3.0)
        np.testing.assert_allclose(with_one, default)


class TestSaturationGrowth:

    def test_formula_scalar(self):
        m = SaturationGrowth()
        theta, kmin, kmax = 1.0, 0.001, 0.04
        result = m.predict(theta=theta, kmin=kmin, kmax=kmax)
        expected = kmin + (kmax - kmin) * theta / (1.0 + theta)
        np.testing.assert_allclose(result, expected)

    def test_formula_with_activity(self):
        m = SaturationGrowth()
        theta, kmin, kmax, activity = 1.0, 0.001, 0.04, 0.5
        result = m.predict(theta=theta, kmin=kmin, kmax=kmax, activity=activity)
        expected = kmin + activity * (kmax - kmin) * theta / (1.0 + theta)
        np.testing.assert_allclose(result, expected)

    def test_zero_theta_returns_kmin(self):
        m = SaturationGrowth()
        np.testing.assert_allclose(m.predict(theta=0.0, kmin=0.005, kmax=0.04), 0.005)

    def test_large_theta_approaches_kmax(self):
        m = SaturationGrowth()
        result = m.predict(theta=1e6, kmin=0.001, kmax=0.04)
        np.testing.assert_allclose(result, 0.04, rtol=1e-4)

    def test_activity_zero_returns_kmin(self):
        m = SaturationGrowth()
        result = m.predict(theta=100.0, kmin=0.005, kmax=0.04, activity=0.0)
        np.testing.assert_allclose(result, 0.005)

    def test_monotone_increasing_in_theta_when_kmax_gt_kmin(self):
        m = SaturationGrowth()
        theta = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        result = m.predict(theta=theta, kmin=0.001, kmax=0.04)
        assert np.all(np.diff(result) > 0)

    def test_vectorized_theta(self):
        m = SaturationGrowth()
        theta = np.array([0.0, 0.5, 1.0, 2.0])
        kmin, kmax = 0.001, 0.04
        result = m.predict(theta=theta, kmin=kmin, kmax=kmax)
        expected = kmin + (kmax - kmin) * theta / (1.0 + theta)
        np.testing.assert_allclose(result, expected)

    def test_vectorized_activity(self):
        m = SaturationGrowth()
        activity = np.array([0.0, 0.5, 1.0])
        kmin, kmax, theta = 0.001, 0.04, 1.0
        result = m.predict(theta=theta, kmin=kmin, kmax=kmax, activity=activity)
        expected = kmin + activity * (kmax - kmin) * theta / (1.0 + theta)
        np.testing.assert_allclose(result, expected)

    def test_default_activity_is_one(self):
        m = SaturationGrowth()
        with_one = m.predict(theta=0.5, kmin=0.001, kmax=0.04, activity=1.0)
        default  = m.predict(theta=0.5, kmin=0.001, kmax=0.04)
        np.testing.assert_allclose(with_one, default)


class TestModelRegistry:

    def test_registry_has_three_keys(self):
        assert set(MODEL_REGISTRY.keys()) == {"linear", "power", "saturation"}

    def test_registry_values_are_classes(self):
        assert MODEL_REGISTRY["linear"] is LinearGrowth
        assert MODEL_REGISTRY["power"] is PowerGrowth
        assert MODEL_REGISTRY["saturation"] is SaturationGrowth


class TestGetGrowthModel:

    @pytest.mark.parametrize("name", list(MODEL_REGISTRY))
    def test_known_names_return_instance(self, name):
        m = get_growth_model(name)
        assert m is not None
        assert hasattr(m, "predict")

    def test_linear_returns_linear_growth(self):
        assert isinstance(get_growth_model("linear"), LinearGrowth)

    def test_power_returns_power_growth(self):
        assert isinstance(get_growth_model("power"), PowerGrowth)

    def test_saturation_returns_saturation_growth(self):
        assert isinstance(get_growth_model("saturation"), SaturationGrowth)

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown growth model"):
            get_growth_model("not_a_model")

    def test_error_message_contains_bad_name(self):
        with pytest.raises(ValueError, match="bad_name"):
            get_growth_model("bad_name")

    def test_error_message_lists_available_models(self):
        with pytest.raises(ValueError, match="Available models"):
            get_growth_model("bad_name")
