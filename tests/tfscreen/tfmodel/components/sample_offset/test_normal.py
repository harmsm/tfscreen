import pytest
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
import optax
from unittest.mock import MagicMock

from tfscreen.tfmodel.generative.components.sample_offset.normal import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_priors,
    get_guesses,
    get_extract_specs,
)


def _make_mock_data(R=2, T=3, CP=1, CS=2, TN=1, TC=4, G=5):
    """Build a minimal GrowthData-like mock with the fields normal.py needs."""
    data = MagicMock()
    data.num_replicate = R
    data.num_time = T
    data.num_condition_pre = CP
    data.num_condition_sel = CS
    data.num_titrant_name = TN
    data.num_titrant_conc = TC
    # t_pre and t_sel: shape (R, T, CP, CS, TN, TC, G)
    data.t_pre = jnp.full((R, T, CP, CS, TN, TC, G), 30.0)
    data.t_sel = jnp.full((R, T, CP, CS, TN, TC, G), 120.0)
    return data


class TestNormalSampleOffset:

    def test_get_hyperparameters_has_sigma_env_scale(self):
        hp = get_hyperparameters()
        assert "sigma_env_scale" in hp
        assert hp["sigma_env_scale"] > 0

    def test_get_priors_returns_model_priors(self):
        p = get_priors()
        assert isinstance(p, ModelPriors)
        assert p.sigma_env_scale > 0

    def test_get_guesses_returns_small_positive(self):
        guesses = get_guesses("sample_offset", None)
        assert "sample_offset_sigma_env" in guesses
        assert float(guesses["sample_offset_sigma_env"]) > 0

    def test_get_extract_specs_returns_list(self):
        assert isinstance(get_extract_specs(None), list)

    def test_define_model_output_shape(self):
        """define_model must return shape (R, T, CP, CS, TN, TC, 1)."""
        R, T, CP, CS, TN, TC = 2, 3, 1, 2, 1, 4
        data = _make_mock_data(R, T, CP, CS, TN, TC)
        priors = get_priors()

        captured = {}

        def model_fn():
            captured["result"] = define_model("sample_offset", data, priors)

        with handlers.seed(rng_seed=0):
            model_fn()

        assert captured["result"].shape == (R, T, CP, CS, TN, TC, 1)

    def test_define_model_sigma_env_positive(self):
        """sigma_env must always be positive (HalfNormal prior)."""
        data = _make_mock_data()
        priors = get_priors()

        sigma_vals = []
        for seed in range(20):
            with handlers.seed(rng_seed=seed):
                with handlers.trace() as tr:
                    define_model("sample_offset", data, priors)
            sigma_vals.append(float(tr["sample_offset_sigma_env"]["value"]))

        assert all(s > 0 for s in sigma_vals)

    def test_define_model_return_scales_with_t(self):
        """Returned value equals delta_k * (t_pre + t_sel)."""
        R, T, CP, CS, TN, TC, G = 1, 1, 1, 1, 1, 1, 3
        data = _make_mock_data(R, T, CP, CS, TN, TC, G)
        priors = ModelPriors(sigma_env_scale=1.0)

        captured = {}
        with handlers.seed(rng_seed=0):
            with handlers.trace() as tr:
                captured["result"] = define_model("sample_offset", data, priors)

        # Single tube: delta_k is a scalar-like array
        delta_k_val = float(jnp.ravel(tr["sample_offset_delta_k"]["value"])[0])
        t_total = 30.0 + 120.0  # t_pre + t_sel from mock data
        expected = delta_k_val * t_total
        assert np.isclose(float(jnp.ravel(captured["result"])[0]), expected, rtol=1e-5)

    def test_guide_registers_variational_params(self):
        """guide must create delta_k_loc, delta_k_scale, sigma_env params."""
        data = _make_mock_data()
        priors = get_priors()

        def model_fn():
            define_model("sample_offset", data, priors)

        def guide_fn():
            guide("sample_offset", data, priors)

        svi = SVI(model_fn, guide_fn, optax.adam(1e-3), Trace_ELBO())
        state = svi.init(jax.random.PRNGKey(2))
        params = svi.get_params(state)
        assert "sample_offset_delta_k_loc" in params
        assert "sample_offset_delta_k_scale" in params
        assert "sample_offset_sigma_env_loc" in params
        assert "sample_offset_sigma_env_scale" in params

    def test_guide_delta_k_loc_shape(self):
        """Variational loc for delta_k must have shape (num_tubes,)."""
        R, T, CP, CS, TN, TC = 2, 3, 1, 2, 1, 4
        data = _make_mock_data(R, T, CP, CS, TN, TC)
        priors = get_priors()
        num_tubes = R * T * CP * CS * TN * TC

        def model_fn():
            define_model("sample_offset", data, priors)

        def guide_fn():
            guide("sample_offset", data, priors)

        svi = SVI(model_fn, guide_fn, optax.adam(1e-3), Trace_ELBO())
        state = svi.init(jax.random.PRNGKey(3))
        params = svi.get_params(state)
        assert params["sample_offset_delta_k_loc"].shape == (num_tubes,)

    def test_guide_delta_k_scale_positive(self):
        """Variational scale for delta_k must be positive after init."""
        data = _make_mock_data()
        priors = get_priors()

        def model_fn():
            define_model("sample_offset", data, priors)

        def guide_fn():
            guide("sample_offset", data, priors)

        svi = SVI(model_fn, guide_fn, optax.adam(1e-3), Trace_ELBO())
        state = svi.init(jax.random.PRNGKey(4))
        params = svi.get_params(state)
        assert np.all(np.array(params["sample_offset_delta_k_scale"]) > 0)
