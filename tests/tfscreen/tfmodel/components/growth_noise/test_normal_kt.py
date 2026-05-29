import pytest
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
import jax

from tfscreen.tfmodel.components.growth_noise.normal_kt import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_priors,
    get_guesses,
    get_extract_specs,
)
from tfscreen.tfmodel.registry import model_registry


class TestNormalKtGrowthNoise:

    def test_get_hyperparameters_has_sigma_k_scale(self):
        hp = get_hyperparameters()
        assert "sigma_k_scale" in hp
        assert hp["sigma_k_scale"] > 0

    def test_get_priors_returns_model_priors(self):
        p = get_priors()
        assert isinstance(p, ModelPriors)
        assert p.sigma_k_scale > 0

    def test_get_guesses_returns_small_positive(self):
        guesses = get_guesses("growth_noise", None)
        assert "growth_noise_sigma_k" in guesses
        val = float(guesses["growth_noise_sigma_k"])
        assert val > 0
        assert val < 0.1  # initialized near zero

    def test_get_extract_specs_returns_list(self):
        specs = get_extract_specs(None)
        assert isinstance(specs, list)
        assert len(specs) == 1
        assert "sigma_k" in specs[0]["params_to_get"]

    def test_define_model_samples_positive_sigma_k(self):
        """define_model should sample sigma_k > 0 under its own prior."""
        priors = get_priors()

        def model_fn():
            return define_model("growth_noise", None, priors)

        predictive = Predictive(model_fn, num_samples=50)
        rng = jax.random.PRNGKey(0)
        samples = predictive(rng)
        sigma_k_vals = np.array(samples["growth_noise_sigma_k"])
        assert np.all(sigma_k_vals > 0), "sigma_k must always be positive"

    def test_define_model_sigma_k_is_scalar(self):
        """define_model should return a scalar (shape ())."""
        priors = get_priors()

        def model_fn():
            return define_model("growth_noise", None, priors)

        predictive = Predictive(model_fn, num_samples=1)
        rng = jax.random.PRNGKey(1)
        samples = predictive(rng)
        assert samples["growth_noise_sigma_k"].shape == (1,)  # one sample, scalar site

    def test_guide_registers_sigma_k_param(self):
        """guide should create variational params for sigma_k."""
        priors = get_priors()

        def model_fn():
            define_model("growth_noise", None, priors)

        def guide_fn():
            guide("growth_noise", None, priors)

        # Run one SVI step to trigger param creation
        from numpyro.infer import SVI, Trace_ELBO
        import optax
        svi = SVI(model_fn, guide_fn, optax.adam(1e-3), Trace_ELBO())
        rng = jax.random.PRNGKey(2)
        state = svi.init(rng)
        params = svi.get_params(state)
        assert "growth_noise_sigma_k_loc" in params
        assert "growth_noise_sigma_k_scale" in params

    def test_guide_sigma_k_scale_is_positive(self):
        """The variational scale param must be positive after init."""
        priors = get_priors()

        def model_fn():
            define_model("growth_noise", None, priors)

        def guide_fn():
            guide("growth_noise", None, priors)

        from numpyro.infer import SVI, Trace_ELBO
        import optax
        svi = SVI(model_fn, guide_fn, optax.adam(1e-3), Trace_ELBO())
        rng = jax.random.PRNGKey(3)
        state = svi.init(rng)
        params = svi.get_params(state)
        assert float(params["growth_noise_sigma_k_scale"]) > 0


class TestGrowthNoiseRegistry:

    def test_growth_noise_slot_in_registry(self):
        assert "growth_noise" in model_registry

    def test_zero_in_registry(self):
        assert "zero" in model_registry["growth_noise"]

    def test_normal_kt_in_registry(self):
        assert "normal_kt" in model_registry["growth_noise"]

    def test_zero_has_required_interface(self):
        mod = model_registry["growth_noise"]["zero"]
        for attr in ("define_model", "guide", "get_priors", "get_guesses",
                     "get_hyperparameters", "get_extract_specs"):
            assert hasattr(mod, attr), f"zero missing: {attr}"

    def test_normal_kt_has_required_interface(self):
        mod = model_registry["growth_noise"]["normal_kt"]
        for attr in ("define_model", "guide", "get_priors", "get_guesses",
                     "get_hyperparameters", "get_extract_specs"):
            assert hasattr(mod, attr), f"normal_kt missing: {attr}"
