import pytest
import jax.numpy as jnp

from tfscreen.tfmodel.components.growth_noise.zero import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_priors,
    get_guesses,
    get_extract_specs,
)


class TestZeroGrowthNoise:

    def test_define_model_returns_zero(self):
        priors = get_priors()
        result = define_model("growth_noise", None, priors)
        assert float(result) == 0.0

    def test_guide_returns_zero(self):
        priors = get_priors()
        result = guide("growth_noise", None, priors)
        assert float(result) == 0.0

    def test_get_hyperparameters_empty(self):
        assert get_hyperparameters() == {}

    def test_get_priors_returns_model_priors(self):
        p = get_priors()
        assert isinstance(p, ModelPriors)

    def test_get_guesses_empty(self):
        assert get_guesses("growth_noise", None) == {}

    def test_get_extract_specs_empty(self):
        assert get_extract_specs(None) == []

    def test_define_model_is_jax_array(self):
        priors = get_priors()
        result = define_model("growth_noise", None, priors)
        # Should be a JAX array or castable scalar — zero
        assert jnp.isclose(result, 0.0)
