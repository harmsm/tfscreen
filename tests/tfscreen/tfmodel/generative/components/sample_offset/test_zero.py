import pytest
import jax.numpy as jnp

from tfscreen.tfmodel.generative.components.sample_offset.zero import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_priors,
    get_guesses,
    get_extract_specs,
)
from tfscreen.tfmodel.generative.registry import model_registry


class TestSampleOffsetZero:

    def test_get_hyperparameters_empty(self):
        assert get_hyperparameters() == {}

    def test_get_priors_returns_model_priors(self):
        assert isinstance(get_priors(), ModelPriors)

    def test_get_guesses_empty(self):
        assert get_guesses("sample_offset", None) == {}

    def test_get_extract_specs_empty(self):
        assert get_extract_specs(None) == []

    def test_define_model_returns_zero(self):
        result = define_model("sample_offset", None, ModelPriors())
        assert float(result) == 0.0

    def test_guide_returns_zero(self):
        result = guide("sample_offset", None, ModelPriors())
        assert float(result) == 0.0


class TestSampleOffsetRegistry:

    def test_sample_offset_slot_in_registry(self):
        assert "sample_offset" in model_registry

    def test_zero_in_registry(self):
        assert "zero" in model_registry["sample_offset"]

    def test_normal_in_registry(self):
        assert "normal" in model_registry["sample_offset"]

    def test_zero_has_required_interface(self):
        mod = model_registry["sample_offset"]["zero"]
        for attr in ("define_model", "guide", "get_priors", "get_guesses",
                     "get_hyperparameters", "get_extract_specs"):
            assert hasattr(mod, attr), f"zero missing: {attr}"

    def test_normal_has_required_interface(self):
        mod = model_registry["sample_offset"]["normal"]
        for attr in ("define_model", "guide", "get_priors", "get_guesses",
                     "get_hyperparameters", "get_extract_specs"):
            assert hasattr(mod, attr), f"normal missing: {attr}"
