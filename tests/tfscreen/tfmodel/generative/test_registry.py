"""
Unit tests for tfscreen.tfmodel.generative.registry.

Verifies that model_registry has the expected top-level keys, that each
component category maps to a dict of name->module, and that every component
module exposes the standard interface attributes.
"""

import types
import pytest

from tfscreen.tfmodel.generative.registry import model_registry


# ---------------------------------------------------------------------------
# Expected structure
# ---------------------------------------------------------------------------

COMPONENT_CATEGORIES = {
    "condition_growth",
    "ln_cfu0",
    "dk_geno",
    "activity",
    "transformation",
    "theta_rescale",
    "theta",
    "theta_growth_noise",
    "theta_binding_noise",
    "growth_noise",
    "sample_offset",
    "growth_transition",
}

OBSERVE_KEYS = {"observe_binding", "observe_growth"}

# Attributes that non-rescale component modules should expose
STANDARD_COMPONENT_ATTRS = {"get_priors", "define_model", "guide", "get_guesses"}

# theta_rescale modules expose only a rescale callable
THETA_RESCALE_ATTR = "rescale"

# observe modules expose these
OBSERVE_ATTRS = {"observe", "guide"}

# Known minimum component counts per category
MIN_COUNTS = {
    "condition_growth": 3,
    "ln_cfu0": 2,
    "dk_geno": 2,
    "activity": 5,
    "transformation": 3,
    "theta_rescale": 2,
    "theta": 4,
    "theta_growth_noise": 3,
    "theta_binding_noise": 2,
    "growth_noise": 2,
    "sample_offset": 2,
    "growth_transition": 5,
}


# ---------------------------------------------------------------------------
# Top-level structure
# ---------------------------------------------------------------------------

class TestRegistryStructure:
    def test_all_component_categories_present(self):
        for key in COMPONENT_CATEGORIES:
            assert key in model_registry, f"Missing category: {key}"

    def test_all_observe_keys_present(self):
        for key in OBSERVE_KEYS:
            assert key in model_registry, f"Missing observe key: {key}"

    def test_no_unexpected_top_level_keys(self):
        expected = COMPONENT_CATEGORIES | OBSERVE_KEYS
        unexpected = set(model_registry.keys()) - expected
        assert not unexpected, f"Unexpected registry keys: {unexpected}"

    def test_component_categories_are_dicts(self):
        for key in COMPONENT_CATEGORIES:
            assert isinstance(model_registry[key], dict), (
                f"model_registry['{key}'] should be a dict"
            )

    def test_observe_values_are_modules(self):
        for key in OBSERVE_KEYS:
            assert isinstance(model_registry[key], types.ModuleType), (
                f"model_registry['{key}'] should be a module"
            )


# ---------------------------------------------------------------------------
# Component counts
# ---------------------------------------------------------------------------

class TestComponentCounts:
    @pytest.mark.parametrize("category,min_count", MIN_COUNTS.items())
    def test_minimum_component_count(self, category, min_count):
        count = len(model_registry[category])
        assert count >= min_count, (
            f"model_registry['{category}'] has {count} entries; expected >= {min_count}"
        )

    def test_all_component_names_are_strings(self):
        for cat, components in model_registry.items():
            if not isinstance(components, dict):
                continue
            for name in components:
                assert isinstance(name, str), (
                    f"Component name in '{cat}' is not a string: {name!r}"
                )


# ---------------------------------------------------------------------------
# Component module interfaces
# ---------------------------------------------------------------------------

class TestComponentInterfaces:
    @pytest.mark.parametrize("name,mod", [
        (name, mod)
        for cat, components in model_registry.items()
        if isinstance(components, dict) and cat != "theta_rescale"
        for name, mod in components.items()
    ])
    def test_standard_components_have_define_model_and_get_guesses(self, name, mod):
        for attr in STANDARD_COMPONENT_ATTRS:
            assert hasattr(mod, attr), (
                f"Component '{name}' ({mod.__name__}) is missing '{attr}'"
            )

    @pytest.mark.parametrize("name,mod", model_registry.get("theta_rescale", {}).items())
    def test_theta_rescale_has_rescale(self, name, mod):
        assert hasattr(mod, THETA_RESCALE_ATTR), (
            f"theta_rescale component '{name}' is missing '{THETA_RESCALE_ATTR}'"
        )
        assert callable(getattr(mod, THETA_RESCALE_ATTR)), (
            f"theta_rescale component '{name}'.rescale is not callable"
        )

    @pytest.mark.parametrize("name,mod", [
        (name, mod)
        for cat, components in model_registry.items()
        if isinstance(components, dict) and cat not in ("theta_rescale",)
        for name, mod in components.items()
    ])
    def test_get_priors_returns_object(self, name, mod):
        priors = mod.get_priors()
        assert priors is not None, f"Component '{name}'.get_priors() returned None"

    @pytest.mark.parametrize("name,mod", model_registry.get("transformation", {}).items())
    def test_transformation_has_update_thetas(self, name, mod):
        assert hasattr(mod, "update_thetas"), (
            f"transformation/{name} missing 'update_thetas'"
        )
        assert callable(mod.update_thetas)

    @pytest.mark.parametrize("key", list(OBSERVE_KEYS))
    def test_observe_modules_have_observe_and_guide(self, key):
        mod = model_registry[key]
        for attr in OBSERVE_ATTRS:
            assert hasattr(mod, attr), (
                f"observe module '{key}' is missing '{attr}'"
            )
            assert callable(getattr(mod, attr)), (
                f"observe module '{key}'.{attr} is not callable"
            )
