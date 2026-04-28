"""
Tests for growth_model/registry.py.

Verifies that:
  - All expected categories are present in model_registry.
  - Every component module under a dict-valued category exposes the three
    required functions: get_priors, define_model, guide.
  - get_priors() returns a non-None object for every component.
  - The flat observe entries expose an `observe` callable.
"""

import pytest
from tfscreen.analysis.hierarchical.growth_model.registry import model_registry

# Categories whose values are {name: module} dicts.
COMPONENT_CATEGORIES = [
    "condition_growth",
    "ln_cfu0",
    "dk_geno",
    "activity",
    "transformation",
    "theta",
    "theta_growth_noise",
    "theta_binding_noise",
    "growth_transition",
]

# Categories whose value is a single module (not a dict).
OBSERVE_CATEGORIES = ["observe_binding", "observe_growth"]

REQUIRED_INTERFACE = ["get_priors", "define_model", "guide"]


# ---------------------------------------------------------------------------
# Top-level presence
# ---------------------------------------------------------------------------

def test_all_component_categories_present():
    for cat in COMPONENT_CATEGORIES:
        assert cat in model_registry, f"Missing category: {cat}"


def test_all_observe_categories_present():
    for cat in OBSERVE_CATEGORIES:
        assert cat in model_registry, f"Missing observe entry: {cat}"


# ---------------------------------------------------------------------------
# Component interface (parametrised over category × entry)
# ---------------------------------------------------------------------------

def _all_component_entries():
    """Yield (category, name, module) for every registered component."""
    for cat in COMPONENT_CATEGORIES:
        entries = model_registry[cat]
        assert isinstance(entries, dict), f"{cat} should be a dict of components"
        for name, module in entries.items():
            yield cat, name, module


@pytest.mark.parametrize("cat,name,module", list(_all_component_entries()))
def test_component_has_required_interface(cat, name, module):
    for fn in REQUIRED_INTERFACE:
        assert hasattr(module, fn), (
            f"{cat}/{name} is missing required attribute '{fn}'"
        )
        assert callable(getattr(module, fn)), (
            f"{cat}/{name}.{fn} is not callable"
        )


@pytest.mark.parametrize("cat,name,module", list(_all_component_entries()))
def test_get_priors_returns_object(cat, name, module):
    priors = module.get_priors()
    assert priors is not None, f"{cat}/{name}.get_priors() returned None"


# ---------------------------------------------------------------------------
# Transformation components also need update_thetas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,module", list(model_registry["transformation"].items()))
def test_transformation_has_update_thetas(name, module):
    assert hasattr(module, "update_thetas"), (
        f"transformation/{name} missing 'update_thetas'"
    )
    assert callable(module.update_thetas)


# ---------------------------------------------------------------------------
# Observe modules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cat", OBSERVE_CATEGORIES)
def test_observe_module_has_observe(cat):
    module = model_registry[cat]
    assert hasattr(module, "observe"), f"{cat} missing 'observe' function"
    assert callable(module.observe)


# ---------------------------------------------------------------------------
# No unexpected categories (guard against leftover or mis-keyed entries)
# ---------------------------------------------------------------------------

def test_no_unexpected_top_level_keys():
    expected = set(COMPONENT_CATEGORIES) | set(OBSERVE_CATEGORIES)
    actual = set(model_registry.keys())
    extra = actual - expected
    assert not extra, f"Unexpected keys in model_registry: {extra}"
