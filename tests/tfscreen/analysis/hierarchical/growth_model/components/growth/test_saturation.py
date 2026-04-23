import pytest
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

from tfscreen.analysis.hierarchical.growth_model.components.growth.saturation import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    SaturationParams,
)

MockGrowthData = namedtuple("MockGrowthData", [
    "num_condition_rep",
    "num_replicate",
    "map_condition_pre",
    "map_condition_sel",
])

@pytest.fixture
def mock_data():
    num_condition_rep = 3
    num_replicate = 2
    map_condition_pre = jnp.array([0, 2, 2, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([1, 0, 1, 2], dtype=jnp.int32)
    return MockGrowthData(
        num_condition_rep=num_condition_rep,
        num_replicate=num_replicate,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel,
    )


def test_get_hyperparameters():
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "min_loc" in params
    assert "max_loc" in params
    assert params["min_loc"] == 0.025


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.min_loc == 0.025
    assert not hasattr(priors, "pinned")


def test_get_guesses(mock_data):
    name = "test_growth"
    guesses = get_guesses(name, mock_data)

    assert isinstance(guesses, dict)
    expected_shape = (mock_data.num_condition_rep,)
    for param in ("min", "max"):
        assert f"{name}_{param}_locs" in guesses
        assert f"{name}_{param}_scales" in guesses
        assert guesses[f"{name}_{param}_locs"].shape == expected_shape


def test_define_model_structure_and_shapes(mock_data):
    name = "test_growth"
    priors = get_priors()

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors
        )
        params = define_model(name=name, data=mock_data, priors=priors)

    for param in ("min", "max"):
        assert f"{name}_{param}" in model_trace
        assert model_trace[f"{name}_{param}"]["type"] == "sample"
        assert model_trace[f"{name}_{param}"]["value"].shape == (mock_data.num_condition_rep,)

    assert params.min_pre.shape == mock_data.map_condition_pre.shape
    assert params.max_pre.shape == mock_data.map_condition_pre.shape
    assert params.min_sel.shape == mock_data.map_condition_sel.shape
    assert params.max_sel.shape == mock_data.map_condition_sel.shape


def test_define_model_mapping(mock_data):
    """Substituted min/max values propagate correctly through condition mapping."""
    name = "g"
    priors = get_priors()

    subs = {
        f"{name}_min": jnp.array([1.0, 2.0, 3.0]),
        f"{name}_max": jnp.array([4.0, 5.0, 6.0]),
    }
    substituted = substitute(define_model, data=subs)
    params = substituted(name=name, data=mock_data, priors=priors)

    expected_min_pre = jnp.array([1.0, 2.0, 3.0])[mock_data.map_condition_pre]
    assert jnp.allclose(params.min_pre, expected_min_pre)


def test_guide_logic_and_shapes(mock_data):
    name = "test_growth_guide"
    priors = get_priors()

    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name, data=mock_data, priors=priors
        )
        params = guide(name=name, data=mock_data, priors=priors)

    for param in ("min", "max"):
        assert f"{name}_{param}_locs" in guide_trace
        assert f"{name}_{param}_scales" in guide_trace
        assert f"{name}_{param}" in guide_trace

    assert params.min_pre.shape == mock_data.map_condition_pre.shape


def test_model_and_guide_have_compatible_sample_sites(mock_data):
    name = "compat"
    priors = get_priors()

    with seed(rng_seed=0):
        m_tr = trace(define_model).get_trace(name=name, data=mock_data, priors=priors)
    with seed(rng_seed=0):
        g_tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    m_samples = {k for k, v in m_tr.items() if v["type"] == "sample"}
    g_samples = {k for k, v in g_tr.items() if v["type"] == "sample"}
    assert m_samples == g_samples
