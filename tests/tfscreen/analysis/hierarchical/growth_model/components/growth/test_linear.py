import pytest
import numpy as np
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

from tfscreen.analysis.hierarchical.growth_model.components.growth.linear import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    LinearParams,
)

# --- Mock Data Fixtures ---

MockGrowthData = namedtuple("MockGrowthData", [
    "num_condition_rep",
    "num_replicate",
    "map_condition_pre",
    "map_condition_sel",
    "ln_cfu",
    "t_sel",
    "good_mask",
])


def _make_ln_cfu(slopes, num_rep=2, num_cond_pre=4, num_tname=1, num_tconc=3, num_geno=3):
    num_cond_sel = len(slopes)
    num_time = 2
    shape = (num_rep, num_time, num_cond_pre, num_cond_sel, num_tname, num_tconc, num_geno)
    arr = np.full(shape, 7.0)
    for cs, slope in enumerate(slopes):
        arr[:, 1, :, cs, :, :, :] = 7.0 + slope
    return jnp.array(arr)


def _make_t_sel(num_rep=2, num_cond_pre=4, num_cond_sel=4, num_tname=1, num_tconc=3, num_geno=3):
    num_time = 2
    shape = (num_rep, num_time, num_cond_pre, num_cond_sel, num_tname, num_tconc, num_geno)
    arr = np.zeros(shape)
    arr[:, 1, ...] = 1.0
    return jnp.array(arr)


@pytest.fixture
def mock_data():
    num_condition_rep = 3
    num_replicate = 2
    map_condition_pre = jnp.array([0, 2, 2, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([1, 0, 1, 2], dtype=jnp.int32)
    ln_cfu = _make_ln_cfu([0.0, 0.0, 0.0, 0.0])
    t_sel  = _make_t_sel()
    good_mask = jnp.ones(ln_cfu.shape, dtype=bool)

    return MockGrowthData(
        num_condition_rep=num_condition_rep,
        num_replicate=num_replicate,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel,
        ln_cfu=ln_cfu,
        t_sel=t_sel,
        good_mask=good_mask,
    )


@pytest.fixture
def mock_data_empirical():
    """
    Two condition_reps with different baseline growth rates:
      cond_rep 0 (cond_sel index 0): slope = 0.030
      cond_rep 1 (cond_sel index 1): slope = 0.020
    """
    num_condition_rep = 2
    num_replicate = 1
    num_cond_pre = 2
    num_cond_sel = 2
    num_tname = 1
    num_tconc = 1
    num_geno = 1
    num_time = 2

    shape = (num_replicate, num_time, num_cond_pre, num_cond_sel, num_tname, num_tconc, num_geno)
    ln_cfu_arr = np.full(shape, 7.0)
    ln_cfu_arr[:, 1, :, 0, :, :, :] = 7.03
    ln_cfu_arr[:, 1, :, 1, :, :, :] = 7.02

    t_arr = np.zeros(shape)
    t_arr[:, 1, ...] = 1.0

    map_condition_pre = jnp.array([0, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([0, 1], dtype=jnp.int32)

    return MockGrowthData(
        num_condition_rep=num_condition_rep,
        num_replicate=num_replicate,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel,
        ln_cfu=jnp.array(ln_cfu_arr),
        t_sel=jnp.array(t_arr),
        good_mask=jnp.ones(shape, dtype=bool),
    )


# --- Test Cases ---

def test_get_hyperparameters():
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "k_loc" in params
    assert "k_scale" in params
    assert "m_loc" in params
    assert "m_scale" in params
    assert params["k_loc"] == 0.025


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.k_loc == 0.025
    assert priors.m_loc == 0.0
    assert not hasattr(priors, "pinned")


def test_get_guesses_keys_and_shapes(mock_data):
    name = "test_growth"
    guesses = get_guesses(name, mock_data)

    assert isinstance(guesses, dict)
    assert f"{name}_k_locs"   in guesses
    assert f"{name}_k_scales" in guesses
    assert f"{name}_m_locs"   in guesses
    assert f"{name}_m_scales" in guesses

    expected_shape = (mock_data.num_condition_rep,)
    assert guesses[f"{name}_k_locs"].shape == expected_shape
    assert guesses[f"{name}_m_locs"].shape == expected_shape


def test_get_guesses_constant_ln_cfu_gives_zero_k(mock_data):
    """With constant ln_cfu (slope=0), k_locs should be near zero."""
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    assert jnp.allclose(guesses[f"{name}_k_locs"], 0.0, atol=1e-6)


def test_get_guesses_m_locs_always_zero(mock_data):
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    assert jnp.all(guesses[f"{name}_m_locs"] == 0.0)


def test_get_guesses_empirical_k_locs(mock_data_empirical):
    """k_locs should equal per-condition OLS slopes."""
    name = "eg"
    guesses = get_guesses(name, mock_data_empirical)
    k_locs = np.array(guesses[f"{name}_k_locs"])
    assert abs(k_locs[0] - 0.030) < 1e-6
    assert abs(k_locs[1] - 0.020) < 1e-6


def test_get_guesses_masked_observations(mock_data_empirical):
    """Fully masked cond_sel falls back to global median, not NaN."""
    name = "eg"
    mask_arr = np.array(mock_data_empirical.good_mask)
    mask_arr[:, :, :, 0, :, :, :] = False
    masked_data = mock_data_empirical._replace(good_mask=jnp.array(mask_arr))

    guesses = get_guesses(name, masked_data)
    k_locs = np.array(guesses[f"{name}_k_locs"])
    assert np.all(np.isfinite(k_locs))


def test_get_guesses_non7d_fallback():
    """Non-7D tensors trigger hard-coded fallback without error."""
    name = "fg"
    MockSimple = namedtuple("MockSimple", [
        "num_condition_rep", "num_replicate",
        "map_condition_pre", "map_condition_sel",
        "ln_cfu", "t_sel", "good_mask",
    ])
    data = MockSimple(
        num_condition_rep=3,
        num_replicate=2,
        map_condition_pre=jnp.array([0, 1, 2]),
        map_condition_sel=jnp.array([0, 1, 2]),
        ln_cfu=jnp.zeros((3,)),
        t_sel=jnp.zeros((3,)),
        good_mask=jnp.ones((3,), dtype=bool),
    )
    guesses = get_guesses(name, data)
    assert guesses[f"{name}_k_locs"].shape == (3,)
    assert jnp.all(guesses[f"{name}_m_locs"] == 0.0)


# --- define_model tests ---

def test_define_model_structure_and_shapes(mock_data):
    """Verifies output shapes and sample site names."""
    name = "test_growth"
    priors = get_priors()

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors
        )

    assert f"{name}_k" in model_trace
    assert f"{name}_m" in model_trace
    assert model_trace[f"{name}_k"]["type"] == "sample"
    assert model_trace[f"{name}_m"]["type"] == "sample"

    k_per_condition = model_trace[f"{name}_k"]["value"]
    m_per_condition = model_trace[f"{name}_m"]["value"]
    assert k_per_condition.shape == (mock_data.num_condition_rep,)
    assert m_per_condition.shape == (mock_data.num_condition_rep,)

    with seed(rng_seed=0):
        params = define_model(name=name, data=mock_data, priors=priors)

    assert params.k_pre.shape == mock_data.map_condition_pre.shape
    assert params.m_pre.shape == mock_data.map_condition_pre.shape
    assert params.k_sel.shape == mock_data.map_condition_sel.shape
    assert params.m_sel.shape == mock_data.map_condition_sel.shape


def test_define_model_calculation_logic(mock_data):
    """Substituted k values propagate correctly through the condition mapping."""
    name = "test_growth"
    priors = get_priors()

    subs = {
        f"{name}_k": jnp.array([10.0, 12.0, 8.0]),
        f"{name}_m": jnp.zeros(mock_data.num_condition_rep),
    }
    substituted = substitute(define_model, data=subs)
    params = substituted(name=name, data=mock_data, priors=priors)

    expected_k_pre = jnp.array([10.0, 12.0, 8.0])[mock_data.map_condition_pre]
    assert jnp.allclose(params.k_pre, expected_k_pre)
    expected_k_sel = jnp.array([10.0, 12.0, 8.0])[mock_data.map_condition_sel]
    assert jnp.allclose(params.k_sel, expected_k_sel)


# --- guide tests ---

def test_guide_logic_and_shapes(mock_data):
    """Guide creates per-condition variational params and samples."""
    name = "test_growth_guide"
    priors = get_priors()

    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name, data=mock_data, priors=priors
        )
        params = guide(name=name, data=mock_data, priors=priors)

    assert f"{name}_k_locs" in guide_trace
    assert f"{name}_k_scales" in guide_trace
    assert f"{name}_m_locs" in guide_trace
    assert f"{name}_m_scales" in guide_trace
    assert f"{name}_k" in guide_trace
    assert f"{name}_m" in guide_trace

    k_locs_val = guide_trace[f"{name}_k_locs"]["value"]
    assert k_locs_val.shape == (mock_data.num_condition_rep,)

    assert params.k_pre.shape == mock_data.map_condition_pre.shape
    assert params.m_pre.shape == mock_data.map_condition_pre.shape
    assert params.k_sel.shape == mock_data.map_condition_sel.shape
    assert params.m_sel.shape == mock_data.map_condition_sel.shape


def test_model_and_guide_have_compatible_sample_sites(mock_data):
    """Guide and model must agree on which sample sites exist."""
    name = "compat"
    priors = get_priors()

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors
        )
    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name, data=mock_data, priors=priors
        )

    model_samples = {
        n for n, s in model_trace.items()
        if s["type"] == "sample" and not s.get("is_observed", False)
    }
    guide_samples = {
        n for n, s in guide_trace.items() if s["type"] == "sample"
    }

    assert model_samples == guide_samples, (
        f"model and guide sample sites differ:\n"
        f"  model only: {model_samples - guide_samples}\n"
        f"  guide only: {guide_samples - model_samples}"
    )
