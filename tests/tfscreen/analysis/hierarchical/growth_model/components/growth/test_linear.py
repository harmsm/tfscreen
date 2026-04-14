import pytest
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.growth.linear import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    LinearParams
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
    """
    Build a 7-D ln_cfu array where each cond_sel index has a given OLS slope.

    slopes : sequence of length num_cond_sel
        Desired slope for each cond_sel index (y = 7.0 + slope * t).
    t values are 0.0 at time index 0 and 1.0 at time index 1.
    """
    num_cond_sel = len(slopes)
    num_time = 2
    shape = (num_rep, num_time, num_cond_pre, num_cond_sel, num_tname, num_tconc, num_geno)
    arr = np.full(shape, 7.0)
    for cs, slope in enumerate(slopes):
        arr[:, 1, :, cs, :, :, :] = 7.0 + slope
    return jnp.array(arr)


def _make_t_sel(num_rep=2, num_cond_pre=4, num_cond_sel=4, num_tname=1, num_tconc=3, num_geno=3):
    """Build a 7-D t_sel array with t=0 at index 0 and t=1 at index 1."""
    num_time = 2
    shape = (num_rep, num_time, num_cond_pre, num_cond_sel, num_tname, num_tconc, num_geno)
    arr = np.zeros(shape)
    arr[:, 1, ...] = 1.0
    return jnp.array(arr)


@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    - 3 conditions
    - Maps index into these 3 conditions
    - Constant ln_cfu (slope=0.0) so k_offsets are all zero.
    """
    num_condition_rep = 3
    num_replicate = 2

    # 4 observations mapping into the [0, 1, 2] condition array
    map_condition_pre = jnp.array([0, 2, 2, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([1, 0, 1, 2], dtype=jnp.int32)

    # Constant ln_cfu → slope = 0.0 everywhere → k_offsets = 0.0
    ln_cfu = _make_ln_cfu([0.0, 0.0, 0.0, 0.0])          # 4 cond_sel, slope=0 each
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
      cond_rep 0  (cond_sel index 0): slope = 0.030
      cond_rep 1  (cond_sel index 1): slope = 0.020

    Expected:
      k_per_cond_rep = [0.030, 0.020]
      k_hyper_loc    = median([0.030, 0.020]) = 0.025
      k_offsets      = [(0.030-0.025)/0.1, (0.020-0.025)/0.1] = [0.5, -0.5]
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
    ln_cfu_arr[:, 1, :, 0, :, :, :] = 7.03   # cond_sel=0  slope=0.030
    ln_cfu_arr[:, 1, :, 1, :, :, :] = 7.02   # cond_sel=1  slope=0.020

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
    """Tests that get_hyperparameters returns the correct structure and defaults."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "growth_k_hyper_loc_loc" in params
    assert params["growth_k_hyper_loc_loc"] == 0.025


def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.growth_k_hyper_loc_loc == 0.025
    assert priors.growth_m_hyper_loc_loc == 0.0


def test_get_guesses_keys_and_shapes(mock_data):
    """get_guesses returns correctly named and shaped guesses."""
    name = "test_growth"
    guesses = get_guesses(name, mock_data)

    assert isinstance(guesses, dict)
    assert f"{name}_k_hyper_loc"   in guesses
    assert f"{name}_k_hyper_scale" in guesses
    assert f"{name}_m_hyper_loc"   in guesses
    assert f"{name}_m_hyper_scale" in guesses
    assert f"{name}_k_offset"      in guesses
    assert f"{name}_m_offset"      in guesses

    expected_shape = (mock_data.num_condition_rep,)
    assert guesses[f"{name}_k_offset"].shape == expected_shape
    assert guesses[f"{name}_m_offset"].shape == expected_shape


def test_get_guesses_constant_ln_cfu_gives_zero_offsets(mock_data):
    """With constant ln_cfu (slope=0 everywhere), all k_offsets are zero."""
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    assert jnp.all(guesses[f"{name}_k_offset"] == 0.0)


def test_get_guesses_m_offsets_always_zero(mock_data):
    """m offsets are always zero (cannot estimate m without theta)."""
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    assert jnp.all(guesses[f"{name}_m_offset"] == 0.0)


def test_get_guesses_empirical_k_hyper_loc(mock_data_empirical):
    """k_hyper_loc is the median of per-cond_rep slopes."""
    name = "eg"
    guesses = get_guesses(name, mock_data_empirical)
    # slopes = [0.030, 0.020] → median = 0.025
    assert abs(float(guesses[f"{name}_k_hyper_loc"]) - 0.025) < 1e-6


def test_get_guesses_empirical_k_offsets(mock_data_empirical):
    """k_offsets reflect per-cond_rep deviation from hyper_loc."""
    name = "eg"
    guesses = get_guesses(name, mock_data_empirical)
    offsets = np.array(guesses[f"{name}_k_offset"])
    # cond_rep 0: (0.030 - 0.025) / 0.1 =  0.05
    # cond_rep 1: (0.020 - 0.025) / 0.1 = -0.05
    assert abs(offsets[0] - 0.05)  < 1e-4
    assert abs(offsets[1] - (-0.05)) < 1e-4


def test_get_guesses_masked_observations(mock_data_empirical):
    """Fully masked cond_sel falls back to global median, not NaN."""
    name = "eg"
    # Mask out all observations for cond_sel=0 (cond_rep=0)
    mask_arr = np.array(mock_data_empirical.good_mask)
    mask_arr[:, :, :, 0, :, :, :] = False
    masked_data = mock_data_empirical._replace(good_mask=jnp.array(mask_arr))

    guesses = get_guesses(name, masked_data)
    offsets = np.array(guesses[f"{name}_k_offset"])
    # Only cond_rep=1 has data (slope=0.020).
    # global_k = 0.020, k_hyper_loc = median([0.020, 0.020]) = 0.020
    # cond_rep=0 filled with global_k=0.020, offset = 0.0
    # cond_rep=1 offset = 0.0
    assert np.all(np.isfinite(offsets)), "offsets should not be NaN"
    assert np.allclose(offsets, 0.0, atol=1e-5)


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
        ln_cfu=jnp.zeros((3,)),   # wrong ndim
        t_sel=jnp.zeros((3,)),
        good_mask=jnp.ones((3,), dtype=bool),
    )
    guesses = get_guesses(name, data)
    assert guesses[f"{name}_k_hyper_loc"] == 0.025
    assert guesses[f"{name}_k_offset"].shape == (3,)
    assert jnp.all(guesses[f"{name}_k_offset"] == 0.0)


# --- define_model tests (unchanged logic) ---

def test_define_model_structure_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the hierarchical case.
    Verifies output shapes and deterministic site registration.
    """
    name = "test_growth"
    priors = get_priors()
    guesses = get_guesses(name, mock_data)

    substituted_model = substitute(define_model, data=guesses)
    params = substituted_model(name=name, data=mock_data, priors=priors)
    k_pre = params.k_pre
    m_pre = params.m_pre
    k_sel = params.k_sel
    m_sel = params.m_sel

    model_trace = trace(substituted_model).get_trace(
        name=name, data=mock_data, priors=priors
    )

    assert k_pre.shape == mock_data.map_condition_pre.shape
    assert m_pre.shape == mock_data.map_condition_pre.shape
    assert k_sel.shape == mock_data.map_condition_sel.shape
    assert m_sel.shape == mock_data.map_condition_sel.shape

    k_name = f"{name}_k"
    m_name = f"{name}_m"
    assert k_name in model_trace
    assert m_name in model_trace

    k_per_condition = model_trace[k_name]["value"]
    m_per_condition = model_trace[m_name]["value"]

    expected_shape = (mock_data.num_condition_rep,)
    assert k_per_condition.shape == expected_shape
    assert m_per_condition.shape == expected_shape


def test_define_model_calculation_logic(mock_data):
    """
    Tests that the math (loc + offset * scale) and mapping are correct.
    We inject specific non-zero values to ensure data flows correctly.
    """
    name = "test_growth"
    priors = get_priors()

    custom_guesses = {
        f"{name}_k_hyper_loc":  10.0,
        f"{name}_k_hyper_scale": 2.0,
        f"{name}_m_hyper_loc":   0.0,
        f"{name}_m_hyper_scale": 1.0,
        f"{name}_k_offset": jnp.array([0.0, 1.0, -1.0]),
        f"{name}_m_offset": jnp.zeros(mock_data.num_condition_rep),
    }

    substituted_model = substitute(define_model, data=custom_guesses)
    params = substituted_model(name=name, data=mock_data, priors=priors)
    k_pre = params.k_pre
    k_sel = params.k_sel

    # Expected values per condition: [10, 12, 8]
    expected_k_per_condition = jnp.array([10.0, 12.0, 8.0])

    expected_k_pre = expected_k_per_condition[mock_data.map_condition_pre]
    assert jnp.allclose(k_pre, expected_k_pre)

    expected_k_sel = expected_k_per_condition[mock_data.map_condition_sel]
    assert jnp.allclose(k_sel, expected_k_sel)


def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function shapes, parameter creation, and execution.
    """
    name = "test_growth_guide"
    priors = get_priors()

    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name, data=mock_data, priors=priors
        )
        params = guide(name=name, data=mock_data, priors=priors)

    k_pre = params.k_pre
    m_pre = params.m_pre
    k_sel = params.k_sel
    m_sel = params.m_sel

    assert f"{name}_k_hyper_loc_loc"  in guide_trace
    assert f"{name}_k_hyper_scale_loc" in guide_trace

    assert f"{name}_k_offset_locs" in guide_trace
    k_locs = guide_trace[f"{name}_k_offset_locs"]["value"]
    assert k_locs.shape == (mock_data.num_condition_rep,)

    assert f"{name}_m_offset_scales" in guide_trace
    m_scales = guide_trace[f"{name}_m_offset_scales"]["value"]
    assert m_scales.shape == (mock_data.num_condition_rep,)

    assert f"{name}_k_hyper_loc" in guide_trace
    assert f"{name}_m_hyper_scale" in guide_trace
    assert f"{name}_k_offset" in guide_trace

    assert k_pre.shape == mock_data.map_condition_pre.shape
    assert m_pre.shape == mock_data.map_condition_pre.shape
    assert k_sel.shape == mock_data.map_condition_sel.shape
    assert m_sel.shape == mock_data.map_condition_sel.shape
