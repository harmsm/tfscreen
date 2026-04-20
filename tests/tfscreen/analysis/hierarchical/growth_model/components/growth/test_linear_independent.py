import pytest
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.growth.linear_independent import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_priors,
    get_guesses,
    LinearParams
)

# --- Mock Data Fixtures ---

MockGrowthData = namedtuple("MockGrowthData", [
    "num_condition_rep",
    "num_replicate",
    "map_condition_pre",
    "map_condition_sel",
    "growth_shares_replicates",
    "ln_cfu",
    "t_sel",
    "good_mask",
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    - 2 conditions, 3 replicates (N=2, M=3, total flat indices 0-5)
    - 4 'pre' and 4 'sel' observations
    - 1-D dummy ln_cfu/t_sel triggers the fallback path in get_guesses.
    """
    num_condition_rep = 2
    num_replicate = 3

    map_condition_pre = jnp.array([0, 2, 4, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([5, 3, 1, 0], dtype=jnp.int32)

    return MockGrowthData(
        num_condition_rep=num_condition_rep,
        num_replicate=num_replicate,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel,
        growth_shares_replicates=False,
        ln_cfu=jnp.zeros((1,)),    # wrong ndim → fallback path
        t_sel=jnp.zeros((1,)),
        good_mask=jnp.ones((1,), dtype=bool),
    )


def _make_7d(slopes, num_array_rep=1, num_cond_pre=2,
             num_tname=1, num_tconc=1, num_geno=1):
    """Build (num_array_rep, 2, num_cond_pre, num_cond_sel, ...) ln_cfu and t_sel
    arrays where cond_sel index i gets slope slopes[i].  t=0 at time 0, t=1 at time 1.
    """
    num_cond_sel = len(slopes)
    num_time = 2
    shape = (num_array_rep, num_time, num_cond_pre, num_cond_sel,
             num_tname, num_tconc, num_geno)
    ln_cfu = np.full(shape, 7.0)
    for cs, slope in enumerate(slopes):
        ln_cfu[:, 1, :, cs, :, :, :] = 7.0 + slope
    t_sel = np.zeros(shape)
    t_sel[:, 1, ...] = 1.0
    return jnp.array(ln_cfu), jnp.array(t_sel), jnp.ones(shape, dtype=bool)


@pytest.fixture
def mock_data_empirical():
    """
    2 conditions × 2 replicates (N=2, M=2).  Flat index = c*M + r:
      flat 0 = (c=0, r=0): slope 0.030
      flat 1 = (c=0, r=1): slope 0.020
      flat 2 = (c=1, r=0): slope 0.010
      flat 3 = (c=1, r=1): slope 0.010

    Expected empirical guesses:
      k_hyper_loc  = [[0.025], [0.010]]   (median across reps per condition)
      k_offset     = [[ 0.5, -0.5],       (deviation / DEFAULT_HYPER_SCALE=0.1)
                      [ 0.0,  0.0]]
    """
    num_condition_rep = 2
    num_replicate = 2
    slopes = [0.030, 0.020, 0.010, 0.010]   # one per flat index
    ln_cfu, t_sel, good_mask = _make_7d(slopes)

    map_condition_pre = jnp.array([0, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

    return MockGrowthData(
        num_condition_rep=num_condition_rep,
        num_replicate=num_replicate,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel,
        growth_shares_replicates=False,
        ln_cfu=ln_cfu,
        t_sel=t_sel,
        good_mask=good_mask,
    )


# --- Test Cases ---

def test_get_hyperparameters(mock_data):
    """Tests that get_hyperparameters returns correctly shaped arrays."""
    params = get_hyperparameters(mock_data.num_condition_rep)
    assert isinstance(params, dict)
    
    k_loc = params["growth_k_hyper_loc_loc"]
    assert k_loc.shape == (mock_data.num_condition_rep,)
    assert jnp.allclose(k_loc, 0.025)

def test_get_priors(mock_data):
    """Tests our corrected get_priors function."""
    priors = get_priors(mock_data.num_condition_rep)
    assert isinstance(priors, ModelPriors)
    assert priors.growth_k_hyper_loc_loc.shape == (mock_data.num_condition_rep,)
    assert priors.growth_m_hyper_loc_loc.shape == (mock_data.num_condition_rep,)

def test_get_guesses_shapes(mock_data):
    """get_guesses returns correctly shaped arrays."""
    name = "test_growth"
    guesses = get_guesses(name, mock_data)

    assert isinstance(guesses, dict)

    hyper_shape = (mock_data.num_condition_rep, 1)
    assert guesses[f"{name}_k_hyper_loc"].shape == hyper_shape
    assert guesses[f"{name}_k_hyper_scale"].shape == hyper_shape
    assert guesses[f"{name}_m_hyper_loc"].shape == hyper_shape
    assert guesses[f"{name}_m_hyper_scale"].shape == hyper_shape

    offset_shape = (mock_data.num_condition_rep, mock_data.num_replicate)
    assert guesses[f"{name}_k_offset"].shape == offset_shape
    assert guesses[f"{name}_m_offset"].shape == offset_shape


def test_get_guesses_fallback_default_values(mock_data):
    """Fallback path returns sensible defaults: k≈0.025, m=0."""
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    assert jnp.allclose(guesses[f"{name}_k_hyper_loc"], 0.025)
    assert jnp.allclose(guesses[f"{name}_m_hyper_loc"], 0.0)


def test_get_guesses_m_offsets_always_zero(mock_data):
    """m offsets are always zero (cannot estimate m without knowing theta)."""
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    assert jnp.all(guesses[f"{name}_m_offset"] == 0.0)


def test_get_guesses_empirical_k_hyper_loc(mock_data_empirical):
    """k_hyper_loc is the per-condition median OLS slope across replicates."""
    name = "eg"
    guesses = get_guesses(name, mock_data_empirical)
    k_hl = np.array(guesses[f"{name}_k_hyper_loc"]).ravel()
    # cond 0: median(0.030, 0.020) = 0.025
    # cond 1: median(0.010, 0.010) = 0.010
    assert abs(k_hl[0] - 0.025) < 1e-5
    assert abs(k_hl[1] - 0.010) < 1e-5


def test_get_guesses_empirical_k_offsets(mock_data_empirical):
    """k_offset captures per-(condition, replicate) deviation from k_hyper_loc."""
    name = "eg"
    guesses = get_guesses(name, mock_data_empirical)
    k_off = np.array(guesses[f"{name}_k_offset"])
    # cond 0: (0.030-0.025)/0.1=0.05 and (0.020-0.025)/0.1=-0.05
    # cond 1: both 0.0
    assert abs(k_off[0, 0] -  0.05) < 1e-4
    assert abs(k_off[0, 1] - -0.05) < 1e-4
    assert abs(k_off[1, 0]) < 1e-4
    assert abs(k_off[1, 1]) < 1e-4


def test_get_guesses_masked_observations(mock_data_empirical):
    """Fully masked condition-rep falls back to global median, not NaN."""
    name = "eg"
    mask_arr = np.array(mock_data_empirical.good_mask)
    # Mask all observations for cond_sel=0 (flat index 0 = cond 0, rep 0)
    mask_arr[:, :, :, 0, :, :, :] = False
    masked = mock_data_empirical._replace(good_mask=jnp.array(mask_arr))

    guesses = get_guesses(name, masked)
    k_off = np.array(guesses[f"{name}_k_offset"])
    assert np.all(np.isfinite(k_off)), "offsets must not be NaN"


def test_get_guesses_non7d_fallback():
    """Non-7D tensors trigger the hard-coded fallback without error."""
    name = "fg"
    MockSimple = namedtuple("MockSimple", [
        "num_condition_rep", "num_replicate",
        "map_condition_pre", "map_condition_sel", "growth_shares_replicates",
        "ln_cfu", "t_sel", "good_mask",
    ])
    data = MockSimple(
        num_condition_rep=2, num_replicate=3,
        map_condition_pre=jnp.array([0, 1]),
        map_condition_sel=jnp.array([0, 1, 2, 3, 4, 5]),
        growth_shares_replicates=False,
        ln_cfu=jnp.zeros((3,)),    # wrong ndim
        t_sel=jnp.zeros((3,)),
        good_mask=jnp.ones((3,), dtype=bool),
    )
    guesses = get_guesses(name, data)
    assert jnp.allclose(guesses[f"{name}_k_hyper_loc"], 0.025)
    assert guesses[f"{name}_k_offset"].shape == (2, 3)
    assert jnp.all(guesses[f"{name}_k_offset"] == 0.0)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the independent case.
    """
    name = "test_growth_ind"
    
    # Use our fixed helper functions
    priors = get_priors(mock_data.num_condition_rep)
    guesses = get_guesses(name, mock_data)
    
    # 1. Substitute sample sites with our guess values
    substituted_model = substitute(define_model, data=guesses)
    
    # 2. Run the substituted model to get the final return tuple
    params = substituted_model(name=name, 
                                     data=mock_data, 
                                     priors=priors)
    k_pre = params.k_pre
    m_pre = params.m_pre
    k_sel = params.k_sel
    m_sel = params.m_sel
    
    # 3. Trace the execution to capture intermediate (deterministic) values
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    assert k_pre.shape == mock_data.map_condition_pre.shape
    assert m_pre.shape == mock_data.map_condition_pre.shape
    assert k_sel.shape == mock_data.map_condition_sel.shape
    assert m_sel.shape == mock_data.map_condition_sel.shape

    # --- 2. Check the Per-Condition/Replicate Deterministic Sites ---
    k_name = f"{name}_k"
    m_name = f"{name}_m"
    assert k_name in model_trace
    assert m_name in model_trace
    
    k_per_cond_rep_1d = model_trace[k_name]["value"]
    m_per_cond_rep_1d = model_trace[m_name]["value"]
    
    # Check shape (must be flattened)
    expected_flat_shape = (mock_data.num_condition_rep * mock_data.num_replicate,)
    assert k_per_cond_rep_1d.shape == expected_flat_shape
    
    # --- 3. Check Values ---
    
    # We must replicate the model's logic exactly to get the expected
    # values. This includes the broadcasting logic.
    
    # Get the guess values
    k_hyper_loc = guesses[f"{name}_k_hyper_loc"]
    k_hyper_scale = guesses[f"{name}_k_hyper_scale"]
    k_offset = guesses[f"{name}_k_offset"]
    
    m_hyper_loc = guesses[f"{name}_m_hyper_loc"]
    m_hyper_scale = guesses[f"{name}_m_hyper_scale"]
    m_offset = guesses[f"{name}_m_offset"]
    
    # The model calculates: (shape_2_1 + shape_2_3 * shape_2_1)
    # This broadcasts to shape (2,3)
    expected_k_dist_2d = k_hyper_loc + k_offset * k_hyper_scale
    expected_m_dist_2d = m_hyper_loc + m_offset * m_hyper_scale
    
    # Ravel just like the model does
    expected_k_vals = expected_k_dist_2d.ravel()
    expected_m_vals = expected_m_dist_2d.ravel()
    
    # Now, the comparison should be correct
    assert jnp.allclose(k_per_cond_rep_1d, expected_k_vals)
    assert jnp.allclose(m_per_cond_rep_1d, expected_m_vals)
    
    # --- 4. Check Final Returned (Expanded) Tensors ---
    
    # Spot-check the mapping logic
    assert k_pre[0] == k_per_cond_rep_1d[mock_data.map_condition_pre[0]]
    assert m_sel[1] == m_per_cond_rep_1d[mock_data.map_condition_sel[1]]

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function shapes, parameter creation, and execution.
    """
    name = "test_growth_ind_guide"
    priors = get_priors(mock_data.num_condition_rep)

    # Seed the guide execution to handle sampling
    with seed(rng_seed=0):
        # Trace the guide to inspect parameters and samples
        guide_trace = trace(guide).get_trace(
            name=name,
            data=mock_data,
            priors=priors
        )
        
        # Run guide to check return values
        params = guide(name=name,
                             data=mock_data,
                             priors=priors)
    
    k_pre = params.k_pre
    m_pre = params.m_pre
    k_sel = params.k_sel
    m_sel = params.m_sel

    # --- 1. Check Global Parameter Sites ---
    # Should have shape (num_condition_rep,)
    assert f"{name}_k_hyper_loc_loc" in guide_trace
    k_hl_loc = guide_trace[f"{name}_k_hyper_loc_loc"]["value"]
    assert k_hl_loc.shape == (mock_data.num_condition_rep,)

    # --- 2. Check Local Parameter Sites ---
    # The guide initializes local params with shape (num_replicate, num_condition_rep)
    # due to how the nested plates index into them.
    assert f"{name}_k_offset_locs" in guide_trace
    k_offset_locs = guide_trace[f"{name}_k_offset_locs"]["value"]
    expected_local_shape = (mock_data.num_replicate, mock_data.num_condition_rep)
    assert k_offset_locs.shape == expected_local_shape
    
    # --- 3. Check Sample Sites ---
    # Global samples should have shape (num_condition_rep,)
    assert f"{name}_k_hyper_loc" in guide_trace
    k_hyper = guide_trace[f"{name}_k_hyper_loc"]["value"]
    assert k_hyper.shape == (mock_data.num_condition_rep,)

    # Local samples should broadcast to (num_replicate, num_condition_rep)
    # because of the nested plates dim=-1 and dim=-2
    assert f"{name}_k_offset" in guide_trace
    k_offset = guide_trace[f"{name}_k_offset"]["value"]
    assert k_offset.shape == expected_local_shape

    # --- 4. Check Return Shapes ---
    # Must match the mapping arrays
    assert k_pre.shape == mock_data.map_condition_pre.shape
    assert m_pre.shape == mock_data.map_condition_pre.shape
    assert k_sel.shape == mock_data.map_condition_sel.shape
    assert m_sel.shape == mock_data.map_condition_sel.shape