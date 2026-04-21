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
    LinearParams,
    _PINNABLE_SUFFIXES,
)
from tfscreen.analysis.hierarchical.growth_model.components._pinning import (
    _hyper,
    _pinned_value,
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
    assert "k_hyper_loc_loc" in params
    assert params["k_hyper_loc_loc"] == 0.025


def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.k_hyper_loc_loc == 0.025
    assert priors.m_hyper_loc_loc == 0.0


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
    assert f"{name}_m_hyper_scale_loc" in guide_trace
    assert f"{name}_k_offset" in guide_trace

    assert k_pre.shape == mock_data.map_condition_pre.shape
    assert m_pre.shape == mock_data.map_condition_pre.shape
    assert k_sel.shape == mock_data.map_condition_sel.shape
    assert m_sel.shape == mock_data.map_condition_sel.shape


# ---------------------------------------------------------------------------
# Pinning helpers
# ---------------------------------------------------------------------------

def test_pinnable_suffixes_includes_all_four_hypers():
    """The set of pinnable suffixes is complete and stable."""
    assert set(_PINNABLE_SUFFIXES) == {
        "k_hyper_loc", "k_hyper_scale",
        "m_hyper_loc", "m_hyper_scale",
    }


# ---------------------------------------------------------------------------
# ModelPriors with pinned field
# ---------------------------------------------------------------------------

def test_model_priors_default_pinned_is_empty_dict():
    """ModelPriors() with no `pinned` argument exposes an empty dict."""
    priors = get_priors()
    assert hasattr(priors, "pinned")
    assert priors.pinned == {}


def test_model_priors_accepts_pinned_dict():
    """ModelPriors accepts a `pinned` dict and exposes it on the instance."""
    pinned = {"k_hyper_loc": 0.030, "m_hyper_scale": 0.05}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    assert priors.pinned == pinned


def test_model_priors_replace_preserves_pinned():
    """flax.struct.dataclass.replace preserves the pinned dict."""
    pinned = {"k_hyper_loc": 0.03}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    replaced = priors.replace(k_hyper_loc_loc=0.05)
    assert replaced.pinned == pinned


# ---------------------------------------------------------------------------
# define_model with pinning
# ---------------------------------------------------------------------------

def test_define_model_unpinned_uses_sample_sites(mock_data):
    """Without pinning, all four hyper sites appear as `sample` in the trace."""
    name = "g"
    priors = get_priors()
    guesses = get_guesses(name, mock_data)
    substituted = substitute(define_model, data=guesses)

    with seed(rng_seed=0):
        tr = trace(substituted).get_trace(name=name, data=mock_data, priors=priors)

    for suffix in _PINNABLE_SUFFIXES:
        site = tr[f"{name}_{suffix}"]
        assert site["type"] == "sample", (
            f"{suffix} should be a sample site when not pinned"
        )


def test_define_model_pinned_replaces_sample_with_deterministic(mock_data):
    """Pinning a hyper replaces its sample site with a deterministic site."""
    name = "g"
    pinned = {"k_hyper_loc": 0.040, "m_hyper_scale": 0.15}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    guesses = get_guesses(name, mock_data)
    # Strip pinned sites from guesses so substitute() does not override the
    # deterministic value set by the pinning machinery.
    guesses_no_pinned = {
        k: v for k, v in guesses.items()
        if not any(k == f"{name}_{s}" for s in pinned)
    }
    substituted = substitute(define_model, data=guesses_no_pinned)

    with seed(rng_seed=0):
        tr = trace(substituted).get_trace(name=name, data=mock_data, priors=priors)

    # Pinned sites are deterministic at the pinned value
    assert tr[f"{name}_k_hyper_loc"]["type"] == "deterministic"
    assert float(tr[f"{name}_k_hyper_loc"]["value"]) == pytest.approx(0.040)
    assert tr[f"{name}_m_hyper_scale"]["type"] == "deterministic"
    assert float(tr[f"{name}_m_hyper_scale"]["value"]) == pytest.approx(0.15)

    # Unpinned sites are still sample sites
    assert tr[f"{name}_k_hyper_scale"]["type"] == "sample"
    assert tr[f"{name}_m_hyper_loc"]["type"] == "sample"


def test_define_model_pinned_constant_propagates_to_per_condition(mock_data):
    """
    With every hyper pinned and all offsets = 0, the per-condition outputs
    must equal the pinned hyper-locs (since loc + 0*scale = loc).
    """
    name = "g"
    pinned = {
        "k_hyper_loc":   0.040,
        "k_hyper_scale": 0.10,
        "m_hyper_loc":   0.005,
        "m_hyper_scale": 0.05,
    }
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    # Force offsets to zero so loc is the only contribution
    subs = {
        f"{name}_k_offset": jnp.zeros(mock_data.num_condition_rep),
        f"{name}_m_offset": jnp.zeros(mock_data.num_condition_rep),
    }

    substituted = substitute(define_model, data=subs)
    with seed(rng_seed=0):
        tr = trace(substituted).get_trace(name=name, data=mock_data, priors=priors)

    assert jnp.allclose(tr[f"{name}_k"]["value"], 0.040)
    assert jnp.allclose(tr[f"{name}_m"]["value"], 0.005)


def test_define_model_all_pinned_has_only_offset_sample_sites(mock_data):
    """When every hyper is pinned, only the per-condition offsets remain as samples."""
    name = "g"
    pinned = {s: 0.0 for s in _PINNABLE_SUFFIXES}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    substituted = substitute(define_model,
                             data={f"{name}_k_offset": jnp.zeros(mock_data.num_condition_rep),
                                   f"{name}_m_offset": jnp.zeros(mock_data.num_condition_rep)})

    with seed(rng_seed=0):
        tr = trace(substituted).get_trace(name=name, data=mock_data, priors=priors)

    sample_sites = {n for n, s in tr.items() if s["type"] == "sample"}
    assert sample_sites == {f"{name}_k_offset", f"{name}_m_offset"}


# ---------------------------------------------------------------------------
# guide with pinning
# ---------------------------------------------------------------------------

def test_guide_pinned_drops_variational_params(mock_data):
    """
    A pinned hyper must not register any variational params or sample sites
    in the guide.  Unpinned hypers retain their full param + sample machinery.
    """
    name = "gg"
    pinned = {"k_hyper_loc": 0.030}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    # Pinned hyper has neither sample nor variational params
    assert f"{name}_k_hyper_loc" not in tr
    assert f"{name}_k_hyper_loc_loc" not in tr
    assert f"{name}_k_hyper_loc_scale" not in tr

    # Unpinned hypers still expose their variational params
    assert f"{name}_m_hyper_loc" in tr
    assert f"{name}_m_hyper_loc_loc" in tr
    assert f"{name}_m_hyper_loc_scale" in tr

    # Per-condition offset sample sites are unaffected
    assert f"{name}_k_offset" in tr
    assert f"{name}_m_offset" in tr


def test_guide_all_pinned_keeps_only_offset_machinery(mock_data):
    """When all hypers are pinned, only offset params/samples remain in the guide."""
    name = "gg"
    pinned = {s: 0.0 for s in _PINNABLE_SUFFIXES}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    for suffix in _PINNABLE_SUFFIXES:
        assert f"{name}_{suffix}" not in tr
        assert f"{name}_{suffix}_loc" not in tr
        assert f"{name}_{suffix}_scale" not in tr

    # Offset params and sample sites still present
    assert f"{name}_k_offset_locs" in tr
    assert f"{name}_k_offset_scales" in tr
    assert f"{name}_m_offset_locs" in tr
    assert f"{name}_m_offset_scales" in tr
    assert f"{name}_k_offset" in tr
    assert f"{name}_m_offset" in tr


def test_guide_pinned_returns_pinned_value_in_per_condition(mock_data):
    """
    When all hypers are pinned and the variational offsets sample at zero,
    the guide's per-condition values equal the pinned hyper-locs.
    """
    name = "gg"
    pinned = {
        "k_hyper_loc":   0.040,
        "k_hyper_scale": 0.10,
        "m_hyper_loc":   0.005,
        "m_hyper_scale": 0.05,
    }
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    # Substitute the offset samples to zero so loc dominates
    subs = {
        f"{name}_k_offset": jnp.zeros(mock_data.num_condition_rep),
        f"{name}_m_offset": jnp.zeros(mock_data.num_condition_rep),
    }
    substituted = substitute(guide, data=subs)

    with seed(rng_seed=0):
        params = substituted(name=name, data=mock_data, priors=priors)

    # k_pre is k_per_condition[map_condition_pre]; with offset=0 every
    # entry equals the pinned k_hyper_loc.
    assert jnp.allclose(params.k_pre, 0.040)
    assert jnp.allclose(params.k_sel, 0.040)
    assert jnp.allclose(params.m_pre, 0.005)
    assert jnp.allclose(params.m_sel, 0.005)


# ---------------------------------------------------------------------------
# Model+guide compatibility under pinning (SVI sanity)
# ---------------------------------------------------------------------------

def test_model_and_guide_pinned_have_compatible_sample_sites(mock_data):
    """
    Numpyro requires the guide to have a `sample` site for every model
    `sample` site (excluding `obs`) — and no extras.  After pinning, both
    sides drop the same sites; this test confirms the symmetry.
    """
    name = "compat"
    pinned = {"k_hyper_loc": 0.03, "m_hyper_scale": 0.04}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors
        )
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
