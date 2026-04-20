import pytest
import inspect
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.ln_cfu0.hierarchical import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    _mad_scale,
    _empirical_group_estimates,
    _SCALE_FLOOR,
    _PINNABLE_SUFFIXES,
)

# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

MockGrowthData = namedtuple("MockGrowthData", [
    "num_replicate",
    "num_condition_pre",
    "num_genotype",
    "batch_size",
    "batch_idx",
    "scale_vector",
    "ln_cfu0_spiked_mask",
    "ln_cfu0_wt_mask",
    "ln_cfu",       # (rep, time, cond_pre, cond_sel, tname, tconc, geno)
    "good_mask",    # same shape as ln_cfu, bool
    "map_ln_cfu0",  # kept for API compatibility
])


def _make_ln_cfu(num_replicate, num_condition_pre, num_genotype, per_geno_values):
    """
    Build a (rep, time=2, cond_pre, cond_sel=2, tname=1, tconc=3, geno) array
    where ``per_geno_values[g]`` is the constant ln_cfu for genotype g across
    all other dimensions.
    """
    shape = (num_replicate, 2, num_condition_pre, 2, 1, 3, num_genotype)
    arr = np.zeros(shape)
    for g, v in enumerate(per_geno_values):
        arr[..., g] = v
    return arr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_data():
    """
    4 library genotypes (no spiked, no wt), 2 replicates, 2 condition_pre.
    All genotypes have ln_cfu = 8.0, so hyper_loc will be estimated as 8.0
    and all offsets will be 0.
    """
    num_replicate    = 2
    num_condition_pre = 2
    num_genotype     = 4
    batch_size       = 4

    ln_cfu_vals = [8.0, 8.0, 8.0, 8.0]
    ln_cfu    = _make_ln_cfu(num_replicate, num_condition_pre, num_genotype, ln_cfu_vals)
    good_mask = np.ones_like(ln_cfu, dtype=bool)

    return MockGrowthData(
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=jnp.arange(batch_size, dtype=jnp.int32),
        scale_vector=jnp.ones(batch_size, dtype=float),
        ln_cfu0_spiked_mask=jnp.zeros(num_genotype, dtype=bool),
        ln_cfu0_wt_mask=jnp.zeros(num_genotype, dtype=bool),
        ln_cfu=ln_cfu,
        good_mask=good_mask,
        map_ln_cfu0=jnp.arange(batch_size, dtype=jnp.int32),
    )


@pytest.fixture
def mock_data_with_spiked():
    """
    6 genotypes: indices 0,1 are spiked (ln_cfu=10), indices 2-5 are library
    (ln_cfu=8).  Offsets are 0 because each genotype is exactly at its group
    median.
    """
    num_replicate    = 2
    num_condition_pre = 2
    num_genotype     = 6
    batch_size       = 6

    ln_cfu_vals = [10.0, 10.0, 8.0, 8.0, 8.0, 8.0]
    ln_cfu    = _make_ln_cfu(num_replicate, num_condition_pre, num_genotype, ln_cfu_vals)
    good_mask = np.ones_like(ln_cfu, dtype=bool)

    return MockGrowthData(
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=jnp.arange(batch_size, dtype=jnp.int32),
        scale_vector=jnp.ones(batch_size, dtype=float),
        ln_cfu0_spiked_mask=jnp.array([True, True, False, False, False, False]),
        ln_cfu0_wt_mask=jnp.zeros(num_genotype, dtype=bool),
        ln_cfu=ln_cfu,
        good_mask=good_mask,
        map_ln_cfu0=jnp.arange(batch_size, dtype=jnp.int32),
    )


@pytest.fixture
def mock_data_with_wt():
    """
    6 genotypes: index 5 is wt (ln_cfu=12), indices 0-4 are library (ln_cfu=8).
    Offsets are 0 because each genotype is exactly at its group median.
    """
    num_replicate    = 2
    num_condition_pre = 2
    num_genotype     = 6
    batch_size       = 6

    ln_cfu_vals = [8.0, 8.0, 8.0, 8.0, 8.0, 12.0]
    ln_cfu    = _make_ln_cfu(num_replicate, num_condition_pre, num_genotype, ln_cfu_vals)
    good_mask = np.ones_like(ln_cfu, dtype=bool)

    return MockGrowthData(
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=jnp.arange(batch_size, dtype=jnp.int32),
        scale_vector=jnp.ones(batch_size, dtype=float),
        ln_cfu0_spiked_mask=jnp.zeros(num_genotype, dtype=bool),
        ln_cfu0_wt_mask=jnp.array([False, False, False, False, False, True]),
        ln_cfu=ln_cfu,
        good_mask=good_mask,
        map_ln_cfu0=jnp.arange(batch_size, dtype=jnp.int32),
    )


@pytest.fixture
def mock_data_empirical():
    """
    6 genotypes with varied ln_cfu for testing empirical estimation:
      index 0: wt,      ln_cfu = 12.0
      index 1: spiked,  ln_cfu = 10.0
      index 2: spiked,  ln_cfu = 11.0   ← differs from index 1
      index 3: library, ln_cfu =  8.0
      index 4: library, ln_cfu =  7.0
      index 5: library, ln_cfu =  9.0

    Expected group medians:
      wt_loc     = 12.0
      spiked_loc = median(10.0, 11.0) = 10.5
      hyper_loc  = median(8.0, 7.0, 9.0) = 8.0
    """
    num_replicate    = 2
    num_condition_pre = 2
    num_genotype     = 6
    batch_size       = 6

    ln_cfu_vals = [12.0, 10.0, 11.0, 8.0, 7.0, 9.0]
    ln_cfu    = _make_ln_cfu(num_replicate, num_condition_pre, num_genotype, ln_cfu_vals)
    good_mask = np.ones_like(ln_cfu, dtype=bool)

    return MockGrowthData(
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=jnp.arange(batch_size, dtype=jnp.int32),
        scale_vector=jnp.ones(batch_size, dtype=float),
        ln_cfu0_spiked_mask=jnp.array([False, True, True, False, False, False]),
        ln_cfu0_wt_mask=jnp.array([True, False, False, False, False, False]),
        ln_cfu=ln_cfu,
        good_mask=good_mask,
        map_ln_cfu0=jnp.arange(batch_size, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Tests: _mad_scale helper
# ---------------------------------------------------------------------------

def test_mad_scale_uses_fallback_for_empty_input():
    """An empty array returns the supplied fallback (no MAD computable)."""
    assert _mad_scale(np.array([]), fallback=2.5) == 2.5


def test_mad_scale_uses_fallback_for_all_nan():
    """All-NaN input returns the fallback."""
    assert _mad_scale(np.array([np.nan, np.nan]), fallback=2.5) == 2.5


def test_mad_scale_floors_at_scale_floor():
    """Identical values produce zero MAD; result is clipped to the floor."""
    assert _mad_scale(np.array([5.0, 5.0, 5.0]), fallback=99.0) == _SCALE_FLOOR


def test_mad_scale_returns_consistent_estimate():
    """For symmetric deviations, scale = 1.4826 * MAD."""
    # MAD of [-1, -1, +1, +1] is 1.0; expected scale = 1.4826
    result = _mad_scale(np.array([-1.0, -1.0, 1.0, 1.0]), fallback=99.0)
    assert result == pytest.approx(1.4826, rel=1e-4)


def test_mad_scale_ignores_nans():
    """NaN entries are ignored when computing the MAD."""
    result = _mad_scale(np.array([np.nan, -1.0, 1.0, np.nan]), fallback=99.0)
    assert result == pytest.approx(1.4826, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests: get_hyperparameters / get_priors
# ---------------------------------------------------------------------------

def test_get_hyperparameters():
    """get_hyperparameters returns the correct keys and default values."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "ln_cfu0_hyper_loc_loc" in params
    assert params["ln_cfu0_hyper_loc_loc"] == 6.0
    assert "ln_cfu0_spiked_loc_loc" in params
    assert params["ln_cfu0_spiked_loc_loc"] == 12.0


def test_get_hyperparameters_includes_new_fixed_scales():
    """The restructured ModelPriors exposes fixed wt/spiked scales."""
    params = get_hyperparameters()
    assert "ln_cfu0_wt_scale" in params
    assert "ln_cfu0_spiked_scale" in params
    # Defaults are positive floats
    assert isinstance(params["ln_cfu0_wt_scale"], float)
    assert isinstance(params["ln_cfu0_spiked_scale"], float)
    assert params["ln_cfu0_wt_scale"] > 0
    assert params["ln_cfu0_spiked_scale"] > 0


def test_get_priors_no_data_returns_defaults():
    """get_priors() with no data returns the hyperparameter defaults."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.ln_cfu0_hyper_loc_loc == 6.0
    assert priors.ln_cfu0_spiked_loc_loc == 12.0
    # New scale fields populated from defaults
    assert priors.ln_cfu0_wt_scale == get_hyperparameters()["ln_cfu0_wt_scale"]
    assert priors.ln_cfu0_spiked_scale == get_hyperparameters()["ln_cfu0_spiked_scale"]


def test_get_priors_accepts_data_keyword():
    """get_priors must accept an optional `data` keyword (model_class detects this)."""
    sig = inspect.signature(get_priors)
    assert "data" in sig.parameters
    assert sig.parameters["data"].default is None


def test_get_priors_with_data_overrides_subgroup_scales(mock_data_empirical):
    """
    When data is supplied, ln_cfu0_wt_scale and ln_cfu0_spiked_scale are
    derived from the data instead of the defaults.  Other priors stay at
    their hyperparameter defaults.
    """
    priors = get_priors(data=mock_data_empirical)

    # spiked: vals [10, 11], loc=10.5, deviations=[-0.5*4, +0.5*4],
    # MAD=0.5, scale = 1.4826*0.5 = 0.7413
    assert priors.ln_cfu0_spiked_scale == pytest.approx(1.4826 * 0.5, rel=1e-4)

    # wt: only one genotype with constant value -> deviations all zero ->
    # scale floored at _SCALE_FLOOR
    assert priors.ln_cfu0_wt_scale == pytest.approx(_SCALE_FLOOR)

    # Non-scale priors unchanged from defaults
    defaults = get_hyperparameters()
    assert priors.ln_cfu0_hyper_loc_loc == defaults["ln_cfu0_hyper_loc_loc"]
    assert priors.ln_cfu0_wt_loc_loc == defaults["ln_cfu0_wt_loc_loc"]


def test_get_priors_with_unparseable_data_falls_back_to_defaults():
    """
    A stub data object that lacks a 7-D ln_cfu tensor must fall through to
    the hyperparameter defaults rather than raising.
    """
    StubData = namedtuple(
        "StubData",
        ["ln_cfu", "good_mask", "ln_cfu0_spiked_mask", "ln_cfu0_wt_mask"],
    )
    stub = StubData(
        ln_cfu=np.zeros((1, 1)),                       # not 7-D
        good_mask=np.ones((1, 1), dtype=bool),
        ln_cfu0_spiked_mask=np.zeros(1, dtype=bool),
        ln_cfu0_wt_mask=np.zeros(1, dtype=bool),
    )
    priors = get_priors(data=stub)
    defaults = get_hyperparameters()
    assert priors.ln_cfu0_wt_scale == defaults["ln_cfu0_wt_scale"]
    assert priors.ln_cfu0_spiked_scale == defaults["ln_cfu0_spiked_scale"]


# ---------------------------------------------------------------------------
# Tests: _empirical_group_estimates
# ---------------------------------------------------------------------------

def test_empirical_group_estimates_returns_none_for_non_7d_input():
    """Non-7-D ln_cfu signals 'no usable data' and the helper returns None."""
    StubData = namedtuple(
        "StubData",
        ["ln_cfu", "good_mask", "ln_cfu0_spiked_mask", "ln_cfu0_wt_mask"],
    )
    stub = StubData(
        ln_cfu=np.zeros((1, 1)),
        good_mask=np.ones((1, 1), dtype=bool),
        ln_cfu0_spiked_mask=np.zeros(1, dtype=bool),
        ln_cfu0_wt_mask=np.zeros(1, dtype=bool),
    )
    assert _empirical_group_estimates(stub) is None


def test_empirical_group_estimates_locs_and_scales(mock_data_empirical):
    """
    Group locs come from per-genotype medians; group scales come from
    1.4826 * MAD of within-group per-(rep, cond_pre, geno) deviations from
    the group loc.
    """
    est = _empirical_group_estimates(mock_data_empirical)
    assert est is not None
    assert est["wt_loc"] == pytest.approx(12.0)
    assert est["spiked_loc"] == pytest.approx(10.5)
    assert est["hyper_loc"] == pytest.approx(8.0)

    # Library: vals 8, 7, 9; MAD of deviations [0, -1, +1] over 4 (rep,cond)
    # repeats = 1.0; scale = 1.4826
    assert est["hyper_scale"] == pytest.approx(1.4826, rel=1e-4)
    # Spiked: vals 10, 11; MAD = 0.5; scale = 0.7413
    assert est["spiked_scale"] == pytest.approx(1.4826 * 0.5, rel=1e-4)
    # Wt: single value -> floor
    assert est["wt_scale"] == pytest.approx(_SCALE_FLOOR)


# ---------------------------------------------------------------------------
# Tests: get_guesses – structure
# ---------------------------------------------------------------------------

def test_get_guesses_keys_and_offset_shape(mock_data):
    """get_guesses returns all required keys with the correct offset shape."""
    name   = "test_ln_cfu0"
    guesses = get_guesses(name, mock_data)

    assert isinstance(guesses, dict)
    for key in [f"{name}_hyper_loc", f"{name}_hyper_scale",
                f"{name}_spiked_loc", f"{name}_wt_loc", f"{name}_offset"]:
        assert key in guesses

    expected_shape = (mock_data.num_replicate,
                      mock_data.num_condition_pre,
                      mock_data.num_genotype)
    assert guesses[f"{name}_offset"].shape == expected_shape


# ---------------------------------------------------------------------------
# Tests: get_guesses – empirical group-level estimates
# ---------------------------------------------------------------------------

def test_get_guesses_hyper_loc_from_library(mock_data):
    """hyper_loc is the median of library genotype ln_cfu values (all 8.0)."""
    guesses = get_guesses("x", mock_data)
    assert guesses["x_hyper_loc"] == pytest.approx(8.0)


def test_get_guesses_spiked_loc_from_data(mock_data_with_spiked):
    """spiked_loc is the median of spiked-genotype ln_cfu values (10.0)."""
    guesses = get_guesses("x", mock_data_with_spiked)
    assert guesses["x_spiked_loc"] == pytest.approx(10.0)


def test_get_guesses_wt_loc_from_data(mock_data_with_wt):
    """wt_loc is the median of wt-genotype ln_cfu values (12.0)."""
    guesses = get_guesses("x", mock_data_with_wt)
    assert guesses["x_wt_loc"] == pytest.approx(12.0)


def test_get_guesses_group_medians_with_variation(mock_data_empirical):
    """
    With different ln_cfu per genotype, group medians are computed correctly:
      wt_loc     = 12.0
      spiked_loc = 10.5  (median of 10.0 and 11.0)
      hyper_loc  =  8.0  (median of 7.0, 8.0, 9.0)
    """
    guesses = get_guesses("x", mock_data_empirical)
    assert guesses["x_wt_loc"]     == pytest.approx(12.0)
    assert guesses["x_spiked_loc"] == pytest.approx(10.5)
    assert guesses["x_hyper_loc"]  == pytest.approx(8.0)


def test_get_guesses_hyper_scale_is_data_derived(mock_data_empirical):
    """
    The hyper_scale guess now comes from a MAD-based estimator, not a
    hard-coded constant.  For mock_data_empirical the library deviations
    have MAD=1, so hyper_scale = 1.4826.
    """
    guesses = get_guesses("x", mock_data_empirical)
    assert guesses["x_hyper_scale"] == pytest.approx(1.4826, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests: get_guesses – per-genotype offsets (per-group scale, not constant)
# ---------------------------------------------------------------------------

def test_get_guesses_offsets_zero_when_at_group_median(mock_data_with_spiked):
    """
    When every genotype's ln_cfu equals its group median exactly, all
    non-centred offsets are 0 regardless of the scale used.
    """
    guesses = get_guesses("x", mock_data_with_spiked)
    assert jnp.allclose(guesses["x_offset"], 0.0)


def test_get_guesses_offsets_use_per_group_scale(mock_data_empirical):
    """
    Offsets = (per_rep_cond_geno - group_loc) / group_scale, where
    ``group_scale`` is the *per-group* MAD-based scale (NOT a global
    constant).  For mock_data_empirical:

      wt:    val=12, loc=12, scale=floor → diff=0  → offset=0
      spiked geno 1: val=10, loc=10.5, scale=0.7413 → -0.5/0.7413 = -1/1.4826
      spiked geno 2: val=11, loc=10.5, scale=0.7413 → +0.5/0.7413 = +1/1.4826
      lib geno 3:    val=8,  loc=8,    scale=1.4826 → 0
      lib geno 4:    val=7,  loc=8,    scale=1.4826 → -1/1.4826
      lib geno 5:    val=9,  loc=8,    scale=1.4826 → +1/1.4826
    """
    guesses  = get_guesses("x", mock_data_empirical)
    offsets  = np.array(guesses["x_offset"])   # (rep, cond_pre, geno)

    expected_unit = 1.0 / 1.4826  # ≈ 0.6745

    assert offsets[..., 0] == pytest.approx(0.0,            abs=1e-6)
    assert offsets[..., 1] == pytest.approx(-expected_unit, rel=1e-4)
    assert offsets[..., 2] == pytest.approx(+expected_unit, rel=1e-4)
    assert offsets[..., 3] == pytest.approx(0.0,            abs=1e-6)
    assert offsets[..., 4] == pytest.approx(-expected_unit, rel=1e-4)
    assert offsets[..., 5] == pytest.approx(+expected_unit, rel=1e-4)


def test_get_guesses_offsets_invariant_to_group_scaling():
    """
    Doubling the within-group spread of *one* subgroup must change only that
    subgroup's offset normalisation — it must not affect the other groups.
    Confirms that scales are computed and applied per-group, not globally.
    """
    num_replicate    = 1
    num_condition_pre = 1
    num_genotype     = 6
    batch_size       = num_genotype

    # Library: 8, 7, 9 (deviations -1/0/+1; MAD=1; scale=1.4826)
    # Spiked:  10, 11   (deviations -0.5/+0.5; MAD=0.5; scale=0.7413)
    # Wt:      12       (single value; floored)
    base = MockGrowthData(
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=jnp.arange(batch_size, dtype=jnp.int32),
        scale_vector=jnp.ones(batch_size, dtype=float),
        ln_cfu0_spiked_mask=jnp.array([False, True, True, False, False, False]),
        ln_cfu0_wt_mask=jnp.array([True, False, False, False, False, False]),
        ln_cfu=_make_ln_cfu(num_replicate, num_condition_pre, num_genotype,
                            [12.0, 10.0, 11.0, 8.0, 7.0, 9.0]),
        good_mask=np.ones((num_replicate, 2, num_condition_pre, 2, 1, 3, num_genotype), dtype=bool),
        map_ln_cfu0=jnp.arange(batch_size, dtype=jnp.int32),
    )

    # Variant: spread the spiked group twice as wide.  Library and wt
    # genotypes are unchanged.
    spread = MockGrowthData(
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=jnp.arange(batch_size, dtype=jnp.int32),
        scale_vector=jnp.ones(batch_size, dtype=float),
        ln_cfu0_spiked_mask=jnp.array([False, True, True, False, False, False]),
        ln_cfu0_wt_mask=jnp.array([True, False, False, False, False, False]),
        ln_cfu=_make_ln_cfu(num_replicate, num_condition_pre, num_genotype,
                            [12.0, 9.5, 11.5, 8.0, 7.0, 9.0]),  # spiked spread doubled
        good_mask=np.ones((num_replicate, 2, num_condition_pre, 2, 1, 3, num_genotype), dtype=bool),
        map_ln_cfu0=jnp.arange(batch_size, dtype=jnp.int32),
    )

    g_base   = get_guesses("x", base)
    g_spread = get_guesses("x", spread)

    base_offsets   = np.array(g_base["x_offset"])
    spread_offsets = np.array(g_spread["x_offset"])

    # Library and wt offsets are *identical* under the spiked-only change
    library_idx = np.array([3, 4, 5])
    assert np.allclose(base_offsets[..., library_idx],
                       spread_offsets[..., library_idx])
    assert np.allclose(base_offsets[..., 0], spread_offsets[..., 0])

    # Spiked offsets remain ±1/1.4826 because both the deviations and the
    # group scale doubled in lock-step (numerator and denominator change by
    # the same factor).
    expected_unit = 1.0 / 1.4826
    assert spread_offsets[..., 1] == pytest.approx(-expected_unit, rel=1e-4)
    assert spread_offsets[..., 2] == pytest.approx(+expected_unit, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests: get_guesses – fallback when a group is absent
# ---------------------------------------------------------------------------

def test_get_guesses_fallback_spiked_when_no_spiked_genotypes(mock_data):
    """
    When there are no spiked genotypes, spiked_loc falls back to the
    hard-coded default (12.0).
    """
    guesses = get_guesses("x", mock_data)
    assert guesses["x_spiked_loc"] == pytest.approx(12.0)


def test_get_guesses_fallback_wt_when_no_wt_genotype(mock_data):
    """
    When there is no wt genotype, wt_loc falls back to the
    hard-coded default (13.0).
    """
    guesses = get_guesses("x", mock_data)
    assert guesses["x_wt_loc"] == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# Tests: get_guesses – masked (invalid) observations
# ---------------------------------------------------------------------------

def test_get_guesses_ignores_masked_observations():
    """
    Observations excluded by good_mask (False) do not influence the estimate.
    Only valid observations contribute to the group medians and offsets.
    """
    num_replicate    = 2
    num_condition_pre = 2
    num_genotype     = 3
    batch_size       = 3

    ln_cfu_vals = [9.0, 9.0, 9.0]   # underlying true values
    ln_cfu    = _make_ln_cfu(num_replicate, num_condition_pre, num_genotype, ln_cfu_vals)

    # Corrupt the first time slice with extreme values; then mask it out
    ln_cfu_corrupted = ln_cfu.copy()
    ln_cfu_corrupted[:, 0, :, :, :, :, :] = 999.0

    good_mask = np.ones_like(ln_cfu_corrupted, dtype=bool)
    good_mask[:, 0, :, :, :, :, :] = False   # mask out the corrupted slice

    data = MockGrowthData(
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=jnp.arange(batch_size, dtype=jnp.int32),
        scale_vector=jnp.ones(batch_size, dtype=float),
        ln_cfu0_spiked_mask=jnp.zeros(num_genotype, dtype=bool),
        ln_cfu0_wt_mask=jnp.zeros(num_genotype, dtype=bool),
        ln_cfu=ln_cfu_corrupted,
        good_mask=good_mask,
        map_ln_cfu0=jnp.arange(batch_size, dtype=jnp.int32),
    )

    guesses = get_guesses("x", data)
    # hyper_loc should reflect the valid value (9.0), not the corrupted slice
    assert guesses["x_hyper_loc"] == pytest.approx(9.0)
    assert jnp.allclose(guesses["x_offset"], 0.0)


# ---------------------------------------------------------------------------
# Tests: define_model – logic and shapes
# ---------------------------------------------------------------------------

def test_define_model_logic_and_shapes(mock_data):
    """
    With all-library genotypes and zero offsets (each genotype at hyper_loc),
    every ln_cfu0 value should equal hyper_loc.
    """
    name   = "test_ln_cfu0"
    priors = get_priors()

    base_guesses  = get_guesses(name, mock_data)
    batch_guesses = base_guesses.copy()
    batch_guesses[f"{name}_offset"] = base_guesses[f"{name}_offset"][..., mock_data.batch_idx]

    substituted_model = substitute(define_model, data=batch_guesses)
    final_ln_cfu0     = substituted_model(name=name, data=mock_data, priors=priors)

    model_trace = trace(substituted_model).get_trace(name=name, data=mock_data, priors=priors)

    # Deterministic site shape: (rep, cond_pre, batch)
    ln_cfu0_site = model_trace[name]["value"]
    assert ln_cfu0_site.shape == (mock_data.num_replicate,
                                  mock_data.num_condition_pre,
                                  mock_data.batch_size)

    # All genotypes at hyper_loc (offsets == 0 and all library)
    hyper_loc = base_guesses[f"{name}_hyper_loc"]
    assert jnp.allclose(ln_cfu0_site, hyper_loc)

    # Expanded return shape: (rep, 1, cond_pre, 1, 1, 1, batch)
    assert final_ln_cfu0.shape == (mock_data.num_replicate, 1,
                                   mock_data.num_condition_pre,
                                   1, 1, 1, mock_data.batch_size)
    assert jnp.allclose(final_ln_cfu0, hyper_loc)


def test_define_model_spiked_genotypes(mock_data_with_spiked):
    """
    Spiked genotypes receive spiked_loc; library genotypes receive hyper_loc.
    """
    name   = "test_ln_cfu0_spiked"
    priors = get_priors()
    data   = mock_data_with_spiked

    base_guesses  = get_guesses(name, data)
    batch_guesses = base_guesses.copy()
    batch_guesses[f"{name}_offset"] = base_guesses[f"{name}_offset"][..., data.batch_idx]

    substituted_model = substitute(define_model, data=batch_guesses)
    model_trace = trace(substituted_model).get_trace(name=name, data=data, priors=priors)
    ln_cfu0_site = model_trace[name]["value"]

    hyper_loc  = float(base_guesses[f"{name}_hyper_loc"])
    spiked_loc = float(base_guesses[f"{name}_spiked_loc"])

    assert jnp.allclose(ln_cfu0_site[..., 0], spiked_loc)
    assert jnp.allclose(ln_cfu0_site[..., 1], spiked_loc)
    assert jnp.allclose(ln_cfu0_site[..., 2], hyper_loc)
    assert jnp.allclose(ln_cfu0_site[..., 5], hyper_loc)


def test_define_model_wt_genotype(mock_data_with_wt):
    """
    Wildtype genotype receives wt_loc; library genotypes receive hyper_loc.
    """
    name   = "test_ln_cfu0_wt"
    priors = get_priors()
    data   = mock_data_with_wt

    base_guesses  = get_guesses(name, data)
    batch_guesses = base_guesses.copy()
    batch_guesses[f"{name}_offset"] = base_guesses[f"{name}_offset"][..., data.batch_idx]

    substituted_model = substitute(define_model, data=batch_guesses)
    model_trace = trace(substituted_model).get_trace(name=name, data=data, priors=priors)
    ln_cfu0_site = model_trace[name]["value"]

    hyper_loc = float(base_guesses[f"{name}_hyper_loc"])
    wt_loc    = float(base_guesses[f"{name}_wt_loc"])

    assert jnp.allclose(ln_cfu0_site[..., 0], hyper_loc)
    assert jnp.allclose(ln_cfu0_site[..., 4], hyper_loc)
    assert jnp.allclose(ln_cfu0_site[..., 5], wt_loc)


def test_define_model_uses_per_genotype_scale_for_subgroups():
    """
    The wt and spiked genotypes apply the *fixed* prior scales
    (priors.ln_cfu0_wt_scale / ln_cfu0_spiked_scale) when reconstructing
    ln_cfu0 from the offsets — NOT the (sampled) library hyper_scale.
    Library genotypes still apply the library hyper_scale.
    """
    num_replicate    = 1
    num_condition_pre = 1
    num_genotype     = 3
    batch_size       = 3

    data = MockGrowthData(
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=jnp.arange(batch_size, dtype=jnp.int32),
        scale_vector=jnp.ones(batch_size, dtype=float),
        ln_cfu0_spiked_mask=jnp.array([False, True, False]),
        ln_cfu0_wt_mask=jnp.array([True, False, False]),
        ln_cfu=_make_ln_cfu(num_replicate, num_condition_pre, num_genotype,
                            [12.0, 10.0, 8.0]),
        good_mask=np.ones((num_replicate, 2, num_condition_pre, 2, 1, 3, num_genotype), dtype=bool),
        map_ln_cfu0=jnp.arange(batch_size, dtype=jnp.int32),
    )

    # Pick distinctive prior scales so each per-genotype scale is identifiable
    priors = ModelPriors(
        ln_cfu0_hyper_loc_loc=8.0,
        ln_cfu0_hyper_loc_scale=1.0,
        ln_cfu0_hyper_scale_loc=1.0,
        ln_cfu0_spiked_loc_loc=10.0,
        ln_cfu0_spiked_loc_scale=1.0,
        ln_cfu0_spiked_scale=2.0,    # ← fixed
        ln_cfu0_wt_loc_loc=12.0,
        ln_cfu0_wt_loc_scale=1.0,
        ln_cfu0_wt_scale=5.0,        # ← fixed (large to isolate wt)
    )

    name = "test_ln_cfu0_scales"

    # Substitute deterministic values: hyper_loc=8, spiked_loc=10, wt_loc=12,
    # hyper_scale=7 (large to isolate library), and offset=1 everywhere.
    subs = {
        f"{name}_hyper_loc":   jnp.array(8.0),
        f"{name}_hyper_scale": jnp.array(7.0),     # library hyper_scale
        f"{name}_spiked_loc":  jnp.array(10.0),
        f"{name}_wt_loc":      jnp.array(12.0),
        f"{name}_offset":      jnp.ones((num_replicate, num_condition_pre, batch_size)),
    }

    substituted_model = substitute(define_model, data=subs)
    model_trace = trace(substituted_model).get_trace(name=name, data=data, priors=priors)
    ln_cfu0_site = model_trace[name]["value"]    # (rep, cond_pre, geno)

    # geno 0 (wt):     12 + 1 * 5  = 17  (uses priors.ln_cfu0_wt_scale)
    # geno 1 (spiked): 10 + 1 * 2  = 12  (uses priors.ln_cfu0_spiked_scale)
    # geno 2 (lib):     8 + 1 * 7  = 15  (uses sampled hyper_scale)
    assert jnp.allclose(ln_cfu0_site[..., 0], 17.0)
    assert jnp.allclose(ln_cfu0_site[..., 1], 12.0)
    assert jnp.allclose(ln_cfu0_site[..., 2], 15.0)


def test_define_model_eliminates_shared_hyper_scale_for_subgroups():
    """
    Regression guard: changing the *library* hyper_scale must NOT change the
    reconstructed ln_cfu0 of wt or spiked genotypes.  The two groups now use
    fixed prior-supplied scales rather than sharing the library hyper_scale.
    """
    num_replicate    = 1
    num_condition_pre = 1
    num_genotype     = 3
    batch_size       = 3

    data = MockGrowthData(
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=jnp.arange(batch_size, dtype=jnp.int32),
        scale_vector=jnp.ones(batch_size, dtype=float),
        ln_cfu0_spiked_mask=jnp.array([False, True, False]),
        ln_cfu0_wt_mask=jnp.array([True, False, False]),
        ln_cfu=_make_ln_cfu(num_replicate, num_condition_pre, num_genotype,
                            [12.0, 10.0, 8.0]),
        good_mask=np.ones((num_replicate, 2, num_condition_pre, 2, 1, 3, num_genotype), dtype=bool),
        map_ln_cfu0=jnp.arange(batch_size, dtype=jnp.int32),
    )

    priors = ModelPriors(**{**get_hyperparameters(),
                            "ln_cfu0_wt_scale":     2.0,
                            "ln_cfu0_spiked_scale": 3.0})

    name = "test_ln_cfu0_invariance"

    def _trace_with_hyper_scale(hyper_scale_value):
        subs = {
            f"{name}_hyper_loc":   jnp.array(8.0),
            f"{name}_hyper_scale": jnp.array(hyper_scale_value),
            f"{name}_spiked_loc":  jnp.array(10.0),
            f"{name}_wt_loc":      jnp.array(12.0),
            f"{name}_offset":      jnp.ones((num_replicate, num_condition_pre, batch_size)),
        }
        m = substitute(define_model, data=subs)
        tr = trace(m).get_trace(name=name, data=data, priors=priors)
        return np.array(tr[name]["value"])

    site_a = _trace_with_hyper_scale(1.0)
    site_b = _trace_with_hyper_scale(50.0)   # vastly different library scale

    # wt and spiked outputs are unchanged when hyper_scale changes
    assert site_a[..., 0] == pytest.approx(site_b[..., 0])
    assert site_a[..., 1] == pytest.approx(site_b[..., 1])

    # Library output DID change with hyper_scale (sanity check)
    assert site_a[..., 2] != pytest.approx(site_b[..., 2])


# ---------------------------------------------------------------------------
# Tests: guide – logic and shapes
# ---------------------------------------------------------------------------

def test_guide_logic_and_shapes(mock_data):
    """
    Guide returns correct parameter shapes and a correctly expanded ln_cfu0.
    """
    name   = "test_ln_cfu0_guide"
    priors = get_priors()

    with seed(rng_seed=0):
        guide_trace   = trace(guide).get_trace(name=name, data=mock_data, priors=priors)
        final_ln_cfu0 = guide(name=name, data=mock_data, priors=priors)

    # Offset locs/scales cover ALL genotypes
    assert f"{name}_offset_locs" in guide_trace
    assert guide_trace[f"{name}_offset_locs"]["value"].shape == (
        mock_data.num_replicate, mock_data.num_condition_pre, mock_data.num_genotype)

    # Variational parameters for spiked/wt locs exist
    for suffix in ["spiked_loc_loc", "spiked_loc_scale", "spiked_loc",
                   "wt_loc_loc",     "wt_loc_scale",     "wt_loc"]:
        assert f"{name}_{suffix}" in guide_trace

    # Sampled offsets match BATCH size
    assert guide_trace[f"{name}_offset"]["value"].shape == (
        mock_data.num_replicate, mock_data.num_condition_pre, mock_data.batch_size)

    # Return shape: (rep, 1, cond_pre, 1, 1, 1, batch)
    assert final_ln_cfu0.shape == (mock_data.num_replicate, 1,
                                   mock_data.num_condition_pre,
                                   1, 1, 1, mock_data.batch_size)


def test_guide_does_not_introduce_subgroup_scale_params(mock_data_empirical):
    """
    The fixed wt/spiked scales come from the priors dataclass, not from
    sampled or learnable variational parameters.  The guide must not
    expose ``*_wt_scale`` or ``*_spiked_scale`` as ``pyro.param`` sites.
    """
    name   = "test_ln_cfu0_no_extra_params"
    priors = get_priors(data=mock_data_empirical)

    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(name=name, data=mock_data_empirical, priors=priors)

    forbidden = [f"{name}_wt_scale", f"{name}_spiked_scale"]
    for site in forbidden:
        assert site not in guide_trace, (
            f"Guide should not expose {site}; the scale comes from priors."
        )


# ---------------------------------------------------------------------------
# Tests: pinning – sanity / ModelPriors
# ---------------------------------------------------------------------------

def test_pinnable_suffixes():
    """The module exposes the expected pinnable suffix tuple.

    Only the *library* subgroup hyperpriors are pinnable; the spiked and
    wt subgroups are non-hierarchical after the Step 2 restructure.
    """
    assert set(_PINNABLE_SUFFIXES) == {"hyper_loc", "hyper_scale"}


def test_model_priors_default_pinned_is_empty_dict():
    """ModelPriors() with no `pinned` argument exposes an empty dict."""
    priors = ModelPriors(**get_hyperparameters())
    assert hasattr(priors, "pinned")
    assert priors.pinned == {}


def test_model_priors_accepts_pinned_dict():
    """ModelPriors accepts a `pinned` dict and exposes it on the instance."""
    pinned = {"hyper_loc": 8.0, "hyper_scale": 1.5}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    assert priors.pinned == pinned


# ---------------------------------------------------------------------------
# Tests: define_model under pinning
# ---------------------------------------------------------------------------

def test_define_model_unpinned_uses_sample_sites(mock_data):
    """Without any pinning, every pinnable suffix is a sample site."""
    name = "lc"
    priors = get_priors()  # pinned = {}

    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors
        )

    for suffix in _PINNABLE_SUFFIXES:
        site_name = f"{name}_{suffix}"
        assert site_name in tr
        assert tr[site_name]["type"] == "sample"

    # Spiked/wt loc remain sampled (NOT pinnable)
    assert tr[f"{name}_spiked_loc"]["type"] == "sample"
    assert tr[f"{name}_wt_loc"]["type"] == "sample"


def test_define_model_pinned_replaces_with_deterministic(mock_data):
    """Pinned suffixes become deterministic sites with the pinned value."""
    name = "lc"
    pinned = {"hyper_loc": 8.0, "hyper_scale": 1.5}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    base_guesses = get_guesses(name, mock_data)
    # Drop pinned keys so substitute doesn't override deterministic values
    sub_data = {
        k: v for k, v in base_guesses.items()
        if not any(k == f"{name}_{s}" for s in pinned)
    }

    substituted = substitute(define_model, data=sub_data)
    with seed(rng_seed=0):
        tr = trace(substituted).get_trace(
            name=name, data=mock_data, priors=priors
        )

    for suffix, val in pinned.items():
        site_name = f"{name}_{suffix}"
        assert site_name in tr
        assert tr[site_name]["type"] == "deterministic"
        assert jnp.allclose(tr[site_name]["value"], val)

    # Non-pinnable subgroup locs remain sample sites
    assert tr[f"{name}_spiked_loc"]["type"] == "sample"
    assert tr[f"{name}_wt_loc"]["type"] == "sample"


def test_define_model_all_library_pinned(mock_data):
    """When both library hypers are pinned, only spiked/wt locs and the
    per-(rep,cond,geno) offset remain as sample sites."""
    name = "lc"
    pinned = {"hyper_loc": 8.0, "hyper_scale": 1.5}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    base_guesses = get_guesses(name, mock_data)
    sub_data = {
        k: v for k, v in base_guesses.items()
        if not any(k == f"{name}_{s}" for s in pinned)
    }
    substituted = substitute(define_model, data=sub_data)
    with seed(rng_seed=0):
        tr = trace(substituted).get_trace(
            name=name, data=mock_data, priors=priors
        )

    sample_sites = {n for n, s in tr.items() if s["type"] == "sample"}
    assert sample_sites == {
        f"{name}_spiked_loc", f"{name}_wt_loc", f"{name}_offset"
    }


# ---------------------------------------------------------------------------
# Tests: guide under pinning
# ---------------------------------------------------------------------------

def test_guide_pinned_drops_variational_params(mock_data):
    """
    A pinned hyper must not register any variational params or sample sites
    in the guide.  Unpinned hypers retain their full param + sample machinery.
    """
    name = "lc"
    pinned = {"hyper_loc": 8.0}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    # Pinned hyper has neither sample nor variational params
    assert f"{name}_hyper_loc" not in tr
    assert f"{name}_hyper_loc_loc" not in tr
    assert f"{name}_hyper_loc_scale" not in tr

    # Unpinned library hyper still exposes its variational params
    assert f"{name}_hyper_scale" in tr
    assert f"{name}_hyper_scale_loc" in tr
    assert f"{name}_hyper_scale_scale" in tr

    # Spiked/wt loc machinery untouched (never pinnable)
    for suffix in ("spiked_loc", "spiked_loc_loc", "spiked_loc_scale",
                   "wt_loc",     "wt_loc_loc",     "wt_loc_scale"):
        assert f"{name}_{suffix}" in tr

    # Per-(rep,cond,geno) offset machinery untouched
    assert f"{name}_offset" in tr
    assert f"{name}_offset_locs" in tr
    assert f"{name}_offset_scales" in tr


def test_guide_all_library_pinned(mock_data):
    """When both library hypers are pinned, library hyper machinery is gone
    but spiked/wt and offset machinery is intact."""
    name = "lc"
    pinned = {"hyper_loc": 8.0, "hyper_scale": 1.5}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    for suffix in _PINNABLE_SUFFIXES:
        assert f"{name}_{suffix}" not in tr
        assert f"{name}_{suffix}_loc" not in tr
        assert f"{name}_{suffix}_scale" not in tr

    # Spiked/wt locs remain
    for suffix in ("spiked_loc", "wt_loc"):
        assert f"{name}_{suffix}" in tr

    assert f"{name}_offset" in tr
    assert f"{name}_offset_locs" in tr
    assert f"{name}_offset_scales" in tr


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
    pinned = {"hyper_loc": 8.0, "hyper_scale": 1.5}
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
