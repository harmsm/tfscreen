import pytest
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
    get_priors
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


def test_get_priors():
    """get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.ln_cfu0_hyper_loc_loc == 6.0
    assert priors.ln_cfu0_spiked_loc_loc == 12.0


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


# ---------------------------------------------------------------------------
# Tests: get_guesses – per-genotype offsets
# ---------------------------------------------------------------------------

def test_get_guesses_offsets_zero_when_at_group_median(mock_data_with_spiked):
    """
    When every genotype's ln_cfu equals its group median exactly, all
    non-centred offsets are 0.
    """
    guesses = get_guesses("x", mock_data_with_spiked)
    assert jnp.allclose(guesses["x_offset"], 0.0)


def test_get_guesses_offsets_reflect_per_genotype_deviation(mock_data_empirical):
    """
    Offsets = (per_rep_cond_geno - group_loc) / DEFAULT_HYPER_SCALE (3.0).

    For mock_data_empirical (constant across rep/cond/time/conc):
      geno 0 (wt,     val=12.0, loc=12.0): offset = 0.0
      geno 1 (spiked, val=10.0, loc=10.5): offset = (10.0-10.5)/3.0 = -1/6
      geno 2 (spiked, val=11.0, loc=10.5): offset = (11.0-10.5)/3.0 = +1/6
      geno 3 (lib,    val= 8.0, loc= 8.0): offset = 0.0
      geno 4 (lib,    val= 7.0, loc= 8.0): offset = (7.0-8.0)/3.0  = -1/3
      geno 5 (lib,    val= 9.0, loc= 8.0): offset = (9.0-8.0)/3.0  = +1/3
    """
    guesses  = get_guesses("x", mock_data_empirical)
    offsets  = np.array(guesses["x_offset"])   # (rep, cond_pre, geno)

    assert offsets[..., 0] == pytest.approx(0.0,        abs=1e-6)
    assert offsets[..., 1] == pytest.approx(-1.0/6.0,   abs=1e-6)
    assert offsets[..., 2] == pytest.approx(+1.0/6.0,   abs=1e-6)
    assert offsets[..., 3] == pytest.approx(0.0,        abs=1e-6)
    assert offsets[..., 4] == pytest.approx(-1.0/3.0,   abs=1e-6)
    assert offsets[..., 5] == pytest.approx(+1.0/3.0,   abs=1e-6)


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
# Tests: define_model – logic and shapes (unchanged from original)
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


# ---------------------------------------------------------------------------
# Tests: guide – logic and shapes (unchanged from original)
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
