import pytest
import numpy as np
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.dk_geno.hierarchical import (
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
    "num_genotype",
    "batch_size",
    "batch_idx",
    "wt_indexes",
    "scale_vector",
    "map_genotype",
    "num_not_wt",
    "not_wt_mask",
    "ln_cfu",     # (rep, time, cond_pre, cond_sel, tname, tconc, geno)
    "t_sel",      # same shape
    "good_mask",  # same shape, bool
])


def _make_growth_tensors(num_rep, num_time, num_cond_pre, num_cond_sel,
                         num_tname, num_tconc, t_sel_values, per_geno_slopes,
                         per_geno_intercepts):
    """
    Build (ln_cfu, t_sel, good_mask) tensors with shape
    (rep, time, cond_pre, cond_sel, tname, tconc, geno).

    ``t_sel_values`` is a 1-D array of length num_time.
    ``per_geno_slopes[g]`` and ``per_geno_intercepts[g]`` define
    ln_cfu = intercept + slope * t_sel for genotype g, broadcast across
    all other dimensions.
    """
    shape = (num_rep, num_time, num_cond_pre, num_cond_sel,
             num_tname, num_tconc, len(per_geno_slopes))

    t_arr = np.zeros(shape)
    y_arr = np.zeros(shape)

    for i, t in enumerate(t_sel_values):
        t_arr[:, i, :, :, :, :, :] = t

    for g, (slope, intercept) in enumerate(zip(per_geno_slopes, per_geno_intercepts)):
        y_arr[..., g] = intercept + slope * t_arr[..., g]

    mask = np.ones(shape, dtype=bool)
    return y_arr, t_arr, mask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_data():
    """
    4 genotypes (0 = WT, 1-3 = mutants), batch_size=8, all genotypes have
    the same growth slope (0) so dk_geno = 0 and offsets fall back to the
    neutral value (~-0.824).
    """
    num_genotype = 4
    batch_size   = 8
    t_sel_values = [0.0, 5.0, 10.0]

    # All genotypes: slope=0, intercept=10 → constant ln_cfu → slope=0
    per_geno_slopes     = [0.0, 0.0, 0.0, 0.0]
    per_geno_intercepts = [10.0, 10.0, 10.0, 10.0]

    ln_cfu, t_sel, good_mask = _make_growth_tensors(
        num_rep=1, num_time=3, num_cond_pre=1, num_cond_sel=1,
        num_tname=1, num_tconc=1,
        t_sel_values=t_sel_values,
        per_geno_slopes=per_geno_slopes,
        per_geno_intercepts=per_geno_intercepts,
    )

    batch_idx = jnp.array([0, 1, 2, 3, 1, 0, 2, 3], dtype=jnp.int32)

    return MockGrowthData(
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=batch_idx,
        wt_indexes=jnp.array([0], dtype=jnp.int32),
        scale_vector=jnp.ones(batch_size, dtype=float),
        map_genotype=batch_idx,
        num_not_wt=3,
        not_wt_mask=jnp.array([False, True, True, True]),
        ln_cfu=ln_cfu,
        t_sel=t_sel,
        good_mask=good_mask,
    )


@pytest.fixture
def mock_data_empirical():
    """
    3 genotypes with distinct growth slopes for testing empirical estimation:
      geno 0: WT,   slope = 0.020
      geno 1: mut1, slope = 0.010  → dk_geno ≈ −0.010
      geno 2: mut2, slope = 0.025  → dk_geno ≈ +0.005

    1 replicate, 3 time points (t_sel = 0, 5, 10), 1 condition.
    """
    t_sel_values        = [0.0, 5.0, 10.0]
    per_geno_slopes     = [0.020, 0.010, 0.025]
    per_geno_intercepts = [10.0,  10.0,  10.0]

    ln_cfu, t_sel, good_mask = _make_growth_tensors(
        num_rep=1, num_time=3, num_cond_pre=1, num_cond_sel=1,
        num_tname=1, num_tconc=1,
        t_sel_values=t_sel_values,
        per_geno_slopes=per_geno_slopes,
        per_geno_intercepts=per_geno_intercepts,
    )

    return MockGrowthData(
        num_genotype=3,
        batch_size=3,
        batch_idx=jnp.arange(3, dtype=jnp.int32),
        wt_indexes=jnp.array([0], dtype=jnp.int32),
        scale_vector=jnp.ones(3, dtype=float),
        map_genotype=jnp.arange(3, dtype=jnp.int32),
        num_not_wt=2,
        not_wt_mask=jnp.array([False, True, True]),
        ln_cfu=ln_cfu,
        t_sel=t_sel,
        good_mask=good_mask,
    )


# ---------------------------------------------------------------------------
# Helper: expected offset for a given dk_geno
# ---------------------------------------------------------------------------

def _expected_offset(dk_geno,
                     hyper_loc=-3.5, hyper_scale=0.5, shift=0.02):
    arg = np.clip(shift - dk_geno, 1e-6, None)
    return (np.log(arg) - hyper_loc) / hyper_scale


# ---------------------------------------------------------------------------
# Tests: get_hyperparameters / get_priors
# ---------------------------------------------------------------------------

def test_get_hyperparameters():
    """get_hyperparameters returns the correct keys and default values."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "dk_geno_hyper_loc_loc" in params
    assert params["dk_geno_hyper_loc_loc"] == -3.5


def test_get_priors():
    """get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.dk_geno_hyper_loc_loc == -3.5
    assert priors.dk_geno_hyper_shift_loc == 0.02


# ---------------------------------------------------------------------------
# Tests: get_guesses – structure
# ---------------------------------------------------------------------------

def test_get_guesses_keys_and_offset_shape(mock_data):
    """get_guesses returns all required keys with the correct offset shape."""
    name    = "test_dk"
    guesses = get_guesses(name, mock_data)

    assert isinstance(guesses, dict)
    for key in [f"{name}_hyper_loc", f"{name}_hyper_scale",
                f"{name}_shift", f"{name}_offset"]:
        assert key in guesses

    assert guesses[f"{name}_offset"].shape == (mock_data.num_genotype,)


def test_get_guesses_hyperparams_unchanged(mock_data):
    """Hyper-level parameters keep their default values."""
    guesses = get_guesses("x", mock_data)
    assert guesses["x_hyper_loc"]   == pytest.approx(-3.5)
    assert guesses["x_hyper_scale"] == pytest.approx(0.5)
    assert guesses["x_shift"]       == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Tests: get_guesses – neutral (dk_geno = 0) case
# ---------------------------------------------------------------------------

def test_get_guesses_neutral_offset_when_all_same_slope(mock_data):
    """
    When all genotypes have the same slope, dk_geno = 0 for all, and each
    offset equals the value that maps to dk_geno = 0.
    """
    guesses = get_guesses("x", mock_data)
    offsets = np.array(guesses["x_offset"])

    expected = _expected_offset(0.0)   # ≈ -0.8240
    assert offsets == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: get_guesses – empirical estimation
# ---------------------------------------------------------------------------

def test_get_guesses_empirical_wt_offset(mock_data_empirical):
    """
    WT genotype has dk_geno = 0 → neutral offset.
    """
    guesses = get_guesses("x", mock_data_empirical)
    offsets = np.array(guesses["x_offset"])

    expected_wt = _expected_offset(0.0)
    assert offsets[0] == pytest.approx(expected_wt, abs=1e-4)


def test_get_guesses_empirical_mut1_offset(mock_data_empirical):
    """
    mut1 has slope 0.010 vs WT 0.020 → dk_geno ≈ −0.010.
    offset should equal (log(0.02 − (−0.010)) + 3.5) / 0.5.
    """
    guesses = get_guesses("x", mock_data_empirical)
    offsets = np.array(guesses["x_offset"])

    expected = _expected_offset(-0.010)
    assert offsets[1] == pytest.approx(expected, abs=1e-4)


def test_get_guesses_empirical_mut2_offset(mock_data_empirical):
    """
    mut2 has slope 0.025 vs WT 0.020 → dk_geno ≈ +0.005.
    offset should equal (log(0.02 − 0.005) + 3.5) / 0.5.
    """
    guesses = get_guesses("x", mock_data_empirical)
    offsets = np.array(guesses["x_offset"])

    expected = _expected_offset(0.005)
    assert offsets[2] == pytest.approx(expected, abs=1e-4)


def test_get_guesses_offsets_ordered(mock_data_empirical):
    """
    Mutant with lower slope (mut1) gets offset closer to 0 (less negative)
    than neutral, while mutant with higher slope (mut2) gets a more negative
    offset (penalised less by the log-normal prior).

    Specifically: offset[mut2] < offset[wt] < offset[mut1].
    """
    guesses = get_guesses("x", mock_data_empirical)
    offsets = np.array(guesses["x_offset"])

    # mut1 (dk_geno=-0.01): lognormal_arg = 0.03 > 0.02 → log bigger → offset bigger (less negative)
    # mut2 (dk_geno=+0.005): lognormal_arg = 0.015 < 0.02 → log smaller → offset smaller (more negative)
    assert offsets[2] < offsets[0] < offsets[1]


# ---------------------------------------------------------------------------
# Tests: get_guesses – masking and edge cases
# ---------------------------------------------------------------------------

def test_get_guesses_ignores_masked_observations():
    """
    Masked observations (good_mask=False) are excluded from slope estimation.
    Corrupting them with extreme values should not change the result.
    """
    t_sel_values        = [0.0, 5.0, 10.0]
    per_geno_slopes     = [0.020, 0.010, 0.025]
    per_geno_intercepts = [10.0,  10.0,  10.0]

    ln_cfu_clean, t_sel, mask_clean = _make_growth_tensors(
        1, 3, 1, 1, 1, 1, t_sel_values, per_geno_slopes, per_geno_intercepts)

    # Corrupt the first time slice; then mask it out
    ln_cfu_corrupt = ln_cfu_clean.copy()
    ln_cfu_corrupt[:, 0, ...] = 9999.0
    mask_corrupt = mask_clean.copy()
    mask_corrupt[:, 0, ...] = False

    data_clean = MockGrowthData(
        num_genotype=3, batch_size=3,
        batch_idx=jnp.arange(3, dtype=jnp.int32),
        wt_indexes=jnp.array([0], dtype=jnp.int32),
        scale_vector=jnp.ones(3, dtype=float),
        map_genotype=jnp.arange(3, dtype=jnp.int32),
        num_not_wt=2, not_wt_mask=jnp.array([False, True, True]),
        ln_cfu=ln_cfu_clean, t_sel=t_sel, good_mask=mask_clean,
    )
    data_corrupt = MockGrowthData(
        num_genotype=3, batch_size=3,
        batch_idx=jnp.arange(3, dtype=jnp.int32),
        wt_indexes=jnp.array([0], dtype=jnp.int32),
        scale_vector=jnp.ones(3, dtype=float),
        map_genotype=jnp.arange(3, dtype=jnp.int32),
        num_not_wt=2, not_wt_mask=jnp.array([False, True, True]),
        ln_cfu=ln_cfu_corrupt, t_sel=t_sel, good_mask=mask_corrupt,
    )

    offsets_clean   = np.array(get_guesses("x", data_clean)["x_offset"])
    offsets_corrupt = np.array(get_guesses("x", data_corrupt)["x_offset"])
    assert offsets_clean == pytest.approx(offsets_corrupt, abs=1e-4)


def test_get_guesses_fallback_when_no_time_variation():
    """
    If all t_sel values are identical (no time variation), slopes cannot be
    computed and offsets fall back to the neutral value for all genotypes.
    """
    # All t_sel = 0 → var_t = 0 → slope = NaN → dk_geno = 0 → neutral offset
    ln_cfu_vals = [10.0, 10.0, 10.0]
    t_sel_values = [0.0, 0.0, 0.0]   # no time variation

    ln_cfu, t_sel, mask = _make_growth_tensors(
        1, 3, 1, 1, 1, 1, t_sel_values, [0.0, 0.0, 0.0], ln_cfu_vals)

    data = MockGrowthData(
        num_genotype=3, batch_size=3,
        batch_idx=jnp.arange(3, dtype=jnp.int32),
        wt_indexes=jnp.array([0], dtype=jnp.int32),
        scale_vector=jnp.ones(3, dtype=float),
        map_genotype=jnp.arange(3, dtype=jnp.int32),
        num_not_wt=2, not_wt_mask=jnp.array([False, True, True]),
        ln_cfu=ln_cfu, t_sel=t_sel, good_mask=mask,
    )

    offsets = np.array(get_guesses("x", data)["x_offset"])
    expected = _expected_offset(0.0)
    assert offsets == pytest.approx(expected, abs=1e-5)


def test_get_guesses_multi_rep_and_cond_averaged(mock_data_empirical):
    """
    With multiple replicates/conditions that agree on slopes, the result
    should equal the single-rep/cond case.
    """
    # Build a 2-rep version of mock_data_empirical
    t_sel_values        = [0.0, 5.0, 10.0]
    per_geno_slopes     = [0.020, 0.010, 0.025]
    per_geno_intercepts = [10.0,  10.0,  10.0]

    ln_cfu, t_sel, mask = _make_growth_tensors(
        2, 3, 2, 1, 1, 1, t_sel_values, per_geno_slopes, per_geno_intercepts)

    data_multi = MockGrowthData(
        num_genotype=3, batch_size=3,
        batch_idx=jnp.arange(3, dtype=jnp.int32),
        wt_indexes=jnp.array([0], dtype=jnp.int32),
        scale_vector=jnp.ones(3, dtype=float),
        map_genotype=jnp.arange(3, dtype=jnp.int32),
        num_not_wt=2, not_wt_mask=jnp.array([False, True, True]),
        ln_cfu=ln_cfu, t_sel=t_sel, good_mask=mask,
    )

    offsets_single = np.array(get_guesses("x", mock_data_empirical)["x_offset"])
    offsets_multi  = np.array(get_guesses("x", data_multi)["x_offset"])
    assert offsets_single == pytest.approx(offsets_multi, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: define_model – logic and shapes (unchanged from original)
# ---------------------------------------------------------------------------

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model using handlers.
    With all genotypes having dk_geno = 0, mutant values should be ≈ 0.
    """
    name   = "test_dk"
    priors = get_priors()

    base_guesses  = get_guesses(name, mock_data)
    batch_guesses = base_guesses.copy()

    genotype_offsets         = base_guesses[f"{name}_offset"]
    batch_guesses[f"{name}_offset"] = genotype_offsets[mock_data.batch_idx]

    substituted_model = substitute(define_model, data=batch_guesses)
    final_dk_geno     = substituted_model(name=name, data=mock_data, priors=priors)

    model_trace = trace(substituted_model).get_trace(
        name=name, data=mock_data, priors=priors)

    # Deterministic site shape: (batch_size,)
    assert name in model_trace
    dk_geno_site = model_trace[name]["value"]
    assert dk_geno_site.shape == (mock_data.batch_size,)

    # WT entries must be exactly 0
    wt_in_batch = jnp.where(jnp.isin(mock_data.batch_idx, mock_data.wt_indexes))[0]
    assert jnp.all(dk_geno_site[wt_in_batch] == 0.0)

    # Mutant entries: compute expected value from the guesses
    mut_in_batch = jnp.where(~jnp.isin(mock_data.batch_idx, mock_data.wt_indexes))[0]
    hyper_loc    = base_guesses[f"{name}_hyper_loc"]
    hyper_scale  = base_guesses[f"{name}_hyper_scale"]
    hyper_shift  = base_guesses[f"{name}_shift"]
    offset_val   = genotype_offsets[1]   # any non-WT genotype; all offsets equal here

    expected_lognormal = jnp.clip(jnp.exp(hyper_loc + offset_val * hyper_scale), max=1e30)
    expected_mut_val   = hyper_shift - expected_lognormal

    assert jnp.allclose(dk_geno_site[mut_in_batch], expected_mut_val)
    assert jnp.allclose(dk_geno_site[mut_in_batch], 0.0, atol=1e-4)

    # Expanded return shape: (1, 1, 1, 1, 1, 1, batch_size)
    assert final_dk_geno.shape == (1, 1, 1, 1, 1, 1, mock_data.batch_size)


def test_guide_logic_and_shapes(mock_data):
    """Guide function returns the correct expanded shape."""
    name   = "test_dk_guide"
    priors = get_priors()

    with seed(rng_seed=0):
        final_dk_geno = guide(name=name, data=mock_data, priors=priors)

    assert final_dk_geno.shape == (1, 1, 1, 1, 1, 1, mock_data.batch_size)
