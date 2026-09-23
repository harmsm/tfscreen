import pytest
import pandas as pd
import numpy as np
import numpy.ma as ma
from numpy.random import Generator
from scipy.stats import gmean

from pathlib import Path
from unittest.mock import MagicMock

# Import your module
from tfscreen.simulate.selection_experiment import (
    _check_dict_number,
    _check_cf,
    _check_lib_spec,
    _sim_plasmid_probabilities,
    _sim_index_hop,
    _sim_transform,
    _sim_transform_and_mix,
    _sim_growth,
    _cell_kt,
    _check_cell_components,
    SIMULATE_KNOWN_KEYS,
    _sim_sequencing,
    _calc_genotype_cfu0,
    _plasmid_shares,
    _compute_kt,
    _simulate_library_group,
    selection_experiment
)

# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def rng() -> Generator:
    """Provides a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)

@pytest.fixture
def base_config() -> dict:
    return {
        "prob_index_hop": 0.01,
        "lib_assembly_skew_sigma": 0.5,
        "transformation_poisson_lambda": 0.8,
        "tube_noise_sigma": 0.002,
        "seed": 42,
        "cfu0": 1.0e7,
        "total_num_reads": 5_000_000,
        "transform_sizes": {
            "libA": 100_000,
            "libB": 150_000,
        },
        "library_mixture": {
            "libA": 0.6,
            "libB": 0.4,
        },
        "growth": {
            "M9": {"model": "linear", "b": 0.09, "m": 0.0},
            "M9 + Ab": {"model": "linear", "b": 0.5, "m": 0.6},
        },
        "condition_selector": [
            "titrant_name", "titrant_conc",
            "condition_pre", "t_pre",
            "condition_sel", "t_sel"
        ],
        "library_selector": ["replicate", "library"],
    }

@pytest.fixture
def base_library_df() -> pd.DataFrame:
    data = {
        "library_origin": ["libA", "libA", "libB", "libB", "libB"],
        "genotype":       ["A1V", "A2V", "A2V", "A3V", "A4V"],
        "weight":         [1.0, 0.8, 0.5, 1.2, 0.9],
    }
    return pd.DataFrame(data)

@pytest.fixture
def base_phenotype_df(base_library_df: pd.DataFrame) -> pd.DataFrame:
    """
    Phenotypes consistent with base_config["growth"]: k = b + m*theta + dk_geno.
    """
    genotypes = sorted(list(pd.unique(base_library_df["genotype"])))
    conditions = [
        {"titrant_name": "IPTG", "titrant_conc": 10.0},
        {"titrant_name": "IPTG", "titrant_conc": 100.0},
    ]
    theta_by_geno = {"A1V": (0.2, 0.1), "A2V": (0.9, 0.5),
                     "A3V": (0.6, 0.3), "A4V": (0.4, 0.05)}
    dk_by_geno = {"A1V": 0.01, "A2V": -0.02, "A3V": 0.0, "A4V": 0.005}

    records = []
    for g in genotypes:
        for i, c in enumerate(conditions):
            theta = theta_by_geno[g][i]
            dk = dk_by_geno[g]
            record = {
                "genotype": g,
                "replicate": 1,
                "library": "test_selection",
                "condition_pre": "M9",
                "t_pre": 4.0,
                "condition_sel": "M9 + Ab",
                "t_sel": 18.0,
                "k_pre": 0.09 + dk,
                "k_sel": 0.5 + 0.6*theta + dk,
                "dk_geno": dk,
                "activity": 1.0,
                "theta": theta,
                **c
            }
            records.append(record)

    return pd.DataFrame(records)

# =============================================================================
# Unit Tests
# =============================================================================

# ----------------------------------------------------------------------------
# test _check_dict_number
# ----------------------------------------------------------------------------

# Test cases for successful validation
@pytest.mark.parametrize("input_dict, key, kwargs, expected_value", [
    # Test basic float casting and validation
    ({"value": 5}, "value", {}, 5.0),
    # Test integer casting
    ({"value": 5.5}, "value", {"cast_type": int}, 5),
    # Test inclusive bounds
    ({"value": 0}, "value", {"min_allowed": 0}, 0.0),
    ({"value": 10}, "value", {"max_allowed": 10}, 10.0),
    # Test exclusive bounds
    ({"value": 0.1}, "value", {"min_allowed": 0, "inclusive_min": False}, 0.1),
    ({"value": 9.9}, "value", {"max_allowed": 10, "inclusive_max": False}, 9.9),
    # Test allow_none=True when key is missing
    ({}, "value", {"allow_none": True}, None),
    # Test allow_none=True when value is None
    ({"value": None}, "value", {"allow_none": True}, None),
])
def test_check_dict_number_success(input_dict, key, kwargs, expected_value):
    """
    Tests that _check_dict_number correctly validates and casts values
    under various success conditions.
    """
    result_dict = _check_dict_number(key, input_dict, **kwargs)
    assert result_dict[key] == expected_value

# Test cases for expected failures
@pytest.mark.parametrize("input_dict, key, kwargs, match_error", [
    # Test missing key
    ({}, "value", {}, "Required key 'value' not found"),
    # Test non-scalar value
    ({"value": [1, 2]}, "value", {}, "Value must be a scalar"),
    # Test value that cannot be cast
    ({"value": "abc"}, "value", {}, "Could not process key 'value'"),
    # Test below inclusive minimum
    ({"value": -1}, "value", {"min_allowed": 0}, "must be >= 0"),
    # Test at exclusive minimum
    ({"value": 0}, "value", {"min_allowed": 0, "inclusive_min": False}, "must be > 0"),
    # Test above inclusive maximum
    ({"value": 11}, "value", {"max_allowed": 10}, "must be <= 10"),
    # Test at exclusive maximum
    ({"value": 10}, "value", {"max_allowed": 10, "inclusive_max": False}, "must be < 10"),
])
def test_check_dict_number_failures(input_dict, key, kwargs, match_error):
    """
    Tests that _check_dict_number correctly raises ValueError for invalid inputs.
    """
    with pytest.raises(ValueError, match=match_error):
        _check_dict_number(key, input_dict, **kwargs)


# ----------------------------------------------------------------------------
# test _check_cf
# ----------------------------------------------------------------------------

def test_check_cf_success(base_config: dict):
    """
    Tests that a valid configuration dictionary passes validation.
    """
    # Make a copy to avoid modifying the fixture for other tests
    config_copy = base_config.copy()
    
    # This should run without raising an exception
    validated_cf = _check_cf(config_copy)
    
    # Check that a key value remains correct
    assert validated_cf["cfu0"] == 1.0e7
    # Check that default selectors were added
    assert "condition_selector" in validated_cf
    assert isinstance(validated_cf["condition_selector"], list)

def test_check_cf_loads_from_path(mocker, base_config: dict):
    """
    Tests that _check_cf correctly calls the YAML loader when given a path.
    """
    # Mock the external function that loads a YAML file
    mock_loader = mocker.patch(
        "tfscreen.simulate.selection_experiment.read_yaml",
        return_value=base_config
    )
    
    dummy_path = Path("config.yaml")
    validated_cf = _check_cf(dummy_path)

    # Assert that the loader was called correctly
    mock_loader.assert_called_once_with(dummy_path)
    # Assert that the loaded config was processed
    assert validated_cf["seed"] == 42

@pytest.mark.parametrize("key_to_pop, nested_key_to_pop, bad_value, match_error", [
    # Test required numerical key missing
    ("cfu0", None, None, "Required key 'cfu0' not found"),
    # Test required nested dictionary missing
    ("transform_sizes", None, None, "Configuration key 'transform_sizes' is invalid"),
    # Test invalid value in nested dictionary
    ("transform_sizes", "libA", "bad_string", "Could not process key 'libA'"),
    # Test selector with wrong type
    ("condition_selector", None, "not_a_list", "condition_selector must be a list"),
])
def test_check_cf_failures(base_config: dict, key_to_pop, nested_key_to_pop, bad_value, match_error):
    """
    Tests various failure modes for the configuration validation.
    """
    config_copy = base_config.copy()

    if key_to_pop and nested_key_to_pop is None:
        config_copy.pop(key_to_pop, None)
    
    if bad_value:
        if nested_key_to_pop:
            config_copy[key_to_pop][nested_key_to_pop] = bad_value
        else:
            config_copy[key_to_pop] = bad_value
            
    with pytest.raises(ValueError, match=match_error):
        _check_cf(config_copy)


# ----------------------------------------------------------------------------
# test _check_lib_spec
# ----------------------------------------------------------------------------

def test_check_lib_spec_success(base_config: dict, 
                                base_library_df: pd.DataFrame, 
                                base_phenotype_df: pd.DataFrame):
    """
    Tests that validation passes with consistent config and dataframes.
    """
    # Use copies to avoid modifying fixtures
    cf_copy = base_config.copy()
    lib_df_copy = base_library_df.copy()
    pheno_df_copy = base_phenotype_df.copy()

    # This should run without raising an exception
    _check_lib_spec(cf_copy, lib_df_copy, pheno_df_copy)
    assert True # Indicates successful run without errors

def test_check_cf_rejects_multi_plasmid_combine_fcn(base_config: dict):
    """The removed key fails with a message naming its replacements."""
    cf = dict(base_config)
    cf["multi_plasmid_combine_fcn"] = {"test_selection": "mean"}
    with pytest.raises(ValueError, match="congression_theta_rule"):
        _check_cf(cf)


def test_check_cf_congression_rule_defaults(base_config: dict):
    """Omitted congression rules default to max theta and dilution dk."""
    cf = _check_cf(dict(base_config))
    assert cf["congression_theta_rule"] == "max"
    assert cf["congression_dk_rule"] == "dilution"


@pytest.mark.parametrize("key", ["congression_theta_rule", "congression_dk_rule"])
def test_check_cf_bad_congression_rule(base_config: dict, key):
    cf = dict(base_config)
    cf[key] = "not_a_rule"
    with pytest.raises(ValueError, match=key):
        _check_cf(cf)


@pytest.mark.parametrize("modification_lambda, match_error", [
    # Genotype in library_df is missing from phenotype_df
    # FIX: Modify phenotype_df to remove a genotype that library_df has.
    (lambda cf, lib, pheno: pheno.drop(pheno[pheno.genotype == "A2V"].index, inplace=True), "missing from phenotype_df"),
    
    # library_df is missing a required column
    # FIX: Use inplace=True to modify the DataFrame directly.
    (lambda cf, lib, pheno: lib.drop(columns=["weight"], inplace=True), "missing required columns"),
    
    # Mismatch between keys in transform_sizes and library_mixture
    (lambda cf, lib, pheno: cf["transform_sizes"].pop("libA"), "must be identical"),
    
    # Keys in config are not a subset of library_origin values
    (lambda cf, lib, pheno: cf.update({"transform_sizes": {"libC": 1}, "library_mixture": {"libC": 1}}), "must be a subset"),
    
    # Ratios in library_mixture sum to zero
    (lambda cf, lib, pheno: cf.update({"library_mixture": {"libA": 0, "libB": 0}}), "cannot be zero"),

])
def test_check_lib_spec_failures(base_config: dict, 
                                 base_library_df: pd.DataFrame, 
                                 base_phenotype_df: pd.DataFrame,
                                 modification_lambda, match_error):
    """
    Tests that _check_lib_spec correctly fails on a variety of inconsistent inputs.
    """
    cf_copy = base_config.copy()
    lib_df_copy = base_library_df.copy()
    pheno_df_copy = base_phenotype_df.copy()
    
    # Apply the specific modification for the test case
    modification_lambda(cf_copy, lib_df_copy, pheno_df_copy)
    
    with pytest.raises(ValueError, match=match_error):
        _check_lib_spec(cf_copy, lib_df_copy, pheno_df_copy)

# ----------------------------------------------------------------------------
# test _sim_plasmid_probabilities
# ----------------------------------------------------------------------------

def test_sim_plasmid_probabilities_no_skew():
    """
    Tests that the function correctly normalizes frequencies without applying skew.
    """
    frequencies = np.array([10, 30, 60])
    expected_probs = np.array([0.1, 0.3, 0.6])
    
    result = _sim_plasmid_probabilities(frequencies, skew_sigma=None)
    
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected_probs)

def test_sim_plasmid_probabilities_with_skew(rng: Generator):
    """
    Tests that the function correctly applies skew using a seeded RNG,
    producing a deterministic, valid probability distribution.
    """
    frequencies = np.array([10, 30, 60])
    
    # Pre-calculated expected output with rng seeded to 42 and skew_sigma=0.5
    expected_skewed_probs = np.array([0.099706, 0.152702, 0.747591])
    
    result = _sim_plasmid_probabilities(frequencies, skew_sigma=0.5, rng=rng)
    
    np.testing.assert_allclose(result, expected_skewed_probs,rtol=1e-5)
    # The result must still be a valid probability distribution
    assert np.isclose(np.sum(result), 1.0)
    assert np.all(result >= 0)

@pytest.mark.parametrize("bad_frequencies", [
    np.array([-10, 50, 60]),   # Contains a negative number
    np.array([10, np.nan, 20]),# Contains NaN
    np.array([0, 0, 0]),       # Sums to zero
])
def test_sim_plasmid_probabilities_failures(bad_frequencies: np.ndarray):
    """
    Tests that the function correctly raises ValueError for invalid inputs.
    """
    with pytest.raises(ValueError):
        _sim_plasmid_probabilities(bad_frequencies)


# ----------------------------------------------------------------------------
# test _sim_index_hop
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("hop_prob", [None, 0.0])
def test_sim_index_hop_no_hopping(hop_prob):
    """
    Tests that no hopping occurs if prob is None or 0.
    It should return a copy of the original array.
    """
    counts = np.array([100, 200, 700])
    result = _sim_index_hop(counts, index_hop_prob=hop_prob)
    
    np.testing.assert_array_equal(result, counts)
    # Ensure it's a copy, not the same object
    assert id(result) != id(counts)

def test_sim_index_hop_deterministic(rng: Generator):
    """
    Tests the main hopping logic with a seeded RNG to ensure it's
    deterministic and conserves the total number of counts.
    """
    counts = np.array([1000, 2000, 7000])
    total_counts = np.sum(counts)
    
    # Pre-calculated result with rng seeded to 42 and 10% hopping
    expected_hopped_counts = np.array([1248, 2125, 6627])
    
    result = _sim_index_hop(counts, index_hop_prob=0.1, rng=rng)
    
    # The most important invariant: total counts must be conserved.
    assert np.sum(result) == total_counts
    np.testing.assert_array_equal(result, expected_hopped_counts)

@pytest.mark.parametrize("edge_case_counts", [
    np.array([], dtype=int),
    np.array([0, 0, 0]),
])
def test_sim_index_hop_edge_cases(edge_case_counts: np.ndarray):
    """
    Tests edge cases like empty arrays or all-zero counts.
    """
    total_counts = np.sum(edge_case_counts)
    result = _sim_index_hop(edge_case_counts, index_hop_prob=0.1)
    
    assert np.sum(result) == total_counts
    np.testing.assert_array_equal(result, edge_case_counts)



# ----------------------------------------------------------------------------
# test _sim_transform
# ----------------------------------------------------------------------------

def test_sim_transform_single_plasmid(rng: Generator):
    """
    Tests the case where each transformant receives exactly one plasmid.
    """
    probs = np.array([0.1, 0.2, 0.7])
    num_transformants = 5
    
    transformants, mask = _sim_transform(probs, num_transformants, rng=rng)
    
    # Expected output with a seeded rng
    expected_transformants = np.array([[2], [2], [2], [2], [0]])
    
    assert transformants.shape == (5, 1)
    assert mask.shape == (5, 1)
    # The mask should be all False, as every cell has one valid plasmid
    assert not np.any(mask)
    np.testing.assert_array_equal(transformants, expected_transformants)

def test_sim_transform_multi_plasmid(mocker, rng: Generator):
    """
    Tests the case where transformants receive a variable number of plasmids.
    This test mocks the output of `zero_truncated_poisson` to test the
    masking and array generation logic in isolation.
    """
    probs = np.array([0.5, 0.5])
    num_transformants = 3
    
    # Mock the poisson function to return a deterministic number of plasmids
    # for each of the 3 cells.
    mocked_num_plasmids = np.array([1, 3, 2])
    mocker.patch(
        "tfscreen.simulate.selection_experiment.zero_truncated_poisson",
        return_value=mocked_num_plasmids
    )
    
    transformants, mask = _sim_transform(
        probs, num_transformants, transformation_poisson_lambda=0.8, rng=rng
    )
    
    # With 3 cells and a max of 3 plasmids, the shape should be (3, 3)
    assert transformants.shape == (3, 3)
    assert mask.shape == (3, 3)
    
    # Check the mask logic specifically
    expected_mask = np.array([
        [False, True,  True],  # Cell 1 has 1 plasmid
        [False, False, False], # Cell 2 has 3 plasmids
        [False, False, True],  # Cell 3 has 2 plasmids
    ])
    np.testing.assert_array_equal(mask, expected_mask)
    
    # Check that the number of unmasked entries matches the mock
    assert np.sum(~mask) == np.sum(mocked_num_plasmids)

def test_sim_transform_zero_transformants(rng: Generator):
    """
    Tests the edge case where num_transformants is 0.
    """
    probs = np.array([0.1, 0.2, 0.7])
    num_transformants = 0
    
    # This should run without error
    transformants, mask = _sim_transform(
        probs, num_transformants, rng=rng
    )

    # Expect empty arrays as output
    assert transformants.shape == (0, 0)
    assert mask.shape == (0, 0)


# ----------------------------------------------------------------------------
# test _sim_transform_and_mix (UPDATED)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("groupby_key", [
    "library_origin",  
    ["library_origin"], 
])
def test_sim_transform_and_mix(mocker, base_library_df: pd.DataFrame, 
                               base_config: dict, rng: Generator, groupby_key):
    """
    Tests transformation mixing logic.
    UPDATED: Converts grouper to dict to match updated code signature.
    """
    # 1. ARRANGE
    df = base_library_df.copy()
    df["probs"] = 1/len(df)
    
    # FIX: Convert to dict, as the code now iterates over .items()
    lib_origin_dict = dict(list(df.groupby(groupby_key)))
    
    mock_return_libA = (np.ones((2, 2)), np.zeros((2, 2), dtype=bool))
    mock_return_libB = (np.ones((3, 1)), np.zeros((3, 1), dtype=bool))

    mock_sim_transform = mocker.patch(
        "tfscreen.simulate.selection_experiment._sim_transform",
        side_effect=[mock_return_libA, mock_return_libB]
    )
    # We don't necessarily need to mock vstack_padded unless strictly testing flow
    # but let's keep it to verify return value propagation
    mocker.patch(
        "tfscreen.simulate.selection_experiment.vstack_padded",
        return_value="mocked_array"
    )
    
    # 2. ACT
    transformants, mask, probs = _sim_transform_and_mix(
        lib_origin_dict, # Passed as dict
        base_config["transform_sizes"],
        base_config["library_mixture"],
        base_config["transformation_poisson_lambda"],
        rng
    )

    # 3. ASSERT
    assert mock_sim_transform.call_count == 2
    assert transformants == "mocked_array"
    assert mask == "mocked_array"
    
    # Each origin's total weight must equal library_mixture[origin],
    # independent of how many cells (transform_sizes) it was split across.
    weights_libA = np.full(2, 0.6 / 2)
    weights_libB = np.full(3, 0.4 / 3)
    expected_weights = np.concatenate([weights_libA, weights_libB])
    expected_probs = expected_weights / np.sum(expected_weights)

    np.testing.assert_allclose(probs, expected_probs)


def test_sim_transform_and_mix_invariant_to_transform_sizes(rng: Generator):
    """
    An origin's total contribution to the pooled library must depend only on
    library_mixture, not on transform_sizes -- regression test for a bug
    where per-cell weights weren't normalized by the number of transformants
    drawn, letting transform_sizes leak into the mixing ratio.
    """
    df = pd.DataFrame({
        "library_origin": ["A"] * 5 + ["B"] * 5,
        "genotype": [f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)],
    })
    df["probs"] = 0.2
    lib_origin_dict = dict(list(df.groupby("library_origin")))

    library_mixture = {"A": 1.0, "B": 1.0}

    for transform_sizes in (
        {"A": 1000, "B": 1000},
        {"A": 1000, "B": 10000},
        {"A": 1000, "B": 100000},
    ):
        _, _, probs = _sim_transform_and_mix(
            lib_origin_dict, transform_sizes, library_mixture, None, rng
        )
        n_a = transform_sizes["A"]
        np.testing.assert_allclose(probs[:n_a].sum(), 0.5, atol=1e-9)
        np.testing.assert_allclose(probs[n_a:].sum(), 0.5, atol=1e-9)


# ----------------------------------------------------------------------------
# test _sim_growth
# ----------------------------------------------------------------------------

def test_sim_growth():
    """Final CFU is initial CFU times exp(cell k*t)."""
    trans_freq = np.array([0.2, 0.3, 0.5])
    total_cfu0 = 1.0e6
    cell_kt = np.array([[0.1, 0.5],
                        [0.2, 0.6],
                        [0.3, 0.7]])

    expected_cfu = (total_cfu0 * trans_freq)[:, np.newaxis] * np.exp(cell_kt)
    result_cfu = _sim_growth(cell_kt, trans_freq, total_cfu0)

    np.testing.assert_allclose(result_cfu, expected_cfu)


# ----------------------------------------------------------------------------
# test _cell_kt
# ----------------------------------------------------------------------------

def _cell_kt_inputs(growth_transition=None):
    """
    Three genotypes, two conditions, linear growth k = b + m*A*theta + dk.
    Pre-growth has m = 0, so only selection depends on theta.
    """
    cf = {
        "growth": {"pre": {"model": "linear", "b": 0.1, "m": 0.0},
                   "sel": {"model": "linear", "b": 0.2, "m": 0.5}},
        "theta_rescale": "passthrough",
        "growth_transition": growth_transition,
        "congression_theta_rule": "max",
        "congression_dk_rule": "dilution",
    }
    theta = np.array([[0.1, 0.9],    # g0
                      [0.8, 0.2],    # g1
                      [0.5, 0.5]])   # g2
    activity = np.array([1.0, 0.5, 1.0])
    dk = np.array([0.0, -0.04, 0.02])
    condition_info = pd.DataFrame({"condition_pre": ["pre", "pre"],
                                   "condition_sel": ["sel", "sel"],
                                   "t_pre": [1.0, 1.0],
                                   "t_sel": [10.0, 10.0]})

    def kt(theta_, activity_, dk_):
        k_pre = 0.1 + dk_
        k_sel = 0.2 + 0.5*activity_*theta_ + dk_
        return k_pre*1.0 + k_sel*10.0

    genotype_vs_kt = np.array([[kt(theta[g, c], activity[g], dk[g])
                                for c in range(2)] for g in range(3)])
    return cf, theta, activity, dk, condition_info, genotype_vs_kt, kt


def test_cell_kt_single_plasmid_cells_match_genotypes():
    cf, theta, activity, dk, info, gkt, _ = _cell_kt_inputs()
    transformants = np.array([[0], [1], [2]])
    trans_mask = np.zeros((3, 1), dtype=bool)
    result = _cell_kt(transformants, trans_mask, gkt, theta, activity, dk,
                      info, np.zeros(2), cf)
    np.testing.assert_allclose(result, gkt)


def test_cell_kt_co_transformed_cell_uses_cell_physics():
    """
    A {g0, g1} cell: theta and activity from the higher-theta plasmid at each
    condition, dk diluted by share. Not any combination of the two
    genotypes' k*t.
    """
    cf, theta, activity, dk, info, gkt, kt = _cell_kt_inputs()
    transformants = np.array([[0, 1], [2, 0]])
    trans_mask = np.array([[False, False], [False, True]])
    tube_kt = np.array([0.01, -0.02])

    result = _cell_kt(transformants, trans_mask, gkt + tube_kt, theta,
                      activity, dk, info, tube_kt, cf)

    dk_cell = 0.5*dk[0] + 0.5*dk[1]
    # condition 0: g1 has the higher theta (0.8, activity 0.5)
    # condition 1: g0 has the higher theta (0.9, activity 1.0)
    expected_cell0 = [kt(0.8, 0.5, dk_cell) + tube_kt[0],
                      kt(0.9, 1.0, dk_cell) + tube_kt[1]]
    np.testing.assert_allclose(result[0], expected_cell0)
    # Single-plasmid cell keeps its genotype's k*t (tube noise included)
    np.testing.assert_allclose(result[1], gkt[2] + tube_kt)


def test_cell_kt_duplicate_genotype_matches_genotype():
    """A cell with two copies of one genotype grows like that genotype."""
    cf, theta, activity, dk, info, gkt, _ = _cell_kt_inputs()
    transformants = np.array([[1, 1]])
    trans_mask = np.zeros((1, 2), dtype=bool)
    result = _cell_kt(transformants, trans_mask, gkt, theta, activity, dk,
                      info, np.zeros(2), cf)
    np.testing.assert_allclose(result[0], gkt[1])


def test_cell_kt_applies_growth_transition():
    """Co-transformed cells go through the growth transition model too."""
    gt = [{"condition_pre": "pre", "model": "memory",
           "tau0": 3.0, "k1": 1.0, "k2": 1.0}]
    cf, theta, activity, dk, info, gkt, _ = _cell_kt_inputs(growth_transition=gt)

    # Genotype k*t with the transition model, via the shared path
    rows = []
    for g in range(3):
        for c in range(2):
            rows.append({"k_pre": 0.1 + dk[g],
                         "k_sel": 0.2 + 0.5*activity[g]*theta[g, c] + dk[g],
                         "t_pre": 1.0, "t_sel": 10.0, "theta": theta[g, c],
                         "condition_pre": "pre"})
    gkt = _compute_kt(pd.DataFrame(rows), gt).reshape(3, 2)

    # A {g2, g2} cell must reproduce g2's transition-model k*t exactly.
    result = _cell_kt(np.array([[2, 2]]), np.zeros((1, 2), dtype=bool), gkt,
                      theta, activity, dk, info, np.zeros(2), cf)
    np.testing.assert_allclose(result[0], gkt[2])


def test_cell_kt_nan_theta_gives_zero_growth():
    cf, theta, activity, dk, info, gkt, _ = _cell_kt_inputs()
    theta = theta.copy()
    theta[1, 0] = np.nan
    result = _cell_kt(np.array([[0, 1]]), np.zeros((1, 2), dtype=bool), gkt,
                      theta, activity, dk, info, np.zeros(2), cf)
    assert result[0, 0] == 0.0
    assert np.isfinite(result[0, 1]) and result[0, 1] != 0.0


def test_cell_kt_no_cells():
    cf, theta, activity, dk, info, gkt, _ = _cell_kt_inputs()
    result = _cell_kt(np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=bool),
                      gkt, theta, activity, dk, info, np.zeros(2), cf)
    assert result.shape == (0, 2)


# ----------------------------------------------------------------------------
# test _check_cell_components
# ----------------------------------------------------------------------------

def test_check_cell_components_passes(base_config, base_phenotype_df):
    _check_cell_components(_check_cf(dict(base_config)), base_phenotype_df)


def test_check_cell_components_needs_growth(base_config, base_phenotype_df):
    cf = _check_cf(dict(base_config))
    cf.pop("growth")
    with pytest.raises(ValueError, match="'growth' block"):
        _check_cell_components(cf, base_phenotype_df)


def test_check_cell_components_needs_columns(base_config, base_phenotype_df):
    cf = _check_cf(dict(base_config))
    with pytest.raises(ValueError, match="activity"):
        _check_cell_components(cf, base_phenotype_df.drop(columns=["activity"]))


def test_check_cell_components_mismatch(base_config, base_phenotype_df):
    cf = _check_cf(dict(base_config))
    df = base_phenotype_df.copy()
    df.loc[0, "k_sel"] += 0.1
    with pytest.raises(ValueError, match="k_sel"):
        _check_cell_components(cf, df)


# ----------------------------------------------------------------------------
# test _sim_sequencing
# ----------------------------------------------------------------------------

def test_sim_sequencing_deterministic(rng: Generator):
    """
    Tests the main sequencing logic with a seeded RNG for a deterministic outcome.
    """
    # 2 cells: cell 0 has geno0, cell 1 has A1V
    transformants = np.array([[0], [1]])
    trans_mask = np.array([[False], [False]])
    num_genotypes = 2
    reads_per_sample = 1000

    # Final CFU: In condition 0, cell 1 is twice as abundant as cell 0.
    trans_cfu = np.array([[100], [200]]) # Shape is (num_cells, num_conditions)

    # In this case, geno_probs for condition 0 will be [100, 200] -> [1/3, 2/3]
    # We can pre-calculate the expected counts from rng.choice with these probs.
    expected_counts = np.array([[346], [654]])

    result = _sim_sequencing(
        transformants, trans_mask, trans_cfu, num_genotypes, reads_per_sample, rng
    )

    # Total reads must be conserved
    assert np.sum(result) == reads_per_sample
    np.testing.assert_array_equal(result, expected_counts)

def test_sim_sequencing_zero_abundance_condition(rng: Generator):
    """
    Tests that a condition with zero total CFU results in zero reads.
    """
    transformants = np.array([[0], [1]])
    trans_mask = np.array([[False], [False]])
    num_genotypes = 2
    reads_per_sample = 1000
    
    # Condition 0 has cells, Condition 1 is a lethal condition (all zeros)
    trans_cfu = np.array([
        [100, 0],
        [200, 0]
    ])

    result = _sim_sequencing(
        transformants, trans_mask, trans_cfu, num_genotypes, reads_per_sample, rng
    )

    # First condition should have reads
    assert np.sum(result[:, 0]) == reads_per_sample
    # Second condition should have zero reads
    assert np.sum(result[:, 1]) == 0

def test_sim_sequencing_single_survivor(rng: Generator):
    """
    Tests that if only one genotype survives, it gets all the reads.
    """
    transformants = np.array([[0], [1]])
    trans_mask = np.array([[False], [False]])
    num_genotypes = 2
    reads_per_sample = 1000
    
    # Only cell 0 (genotype 0) survives
    trans_cfu = np.array([[100], [0]])

    result = _sim_sequencing(
        transformants, trans_mask, trans_cfu, num_genotypes, reads_per_sample, rng
    )
    
    expected_counts = np.array([[reads_per_sample], [0]])
    np.testing.assert_array_equal(result, expected_counts)

def test_plasmid_shares():
    """Each valid slot is 1/n of its cell; masked slots are 0."""
    trans_mask = np.array([[False, True, True],
                           [False, False, True],
                           [False, False, False]])
    expected = np.array([[1.0, 0.0, 0.0],
                         [0.5, 0.5, 0.0],
                         [1/3, 1/3, 1/3]])
    np.testing.assert_allclose(_plasmid_shares(trans_mask), expected)
    np.testing.assert_allclose(_plasmid_shares(trans_mask).sum(axis=1), 1.0)


def test_plasmid_shares_empty():
    """An empty transformant array gives an empty share array."""
    result = _plasmid_shares(np.empty((0, 0), dtype=bool))
    assert result.shape == (0, 0)


def test_sim_sequencing_splits_cell_among_plasmids(mocker):
    """
    A co-transformed cell's abundance is split among its plasmids, so it
    contributes one cell's worth of reads in total.
    """
    # Cell 0: {g0}, abundance 100. Cell 1: {g0, g1}, abundance 100.
    transformants = np.array([[0, 0], [0, 1]])
    trans_mask = np.array([[False, True], [False, False]])
    trans_cfu = np.array([[100.0], [100.0]])

    # g0 = 100 + 50, g1 = 50 -> probabilities 0.75 / 0.25.
    rng = mocker.Mock()
    rng.choice.return_value = np.zeros(10, dtype=int)
    _sim_sequencing(transformants, trans_mask, trans_cfu, 2, 10, rng)

    np.testing.assert_allclose(rng.choice.call_args.kwargs["p"], [0.75, 0.25])


def test_sim_sequencing_duplicate_genotype_in_cell(mocker):
    """Two copies of one genotype in a cell give it the whole cell."""
    transformants = np.array([[0, 0], [1, 0]])
    trans_mask = np.array([[False, False], [False, True]])
    trans_cfu = np.array([[100.0], [100.0]])

    rng = mocker.Mock()
    rng.choice.return_value = np.zeros(10, dtype=int)
    _sim_sequencing(transformants, trans_mask, trans_cfu, 2, 10, rng)

    np.testing.assert_allclose(rng.choice.call_args.kwargs["p"], [0.5, 0.5])


# ----------------------------------------------------------------------------
# test _calc_genotype_cfu0
# ----------------------------------------------------------------------------

def test_calc_genotype_cfu0_single_plasmid():
    """
    Tests the simple case where each cell has one plasmid. The genotype
    frequencies should mirror the cell frequencies.
    """
    # 3 cells, each with a single, unique plasmid (genotypes 0, 1, 2)
    transformants = np.array([[0], [1], [2]])
    trans_mask = np.array([[False], [False], [False]])
    trans_freq = np.array([0.2, 0.3, 0.5])
    total_cfu0 = 1.0e7
    num_genotypes = 3
    
    expected_cfu0 = np.array([0.2 * total_cfu0, 
                              0.3 * total_cfu0, 
                              0.5 * total_cfu0])

    result = _calc_genotype_cfu0(
        transformants, trans_mask, trans_freq, total_cfu0, num_genotypes
    )
    
    np.testing.assert_allclose(result, expected_cfu0)

def test_calc_genotype_cfu0_multi_plasmid():
    """
    Tests the main logic where multiple cells can contain the same genotype,
    requiring correct aggregation of frequencies.
    """
    # 3 cells. Cell 0: {g0}, Cell 1: {g0, g1}, Cell 2: {g1}
    transformants = np.array([[0, 0], [0, 1], [1, 0]])
    trans_mask = np.array([[False, True], [False, False], [False, True]])
    trans_freq = np.array([0.2, 0.5, 0.3]) # Cell frequencies
    total_cfu0 = 1.0e7
    num_genotypes = 2

    # Expected logic: a cell splits its frequency among its plasmids.
    # Plasmids:    [g0,   g0,   g1,   g1]
    # From cells:  [c0,   c1,   c1,   c2]
    # Share:       [1,    1/2,  1/2,  1]
    # Weight:      [0.2,  0.25, 0.25, 0.3]
    #
    # g0 = 0.2 + 0.25 = 0.45
    # g1 = 0.25 + 0.3 = 0.55
    # Total = 1.0 (one cell's worth per cell)
    expected_cfu0 = np.array([0.45 * total_cfu0,
                              0.55 * total_cfu0])
    
    result = _calc_genotype_cfu0(
        transformants, trans_mask, trans_freq, total_cfu0, num_genotypes
    )
    
    np.testing.assert_allclose(result, expected_cfu0)
    
@pytest.mark.parametrize("lam", [None, 0.357, 1.5])
def test_calc_genotype_cfu0_matches_design_for_any_lambda(lam):
    """
    With plasmid copies shared within a cell, simulated genotype abundance
    matches the library design's pool_fraction whatever lambda is. Bulk
    origins are transformed with lambda; the spiked origin is monoclonal.
    """
    from tfscreen.genetics.library_design import expected_library_composition

    genotypes = ["wt", "A1V", "B2C", "S3T"]
    weights = {"bulk": [0.4, 0.3, 0.3, 0.0],
               "spiked": [0.5, 0.0, 0.0, 0.5]}
    library_df = pd.DataFrame([
        {"library_origin": origin, "genotype": g, "weight": w, "probs": w}
        for origin, ws in weights.items() for g, w in zip(genotypes, ws)
    ])
    library_mixture = {"bulk": 3.0, "spiked": 1.0}
    lib_origin_dict = dict(list(library_df.groupby(["library_origin"])))

    rng = np.random.default_rng(0)
    transformants, trans_mask, trans_freq = _sim_transform_and_mix(
        lib_origin_dict,
        {"bulk": 200_000, "spiked": 200_000},
        library_mixture,
        lam,
        rng)
    cfu0 = _calc_genotype_cfu0(transformants, trans_mask, trans_freq,
                               1.0, len(genotypes))

    expected = (expected_library_composition(library_df, library_mixture)
                .set_index("genotype")
                .loc[genotypes, "pool_fraction"]
                .to_numpy())
    np.testing.assert_allclose(cfu0, expected, atol=0.005)


def test_calc_genotype_cfu0_no_transformants():
    """
    Tests the edge case where there are no successful transformations.
    """
    transformants = np.empty((0, 0), dtype=int)
    trans_mask = np.empty((0, 0), dtype=bool)
    trans_freq = np.array([])
    total_cfu0 = 1.0e7
    num_genotypes = 5
    
    expected_cfu0 = np.zeros(num_genotypes)
    
    result = _calc_genotype_cfu0(
        transformants, trans_mask, trans_freq, total_cfu0, num_genotypes
    )
    
    np.testing.assert_array_equal(result, expected_cfu0)


# ----------------------------------------------------------------------------
# test _simulate_library_group (UPDATED)
# ----------------------------------------------------------------------------

def test_simulate_library_group_integration(base_config: dict, 
                                            base_library_df: pd.DataFrame, 
                                            base_phenotype_df: pd.DataFrame, 
                                            rng: Generator):
    """
    Integration test for _simulate_library_group.
    UPDATED: Passes lib_origin_dict and fixes column assertion.
    """
    # 1. ARRANGE
    sub_df = base_phenotype_df.copy()
    sub_df["kt"] = sub_df["t_sel"]*sub_df["k_sel"]
    
    lib_df = base_library_df.copy()
    lib_df["probs"] = lib_df.groupby("library_origin")["weight"].transform(
        lambda w: w / w.sum()
    )
    
    # FIX: Create dict instead of grouper
    lib_origin_dict = dict(list(lib_df.groupby("library_origin")))
    
    ordered_genotypes = pd.unique(lib_df["genotype"])
    num_conditions = sub_df.groupby(base_config.get("condition_selector", [])).ngroups
    total_reads = base_config["total_num_reads"]
    reads_per_sample = int(np.round(total_reads / num_conditions))
    index_offset = 100
    
    # 2. ACT
    sample_df, counts_df = _simulate_library_group(
        sub_df,
        index_offset,
        lib_origin_dict, # Passed as dict
        ordered_genotypes,
        reads_per_sample,
        _check_cf(dict(base_config)),
        rng
    )
    
    # 3. ASSERT
    assert isinstance(sample_df, pd.DataFrame)
    assert "sample" in sample_df.columns
    
    # FIX: Code produces 'sample_cfu', not 'cfu_per_mL'
    assert "sample_cfu" in sample_df.columns 
    
    assert sample_df.shape[0] == num_conditions 
    assert sample_df["sample"].min() == index_offset
    assert np.all(sample_df["sample_cfu"] > 0)
    
    # Counts checks
    assert isinstance(counts_df, pd.DataFrame)
    assert "sample" in counts_df.columns
    assert "counts" in counts_df.columns
    assert "ln_cfu_0" in counts_df.columns
    assert counts_df.shape[0] == sub_df.shape[0]
    assert np.all(counts_df["counts"] >= 0)
    
    expected_total_counts = num_conditions * reads_per_sample
    assert np.isclose(counts_df["counts"].sum(), expected_total_counts, rtol=1)
    assert np.any(counts_df["ln_cfu_0"] > -np.inf)

@pytest.fixture
def sparse_phenotype_df(base_phenotype_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a sparse version of the phenotype dataframe.
    
    This fixture takes the full, dense base_phenotype_df and removes several
    rows to simulate a dataset where not all genotypes are present in all
    conditions. This is the specific edge case we want to test.
    """
    df = base_phenotype_df.copy()
    
    # Drop a specific genotype ('A1V') from one condition (titrant_conc == 100.0)
    condition_to_drop = (df["genotype"] == "A1V") & (df["titrant_conc"] == 100.0)
    df = df[~condition_to_drop]
    
    # Drop another genotype ('A4V') entirely from the dataset
    df = df[df["genotype"] != "A4V"]
    
    return df

def test_simulate_library_group_handles_sparse_data(
    base_config: dict,
    base_library_df: pd.DataFrame,
    # Assuming sparse_phenotype_df fixture is defined as in your snippet
    base_phenotype_df: pd.DataFrame, 
    rng: Generator
):
    """
    Tests sparse data handling.
    UPDATED: Passes lib_origin_dict.
    """
    # Create sparse DF manually here if fixture isn't available in scope
    sparse_phenotype_df = base_phenotype_df.copy()
    condition_to_drop = (sparse_phenotype_df["genotype"] == "A1V") & (sparse_phenotype_df["titrant_conc"] == 100.0)
    sparse_phenotype_df = sparse_phenotype_df[~condition_to_drop]
    sparse_phenotype_df = sparse_phenotype_df[sparse_phenotype_df["genotype"] != "A4V"]

    # 1. ARRANGE
    sub_df = sparse_phenotype_df.copy()
    sub_df["kt"] = sub_df["t_sel"] * sub_df["k_sel"]

    lib_df = base_library_df.copy()
    lib_df["probs"] = lib_df.groupby("library_origin")["weight"].transform(
        lambda w: w / w.sum()
    )
    
    # FIX: Create dict
    lib_origin_dict = dict(list(lib_df.groupby("library_origin")))

    ordered_genotypes = np.sort(pd.unique(lib_df["genotype"]))
    
    num_conditions = sub_df.groupby(base_config["condition_selector"]).ngroups
    reads_per_sample = int(np.round(base_config["total_num_reads"] / num_conditions))
    
    # 2. ACT
    sample_df, counts_df = _simulate_library_group(
        sub_df=sub_df,
        index_offset=0,
        lib_origin_dict=lib_origin_dict, # Passed as dict
        ordered_genotypes=ordered_genotypes,
        reads_per_sample=reads_per_sample,
        cf=_check_cf(dict(base_config)),
        rng=rng
    )

    # 3. ASSERT
    assert counts_df.shape[0] == sub_df.shape[0]
    assert not counts_df["counts"].isna().any()
    assert counts_df["genotype"].nunique() == sub_df["genotype"].nunique()

# ----------------------------------------------------------------------------
# test selection_experiment (End-to-End)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# test _check_cf — growth_transition validation
# ----------------------------------------------------------------------------

def test_check_cf_growth_transition_absent(base_config: dict):
    """Absent growth_transition key should result in None, not an error."""
    cf = base_config.copy()
    cf.pop("growth_transition", None)
    result = _check_cf(cf)
    assert result["growth_transition"] is None


def test_check_cf_growth_transition_none(base_config: dict):
    """Explicit None for growth_transition should be accepted."""
    cf = base_config.copy()
    cf["growth_transition"] = None
    result = _check_cf(cf)
    assert result["growth_transition"] is None


def test_check_cf_growth_transition_instant(base_config: dict):
    """A single instant entry should pass validation without error."""
    cf = base_config.copy()
    cf["growth_transition"] = [{"condition_pre": "M9", "model": "instant"}]
    result = _check_cf(cf)
    assert len(result["growth_transition"]) == 1
    assert result["growth_transition"][0]["model"] == "instant"


def test_check_cf_growth_transition_memory(base_config: dict):
    """A valid memory entry should pass and cast tau0/k1/k2 to float."""
    cf = base_config.copy()
    cf["growth_transition"] = [{
        "condition_pre": "M9",
        "model": "memory",
        "tau0": "120",   # str — should be cast to float
        "k1": 2,         # int — should be cast to float
        "k2": 0.5,
    }]
    result = _check_cf(cf)
    entry = result["growth_transition"][0]
    assert isinstance(entry["tau0"], float)
    assert isinstance(entry["k1"], float)
    assert isinstance(entry["k2"], float)
    assert entry["tau0"] == 120.0


def test_check_cf_growth_transition_multiple_conditions(base_config: dict):
    """Multiple distinct condition_pre entries should all pass."""
    cf = base_config.copy()
    cf["growth_transition"] = [
        {"condition_pre": "M9", "model": "instant"},
        {"condition_pre": "M9+Ab", "model": "memory", "tau0": 100.0, "k1": 1.0, "k2": 1.0},
    ]
    result = _check_cf(cf)
    assert len(result["growth_transition"]) == 2


@pytest.mark.parametrize("bad_gt, match_error", [
    # Not a list
    ({"condition_pre": "M9", "model": "instant"}, "must be a list"),
    # Entry is not a dict
    (["not_a_dict"], "must be a dict"),
    # Missing condition_pre
    ([{"model": "instant"}], "missing required key 'condition_pre'"),
    # Missing model
    ([{"condition_pre": "M9"}], "missing required key 'model'"),
    # condition_pre not a string
    ([{"condition_pre": 42, "model": "instant"}], "'condition_pre' must be a string"),
    # Unknown model name
    ([{"condition_pre": "M9", "model": "not_a_model"}], "must be one of"),
    # Duplicate condition_pre
    (
        [{"condition_pre": "M9", "model": "instant"},
         {"condition_pre": "M9", "model": "instant"}],
        "duplicate condition_pre"
    ),
    # memory missing tau0
    ([{"condition_pre": "M9", "model": "memory", "k1": 1.0, "k2": 1.0}],
     "requires 'tau0'"),
    # memory missing k1
    ([{"condition_pre": "M9", "model": "memory", "tau0": 100.0, "k2": 1.0}],
     "requires 'k1'"),
    # memory missing k2
    ([{"condition_pre": "M9", "model": "memory", "tau0": 100.0, "k1": 1.0}],
     "requires 'k2'"),
    # memory k2 == 0
    ([{"condition_pre": "M9", "model": "memory", "tau0": 100.0, "k1": 1.0, "k2": 0.0}],
     "'k2' must be > 0"),
    # memory k2 < 0
    ([{"condition_pre": "M9", "model": "memory", "tau0": 100.0, "k1": 1.0, "k2": -1.0}],
     "'k2' must be > 0"),
    # memory tau0 not a number
    ([{"condition_pre": "M9", "model": "memory", "tau0": "bad", "k1": 1.0, "k2": 1.0}],
     "'tau0' must be a number"),
    # baranyi missing tau_lag
    ([{"condition_pre": "M9", "model": "baranyi", "k_sharp": 0.2}],
     "requires 'tau_lag'"),
    # baranyi missing k_sharp
    ([{"condition_pre": "M9", "model": "baranyi", "tau_lag": 100.0}],
     "requires 'k_sharp'"),
    # baranyi k_sharp == 0
    ([{"condition_pre": "M9", "model": "baranyi", "tau_lag": 100.0, "k_sharp": 0.0}],
     "'k_sharp' must be > 0"),
    # two_pop missing k_trans
    ([{"condition_pre": "M9", "model": "two_pop"}],
     "requires 'k_trans'"),
    # two_pop k_trans == 0
    ([{"condition_pre": "M9", "model": "two_pop", "k_trans": 0.0}],
     "'k_trans' must be > 0"),
    # two_pop k_trans not a number
    ([{"condition_pre": "M9", "model": "two_pop", "k_trans": "bad"}],
     "'k_trans' must be a number"),
])
def test_check_cf_growth_transition_failures(base_config: dict, bad_gt, match_error):
    """growth_transition validation should raise ValueError for invalid inputs."""
    cf = base_config.copy()
    cf["growth_transition"] = bad_gt
    with pytest.raises(ValueError, match=match_error):
        _check_cf(cf)


def test_check_cf_growth_transition_baranyi(base_config: dict):
    """A valid baranyi entry should pass and cast tau_lag/k_sharp to float."""
    cf = base_config.copy()
    cf["growth_transition"] = [{
        "condition_pre": "M9",
        "model": "baranyi",
        "tau_lag": "120",   # str — should be cast to float
        "k_sharp": 2,       # int — should be cast to float
    }]
    result = _check_cf(cf)
    entry = result["growth_transition"][0]
    assert isinstance(entry["tau_lag"], float)
    assert isinstance(entry["k_sharp"], float)
    assert entry["tau_lag"] == 120.0


def test_check_cf_growth_transition_two_pop(base_config: dict):
    """A valid two_pop entry should pass and cast k_trans to float."""
    cf = base_config.copy()
    cf["growth_transition"] = [{
        "condition_pre": "M9",
        "model": "two_pop",
        "k_trans": "0.001",
    }]
    result = _check_cf(cf)
    entry = result["growth_transition"][0]
    assert isinstance(entry["k_trans"], float)
    assert entry["k_trans"] == 0.001


def test_compute_kt_baranyi(single_row_df: pd.DataFrame):
    """baranyi model should return same result as BaranyiTransition.compute_kt."""
    from tfscreen.simulate.growth.transition_linkage import BaranyiTransition
    tau_lag, k_sharp = 50.0, 0.2
    gt = [{"condition_pre": "M9", "model": "baranyi",
           "tau_lag": tau_lag, "k_sharp": k_sharp}]
    result = _compute_kt(single_row_df, gt)
    expected = BaranyiTransition().compute_kt(
        0.1, 0.5, 30.0, 100.0, tau_lag=tau_lag, k_sharp=k_sharp
    )
    np.testing.assert_allclose(result, [expected])


def test_compute_kt_two_pop(single_row_df: pd.DataFrame):
    """two_pop model should return same result as TwoPopTransition.compute_kt."""
    from tfscreen.simulate.growth.transition_linkage import TwoPopTransition
    k_trans = 0.001
    gt = [{"condition_pre": "M9", "model": "two_pop", "k_trans": k_trans}]
    result = _compute_kt(single_row_df, gt)
    expected = TwoPopTransition().compute_kt(
        0.1, 0.5, 30.0, 100.0, k_trans=k_trans
    )
    np.testing.assert_allclose(result, [expected])


# ----------------------------------------------------------------------------
# test _compute_kt
# ----------------------------------------------------------------------------

@pytest.fixture
def single_row_df() -> pd.DataFrame:
    """A one-row phenotype DataFrame for simple _compute_kt tests."""
    return pd.DataFrame([{
        "condition_pre": "M9",
        "k_pre": 0.1,
        "k_sel": 0.5,
        "t_pre": 30.0,
        "t_sel": 100.0,
        "theta": 0.5,
    }])


def test_compute_kt_none_gives_instant(single_row_df: pd.DataFrame):
    """growth_transition=None should return the instant-transition formula."""
    result = _compute_kt(single_row_df, growth_transition=None)
    expected = 0.1 * 30.0 + 0.5 * 100.0   # 3 + 50 = 53
    np.testing.assert_allclose(result, [expected])


def test_compute_kt_instant_model(single_row_df: pd.DataFrame):
    """model='instant' should return the same result as growth_transition=None."""
    gt = [{"condition_pre": "M9", "model": "instant"}]
    result = _compute_kt(single_row_df, growth_transition=gt)
    expected = 0.1 * 30.0 + 0.5 * 100.0
    np.testing.assert_allclose(result, [expected])


def test_compute_kt_memory_lag_not_expired(single_row_df: pd.DataFrame):
    """When tau > t_sel the bacteria grow at k_pre throughout all of t_sel."""
    # tau = 200 + 1/(0.5+1) = 200.667, t_sel=100 → lag not expired
    gt = [{"condition_pre": "M9", "model": "memory",
           "tau0": 200.0, "k1": 1.0, "k2": 1.0}]
    result = _compute_kt(single_row_df, growth_transition=gt)
    expected = 0.1 * 30.0 + 0.1 * 100.0   # grows at k_pre the whole time: 3 + 10 = 13
    np.testing.assert_allclose(result, [expected])


def test_compute_kt_memory_lag_expired(single_row_df: pd.DataFrame):
    """When tau < t_sel the bacteria transition partway through t_sel."""
    # tau = 50 + 1/(0.5+1) = 50 + 2/3 ≈ 50.667, t_sel=100 → lag expires
    tau0, k1, k2, theta = 50.0, 1.0, 1.0, 0.5
    tau = tau0 + k1 / (theta + k2)          # ≈ 50.667
    k_pre, k_sel, t_pre, t_sel = 0.1, 0.5, 30.0, 100.0
    expected = k_pre * t_pre + k_pre * tau + k_sel * (t_sel - tau)
    gt = [{"condition_pre": "M9", "model": "memory",
           "tau0": tau0, "k1": k1, "k2": k2}]
    result = _compute_kt(single_row_df, growth_transition=gt)
    np.testing.assert_allclose(result, [expected])


def test_compute_kt_memory_tau_equals_t_sel(single_row_df: pd.DataFrame):
    """When tau == t_sel both branches give the same answer."""
    # Force tau exactly to t_sel=100: tau0=100 - k1/(theta+k2) = 100 - 1/1.5
    k_pre, k_sel, t_pre, t_sel = 0.1, 0.5, 30.0, 100.0
    theta, k1, k2 = 0.5, 1.0, 1.0
    tau0 = t_sel - k1 / (theta + k2)        # tau will equal exactly t_sel
    tau = tau0 + k1 / (theta + k2)
    # Both branches collapse to the same value: k_pre*t_pre + k_pre*tau + 0
    expected = k_pre * t_pre + k_pre * tau  # k_sel * 0 = 0
    gt = [{"condition_pre": "M9", "model": "memory",
           "tau0": tau0, "k1": k1, "k2": k2}]
    result = _compute_kt(single_row_df, growth_transition=gt)
    np.testing.assert_allclose(result, [expected], rtol=1e-10)


def test_compute_kt_memory_higher_theta_shorter_tau():
    """Higher theta should reduce tau (k1>0), meaning earlier transition."""
    df_lo = pd.DataFrame([{"condition_pre": "M9", "k_pre": 0.1, "k_sel": 0.5,
                            "t_pre": 30.0, "t_sel": 200.0, "theta": 0.1}])
    df_hi = pd.DataFrame([{"condition_pre": "M9", "k_pre": 0.1, "k_sel": 0.5,
                            "t_pre": 30.0, "t_sel": 200.0, "theta": 0.9}])
    gt = [{"condition_pre": "M9", "model": "memory",
           "tau0": 50.0, "k1": 10.0, "k2": 1.0}]
    kt_lo = _compute_kt(df_lo, gt)[0]
    kt_hi = _compute_kt(df_hi, gt)[0]
    # Higher theta → smaller tau → more time at k_sel (which > k_pre) → more growth
    assert kt_hi > kt_lo


def test_compute_kt_mixed_conditions():
    """Two condition_pre values, one instant and one memory, computed correctly."""
    df = pd.DataFrame([
        {"condition_pre": "cond_A", "k_pre": 0.1, "k_sel": 0.5,
         "t_pre": 30.0, "t_sel": 100.0, "theta": 0.5},
        {"condition_pre": "cond_B", "k_pre": 0.1, "k_sel": 0.5,
         "t_pre": 30.0, "t_sel": 100.0, "theta": 0.5},
    ])
    tau0, k1, k2, theta = 50.0, 1.0, 1.0, 0.5
    tau = tau0 + k1 / (theta + k2)
    k_pre, k_sel, t_pre, t_sel = 0.1, 0.5, 30.0, 100.0

    expected_instant = k_pre * t_pre + k_sel * t_sel
    expected_memory  = k_pre * t_pre + k_pre * tau + k_sel * (t_sel - tau)

    gt = [
        {"condition_pre": "cond_A", "model": "instant"},
        {"condition_pre": "cond_B", "model": "memory",
         "tau0": tau0, "k1": k1, "k2": k2},
    ]
    result = _compute_kt(df, gt)
    np.testing.assert_allclose(result[0], expected_instant)
    np.testing.assert_allclose(result[1], expected_memory)


def test_compute_kt_memory_less_than_instant(single_row_df: pd.DataFrame):
    """Memory kt must be <= instant kt when k_sel > k_pre (lag costs growth time)."""
    gt_instant = None
    gt_memory  = [{"condition_pre": "M9", "model": "memory",
                   "tau0": 50.0, "k1": 1.0, "k2": 1.0}]
    kt_instant = _compute_kt(single_row_df, gt_instant)[0]
    kt_memory  = _compute_kt(single_row_df, gt_memory)[0]
    # Lag delays the switch to faster k_sel, so total growth is lower
    assert kt_memory <= kt_instant


# ----------------------------------------------------------------------------
# test selection_experiment — growth_transition coverage error
# ----------------------------------------------------------------------------

def test_selection_experiment_missing_condition_pre_raises(
    mocker,
    base_config: dict,
    base_library_df: pd.DataFrame,
    base_phenotype_df: pd.DataFrame,
):
    """selection_experiment should raise if a condition_pre lacks a growth_transition entry."""
    cfg = base_config.copy()
    # Configure a transition only for a condition that doesn't exist in the data;
    # "M9" (the real condition_pre) is left uncovered → must error.
    cfg["growth_transition"] = [
        {"condition_pre": "not_in_data", "model": "instant"}
    ]

    mocker.patch("tfscreen.simulate.selection_experiment.read_yaml", return_value=cfg)
    mocker.patch(
        "tfscreen.simulate.selection_experiment.read_dataframe",
        side_effect=[base_phenotype_df, base_library_df],
    )

    with pytest.raises(ValueError, match="condition_pre values.*no entry"):
        selection_experiment("dummy.yaml", "dummy_lib.csv", "dummy_pheno.csv")


def test_selection_experiment_end_to_end(mocker, base_config: dict,
                                         base_library_df: pd.DataFrame,
                                         base_phenotype_df: pd.DataFrame):
    """
    Integration test.
    This doesn't need signature updates because selection_experiment 
    handles the dict creation internally.
    """
    base_phenotype_df["kt"] = (
        base_phenotype_df["k_pre"] * base_phenotype_df["t_pre"] +
        base_phenotype_df["k_sel"] * base_phenotype_df["t_sel"]
    )

    mocker.patch(
        "tfscreen.simulate.selection_experiment.read_yaml",
        return_value=base_config
    )
    
    mocker.patch(
        "tfscreen.simulate.selection_experiment.read_dataframe",
        side_effect=[base_phenotype_df, base_library_df]
    )

    sample_df, counts_df = selection_experiment(
        cf="dummy_config.yaml",
        library_df="dummy_library.csv",
        phenotype_df="dummy_phenotype.csv"
    )
    
    assert isinstance(sample_df, pd.DataFrame)
    num_conditions = base_phenotype_df.groupby(base_config["condition_selector"]).ngroups
    assert sample_df.shape[0] == num_conditions
    
    assert isinstance(counts_df, pd.DataFrame)
    assert counts_df.shape[0] == base_phenotype_df.shape[0]
    
    total_reads = base_config["total_num_reads"]
    assert np.isclose(counts_df["counts"].sum(), total_reads, rtol=0.01)


# ----------------------------------------------------------------------------
# test SIMULATE_KNOWN_KEYS / unknown-key validation in _check_cf
# ----------------------------------------------------------------------------

def test_simulate_known_keys_is_frozenset():
    assert isinstance(SIMULATE_KNOWN_KEYS, frozenset)
    assert len(SIMULATE_KNOWN_KEYS) > 0


def test_check_cf_unknown_key_raises(base_config: dict):
    bad = dict(base_config)
    bad["not_a_real_key"] = 42
    with pytest.raises(ValueError, match="not_a_real_key"):
        _check_cf(bad)


def test_check_cf_unknown_key_error_mentions_label(base_config: dict):
    bad = dict(base_config)
    bad["typo_key"] = "oops"
    with pytest.raises(ValueError, match="simulate config"):
        _check_cf(bad)


def test_check_cf_multiple_unknown_keys(base_config: dict):
    bad = dict(base_config)
    bad["key_one"] = 1
    bad["key_two"] = 2
    with pytest.raises(ValueError) as exc_info:
        _check_cf(bad)
    msg = str(exc_info.value)
    assert "key_one" in msg
    assert "key_two" in msg


def test_check_cf_all_known_keys_accepted(base_config: dict):
    # Every key in the fixture must already be a known key; no error should be raised.
    _check_cf(base_config)


def test_check_cf_accepts_base_growth_data(base_config: dict):
    """base_growth_data must be a recognized optional output block (it drives
    tfs-simulate's simulated base_growth CSV -- see
    simulate/base_growth_data.py and simulate/scripts/simulate_cli.py)."""
    cf = dict(base_config)
    cf["base_growth_data"] = {"k_ref": 0.025}
    _check_cf(cf)