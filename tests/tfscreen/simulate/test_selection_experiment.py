import pytest
import pandas as pd
import numpy as np
import numpy.ma as ma
from numpy.random import Generator
from scipy.stats import gmean

from pathlib import Path

from tfscreen.simulate.selection_experiment import _check_dict_number
from tfscreen.simulate.selection_experiment import _check_cf
from tfscreen.simulate.selection_experiment import _check_lib_spec
from tfscreen.simulate.selection_experiment import _sim_plasmid_probabilities
from tfscreen.simulate.selection_experiment import _sim_index_hop
from tfscreen.simulate.selection_experiment import _sim_transform
from tfscreen.simulate.selection_experiment import _sim_transform_and_mix
from tfscreen.simulate.selection_experiment import _sim_growth
from tfscreen.simulate.selection_experiment import MULTI_PLASMID_COMBINE_FCNS
from tfscreen.simulate.selection_experiment import _sim_sequencing
from tfscreen.simulate.selection_experiment import _calc_genotype_cfu0
from tfscreen.simulate.selection_experiment import _simulate_library_group
from tfscreen.simulate.selection_experiment import selection_experiment




# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def rng() -> Generator:
    """Provides a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)

@pytest.fixture
def base_config() -> dict:
    """
    Provides a complete and valid configuration dictionary for tests.
    """
    return {
        "prob_index_hop": 0.01,
        "lib_assembly_skew_sigma": 0.5,
        "transformation_poisson_lambda": 0.8,
        "growth_rate_noise": 0.05,
        "final_cfu_pct_err": 0.03,
        "random_seed": 42,
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
        "multi_plasmid_combine_fcn": {
            "test_selection": "mean"
        },
        # FIX: Add the selector keys that are normally set as defaults.
        "condition_selector": [
            "titrant_name", "titrant_conc",
            "condition_pre", "t_pre",
            "condition_sel", "t_sel"
        ],
        "library_selector": ["replicate", "library"],
    }

@pytest.fixture
def base_library_df() -> pd.DataFrame:
    """
    Provides a small, well-defined library composition DataFrame.
    """
    data = {
        "library_origin": ["libA", "libA", "libB", "libB", "libB"],
        "genotype":       ["A1V", "A2V", "A2V", "A3V", "A4V"],
        "weight":         [1.0, 0.8, 0.5, 1.2, 0.9],
    }
    return pd.DataFrame(data)

@pytest.fixture
def base_phenotype_df(base_library_df: pd.DataFrame) -> pd.DataFrame:
    """
    Provides a small phenotype DataFrame consistent with the other fixtures.
    It defines phenotypes for 4 genotypes across 2 conditions.
    """
    # Get all unique genotypes from the library definition
    genotypes = sorted(list(pd.unique(base_library_df["genotype"])))
    
    # Define two experimental conditions
    conditions = [
        # Condition 1: Titrant at 10 uM
        {"titrant_name": "IPTG", "titrant_conc": 10.0},
        # Condition 2: Titrant at 100 uM
        {"titrant_name": "IPTG", "titrant_conc": 100.0},
    ]

    # Define base growth rates for each genotype
    base_growth_rates = {
        "A1V": 0.8, "A2V": 1.0, "A3V": 1.2, "A4V": 0.6
    }
    
    records = []
    for g in genotypes:
        for c in conditions:
            record = {
                "genotype": g,
                "replicate": 1,
                "library": "test_selection",
                "condition_pre": "M9",
                "t_pre": 4.0,
                "condition_sel": "M9 + Ab",
                "t_sel": 18.0,
                "k_pre": 0.1,  # Assume same pre-growth rate
                # Selection growth rate depends on genotype and condition
                "k_sel": base_growth_rates[g] * (1 + c["titrant_conc"] / 50.0),
                "dk_geno": 0.01,
                "theta": 1.0,
                **c # Add condition-specific columns
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
        "tfscreen.simulate.selection_experiment.load_simulation_config",
        return_value=base_config
    )
    
    dummy_path = Path("config.yaml")
    validated_cf = _check_cf(dummy_path)

    # Assert that the loader was called correctly
    mock_loader.assert_called_once_with(dummy_path)
    # Assert that the loaded config was processed
    assert validated_cf["random_seed"] == 42

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

def test_check_lib_spec_sets_defaults(base_config: dict, 
                                      base_library_df: pd.DataFrame, 
                                      base_phenotype_df: pd.DataFrame):
    """
    Tests that the function correctly sets the default for the
    'multi_plasmid_combine_fcn' if it is not in the config.
    """
    cf_copy = base_config.copy()
    lib_df_copy = base_library_df.copy()
    pheno_df_copy = base_phenotype_df.copy()

    # Remove the key to test default-setting behavior
    cf_copy.pop("multi_plasmid_combine_fcn")

    result_cf = _check_lib_spec(cf_copy, lib_df_copy, pheno_df_copy)

    assert "multi_plasmid_combine_fcn" in result_cf
    # 'test_selection' is the library name in base_phenotype_df
    expected_default = {"test_selection": "mean"}
    assert result_cf["multi_plasmid_combine_fcn"] == expected_default

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
    
    # Invalid multi_plasmid_combine_fcn name
    (lambda cf, lib, pheno: cf["multi_plasmid_combine_fcn"].update({"test_selection": "bad_fcn"}), "not recognized"),
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
    expected_hopped_counts = np.array([1234, 2124, 6642])
    
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
# test _sim_transform_and_mix
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("groupby_key", [
    "library_origin",  # This will produce string group keys
    ["library_origin"], # This will produce tuple group keys
])
def test_sim_transform_and_mix(mocker, base_library_df: pd.DataFrame, 
                               base_config: dict, rng: Generator, groupby_key):
    """
    Tests that the function correctly integrates results from its helpers,
    handling both string and tuple group keys from the grouper.
    """
    # 1. ARRANGE: Set up inputs and mock return values
    
    # Per your fix, add the "probs" column before grouping
    df = base_library_df.copy()
    df["probs"] = 1/len(df) # Value doesn't matter, just needs to exist
    grouper = df.groupby(groupby_key)
    
    # Define mock return values for `_sim_transform`
    mock_return_libA = (np.ones((2, 2)), np.zeros((2, 2), dtype=bool))
    mock_return_libB = (np.ones((3, 1)), np.zeros((3, 1), dtype=bool))

    mock_sim_transform = mocker.patch(
        "tfscreen.simulate.selection_experiment._sim_transform",
        side_effect=[mock_return_libA, mock_return_libB]
    )
    mock_vstack = mocker.patch(
        "tfscreen.simulate.selection_experiment.vstack_padded",
        return_value="mocked_array"
    )
    
    # 2. ACT: Call the function under test
    transformants, mask, probs = _sim_transform_and_mix(
        grouper,
        base_config["transform_sizes"],
        base_config["library_mixture"],
        base_config["transformation_poisson_lambda"],
        rng
    )

    # 3. ASSERT: Check the results and mock calls
    assert mock_sim_transform.call_count == 2
    assert transformants == "mocked_array"
    assert mask == "mocked_array"
    
    weights_libA = np.full(2, 0.6)
    weights_libB = np.full(3, 0.4)
    expected_weights = np.concatenate([weights_libA, weights_libB])
    expected_probs = expected_weights / np.sum(expected_weights)
    
    np.testing.assert_allclose(probs, expected_probs)


# ----------------------------------------------------------------------------
# test _sim_growth
# ----------------------------------------------------------------------------

def test_sim_growth_single_plasmid():
    """
    Tests a simple case where all cells have exactly one plasmid.
    """
    # 3 cells, each with one plasmid (genotypes 0, 1, 2)
    transformants = np.array([[0], [1], [2]])
    trans_mask = np.array([[False], [False], [False]])
    trans_freq = np.array([0.2, 0.3, 0.5])
    total_cfu0 = 1.0e6
    
    # 3 genotypes, 2 conditions
    genotype_vs_kt = np.array([
        [0.1, 0.5], # Genotype 0 k*t values
        [0.2, 0.6], # Genotype 1 k*t values
        [0.3, 0.7], # Genotype 2 k*t values
    ])

    # Manually calculate expected CFU
    initial_cfus = total_cfu0 * trans_freq
    growth_factors = np.exp(genotype_vs_kt)
    expected_cfu = initial_cfus[:, np.newaxis] * growth_factors

    result_cfu = _sim_growth(
        transformants, trans_mask, trans_freq, genotype_vs_kt, total_cfu0, "mean"
    )

    np.testing.assert_allclose(result_cfu, expected_cfu)

@pytest.mark.parametrize("combine_fcn_name", MULTI_PLASMID_COMBINE_FCNS.keys())
def test_sim_growth_multi_plasmid(combine_fcn_name: str):
    """
    Tests that multi-plasmid kt values are combined correctly for all
    supported combination functions.
    """
    # 2 cells: cell 0 has 2 plasmids (geno0, A1V), cell 1 has 1 (A2V)
    transformants = np.array([[0, 1], [2, 0]]) # Second plasmid for cell 1 is masked
    trans_mask = np.array([[False, False], [False, True]])
    trans_freq = np.array([0.4, 0.6])
    total_cfu0 = 1.0e6

    genotype_vs_kt = np.array([
        [0.1, 1.0], # Genotype 0
        [0.5, 0.8], # Genotype 1
        [0.3, 0.6], # Genotype 2
    ])

    # Calculate the expected effective kt for each cell
    # Cell 0 combines kt from genotypes 0 and 1
    kt_cell0 = genotype_vs_kt[[0, 1], :]
    # Cell 1 only has kt from genotype 2
    kt_cell1 = genotype_vs_kt[[2], :]
    
    # Use the actual numpy functions to get the expected combined kt values
    if combine_fcn_name == "gmean":
        from scipy.stats import gmean
        expected_combined_kt_cell0 = gmean(kt_cell0, axis=0)
    else:
        # Get the numpy/numpy.ma function (e.g., np.mean, np.max)
        np_fcn = MULTI_PLASMID_COMBINE_FCNS[combine_fcn_name]
        expected_combined_kt_cell0 = np_fcn(kt_cell0, axis=0)

    effective_kt = np.vstack([expected_combined_kt_cell0, kt_cell1])
    
    # Calculate final expected CFU
    initial_cfus = total_cfu0 * trans_freq
    expected_cfu = initial_cfus[:, np.newaxis] * np.exp(effective_kt)
    
    result_cfu = _sim_growth(
        transformants, trans_mask, trans_freq, genotype_vs_kt, total_cfu0, combine_fcn_name
    )

    np.testing.assert_allclose(result_cfu, expected_cfu)

def test_sim_growth_invalid_fcn():
    """
    Tests that the function raises a ValueError for an invalid combine function.
    """
    # We can use dummy arrays as the function will fail before using them
    dummy_array = np.array([[]])
    with pytest.raises(ValueError, match="not recognized"):
        _sim_growth(dummy_array, dummy_array, dummy_array, 
                    dummy_array, 0, "not_a_real_function")
        

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

    # Expected logic:
    # Plasmids:    [g0,   g0,   g1,   g1]
    # From cells:  [c0,   c1,   c1,   c2]
    # Cell Freqs:  [0.2,  0.5,  0.5,  0.3]
    #
    # Total weighted count for g0 = 0.2 + 0.5 = 0.7
    # Total weighted count for g1 = 0.5 + 0.3 = 0.8
    # Total weight = 0.7 + 0.8 = 1.5
    #
    # Final frequency for g0 = 0.7 / 1.5
    # Final frequency for g1 = 0.8 / 1.5
    expected_cfu0 = np.array([ (0.7/1.5) * total_cfu0,
                               (0.8/1.5) * total_cfu0 ])
    
    result = _calc_genotype_cfu0(
        transformants, trans_mask, trans_freq, total_cfu0, num_genotypes
    )
    
    np.testing.assert_allclose(result, expected_cfu0)
    
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
# test _simulate_library_group
# ----------------------------------------------------------------------------

def test_simulate_library_group_integration(base_config: dict, 
                                            base_library_df: pd.DataFrame, 
                                            base_phenotype_df: pd.DataFrame, 
                                            rng: Generator):
    """
    Performs an integration test on _simulate_library_group.
    
    This test verifies that the function correctly orchestrates its helper
    functions and returns two dataframes with the expected structure, shape,
    and data types.
    """
    # 1. ARRANGE: Prepare all the inputs the function expects
    
    # The sub_df for this test will be the entire phenotype dataframe
    sub_df = base_phenotype_df.copy()
    sub_df["kt"] = sub_df["t_sel"]*sub_df["k_sel"]
    
    # Create the grouper for library origins
    lib_df = base_library_df.copy()
    # Calculate probabilities that sum to 1.0 *within each group*.
    # This mimics the logic of the main `selection_experiment` function.
    lib_df["probs"] = lib_df.groupby("library_origin")["weight"].transform(
        lambda w: w / w.sum()
    )
    lib_origin_grouper = lib_df.groupby("library_origin")
    
    # Get other required inputs
    ordered_genotypes = pd.unique(lib_df["genotype"])
    num_conditions = sub_df.groupby(base_config.get("condition_selector", [])).ngroups
    total_reads = base_config["total_num_reads"]
    reads_per_sample = int(np.round(total_reads / num_conditions))
    index_offset = 100 # Use a non-zero offset to test this logic
    
    # 2. ACT: Run the function
    sample_df, counts_df = _simulate_library_group(
        sub_df,
        index_offset,
        lib_origin_grouper,
        ordered_genotypes,
        reads_per_sample,
        base_config,
        rng
    )
    
    # 3. ASSERT: Check the structure and properties of the output dataframes
    
    # -- Check sample_df --
    assert isinstance(sample_df, pd.DataFrame)
    assert "sample" in sample_df.columns
    assert "cfu_per_mL" in sample_df.columns
    assert sample_df.shape[0] == num_conditions # Should have one row per condition
    assert sample_df["sample"].min() == index_offset # Check offset logic
    assert np.all(sample_df["cfu_per_mL"] > 0) # Should have positive CFU counts
    
    # -- Check counts_df --
    assert isinstance(counts_df, pd.DataFrame)
    assert "sample" in counts_df.columns
    assert "counts" in counts_df.columns
    assert "ln_cfu_0" in counts_df.columns
    assert counts_df.shape[0] == sub_df.shape[0] # One row per input genotype/condition
    assert np.all(counts_df["counts"] >= 0) # Counts must be non-negative
    
    # Total counts should be conserved
    expected_total_counts = num_conditions * reads_per_sample
    assert np.isclose(counts_df["counts"].sum(), expected_total_counts, rtol=1)

    # Some ln_cfu_0 values should have been calculated (i.e., not all are -inf)
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
    sparse_phenotype_df: pd.DataFrame, # Use the new sparse fixture
    rng: Generator
):
    """
    GIVEN a sparse phenotype dataframe missing genotype/condition pairs
    WHEN _simulate_library_group is called
    THEN it should run without error and produce a counts_df with the same
         number of rows as the sparse input.
    """
    # 1. ARRANGE
    sub_df = sparse_phenotype_df
    sub_df["kt"] = sub_df["t_sel"] * sub_df["k_sel"]

    # Use the full library df for transformation simulation
    lib_df = base_library_df.copy()
    lib_df["probs"] = lib_df.groupby("library_origin")["weight"].transform(
        lambda w: w / w.sum()
    )
    lib_origin_grouper = lib_df.groupby("library_origin")

    # The canonical list of ALL genotypes that *could* exist
    ordered_genotypes = np.sort(pd.unique(lib_df["genotype"]))
    
    num_conditions = sub_df.groupby(base_config["condition_selector"]).ngroups
    reads_per_sample = int(np.round(base_config["total_num_reads"] / num_conditions))
    
    # 2. ACT
    sample_df, counts_df = _simulate_library_group(
        sub_df=sub_df,
        index_offset=0,
        lib_origin_grouper=lib_origin_grouper,
        ordered_genotypes=ordered_genotypes,
        reads_per_sample=reads_per_sample,
        cf=base_config,
        rng=rng
    )

    # 3. ASSERT
    # The primary check: The output dataframe's shape must match the sparse input.
    assert counts_df.shape[0] == sub_df.shape[0]
    
    # Secondary check: The merge should not have failed (no NaNs in counts).
    assert not counts_df["counts"].isna().any()
    
    # Final check: The total number of genotypes in the output should match the
    # number of unique genotypes in the sparse input.
    assert counts_df["genotype"].nunique() == sub_df["genotype"].nunique()

# ----------------------------------------------------------------------------
# test selection_experiment
# ----------------------------------------------------------------------------
def test_selection_experiment_end_to_end(mocker, base_config: dict, 
                                         base_library_df: pd.DataFrame, 
                                         base_phenotype_df: pd.DataFrame):
    """
    Performs an end-to-end integration test on the main function.
    
    This test mocks the file loading functions but otherwise runs the entire
    simulation pipeline to ensure all components are correctly integrated.
    """
    # 1. ARRANGE: Mock the file I/O dependencies
    
    base_phenotype_df["kt"] = (
        base_phenotype_df["k_pre"] * base_phenotype_df["t_pre"] +
        base_phenotype_df["k_sel"] * base_phenotype_df["t_sel"]
    )

    mocker.patch(
        "tfscreen.simulate.selection_experiment.load_simulation_config",
        return_value=base_config
    )
    
    # FIX: Reorder the side_effect to match the call order in the function.
    # The first call is for phenotype_df, the second is for library_df.
    mocker.patch(
        "tfscreen.simulate.selection_experiment.read_dataframe",
        side_effect=[base_phenotype_df, base_library_df]
    )

    # 2. ACT: Call the main function with dummy paths
    sample_df, counts_df = selection_experiment(
        cf="dummy_config.yaml",
        library_df="dummy_library.csv",
        phenotype_df="dummy_phenotype.csv"
    )
    
    # 3. ASSERT: (Assertions remain the same)
    assert isinstance(sample_df, pd.DataFrame)
    num_conditions = base_phenotype_df.groupby(base_config["condition_selector"]).ngroups
    assert sample_df.shape[0] == num_conditions
    
    assert isinstance(counts_df, pd.DataFrame)
    assert counts_df.shape[0] == base_phenotype_df.shape[0]
    
    total_reads = base_config["total_num_reads"]
    assert np.isclose(counts_df["counts"].sum(), total_reads, rtol=0.01)