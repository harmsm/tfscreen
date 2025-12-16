
import pytest
import pandas as pd
import numpy as np
from typing import Iterable, Generator
from unittest.mock import MagicMock

# Import the function to be tested
from tfscreen.simulate.thermo_to_growth import _assign_ddG
from tfscreen.simulate.thermo_to_growth import _assign_dk_geno
from tfscreen.simulate.thermo_to_growth import thermo_to_growth

import pytest
import pandas as pd
import numpy as np
from numpy.random import Generator


# ----------------------------------------------------------------------------
# Fixtures for these tests
# ----------------------------------------------------------------------------

@pytest.fixture
def rng() -> Generator:
    """Provides a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)

@pytest.fixture
def simple_genotypes() -> list[str]:
    """A simple list of genotypes for testing helpers."""
    return ["wt", "A1B", "C2D", "A1B/C2D"]

@pytest.fixture
def simple_ddG_df() -> pd.DataFrame:
    """A simple DataFrame of single-mutant ddG effects."""
    return pd.DataFrame({"spec1": [1.0, -0.5]}, index=["A1B", "C2D"])


# ----------------------------------------------------------------------------
# test _assign_ddG
# ----------------------------------------------------------------------------

def test_assign_ddG_calls_combiner(mocker, simple_genotypes, simple_ddG_df):
    """
    Tests that _assign_ddG correctly calls the combine_mutation_effects utility.
    """
    # ARRANGE: Mock the utility function it's supposed to call
    mock_combiner = mocker.patch(
        # FIX: The target must match exactly where the function is imported.
        "tfscreen.simulate.thermo_to_growth.combine_mutation_effects",
        return_value="success"
    )
    
    # ACT: Call the wrapper function
    result = _assign_ddG(simple_genotypes, simple_ddG_df, mut_combine_fcn="mean")
    
    # ASSERT
    mock_combiner.assert_called_once_with(
        unique_genotypes=simple_genotypes,
        single_mutant_effects=simple_ddG_df,
        mut_combine_fcn="mean"
    )
    assert result == "success"

# ----------------------------------------------------------------------------
# test _assign_dk_geno
# ----------------------------------------------------------------------------

def test_assign_dk_geno_sampling_and_integration(rng, simple_genotypes):
    """
    Tests that _assign_dk_geno correctly samples from the gamma distribution
    and returns a correctly structured Series.
    """

    # ACT: Run the function, passing the seeded rng
    result = _assign_dk_geno(simple_genotypes, rng=rng)
    
    # ASSERT: Check the properties of the final output
    assert isinstance(result, pd.Series)
    assert len(result) == len(simple_genotypes)
    assert all(result.index == simple_genotypes)
    
    # Check for specific, correct values using the seeded RNG
    assert result.loc["wt"] == 0.0
    
    # Pre-calculated deterministic values
    expected_A1B = -0.0053917202444200745
    expected_C2D = -0.007178919203856204
    
    assert np.isclose(result.loc["A1B"], expected_A1B, rtol=1e-5)
    assert np.isclose(result.loc["C2D"], expected_C2D, rtol=1e-5)
    assert np.isclose(result.loc["A1B/C2D"], expected_A1B + expected_C2D, rtol=1e-5)


@pytest.fixture
def test_genotypes() -> list[str]:
    """A simple list of genotypes for testing."""
    return ["wt", "A1B", "A1B/C2D"]

@pytest.fixture
def test_sample_df() -> pd.DataFrame:
    """A simple DataFrame with two experimental conditions."""
    return pd.DataFrame({
        "condition_pre": ["M9", "M9"],
        "condition_sel": ["M9+Ab", "M9+Ab"],
        "titrant_name": ["IPTG", "IPTG"],
        "titrant_conc": [10.0, 100.0],
    })


# ----------------------------------------------------------------------------
# test thermo_to_growth
# ----------------------------------------------------------------------------

def test_thermo_to_growth_integration(mocker, test_genotypes, test_sample_df):
    """
    Performs an end-to-end integration test on thermo_to_growth.
    
    Mocks all external dependencies to verify that the function correctly
    wires together the internal data processing steps.
    """
    # 1. ARRANGE: Mock all dependencies and define mock return values
    
    # Mock genotype sorting/standardization to have predictable order
    sorted_genotypes = np.array(["wt", "A1B", "A1B/C2D"])
    mocker.patch("tfscreen.simulate.thermo_to_growth.standardize_genotypes", 
                 return_value=test_genotypes)
    mocker.patch("tfscreen.simulate.thermo_to_growth.argsort_genotypes", 
                 return_value=np.arange(len(sorted_genotypes)))

    # Mock the observable setup
    mock_theta_fcn = mocker.Mock(return_value=np.array([0.5, 0.8])) # Returns theta for 2 conditions
    mock_ddG_df = pd.DataFrame({"mut": ["A1B", "C2D"]})
    mocker.patch("tfscreen.simulate.thermo_to_growth.setup_observable",
                 return_value=(mock_theta_fcn, mock_ddG_df))

    # Mock the internal helper calls
    mock_genotype_ddG = pd.DataFrame({"spec1": [0, 1, 1.5]}, index=sorted_genotypes)
    mocker.patch("tfscreen.simulate.thermo_to_growth._assign_ddG",
                 return_value=mock_genotype_ddG)
    
    mock_dk_geno = pd.Series({"wt": 0, "A1B": -0.01, "A1B/C2D": -0.02}, name="dk_geno")
    mocker.patch("tfscreen.simulate.thermo_to_growth._assign_dk_geno",
                 return_value=mock_dk_geno)

    # Mock the calibration function (called twice)
    mock_k_pre = np.array([0.1] * 6) # 3 genotypes * 2 conditions = 6 rows
    mock_k_sel = np.array([0.9] * 6)
    mocker.patch("tfscreen.simulate.thermo_to_growth.get_wt_k",
                 side_effect=[mock_k_pre, mock_k_sel])
    
    # Mock the final utility call
    mocker.patch("tfscreen.simulate.thermo_to_growth.set_categorical_genotype",
                 side_effect=lambda df: df)

    # 2. ACT: Run the function with test data
    phenotype_df, genotype_ddG_df = thermo_to_growth(
        genotypes=test_genotypes,
        sample_df=test_sample_df,
        observable_calculator="mock_calc",
        observable_calc_kwargs={"e_name": "IPTG"},
        ddG_df="dummy_path.csv",
        calibration_data={}
    )

    # 3. ASSERT: Check the final outputs
    
    # Check shape: 3 genotypes * 2 conditions = 6 rows
    assert isinstance(phenotype_df, pd.DataFrame)
    assert phenotype_df.shape[0] == 6
    
    # Check that key columns were added and calculated correctly
    assert "theta" in phenotype_df.columns
    assert "dk_geno" in phenotype_df.columns
    
    # Verify the final growth rate calculation (k_base + dk_geno)
    # Map the mocked dk_geno to the final df's shape to get expected k
    expected_dk_geno = phenotype_df["genotype"].map(mock_dk_geno).to_numpy()
    np.testing.assert_allclose(phenotype_df["k_pre"], mock_k_pre + expected_dk_geno)
    np.testing.assert_allclose(phenotype_df["k_sel"], mock_k_sel + expected_dk_geno)
    
    # Check the second returned dataframe
    assert isinstance(genotype_ddG_df, pd.DataFrame)
    assert "genotype" in genotype_ddG_df.columns

def test_thermo_to_growth_propagates_rng(mocker, test_genotypes, test_sample_df):
    """
    Tests that the rng argument is correctly propagated to internal functions.
    """
    rng = np.random.default_rng(12345)
    
    # Mock dependencies
    mocker.patch("tfscreen.simulate.thermo_to_growth.standardize_genotypes", return_value=test_genotypes)
    mocker.patch("tfscreen.simulate.thermo_to_growth.argsort_genotypes", return_value=np.arange(len(test_genotypes)))
    
    mocker.patch("tfscreen.simulate.thermo_to_growth.setup_observable", 
                 return_value=(MagicMock(return_value=np.zeros(len(test_sample_df))), pd.DataFrame()))
    
    mocker.patch("tfscreen.simulate.thermo_to_growth._assign_ddG", return_value=pd.DataFrame({"col": [0]*len(test_genotypes)}, index=test_genotypes))
    
    # We want to check this mock call
    mock_assign_dk = mocker.patch("tfscreen.simulate.thermo_to_growth._assign_dk_geno", 
                                  return_value=pd.Series(0, index=test_genotypes))

    mocker.patch("tfscreen.simulate.thermo_to_growth.get_wt_k", return_value=np.zeros(len(test_sample_df)*len(test_genotypes)))
    mocker.patch("tfscreen.simulate.thermo_to_growth.set_categorical_genotype", side_effect=lambda x: x)
    
    thermo_to_growth(
        genotypes=test_genotypes,
        sample_df=test_sample_df,
        observable_calculator="mock_calc",
        observable_calc_kwargs={"e_name": "IPTG"},
        ddG_df="dummy.csv",
        calibration_data={},
        rng=rng
    )
    
    # Verify rng was passed
    args, kwargs = mock_assign_dk.call_args
    assert args[4] is rng