import pytest
import pandas as pd
import numpy as np

# Import the function to be tested
from tfscreen.simulate.setup_observable import setup_observable

# --- Fixtures for Test Data ---

@pytest.fixture
def sample_df_fixture():
    """Provides a valid sample_df DataFrame."""
    return pd.DataFrame({
        "titrant_name": ["iptg", "iptg", "iptg"],
        "titrant_conc": [0.0, 0.1, 1.0]  # mM
    })

@pytest.fixture
def ddg_df_fixture():
    """Provides a valid ddG_df DataFrame with extra columns to test subsetting."""
    return pd.DataFrame({
        "mut": ["A1G", "C2T", "D3F"],
        "R": [0.1, 0.2, 0.3],
        "RI": [1.1, 1.2, 1.3],
        "RS": [2.1, 2.2, 2.3],
        "extra_col": [9, 9, 9] # This should be dropped
    })

@pytest.fixture
def mock_calculator(mocker):
    """Mocks the calculator classes and AVAILABLE_CALCULATORS dictionary."""
    # Create a mock class that mimics the real model classes
    mock_model_class = mocker.MagicMock()
    
    # The instance of the mock class needs 'species' and 'get_obs' attributes
    mock_instance = mock_model_class.return_value
    mock_instance.species = ('R', 'RI', 'RS')
    mock_instance.get_obs = mocker.MagicMock(return_value="observable_function")
    
    # Patch the dictionary in the module-under-test to use our mock
    mocker.patch(
        "tfscreen.simulate.setup_observable.AVAILABLE_CALCULATORS",
        {"eee": mock_model_class}
    )
    return mock_model_class

# --- Test Cases ---

def test_setup_observable_happy_path(sample_df_fixture, ddg_df_fixture, mock_calculator):
    """
    Tests the main success path of the function.
    """
    kwargs = {"e_name": "iptg", "other_param": 123}
    
    obs_fcn, result_ddg_df = setup_observable(
        observable_calculator="eee",
        observable_calc_kwargs=kwargs,
        ddG_spreadsheet=ddg_df_fixture,
        sample_df=sample_df_fixture
    )

    # 1. Check that the calculator was initialized correctly
    expected_calc_kwargs = {
        "other_param": 123,
        # Check that e_total was added and converted from mM to M
        "e_total": np.array([0.0, 0.0001, 0.001])
    }
    mock_calculator.assert_called_once()
    # np.array equality needs special handling
    call_args, call_kwargs = mock_calculator.call_args
    assert call_kwargs.keys() == expected_calc_kwargs.keys()
    assert call_kwargs['other_param'] == 123
    np.testing.assert_array_equal(call_kwargs['e_total'], expected_calc_kwargs['e_total'])

    # 2. Check the returned observable function
    assert obs_fcn == mock_calculator.return_value.get_obs

    # 3. Check the returned and subsetted ddG DataFrame
    expected_cols = ["R", "RI", "RS"]
    assert list(result_ddg_df.columns) == expected_cols
    assert result_ddg_df.shape == (3, 3)
    assert result_ddg_df.index.name == "mut"

def test_invalid_calculator(sample_df_fixture, ddg_df_fixture, mock_calculator):
    """Tests that a ValueError is raised for an unrecognized calculator."""
    with pytest.raises(ValueError, match="not recognized"):
        setup_observable(
            observable_calculator="bad_calculator",
            observable_calc_kwargs={"e_name": "iptg"},
            ddG_spreadsheet=ddg_df_fixture,
            sample_df=sample_df_fixture
        )

def test_missing_e_name(sample_df_fixture, ddg_df_fixture, mock_calculator):
    """Tests that a ValueError is raised if 'e_name' is missing."""
    with pytest.raises(ValueError, match="e_name must be defined"):
        setup_observable(
            observable_calculator="eee",
            observable_calc_kwargs={"other_param": 123}, # 'e_name' is missing
            ddG_spreadsheet=ddg_df_fixture,
            sample_df=sample_df_fixture
        )

def test_multiple_titrants(ddg_df_fixture, mock_calculator):
    """Tests that a ValueError is raised for multiple titrants in sample_df."""
    bad_sample_df = pd.DataFrame({
        "titrant_name": ["iptg", "lactose"],
        "titrant_conc": [0.1, 0.1]
    })
    with pytest.raises(ValueError, match="only supports one titrant_name"):
        setup_observable("eee", {"e_name": "iptg"}, ddg_df_fixture, bad_sample_df)

def test_mismatched_titrant_and_e_name(sample_df_fixture, ddg_df_fixture, mock_calculator):
    """Tests that a ValueError is raised if e_name and titrant_name don't match."""
    with pytest.raises(ValueError, match="does not match the titrant specified"):
        setup_observable("eee", {"e_name": "wrong_name"}, ddg_df_fixture, sample_df_fixture)

def test_missing_mut_column(sample_df_fixture, ddg_df_fixture, mock_calculator):
    """Tests that a ValueError is raised if 'mut' column is missing."""
    bad_ddg_df = ddg_df_fixture.drop(columns=["mut"])
    with pytest.raises(ValueError, match="must have a 'mut' column"):
        setup_observable("eee", {"e_name": "iptg"}, bad_ddg_df, sample_df_fixture)

def test_missing_species_column(sample_df_fixture, ddg_df_fixture, mock_calculator):
    """Tests that a ValueError is raised if a required species column is missing."""
    bad_ddg_df = ddg_df_fixture.drop(columns=["RI"]) # 'RI' is a required species
    with pytest.raises(ValueError, match="not all molecular species in ddG file"):
        setup_observable("eee", {"e_name": "iptg"}, bad_ddg_df, sample_df_fixture)