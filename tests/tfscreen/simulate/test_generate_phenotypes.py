import pytest
import pandas as pd
import numpy as np

# Import the functions to be tested
from tfscreen.simulate.generate_phenotypes import (
    _assign_ddG,
    _assign_dk_geno,
    generate_phenotypes
)

# --- Fixtures for Test Data ---

@pytest.fixture
def genotype_list_fixture():
    """A list of genotypes with wt, singles, doubles, and duplicates."""
    return ["wt", "A1G", "C3F", "A1G", "A1G/C3F"]

@pytest.fixture
def ddg_df_fixture():
    """A sample ddG DataFrame, as returned by setup_observable."""
    df = pd.DataFrame({
        "mut": ["A1G", "C3F", "D4V"],
        "R":  [0.5, 1.0, 1.5],
        "RI": [-0.5, -1.0, -1.5]
    })
    return df.set_index("mut")

@pytest.fixture
def sample_df_fixture():
    """A sample experimental conditions DataFrame."""
    return pd.DataFrame({
        "replicate": [1, 1],
        "titrant_name": ["iptg", "iptg"],
        "titrant_conc": [0.1, 1.0],
        "condition_pre": ["glu", "glu"],
        "condition_sel": ["lactose", "lactose"]
    })

@pytest.fixture
def genotype_df_fixture(genotype_list_fixture):
    """A sample genotype DataFrame, with duplicates."""
    return pd.DataFrame({"genotype": genotype_list_fixture})

@pytest.fixture
def mock_dependencies(mocker, ddg_df_fixture, sample_df_fixture):
    """Mocks all external dependencies for the main generate_phenotypes test."""
    
    # Mock theta function returns a unique array for each ddG array it sees
    mock_theta_fcn = mocker.MagicMock(
        side_effect=[
            np.array([0.5, 0.6]), # for wt
            np.array([0.4, 0.5]), # for A1G
            np.array([0.3, 0.4]), # for C3F
            np.array([0.2, 0.3]), # for A1G/C3F
        ]
    )
    
    # Mock setup_observable to return the mock theta_fcn and a real ddG_df
    mock_setup_observable = mocker.patch(
        "tfscreen.simulate.generate_phenotypes.setup_observable",
        return_value=(mock_theta_fcn, ddg_df_fixture)
    )

    # Mock get_wt_k to return a predictable value
    # Let's say it just returns theta * 10
    mock_get_wt_k = mocker.patch(
        "tfscreen.simulate.generate_phenotypes.get_wt_k",
        side_effect=lambda cond, name, conc, cal, theta: theta * 10
    )

    # Mock set_categorical_genotype to simply pass through the DataFrame
    mock_set_categorical = mocker.patch(
        "tfscreen.simulate.generate_phenotypes.set_categorical_genotype",
        side_effect=lambda df, sort: df.sort_values("genotype").reset_index(drop=True)
    )

    # Mock the random sampling to get predictable dk_geno values
    mock_gamma_rvs = mocker.patch(
        "tfscreen.simulate.generate_phenotypes.gamma.rvs",
        # FIX: Return an array of length 2 to match the 2 unique mutations
        # in the test data (A1G, C3F).
        return_value=np.array([0.001, 0.002])
    )


    # Return a dictionary of mocks to inspect in the test
    return {
        "setup_observable": mock_setup_observable,
        "get_wt_k": mock_get_wt_k,
        "set_categorical_genotype": mock_set_categorical,
        "gamma.rvs": mock_gamma_rvs
    }

# --- Tests for Helper Functions ---

def test_assign_ddg(genotype_list_fixture, ddg_df_fixture):
    """Tests the _assign_ddG helper function."""
    ddg_dict = _assign_ddG(genotype_list_fixture, ddg_df_fixture)

    # Check wt
    assert np.array_equal(ddg_dict["wt"], np.array([0.0, 0.0]))

    # Check a single mutant
    assert np.array_equal(ddg_dict["A1G"], np.array([0.5, -0.5]))

    # Check a double mutant (additivity)
    expected_double = np.array([0.5 + 1.0, -0.5 + -1.0])
    assert np.allclose(ddg_dict["A1G/C3F"], expected_double)

def test_assign_dk_geno(genotype_list_fixture, mocker):
    """Tests the _assign_dk_geno helper function with mocked random values."""
    # Mock gamma.rvs to return predictable values
    mock_rvs = mocker.patch(
        "tfscreen.simulate.generate_phenotypes.gamma.rvs",
        return_value=np.array([-0.1, 0.1]) # Mock values for 'A1G' and 'C3F'
    )
    
    # Run function with a fixed scale parameter for easy testing
    dk_dict = _assign_dk_geno(genotype_list_fixture, scale_param=2.0)

    # Expected values are scale/2 - rvs() -> 1.0 - [-0.1, 0.1]
    expected_a1g = 1.0 - (-0.1)
    expected_c3f = 1.0 - 0.1
    
    assert dk_dict["wt"] == 0
    assert dk_dict["A1G"] == expected_a1g
    assert dk_dict["C3F"] == expected_c3f
    # Check additivity for the double mutant
    assert dk_dict["A1G/C3F"] == expected_a1g + expected_c3f


# --- Integration Test for the Main Function ---

def test_generate_phenotypes(genotype_df_fixture, sample_df_fixture, mock_dependencies):
    """
    Tests the main generate_phenotypes orchestrator function.
    """
    genotype_df_out, phenotype_df_out = generate_phenotypes(
        genotype_df=genotype_df_fixture,
        sample_df=sample_df_fixture,
        observable_calculator="eee",
        observable_calc_kwargs={"e_name": "iptg"},
        ddG_spreadsheet="dummy_path.csv",
        calibration_data={}
    )

    # 1. Check returned DataFrame shapes and types
    assert isinstance(genotype_df_out, pd.DataFrame)
    assert isinstance(phenotype_df_out, pd.DataFrame)
    num_genotypes = len(genotype_df_fixture)
    num_samples = len(sample_df_fixture)
    assert phenotype_df_out.shape[0] == num_genotypes * num_samples

    # 2. Check that dependencies were called
    mock_dependencies["setup_observable"].assert_called_once()
    mock_dependencies["set_categorical_genotype"].assert_called_once()
    assert mock_dependencies["get_wt_k"].call_count == 2 # Once for k_pre, once for k_sel

    # 3. Check genotype_df output
    assert "ddG" in genotype_df_out.columns
    assert "dk_geno" in genotype_df_out.columns
    # Check the ddG for the double mutant
    expected_double_ddg = np.array([1.5, -1.5])
    actual_double_ddg = genotype_df_out[genotype_df_out["genotype"] == "A1G/C3F"]["ddG"].iloc[0]
    assert np.allclose(actual_double_ddg, expected_double_ddg)

    # 4. Check phenotype_df output
    assert "theta" in phenotype_df_out.columns
    assert "k_pre" in phenotype_df_out.columns
    assert "k_sel" in phenotype_df_out.columns

    # Check a specific row for correctness: the first row for 'A1G/C3F'
    test_row = phenotype_df_out[phenotype_df_out["genotype"] == "A1G/C3F"].iloc[0]

    # --- FIX: Update assertions to match the groupby sort order ---
    # theta for 'A1G/C3F' is the 2nd item from the side_effect list.
    # The first value in that array corresponds to the first sample condition.
    assert test_row["theta"] == 0.4

    # dk_geno is unchanged
    expected_dk_a1g = 0.001 - 0.001
    expected_dk_c3f = 0.001 - 0.002
    assert np.isclose(test_row["dk_geno"], expected_dk_a1g + expected_dk_c3f)
    
    # k_pre = (theta * 10) + dk_geno. Update with the correct theta.
    expected_k_pre = (0.4 * 10) + (expected_dk_a1g + expected_dk_c3f)
    assert np.isclose(test_row["k_pre"], expected_k_pre)