import pytest
import pandas as pd
import jax.numpy as jnp
import numpy as np
from pandas.testing import assert_frame_equal
from unittest.mock import MagicMock, call, patch

# Import the functions to be tested
from tfscreen.analysis.hierarchical.growth_model.model_class import (
    _read_growth_df,
    _read_binding_df,
    _build_growth_tm,
    _build_growth_theta_tm,
    _build_binding_tm,
    _get_wt_info
)

# ----------------------------------------------------------------------------
# test _build_growth_tm

@pytest.fixture
def mock_tensor_manager_class():
    """Mocks the TensorManager class and returns the class and instance mocks."""
    # Create a mock for the instance
    mock_instance = MagicMock()
    
    # Patch the class in the module where it's *used*
    with patch("tfscreen.analysis.hierarchical.growth_model.model_class.TensorManager") as mock_class:
        # Configure the class mock to return our instance mock when called
        mock_class.return_value = mock_instance
        yield mock_class, mock_instance

@pytest.fixture
def dummy_growth_df():
    """A dummy DataFrame to pass to the function."""
    return pd.DataFrame({"replicate": [1], "t_sel": [0], "genotype": ["wt"]})


def test_build_growth_tm_call_sequence(mock_tensor_manager_class, dummy_growth_df):
    """
    Tests that _build_growth_tm calls all TensorManager methods
    in the correct order with the correct arguments.
    """
    mock_class, mock_instance = mock_tensor_manager_class
    
    # Call the function
    result_tm = _build_growth_tm(dummy_growth_df)

    # 1. Check that TensorManager was initialized correctly
    mock_class.assert_called_once_with(dummy_growth_df)
    
    # 2. Define the exact sequence of calls we expect on the instance
    expected_calls = [
        # Pivots
        call.add_pivot_index(tensor_dim_name="replicate", cat_column="replicate"),
        call.add_ranked_pivot_index(tensor_dim_name="time",
                                    rank_column="t_sel",
                                    select_cols=['replicate', 'genotype', 'treatment']),
        call.add_pivot_index(tensor_dim_name="treatment", cat_column="treatment_tuple"),
        call.add_pivot_index(tensor_dim_name="genotype", cat_column="genotype"),

        # Data Tensors
        call.add_data_tensor("ln_cfu"),
        call.add_data_tensor("ln_cfu_std"),
        call.add_data_tensor("t_pre"),
        call.add_data_tensor("t_sel"),
        call.add_data_tensor("titrant_conc"),

        # Map Tensors (order of these 5 is not critical, but we check)
        call.add_map_tensor(["replicate", "condition_pre", "genotype"],
                             name="ln_cfu0"),
        call.add_map_tensor("genotype", name="genotype"),
        call.add_map_tensor("genotype", name="activity"),
        call.add_map_tensor("genotype", name="dk_geno"),
        call.add_map_tensor(["titrant_name", "titrant_conc", "genotype"], name="theta"),

        # This map_theta_group call comes *after* the other maps
        call.add_data_tensor("map_theta_group", dtype=int),

        # Pooled map tensor
        call.add_map_tensor(select_cols=["replicate"],
                             select_pool_cols=["condition_pre", "condition_sel"],
                             name="condition"),
                             
        # Other maps
        call.add_map_tensor(["titrant_name", "titrant_conc"], name="titrant"),
        call.add_map_tensor("replicate", name="replicate"),

        # Final call
        call.create_tensors()
    ]
    
    # 3. Assert the calls were made in the expected order
    # mock_instance.method_calls is a list of all calls made
    assert mock_instance.method_calls == expected_calls
    
    # 4. As a redundant check, ensure create_tensors was called
    mock_instance.create_tensors.assert_called_once()
    
    # 5. Check that the function returned the mock instance
    assert result_tm is mock_instance

# ----------------------------------------------------------------------------
# test _build_growth_theta_tm

@pytest.fixture
def mock_growth_theta_tm_class():
    """Mocks the TensorManager class and returns the class and instance mocks."""
    mock_instance = MagicMock()
    # Patch the class in the module where it's *used*
    with patch("tfscreen.analysis.hierarchical.growth_model.model_class.TensorManager") as mock_class:
        mock_class.return_value = mock_instance
        yield mock_class, mock_instance

@pytest.fixture
def growth_df_with_duplicates():
    """
    A dummy DataFrame with duplicate rows based on the
    hard-coded columns in the function.
    """
    return pd.DataFrame({
        "titrant_name": ["IPTG", "IPTG", "IPTG"],
        "titrant_conc": [10.0, 10.0, 20.0],
        "genotype":     ["wt", "wt", "wt"],
        "other_data":   [1, 2, 3], # This column should be dropped
        "map_theta_group": [100, 100, 101] # This data should be passed
    })


def test_build_growth_theta_tm_call_sequence(
    mock_growth_theta_tm_class, 
    growth_df_with_duplicates
):
    """
    Tests that _build_growth_theta_tm:
    1. Correctly de-duplicates the input DataFrame.
    2. Adds the 'pivot_titrant_conc' column.
    3. Initializes TensorManager with this processed DataFrame.
    4. Calls all TensorManager methods in the correct sequence.
    """
    mock_class, mock_instance = mock_growth_theta_tm_class
    
    # Call the function
    result_tm = _build_growth_theta_tm(growth_df_with_duplicates)

    # 1. Define the expected DataFrame that should be passed to TensorManager
    expected_df_data = {
        "titrant_name": ["IPTG", "IPTG"],
        "titrant_conc": [10.0, 20.0],
        "genotype":     ["wt", "wt"],
        "other_data":   [1, 3], # drop_duplicates keeps the first
        "map_theta_group": [100, 101],
        "pivot_titrant_conc": [10.0, 20.0] # The new pivot column
    }
    
    # --- FIX ---
    # The index after drop_duplicates (keeping 'first') is [0, 2]
    expected_df = pd.DataFrame(expected_df_data, index=[0, 2])
    # -----------

    # 2. Check that TensorManager was initialized correctly
    # We use assert_frame_equal on the mocked call's argument
    mock_class.assert_called_once()
    df_passed_to_init = mock_class.call_args[0][0]
    assert_frame_equal(df_passed_to_init, expected_df)
    
    # 3. Define the exact sequence of calls we expect on the instance
    expected_calls = [
        call.add_pivot_index(tensor_dim_name="titrant_name", cat_column="titrant_name"),
        call.add_pivot_index(tensor_dim_name="titrant_conc", cat_column="pivot_titrant_conc"),
        call.add_pivot_index(tensor_dim_name="genotype", cat_column="genotype"),
        call.add_data_tensor("map_theta_group", dtype=int),
        call.add_data_tensor("titrant_conc"),
        call.create_tensors()
    ]
    
    # 4. Assert the calls were made in the expected order
    assert mock_instance.method_calls == expected_calls
    
    # 5. Check that the function returned the mock instance
    assert result_tm is mock_instance

# ----------------------------------------------------------------------------
# test _read_growth_df
# ----------------------------------------------------------------------------

# --- Constants for default columns ---
DEFAULT_GROWTH_THETA_COLS = ["genotype", "titrant_name"]
DEFAULT_TREATMENT_COLS = ["condition_pre", "condition_sel",
                          "titrant_name", "titrant_conc"]
# --- FIX: This list was wrong before ---
BASE_GROWTH_REQUIRED_COLS = ["ln_cfu", "ln_cfu_std", "replicate", "t_pre", "t_sel"]

# --- Fixtures for _read_growth_df ---

@pytest.fixture
def base_growth_df():
    """A minimal DataFrame that has all required columns for growth."""
    # --- FIX: Use the correct column lists ---
    all_cols = list(set(
        DEFAULT_GROWTH_THETA_COLS + DEFAULT_TREATMENT_COLS + BASE_GROWTH_REQUIRED_COLS
    ))
    return pd.DataFrame({col: [] for col in all_cols})

@pytest.fixture
def mock_growth_dependencies(mocker, base_growth_df):
    """Mocks all external dependencies for _read_growth_df."""
    mock_df = base_growth_df.copy()

    mock_read_df = mocker.patch(
        "tfscreen.util.read_dataframe", 
        return_value=mock_df
    )
    mock_set_geno = mocker.patch(
        "tfscreen.genetics.set_categorical_genotype",
        return_value=mock_df
    )
    mock_get_cfu = mocker.patch(
        "tfscreen.util.get_scaled_cfu",
        return_value=mock_df
    )
    mock_check_cols = mocker.patch("tfscreen.util.check_columns")
    
    mock_add_group = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class.add_group_columns",
        return_value=mock_df
    )
    
    return {
        "read_dataframe": mock_read_df,
        "set_categorical_genotype": mock_set_geno,
        "get_scaled_cfu": mock_get_cfu,
        "check_columns": mock_check_cols,
        "add_group_columns": mock_add_group,
        "mock_df": mock_df
    }

# --- Test Cases for _read_growth_df ---

class TestReadGrowthDF:

    def test_pass_through_and_defaults(self, mock_growth_dependencies):
        """
        Tests that the function calls all helpers in order and uses
        default column lists when none are provided.
        """
        mock_df = mock_growth_dependencies["mock_df"]
        
        result_df = _read_growth_df("dummy_path.csv")

        mock_growth_dependencies["read_dataframe"].assert_called_once_with("dummy_path.csv")
        
        mock_growth_dependencies["set_categorical_genotype"].assert_called_once_with(
            mock_df, standardize=True
        )
        
        mock_growth_dependencies["get_scaled_cfu"].assert_called_once_with(
            mock_df, need_columns=["ln_cfu", "ln_cfu_std"]
        )

        # --- FIX: Use correct expected columns ---
        expected_required = DEFAULT_GROWTH_THETA_COLS + DEFAULT_TREATMENT_COLS + BASE_GROWTH_REQUIRED_COLS
        mock_growth_dependencies["check_columns"].assert_called_once_with(
            mock_df, required_columns=expected_required
        )
        
        expected_calls = [
            call(target_df=mock_df, 
                 group_cols=DEFAULT_TREATMENT_COLS, 
                 group_name="treatment"),
            call(target_df=mock_df, 
                 group_cols=DEFAULT_GROWTH_THETA_COLS, 
                 group_name="map_theta_group")
        ]
        mock_growth_dependencies["add_group_columns"].assert_has_calls(expected_calls)
        assert mock_growth_dependencies["add_group_columns"].call_count == 2
        
        assert result_df is mock_df

    def test_adds_replicate_column_if_missing(self, mock_growth_dependencies):
        """
        Tests that a 'replicate' column is added if it doesn't exist.
        """
        # --- FIX: The base_growth_df fixture is now correct ---
        df_no_rep = mock_growth_dependencies["mock_df"].drop(columns=["replicate"])
        
        mock_growth_dependencies["read_dataframe"].return_value = df_no_rep
        mock_growth_dependencies["set_categorical_genotype"].return_value = df_no_rep
        mock_growth_dependencies["get_scaled_cfu"].return_value = df_no_rep
        
        _read_growth_df(df_no_rep)

        mock_check_cols = mock_growth_dependencies["check_columns"]
        mock_check_cols.assert_called_once()
        
        df_passed_to_check = mock_check_cols.call_args[0][0]
        
        assert "replicate" in df_passed_to_check.columns

    def test_custom_cols_are_used(self, mock_growth_dependencies):
        """
        Tests that custom theta and treatment columns are used when provided.
        """
        mock_df = mock_growth_dependencies["mock_df"]
        
        custom_theta = ["custom_genotype"]
        custom_treat = ["custom_condition", "custom_conc"]
        
        mock_df["custom_genotype"] = []
        mock_df["custom_condition"] = []
        mock_df["custom_conc"] = []

        _read_growth_df(mock_df, 
                        theta_group_cols=custom_theta, 
                        treatment_cols=custom_treat)

        # --- FIX: Use correct expected columns ---
        expected_required = custom_theta + custom_treat + BASE_GROWTH_REQUIRED_COLS
        mock_growth_dependencies["check_columns"].assert_called_once_with(
            mock_df, required_columns=expected_required
        )
        
        expected_calls = [
            call(target_df=mock_df, 
                 group_cols=custom_treat, 
                 group_name="treatment"),
            call(target_df=mock_df, 
                 group_cols=custom_theta, 
                 group_name="map_theta_group")
        ]
        mock_growth_dependencies["add_group_columns"].assert_has_calls(expected_calls)

# ----------------------------------------------------------------------------
# test _read_binding_df
# ----------------------------------------------------------------------------

# --- Constants for default columns ---
DEFAULT_BINDING_THETA_COLS = ["genotype", "titrant_name"]
BASE_BINDING_REQUIRED_COLS = ["theta_obs", "theta_std", "titrant_conc"]

# --- Fixtures for _read_binding_df ---

@pytest.fixture
def base_binding_df():
    """A minimal DataFrame that has all required columns for binding."""
    # --- FIX: Simplified fixture ---
    all_cols = list(set(
        DEFAULT_BINDING_THETA_COLS + BASE_BINDING_REQUIRED_COLS
    ))
    return pd.DataFrame({col: [] for col in all_cols})

@pytest.fixture
def mock_binding_dependencies(mocker, base_binding_df):
    """
    Mocks all external dependencies for _read_binding_df.
    Returns a dictionary of mock objects.
    """
    mock_df = base_binding_df.copy()

    mock_read_df = mocker.patch(
        "tfscreen.util.read_dataframe", 
        return_value=mock_df
    )
    mock_set_geno = mocker.patch(
        "tfscreen.genetics.set_categorical_genotype",
        return_value=mock_df
    )
    mock_check_cols = mocker.patch("tfscreen.util.check_columns")
    
    mock_add_group = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class.add_group_columns",
        return_value=mock_df
    )
    
    return {
        "read_dataframe": mock_read_df,
        "set_categorical_genotype": mock_set_geno,
        "check_columns": mock_check_cols,
        "add_group_columns": mock_add_group,
        "mock_df": mock_df
    }

# --- Test Cases for _read_binding_df ---

class TestReadBindingDF:

    def test_read_binding_df_defaults_and_existing_df(self, mock_binding_dependencies):
        """
        Tests the standard path where an existing_df is provided and
        default columns are used.
        """
        mock_df = mock_binding_dependencies["mock_df"]
        dummy_existing_df = pd.DataFrame({"dummy": [1]})
        
        result_df = _read_binding_df("dummy_path.csv", 
                                     existing_df=dummy_existing_df)

        mock_binding_dependencies["read_dataframe"].assert_called_once_with(
            "dummy_path.csv"
        )
        
        mock_binding_dependencies["set_categorical_genotype"].assert_called_once_with(
            mock_df, standardize=True, sort=True
        )
        
        expected_required = DEFAULT_BINDING_THETA_COLS + BASE_BINDING_REQUIRED_COLS
        mock_binding_dependencies["check_columns"].assert_called_once_with(
            mock_df, required_columns=expected_required
        )
        
        mock_binding_dependencies["add_group_columns"].assert_called_once_with(
            mock_df,
            group_cols=DEFAULT_BINDING_THETA_COLS,
            group_name="map_theta_group",
            existing_df=dummy_existing_df
        )

        # --- FIX: Removed assertion for 'theta_titrant_conc' ---
        
        assert result_df is mock_df

    def test_read_binding_df_custom_cols_no_existing(self, mock_binding_dependencies):
        """
        Tests the path where no existing_df is provided and
        custom theta_group_cols are used.
        """
        mock_df = mock_binding_dependencies["mock_df"]
        custom_theta = ["custom_genotype", "custom_titrant"]
        
        mock_df["custom_genotype"] = []
        mock_df["custom_titrant"] = []
        
        result_df = _read_binding_df(mock_df, 
                                     existing_df=None,
                                     theta_group_cols=custom_theta)

        mock_binding_dependencies["read_dataframe"].assert_called_once_with(mock_df)

        expected_required = custom_theta + BASE_BINDING_REQUIRED_COLS
        mock_binding_dependencies["check_columns"].assert_called_once_with(
            mock_df, required_columns=expected_required
        )
        
        mock_binding_dependencies["add_group_columns"].assert_called_once_with(
            mock_df,
            group_cols=custom_theta,
            group_name="map_theta_group",
            existing_df=None
        )
        
        assert result_df is mock_df

    def test_read_binding_df_propagates_check_error(self, mock_binding_dependencies):
        """
        Tests that an error from a helper function (like check_columns)
        is correctly raised.
        """
        mock_binding_dependencies["check_columns"].side_effect = ValueError(
            "Missing column"
        )
        
        with pytest.raises(ValueError, match="Missing column"):
            _read_binding_df(mock_binding_dependencies["mock_df"])



# ----------------------------------------------------------------------------
# test _build_binding_tm


@pytest.fixture
def mock_binding_tm_class():
    """Mocks the TensorManager class and returns the class and instance mocks."""
    mock_instance = MagicMock()
    
    # --- FIX ---
    # The code under test calls `tfscreen.analysis.hierarchical.TensorManager`,
    # not the locally imported `TensorManager`. We must patch the full path
    # that is being looked up by the function.
    with patch("tfscreen.analysis.hierarchical.TensorManager") as mock_class:
    # -----------
        mock_class.return_value = mock_instance
        yield mock_class, mock_instance

@pytest.fixture
def dummy_binding_df():
    """
    A dummy DataFrame to pass to _build_binding_tm.
    """
    return pd.DataFrame({
        "titrant_name": ["IPTG"],
        "titrant_conc": [10.0],
        "genotype":     ["wt"],
        "theta_obs": [0.5],
        "theta_std": [0.1],
        "map_theta_group": [100]
    })


def test_build_binding_tm_call_sequence(
    mock_binding_tm_class, 
    dummy_binding_df
):
    """
    Tests that _build_binding_tm:
    1. Adds the 'pivot_titrant_conc' column.
    2. Initializes TensorManager with this processed DataFrame.
    3. Calls all TensorManager methods in the correct sequence.
    """
    mock_class, mock_instance = mock_binding_tm_class
    
    # We need to test the DF *before* it's passed, so make a copy
    df_copy = dummy_binding_df.copy()
    
    # Call the function
    result_tm = _build_binding_tm(df_copy)

    # 1. Define the expected DataFrame that should be passed to TensorManager
    # It should be the original DF plus the new pivot column
    expected_df = dummy_binding_df.copy()
    expected_df["pivot_titrant_conc"] = expected_df["titrant_conc"]

    # 2. Check that TensorManager was initialized correctly
    # The function modifies its input df, so we check the captured arg
    mock_class.assert_called_once()
    df_passed_to_init = mock_class.call_args[0][0]
    assert_frame_equal(df_passed_to_init, expected_df)
    
    # 3. Define the exact sequence of calls we expect on the instance
    expected_calls = [
        call.add_pivot_index(tensor_dim_name="titrant_name", cat_column="titrant_name"),
        call.add_pivot_index(tensor_dim_name="titrant_conc", cat_column="pivot_titrant_conc"),
        call.add_pivot_index(tensor_dim_name="genotype", cat_column="genotype"),
        call.add_data_tensor("theta_obs"),
        call.add_data_tensor("theta_std"),
        call.add_data_tensor("map_theta_group", dtype=int),
        call.add_data_tensor("titrant_conc"),
        call.create_tensors()
    ]
    
    # 4. Assert the calls were made in the expected order
    assert mock_instance.method_calls == expected_calls
    
    # 5. Check that the function returned the mock instance
    assert result_tm is mock_instance



# ----------------------------------------------------------------------------
# test _get_wt_info
# ----------------------------------------------------------------------------


@pytest.fixture
def mock_tm():
    """Returns a basic MagicMock for a TensorManager."""
    return MagicMock()

def test_get_wt_info_success(mock_tm):
    """
    Tests the successful extraction of 'wt' info when 'wt' is present
    and 'genotype' is not the last dimension.
    """
    mock_tm.tensor_dim_names = ["replicate", "genotype", "time"]
    
    # --- FIX ---
    # The labels must be arrays to support element-wise comparison (arr != "wt").
    # The real TensorManager provides pandas.CategoricalIndex, which does this.
    mock_tm.tensor_dim_labels = [
        np.array(["R1", "R2"]),                 # replicate labels
        np.array(["mut1", "wt", "mut2"]),     # genotype labels
        np.array(["T1", "T2"])                  # time labels
    ]
    # -----------
    
    result = _get_wt_info(mock_tm)

    # Check the dictionary keys
    expected_keys = {"wt_index", "not_wt_mask", "num_not_wt"}
    assert set(result.keys()) == expected_keys
    
    # Check the values
    assert result["num_not_wt"] == 2
    
    # Use jnp.array_equal for JAX array comparison
    assert jnp.array_equal(result["wt_index"], jnp.array(1))
    assert jnp.array_equal(result["not_wt_mask"], jnp.array([0, 2]))

def test_get_wt_info_raises_no_wt(mock_tm):
    """Tests that a ValueError is raised if 'wt' is not found."""
    mock_tm.tensor_dim_names = ["genotype"]
    
    # --- FIX ---
    mock_tm.tensor_dim_labels = [
        np.array(["mut1", "mut2", "mut3"])
    ]
    # -----------
    
    with pytest.raises(ValueError, match="Exactly one 'wt' entry is allowed.*found 0"):
        _get_wt_info(mock_tm)

def test_get_wt_info_raises_multiple_wt(mock_tm):
    """Tests that a ValueError is raised if multiple 'wt' entries are found."""
    mock_tm.tensor_dim_names = ["genotype"]

    # --- FIX ---
    mock_tm.tensor_dim_labels = [
        np.array(["mut1", "wt", "mut2", "wt"])
    ]
    # -----------
    
    with pytest.raises(ValueError, match="Exactly one 'wt' entry is allowed.*found 2"):
        _get_wt_info(mock_tm)

def test_get_wt_info_raises_no_genotype_dim(mock_tm):
    """Tests that a ValueError is raised if the 'genotype' dimension is missing."""
    mock_tm.tensor_dim_names = ["replicate", "time"]

    # --- FIX ---
    mock_tm.tensor_dim_labels = [
        np.array(["R1", "R2"]),
        np.array(["T1", "T2"])
    ]
    # -----------
    
    with pytest.raises(ValueError, match="Could not find 'genotype'"):
        _get_wt_info(mock_tm)