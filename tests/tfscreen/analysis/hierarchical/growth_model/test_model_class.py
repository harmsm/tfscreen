import pytest
import pandas as pd
import jax.numpy as jnp
import numpy as np
from pandas.testing import assert_frame_equal
from unittest.mock import MagicMock, call, patch, ANY

# Import the module under test
from tfscreen.analysis.hierarchical.growth_model.model_class import (
    ModelClass,
    _read_growth_df,
    _read_binding_df,
    _build_growth_tm,
    _build_growth_theta_tm,
    _build_binding_tm,
    _get_wt_info,
    _extract_param_est
)

# ----------------------------------------------------------------------------
# 1. Tests for _build_growth_tm
# ----------------------------------------------------------------------------

@pytest.fixture
def mock_tensor_manager_class():
    """Mocks the TensorManager class locally within model_class."""
    mock_instance = MagicMock()
    with patch("tfscreen.analysis.hierarchical.growth_model.model_class.TensorManager") as mock_class:
        mock_class.return_value = mock_instance
        yield mock_class, mock_instance

@pytest.fixture
def dummy_growth_df():
    return pd.DataFrame({"replicate": [1], "t_sel": [0], "genotype": ["wt"]})

def test_build_growth_tm_call_sequence(mock_tensor_manager_class, dummy_growth_df):
    """Tests the exact call sequence on TensorManager for growth data."""
    mock_class, mock_instance = mock_tensor_manager_class
    
    result_tm = _build_growth_tm(dummy_growth_df)

    mock_class.assert_called_once_with(dummy_growth_df)
    
    expected_calls = [
        # Pivots
        call.add_pivot_index(tensor_dim_name="replicate", cat_column="replicate"),
        call.add_ranked_pivot_index(tensor_dim_name="time",
                                    rank_column="t_sel",
                                    select_cols=['replicate', 'genotype', 'treatment']),
        call.add_pivot_index(tensor_dim_name="treatment", cat_column="treatment_tuple"),
        call.add_pivot_index(tensor_dim_name="genotype", cat_column="genotype"),

        # Data Tensors (Expecting dtype kwarg)
        call.add_data_tensor("ln_cfu", dtype=ANY),
        call.add_data_tensor("ln_cfu_std", dtype=ANY),
        call.add_data_tensor("t_pre", dtype=ANY),
        call.add_data_tensor("t_sel", dtype=ANY),
        call.add_data_tensor("titrant_conc", dtype=ANY),
        call.add_data_tensor("log_titrant_conc", dtype=ANY),

        # Map Tensors
        call.add_map_tensor(["replicate", "condition_pre", "genotype"], name="ln_cfu0"),
        call.add_map_tensor("genotype", name="genotype"),
        call.add_map_tensor("genotype", name="activity"),
        call.add_map_tensor("genotype", name="dk_geno"),
        call.add_map_tensor(["titrant_name", "titrant_conc", "genotype"], name="theta"),

        call.add_data_tensor("map_theta_group", dtype=int),

        call.add_map_tensor(select_cols=["replicate"],
                             select_pool_cols=["condition_pre", "condition_sel"],
                             name="condition"),
                             
        call.add_map_tensor(["titrant_name", "titrant_conc"], name="titrant"),
        call.add_map_tensor("replicate", name="replicate"),

        call.create_tensors()
    ]
    
    assert mock_instance.method_calls == expected_calls
    assert result_tm is mock_instance

# ----------------------------------------------------------------------------
# 2. Tests for _build_growth_theta_tm
# ----------------------------------------------------------------------------

@pytest.fixture
def growth_df_with_duplicates():
    return pd.DataFrame({
        "titrant_name": ["IPTG", "IPTG", "IPTG"],
        "titrant_conc": [10.0, 10.0, 20.0],
        "genotype":     ["wt", "wt", "wt"],
        "other_data":   [1, 2, 3], 
        "map_theta_group": [100, 100, 101]
    })

def test_build_growth_theta_tm_call_sequence(mock_tensor_manager_class, growth_df_with_duplicates):
    """Tests deduplication and TensorManager calls for growth theta."""
    mock_class, mock_instance = mock_tensor_manager_class
    
    result_tm = _build_growth_theta_tm(growth_df_with_duplicates)

    # Check that duplicates were dropped and pivot col added
    expected_df_data = {
        "titrant_name": ["IPTG", "IPTG"],
        "titrant_conc": [10.0, 20.0],
        "genotype":     ["wt", "wt"],
        "other_data":   [1, 3],
        "map_theta_group": [100, 101],
        "pivot_titrant_conc": [10.0, 20.0]
    }
    expected_df = pd.DataFrame(expected_df_data, index=[0, 2])
    
    mock_class.assert_called_once()
    df_passed_to_init = mock_class.call_args[0][0]
    assert_frame_equal(df_passed_to_init, expected_df)
    
    expected_calls = [
        call.add_pivot_index(tensor_dim_name="titrant_name", cat_column="titrant_name"),
        call.add_pivot_index(tensor_dim_name="titrant_conc", cat_column="pivot_titrant_conc"),
        call.add_pivot_index(tensor_dim_name="genotype", cat_column="genotype"),
        call.add_data_tensor("map_theta_group", dtype=int),
        call.add_data_tensor("titrant_conc", dtype=ANY),
        call.add_data_tensor("log_titrant_conc", dtype=ANY),
        call.create_tensors()
    ]
    
    assert mock_instance.method_calls == expected_calls
    assert result_tm is mock_instance

# ----------------------------------------------------------------------------
# 3. Tests for _read_growth_df
# ----------------------------------------------------------------------------

@pytest.fixture
def base_growth_df():
    cols = ["ln_cfu", "ln_cfu_std", "replicate", "t_pre", "t_sel",
            "genotype", "titrant_name", "condition_pre", "condition_sel", 
            "titrant_conc"]
    return pd.DataFrame({c: [] for c in cols})

@pytest.fixture
def mock_growth_dependencies(mocker, base_growth_df):
    mock_df = base_growth_df.copy()
    
    mocks = {
        "read": mocker.patch("tfscreen.util.read_dataframe", return_value=mock_df),
        "set_geno": mocker.patch("tfscreen.genetics.set_categorical_genotype", return_value=mock_df),
        "get_cfu": mocker.patch("tfscreen.util.get_scaled_cfu", return_value=mock_df),
        "check": mocker.patch("tfscreen.util.check_columns"),
        "add_group": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class.add_group_columns", return_value=mock_df),
        "df": mock_df
    }
    return mocks

def test_read_growth_df_defaults(mock_growth_dependencies):
    mock_df = mock_growth_dependencies["df"]
    _read_growth_df("path.csv")
    
    mock_growth_dependencies["add_group"].assert_any_call(
        target_df=mock_df, 
        group_cols=["condition_pre","condition_sel","titrant_name","titrant_conc"], 
        group_name="treatment"
    )
    mock_growth_dependencies["add_group"].assert_any_call(
        target_df=mock_df, 
        group_cols=["genotype","titrant_name"], 
        group_name="map_theta_group"
    )

def test_read_growth_df_adds_replicate(mock_growth_dependencies):
    df_no_rep = mock_growth_dependencies["df"].drop(columns=["replicate"])
    mock_growth_dependencies["read"].return_value = df_no_rep
    mock_growth_dependencies["set_geno"].return_value = df_no_rep
    mock_growth_dependencies["get_cfu"].return_value = df_no_rep
    
    _read_growth_df(df_no_rep)
    
    df_checked = mock_growth_dependencies["check"].call_args[0][0]
    assert "replicate" in df_checked.columns

# ----------------------------------------------------------------------------
# 4. Tests for _read_binding_df
# ----------------------------------------------------------------------------

@pytest.fixture
def base_binding_df():
    cols = ["theta_obs", "theta_std", "titrant_conc", "genotype", "titrant_name"]
    return pd.DataFrame({c: [] for c in cols})

@pytest.fixture
def mock_binding_dependencies(mocker, base_binding_df):
    mock_df = base_binding_df.copy()
    return {
        "read": mocker.patch("tfscreen.util.read_dataframe", return_value=mock_df),
        "set_geno": mocker.patch("tfscreen.genetics.set_categorical_genotype", return_value=mock_df),
        "check": mocker.patch("tfscreen.util.check_columns"),
        "add_group": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class.add_group_columns", return_value=mock_df),
        "df": mock_df
    }

def test_read_binding_df_links_existing(mock_binding_dependencies):
    existing_df = pd.DataFrame()
    mock_df = mock_binding_dependencies["df"]
    
    _read_binding_df("path.csv", existing_df=existing_df)
    
    mock_binding_dependencies["add_group"].assert_called_once_with(
        mock_df,
        group_cols=["genotype", "titrant_name"],
        group_name="map_theta_group",
        existing_df=existing_df
    )

# ----------------------------------------------------------------------------
# 5. Tests for _build_binding_tm
# ----------------------------------------------------------------------------

@pytest.fixture
def mock_binding_tm_class():
    mock_instance = MagicMock()
    # Correctly patch the imported TensorManager
    with patch("tfscreen.analysis.hierarchical.growth_model.model_class.TensorManager") as mock_class:
        mock_class.return_value = mock_instance
        yield mock_class, mock_instance

@pytest.fixture
def dummy_binding_df():
    """Returns a DF matching structure of _read_binding_df output."""
    return pd.DataFrame({
        "titrant_name": ["A"], "titrant_conc": [10.0], 
        "genotype": ["wt"], "theta_obs": [0.5], "theta_std": [0.1], 
        "map_theta_group": [1],
        "log_titrant_conc": [2.3] # FIXED: Added required column
    })

def test_build_binding_tm_call_sequence(mock_binding_tm_class, dummy_binding_df):
    mock_class, mock_instance = mock_binding_tm_class
    
    # We pass a copy because the function modifies it
    df_to_test = dummy_binding_df.copy()
    
    _build_binding_tm(df_to_test)
    
    expected_calls = [
        call.add_pivot_index(tensor_dim_name="titrant_name", cat_column="titrant_name"),
        call.add_pivot_index(tensor_dim_name="titrant_conc", cat_column="pivot_titrant_conc"),
        call.add_pivot_index(tensor_dim_name="genotype", cat_column="genotype"),
        call.add_data_tensor("theta_obs", dtype=ANY),
        call.add_data_tensor("theta_std", dtype=ANY),
        call.add_data_tensor("map_theta_group", dtype=int),
        call.add_data_tensor("titrant_conc", dtype=ANY),
        call.add_data_tensor("log_titrant_conc", dtype=ANY),
        call.create_tensors()
    ]
    assert mock_instance.method_calls == expected_calls

# ----------------------------------------------------------------------------
# 6. Tests for _get_wt_info
# ----------------------------------------------------------------------------

def test_get_wt_info_logic():
    mock_tm = MagicMock()
    mock_tm.tensor_dim_names = ["replicate", "genotype"]
    mock_tm.tensor_dim_labels = [
        np.array(["R1"]),
        np.array(["mut1", "wt", "mut2"])
    ]
    
    result = _get_wt_info(mock_tm)
    
    assert result["wt_index"] == 1
    assert result["num_not_wt"] == 2
    assert np.array_equal(result["not_wt_mask"], [0, 2])

def test_get_wt_info_missing_wt():
    mock_tm = MagicMock()
    mock_tm.tensor_dim_names = ["genotype"]
    mock_tm.tensor_dim_labels = [np.array(["A", "B"])]
    
    with pytest.raises(ValueError, match="Exactly one 'wt' entry"):
        _get_wt_info(mock_tm)

# ----------------------------------------------------------------------------
# 7. Tests for _extract_param_est
# ----------------------------------------------------------------------------

def test_extract_param_est():
    """Tests the parameter extraction mapping logic."""
    input_df = pd.DataFrame({"map_id": [0, 1], "gene_name": ["G1", "G2"]})
    samples = np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
    param_posteriors = {"test_param": samples}
    q_to_get = {"median": 0.5}
    
    results = _extract_param_est(
        input_df=input_df,
        params_to_get=["param"], 
        map_column="map_id",
        get_columns=["gene_name"],
        in_run_prefix="test_",
        param_posteriors=param_posteriors,
        q_to_get=q_to_get
    )
    
    assert "param" in results
    df_res = results["param"]
    assert df_res.iloc[0]["median"] == 10.0

# ----------------------------------------------------------------------------
# 8. Tests for ModelClass
# ----------------------------------------------------------------------------

@pytest.fixture
def model_class_dependencies(mocker):
    """Mocks dependencies required to instantiate ModelClass without real data."""
    mocks = {
        "read_growth": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._read_growth_df"),
        "build_growth": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._build_growth_tm"),
        "build_growth_theta": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._build_growth_theta_tm"),
        "read_binding": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._read_binding_df"),
        "build_binding": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._build_binding_tm"),
        "get_wt": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._get_wt_info"),
        "pop_dataclass": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class.populate_dataclass"),
        "comps": mocker.patch.dict("tfscreen.analysis.hierarchical.growth_model.model.MODEL_COMPONENT_NAMES", {
            "condition_growth": {"hierarchical": (1, MagicMock()), "fixed": (2, MagicMock())},
            "ln_cfu0": {"hierarchical": (1, MagicMock())},
            "dk_geno": {"hierarchical": (1, MagicMock())},
            "activity": {"horseshoe": (1, MagicMock())},
            "theta": {"hill": (1, MagicMock())},
            "theta_growth_noise": {"none": (0, MagicMock())},
            "theta_binding_noise": {"none": (0, MagicMock())}
        }, clear=True)
    }
    
    # Setup return values for TMs
    mock_tm = MagicMock()
    mock_tm.tensor_shape = (1,1,1,1)
    mock_tm.map_sizes = {}
    
    # FIXED: Populate tensors dict with ALL keys accessed by _initialize_data
    mock_tm.tensors = {
        "ln_cfu": MagicMock(), "ln_cfu_std": MagicMock(),
        "t_pre": MagicMock(), "t_sel": MagicMock(),
        "map_ln_cfu0": MagicMock(), "map_condition_pre": MagicMock(),
        "map_condition_sel": MagicMock(), "map_genotype": MagicMock(),
        "map_theta": MagicMock(), "good_mask": MagicMock(),
        "titrant_conc": MagicMock(), "log_titrant_conc": MagicMock(),
        "map_theta_group": MagicMock(), "theta_obs": MagicMock(),
        "theta_std": MagicMock()
    }
    mock_tm.df = pd.DataFrame()
    
    mocks["build_growth"].return_value = mock_tm
    mocks["build_growth_theta"].return_value = mock_tm
    mocks["build_binding"].return_value = mock_tm
    mocks["get_wt"].return_value = {"wt_index":0, "not_wt_mask": [], "num_not_wt": 0}
    
    return mocks

def test_model_class_initialization(model_class_dependencies):
    """Verifies ModelClass orchestrates data loading and object creation."""
    model = ModelClass(
        growth_df="growth.csv", 
        binding_df="binding.csv",
        condition_growth="hierarchical"
    )
    
    assert model.data is not None
    assert model.priors is not None
    assert model.control is not None
    assert model.control.condition_growth == 1

def test_model_class_invalid_component(model_class_dependencies):
    """Verifies that bad component strings raise ValueError."""
    with pytest.raises(ValueError, match="not recognized"):
        ModelClass("g.csv", "b.csv", condition_growth="invalid_name")