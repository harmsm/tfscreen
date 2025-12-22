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
    _build_binding_tm,
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
    # Needs columns required for pivots: replicate, t_sel, condition_pre, 
    # condition_sel_reduced, titrant_name, titrant_conc, genotype
    return pd.DataFrame({
        "replicate": [1], 
        "t_sel": [0], 
        "genotype": ["wt"],
        "condition_pre": ["A"],
        "condition_sel_reduced": [0],
        "titrant_name": ["T1"],
        "titrant_conc": [0.1]
    })

def test_build_growth_tm_call_sequence(mock_tensor_manager_class, dummy_growth_df):
    """Tests the exact call sequence on TensorManager for growth data."""
    mock_class, mock_instance = mock_tensor_manager_class
    
    # Pass copy to avoid modification issues
    result_tm = _build_growth_tm(dummy_growth_df.copy())

    # Check pivots
    expected_calls = [
        call.add_pivot_index(tensor_dim_name="replicate", cat_column="replicate"),
        call.add_ranked_pivot_index(tensor_dim_name="time",
                                    rank_column="t_sel",
                                    select_cols=['replicate','genotype','treatment']),
        call.add_pivot_index(tensor_dim_name="condition_pre", cat_column="condition_pre"),
        call.add_pivot_index(tensor_dim_name="condition_sel", cat_column="condition_sel_reduced"),
        call.add_pivot_index(tensor_dim_name="titrant_name", cat_column="titrant_name"),
        call.add_pivot_index(tensor_dim_name="titrant_conc", cat_column="pivot_titrant_conc"),
        call.add_pivot_index(tensor_dim_name="genotype", cat_column="genotype"),
        
        # Data Tensors
        call.add_data_tensor("ln_cfu", dtype=ANY),
        call.add_data_tensor("ln_cfu_std", dtype=ANY),
        call.add_data_tensor("t_pre", dtype=ANY),
        call.add_data_tensor("t_sel", dtype=ANY),

        # Map Tensors
        # 1. Condition map
        call.add_map_tensor(select_cols=["replicate"],
                            select_pool_cols=["condition_pre","condition_sel"],
                            name="condition"),
        # 2. Param maps
        call.add_map_tensor(["replicate","condition_pre","genotype"], name="ln_cfu0"),
        call.add_map_tensor("genotype", name="genotype"),
        call.add_map_tensor(["titrant_name","titrant_conc","genotype"], name="theta"),

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
        "read": mocker.patch("tfscreen.util.io.read_dataframe", return_value=mock_df),
        "set_geno": mocker.patch("tfscreen.genetics.set_categorical_genotype", return_value=mock_df),
        "get_cfu": mocker.patch("tfscreen.util.dataframe.get_scaled_cfu", return_value=mock_df),
        "check": mocker.patch("tfscreen.util.dataframe.check_columns"),
        "add_group": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class.add_group_columns", return_value=mock_df),
        "df": mock_df
    }
    return mocks

def test_read_growth_df_defaults(mock_growth_dependencies):
    mock_df = mock_growth_dependencies["df"]
    _read_growth_df("path.csv")
    
    # Verify add_group_columns called for 'treatment' and 'map_theta_group'
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
    
    # Mocking check_columns to avoid errors since we aren't running real validation logic
    
    result = _read_growth_df(df_no_rep)
    
    # The function adds 'replicate' = 1 if missing
    # Since we mocked internal functions to return 'df_no_rep', we check the object passed to check_columns
    # or verify the logic ran.
    # Actually, in the code: `growth_df["replicate"] = 1`. This modifies the df in place.
    # Since we return a mock_df from read/set/get, let's verify replicate is in there.
    assert "replicate" in result.columns

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
        "read": mocker.patch("tfscreen.util.io.read_dataframe", return_value=mock_df),
        "set_geno": mocker.patch("tfscreen.genetics.set_categorical_genotype", return_value=mock_df),
        "check": mocker.patch("tfscreen.util.dataframe.check_columns"),
        "df": mock_df
    }

def test_read_binding_df_basic(mock_binding_dependencies):
    """
    Tests basic reading logic. Note: _read_binding_df verifies consistency 
    between growth and binding dataframes. We need to mock that check or provide compatible DFs.
    """
    growth_df = pd.DataFrame({"genotype": ["A"], "titrant_name": ["T1"]})
    binding_df = pd.DataFrame({
        "genotype": ["A"], "titrant_name": ["T1"], 
        "theta_obs": [0.5], "theta_std": [0.1], "titrant_conc": [1.0]
    })
    
    mock_binding_dependencies["read"].return_value = binding_df
    mock_binding_dependencies["set_geno"].return_value = binding_df
    
    _read_binding_df("path.csv", growth_df=growth_df)
    
    mock_binding_dependencies["check"].assert_called()

def test_read_binding_df_raises_on_mismatch(mock_binding_dependencies):
    """Ensures it raises ValueError if binding has genotypes not in growth."""
    growth_df = pd.DataFrame({"genotype": ["A"], "titrant_name": ["T1"]})
    binding_df = pd.DataFrame({
        "genotype": ["B"], "titrant_name": ["T1"], # "B" not in growth
        "theta_obs": [0.5], "theta_std": [0.1], "titrant_conc": [1.0]
    })
    
    mock_binding_dependencies["read"].return_value = binding_df
    mock_binding_dependencies["set_geno"].return_value = binding_df
    
    with pytest.raises(ValueError, match="not seen in the growth_df"):
        _read_binding_df("path.csv", growth_df=growth_df)

# ----------------------------------------------------------------------------
# 5. Tests for _build_binding_tm
# ----------------------------------------------------------------------------

@pytest.fixture
def mock_binding_tm_class():
    mock_instance = MagicMock()
    with patch("tfscreen.analysis.hierarchical.growth_model.model_class.TensorManager") as mock_class:
        mock_class.return_value = mock_instance
        yield mock_class, mock_instance

@pytest.fixture
def dummy_binding_df():
    return pd.DataFrame({
        "titrant_name": ["A"], "titrant_conc": [10.0], 
        "genotype": ["wt"], "theta_obs": [0.5], "theta_std": [0.1]
    })

def test_build_binding_tm_call_sequence(mock_binding_tm_class, dummy_binding_df):
    mock_class, mock_instance = mock_binding_tm_class
    
    df_to_test = dummy_binding_df.copy()
    _build_binding_tm(df_to_test)
    
    expected_calls = [
        call.add_pivot_index(tensor_dim_name="titrant_name", cat_column="titrant_name"),
        call.add_pivot_index(tensor_dim_name="titrant_conc", cat_column="pivot_titrant_conc"),
        call.add_pivot_index(tensor_dim_name="genotype", cat_column="genotype"),
        call.add_data_tensor("theta_obs", dtype=ANY),
        call.add_data_tensor("theta_std", dtype=ANY),
        call.create_tensors()
    ]
    assert mock_instance.method_calls == expected_calls


# ----------------------------------------------------------------------------
# 7. Tests for _extract_param_est
# ----------------------------------------------------------------------------

def test_extract_param_est():
    """Tests the parameter extraction mapping logic."""
    input_df = pd.DataFrame({"map_id": [0, 1], "gene_name": ["G1", "G2"]})
    samples = np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
    # The function flattens the posterior, so we mock specific keys
    # param 'test_param' -> samples
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
    # Check that it pulled the correct values
    assert df_res.iloc[0]["median"] == 10.0
    assert df_res.iloc[1]["median"] == 20.0

# ----------------------------------------------------------------------------
# 8. Tests for ModelClass
# ----------------------------------------------------------------------------

@pytest.fixture
def model_class_dependencies(mocker):
    """Mocks dependencies required to instantiate ModelClass without real data."""
    mocks = {
        "read_growth": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._read_growth_df"),
        "build_growth": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._build_growth_tm"),
        "read_binding": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._read_binding_df"),
        "build_binding": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._build_binding_tm"),
        "pop_dataclass": mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class.populate_dataclass"),
        "comps": mocker.patch.dict("tfscreen.analysis.hierarchical.growth_model.model_class.model_registry", {
            "condition_growth": {"hierarchical": MagicMock(), "fixed": MagicMock()},
            "ln_cfu0": {"hierarchical": MagicMock()},
            "dk_geno": {"hierarchical": MagicMock()},
            "activity": {"horseshoe": MagicMock()},
            "theta": {"hill": MagicMock()},
            "transformation": {"congression": MagicMock(), "single": MagicMock()},
            "theta_growth_noise": {"none": MagicMock()},
            "theta_binding_noise": {"none": MagicMock()},
            "observe_binding": MagicMock(),
            "observe_growth": MagicMock()
        }, clear=True)
    }
    
    # Configure Component Mocks to return dicts for get_priors/get_guesses
    for k, v in mocks["comps"].items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                sub_v.get_priors.return_value = {}
                sub_v.get_guesses.return_value = {}
    
    # Setup return values for TMs
    # Growth TM needs explicit labels for wt finding and titrant extraction
    mock_growth_tm = MagicMock()
    mock_growth_tm.tensor_shape = (1,1,1,1,1,1,1) # 7 dims per code
    mock_growth_tm.map_sizes = {}
    
    # Provide tensor_dim_names/labels for WT search and Titrant conc extraction
    # indices: ... 5=titrant_conc, 6=genotype
    mock_growth_tm.tensor_dim_names = ["replicate", "time", "pre", "sel", "name", "titrant_conc", "genotype"]
    mock_growth_tm.tensor_dim_labels = [
        [], [], [], [], [],
        [0.0, 1.0, 10.0], # titrant_conc labels
        np.array(["wt", "mut1"]) # genotype labels
    ]
    
    mock_growth_tm.tensors = {
        "ln_cfu": MagicMock(), "ln_cfu_std": MagicMock(),
        "t_pre": MagicMock(), "t_sel": MagicMock(),
        "map_condition_pre": MagicMock(), "map_condition_sel": MagicMock(),
        "good_mask": MagicMock(), 
        # map tensors
        "ln_cfu0": MagicMock(), "genotype": MagicMock(), "theta": MagicMock(), "condition": MagicMock()
    }
    mock_growth_tm.df = pd.DataFrame()
    
    # Binding TM
    mock_binding_tm = MagicMock()
    mock_binding_tm.tensor_shape = (1,1,1)
    mock_binding_tm.tensor_dim_names = ["name", "titrant_conc", "genotype"]
    mock_binding_tm.tensor_dim_labels = [
        [], 
        [0.0, 1.0, 10.0], # titrant_conc
        np.array(["wt"]) # genotype labels (subset of growth)
    ]
    mock_binding_tm.tensors = {"theta_obs": MagicMock(), "theta_std": MagicMock()}

    mocks["build_growth"].return_value = mock_growth_tm
    mocks["build_binding"].return_value = mock_binding_tm
    
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
    assert "condition_growth" in model.settings
    assert model.settings["condition_growth"] == "hierarchical"

def test_model_class_invalid_component(model_class_dependencies):
    """Verifies that bad component strings raise ValueError."""
    # We used patch.dict so looking up "invalid_name" in the mock registry will fail
    with pytest.raises(ValueError, match="not recognized"):
        ModelClass("g.csv", "b.csv", condition_growth="invalid_name")

# ----------------------------------------------------------------------------
# Tests for _extract_param_est (UPDATED)
# ----------------------------------------------------------------------------

def test_extract_param_est_basic():
    """Tests the parameter extraction mapping logic with sorted inputs."""
    # Map IDs: 0, 1. Corresponds to posterior columns 0, 1.
    input_df = pd.DataFrame({"map_id": [0, 1], "gene_name": ["G1", "G2"]})
    
    # 3 samples, 2 parameters
    samples = np.array([
        [10.0, 20.0], 
        [10.0, 20.0], 
        [10.0, 20.0]
    ])
    
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
    
    # Check shape
    assert len(df_res) == 2
    
    # Check G1 (id 0) -> 10.0
    row_g1 = df_res[df_res["gene_name"] == "G1"].iloc[0]
    assert row_g1["median"] == 10.0
    
    # Check G2 (id 1) -> 20.0
    row_g2 = df_res[df_res["gene_name"] == "G2"].iloc[0]
    assert row_g2["median"] == 20.0

def test_extract_param_est_sorting_logic():
    """
    CRITICAL: Tests that extraction maps correctly even if input DF is unsorted 
    or has gaps/duplicates.
    """
    # Input DF is scrambled. 
    # Map ID 0 -> 'A', Map ID 2 -> 'C', Map ID 1 -> 'B'
    input_df = pd.DataFrame({
        "map_id": [2, 0, 1, 0], # Duplicate 0 to check duplicate drop
        "label": ["C", "A", "B", "A"]
    })
    
    # Posterior: (1 sample, 3 params). 
    # Index 0 -> 100 (A)
    # Index 1 -> 200 (B)
    # Index 2 -> 300 (C)
    samples = np.array([[100.0, 200.0, 300.0]])
    
    param_posteriors = {"p_val": samples}
    
    res = _extract_param_est(
        input_df=input_df,
        params_to_get=["val"],
        map_column="map_id",
        get_columns=["label"],
        in_run_prefix="p_",
        param_posteriors=param_posteriors,
        q_to_get={"mean": 0.5}
    )
    
    df = res["val"]
    
    # Should have 3 unique rows (A, B, C)
    assert len(df) == 3
    
    # Verify mappings
    val_A = df.loc[df["label"] == "A", "mean"].iloc[0]
    val_B = df.loc[df["label"] == "B", "mean"].iloc[0]
    val_C = df.loc[df["label"] == "C", "mean"].iloc[0]
    
    assert val_A == 100.0
    assert val_B == 200.0
    assert val_C == 300.0

def test_extract_param_est_reshaping():
    """
    Tests that multi-dimensional posterior arrays (e.g. from multiple chains)
    are flattened correctly before quantile calculation.
    """
    input_df = pd.DataFrame({"map_id": [0]})
    
    # Shape: (2 chains, 5 samples, 1 param) -> Should flatten to (10 samples, 1 param)
    samples = np.ones((2, 5, 1)) * 5.0 
    
    param_posteriors = {"p_x": samples}
    
    res = _extract_param_est(
        input_df=input_df,
        params_to_get=["x"],
        map_column="map_id",
        get_columns=[],
        in_run_prefix="p_",
        param_posteriors=param_posteriors,
        q_to_get={"val": 0.5}
    )
    
    assert res["x"].iloc[0]["val"] == 5.0

# ----------------------------------------------------------------------------
# Tests for ModelClass.extract_parameters (NEW)
# ----------------------------------------------------------------------------

@pytest.fixture
def initialized_model_class():
    """
    Creates a ModelClass instance with mocked internals to test extract_parameters
    without running the full initialization pipeline.
    """
    # Create a barebones object
    model = MagicMock(spec=ModelClass)
    model._transformation = "none"
    
    # Mock the TensorManager and its dataframes
    mock_tm = MagicMock()
    mock_tm.df = pd.DataFrame({
        "genotype": ["wt", "mut"], 
        "map_theta_group": [0, 1],
        "map_theta": [0, 1],
        "map_genotype": [0, 1],
        "map_ln_cfu0": [0, 1],
        "titrant_name": ["T1", "T1"],
        "titrant_conc": [0, 10],
        "replicate": [1, 1],
        "condition_pre": ["A", "A"]
    })
    
    # Mock map groups for 'independent' condition models
    mock_tm.map_groups = {
        "condition": pd.DataFrame({
            "map_condition": [0], "replicate": [1], "condition": ["A"]
        })
    }
    
    model.growth_tm = mock_tm
    
    # Bind the method under test to the mock instance
    # We use __get__ to bind the function to the object instance
    model.extract_parameters = ModelClass.extract_parameters.__get__(model, ModelClass)
    
    return model

def test_extract_parameters_hill_config(initialized_model_class):
    """Test extraction logic when theta_model is 'hill'."""
    model = initialized_model_class
    
    # Configure model settings
    model._theta = "hill"
    model._condition_growth = "hierarchical" # Should extract condition params
    model._dk_geno = "none"                  # Should SKIP dk_geno
    model._activity = "fixed"                # Should SKIP activity
    
    # Mock posteriors
    # We need keys for hill (theta_n, etc) and condition (condition_growth_m, etc)
    # Shapes must match map sizes. map_theta_group has max index 1 -> size 2.
    posteriors = {
        "theta_hill_n": np.zeros((1, 2)),
        "theta_log_hill_K": np.zeros((1, 2)),
        "theta_theta_high": np.zeros((1, 2)),
        "theta_theta_low": np.zeros((1, 2)),
        "condition_growth_m": np.zeros((1, 1)),
        "condition_growth_k": np.zeros((1, 1)),
        "ln_cfu0": np.zeros((1, 2)) # Default inclusion
    }
    
    # Run
    params = model.extract_parameters(posteriors)
    
    # Check Keys
    assert "hill_n" in params
    assert "theta_high" in params
    assert "growth_m" in params
    assert "dk_geno" not in params
    assert "activity" not in params
    assert "theta" not in params # categorical param

def test_extract_parameters_categorical_config(initialized_model_class):
    """Test extraction logic when theta_model is 'categorical'."""
    model = initialized_model_class
    model._theta = "categorical"
    model._condition_growth = "none" # Should skip condition params
    model._dk_geno = "hierarchical"  # Should extract dk_geno
    model._activity = "horseshoe"    # Should extract activity
    
    posteriors = {
        "theta_theta": np.zeros((1, 2)),
        "dk_geno": np.zeros((1, 2)),
        "activity": np.zeros((1, 2)),
        "ln_cfu0": np.zeros((1, 2))
    }
    
    params = model.extract_parameters(posteriors)
    
    assert "theta" in params
    assert "dk_geno" in params
    assert "activity" in params
    assert "hill_n" not in params
    assert "growth_m" not in params

def test_extract_parameters_file_loading(initialized_model_class):
    """Test that passing a filename string loads the numpy file."""
    model = initialized_model_class
    # Minimal config to trigger at least one extraction
    model._theta = "hill"
    model._condition_growth = "none"
    model._dk_geno = "none"
    model._activity = "fixed"
    
    mock_npz = {
        "theta_hill_n": np.zeros((1, 2)),
        "theta_log_hill_K": np.zeros((1, 2)),
        "theta_theta_high": np.zeros((1, 2)),
        "theta_theta_low": np.zeros((1, 2))
    }
    
    with patch("numpy.load", return_value=mock_npz) as mock_load:
        model.extract_parameters("posterior.npz")
        mock_load.assert_called_once_with("posterior.npz")

def test_extract_parameters_invalid_quantiles(initialized_model_class):
    """Test error handling for bad quantile input."""
    model = initialized_model_class
    with pytest.raises(ValueError, match="q_to_get should be a dictionary"):
        model.extract_parameters({}, q_to_get=[0.5]) # List instead of dict