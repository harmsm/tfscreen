import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from tfscreen.analysis.independent.cfu_to_theta import (
    #_prep_inference_df <- tested in its own file
    _prep_param_guesses,
    _build_param_df,
    _setup_inference,
    _run_inference,
    cfu_to_theta
)

# --- Test Fixtures ---
# Fixtures provide a consistent, reusable setup for tests.

import itertools

@pytest.fixture
def base_df():
    """
    A DENSE, valid DataFrame fixture for input.
    Grid: 2 genotypes x 1 replicate x 2 libraries x 2 conditions = 8 rows.
    """
    # Define the dimensions of the dense grid
    genotypes = ["wt", "V30A"]
    replicates = [1]
    libraries = ["L1", "L2"]
    # Use titrant as the varying condition for simplicity
    conditions = [
        ("iptg", 0.1, "pre1", "sel1"), 
        ("none", 0.0, "pre2", "sel2")
    ]

    # Create all combinations of the primary identifiers
    product = list(itertools.product(genotypes, replicates, libraries, conditions))
    
    # Unpack the product into a list of dictionaries to build the DataFrame
    data_list = []
    for genotype, replicate, library, (titrant, conc, pre, sel) in product:
        data_list.append({
            "genotype": genotype,
            "replicate": replicate,
            "library": library,
            "titrant_name": titrant,
            "titrant_conc": conc,
            "condition_pre": pre,
            "condition_sel": sel,
        })
    
    df = pd.DataFrame(data_list)
    
    # Add the value columns required for the function
    num_rows = len(df)
    df["ln_cfu"] = np.linspace(10.1, 11.5, num_rows)
    df["ln_cfu_std"] = 0.1
    df["t_pre"] = 2.0
    df["t_sel"] = 4.0
    
    return df

@pytest.fixture
def calibration_data():
    """A correctly structured calibration data dictionary fixture."""
    # Mock dataframe for background growth rates (indexed by titrant_name)
    k_bg_df = pd.DataFrame({
        "m": [0.1, 0.0], "b": [1.0, 1.1]
    }, index=["iptg", "none"])

    # Mock dataframe for condition-specific growth effects
    # (indexed by condition name)
    dk_cond_df = pd.DataFrame({
        "m": [0.2, 0.3], "b": [2.0, 3.0]
    }, index=["pre1", "pre2"])
    
    # Add the selection conditions to the index as well
    dk_cond_df = pd.concat([
        dk_cond_df, 
        dk_cond_df.rename(index={"pre1":"sel1", "pre2":"sel2"})
    ])

    return {"k_bg_df": k_bg_df, "dk_cond_df": dk_cond_df}

# --- Test Cases ---



@pytest.fixture
def prepped_df():
    """
    A fixture simulating the DataFrame after _prep_inference_df has run.
    Crucially, 'genotype' is an ordered categorical.
    """
    genotype_order = ["wt", "V30A", "V30C"]
    cat_type = pd.api.types.CategoricalDtype(
        categories=genotype_order, ordered=True
    )
    data = {
        "genotype": pd.Series(["wt", "V30A", "wt", "V30C"], dtype=cat_type),
        "library": ["L1", "L1", "L2", "L1"],
        "replicate": pd.Series([1, 1, 1, 2], dtype="Int64"),
        "condition_sel": ["sel1", "sel1", "sel2", "sel1"],
        "titrant_name": ["iptg", "iptg", "iptg", "none"],
        "titrant_conc": [0.1, 0.1, 0.1, 0.0],
    }
    return pd.DataFrame(data)

# --- Test Cases ---

def test_prep_param_guesses_happy_path(mocker, prepped_df, calibration_data):
    """
    Tests the main success path of _prep_param_guesses.
    """
    # --- Mock external dependencies ---

    # 1. Mock the return value of the individual fitting function.
    #    Note: 'genotype' is intentionally an 'object' dtype here to test
    #    that our function correctly recasts it.
    mock_indiv_params = pd.DataFrame({
        "genotype": ["wt", "V30A", "V30C"],
        "library": ["L1", "L1", "L1"],
        "replicate": pd.Series([1, 1, 2], dtype="Int64"),
        "lnA0_est": [10.5, 11.0, 11.5],
        "dk_geno": [0.0, -0.5, -0.8],
    })
    # We only care about the first return value
    mock_get_indiv = mocker.patch(
        "tfscreen.analysis.independent.cfu_to_theta.get_indiv_growth",
        return_value=(mock_indiv_params, None)
    )

    # 2. Mock the wild-type theta calculation to return a fixed array
    mock_get_theta = mocker.patch(
        "tfscreen.analysis.independent.cfu_to_theta.get_wt_theta",
        return_value=np.array([0.8, 0.8, 0.8, 0.0])
    )

    # --- Run the function ---
    non_sel_conditions = ["sel1"]
    result_df = _prep_param_guesses(
        df=prepped_df.copy(), # Pass a copy to avoid side effects
        non_sel_conditions=non_sel_conditions,
        calibration_data=calibration_data
    )

    # --- Assertions ---

    # 1. Verify the non-selection mask was created correctly
    assert "_dk_geno_mask" in result_df.columns
    expected_mask = [True, True, False, True] # True where condition_sel is 'sel1'
    assert result_df["_dk_geno_mask"].tolist() == expected_mask

    # 2. Verify get_indiv_growth was called correctly
    mock_get_indiv.assert_called_once()
    # Check that the dataframe passed to the mock had the mask in it
    call_args, _ = mock_get_indiv.call_args
    assert "_dk_geno_mask" in call_args[0].columns

    # 3. Verify the crucial genotype dtype was preserved after the merge
    assert result_df["genotype"].dtype == prepped_df["genotype"].dtype
    assert result_df["genotype"].cat.ordered

    # 4. Verify the guess values were merged correctly
    assert "lnA0_est" in result_df.columns
    assert "dk_geno" in result_df.columns
    # Check a value: the second row (V30A, L1, 1) should get lnA0_est of 11.0
    assert result_df.loc[1, "lnA0_est"] == 11.0
    # The third row (wt, L2, 1) does not exist in our mock indiv_param_df,
    # so its guess should be NaN after the left merge.
    assert pd.isna(result_df.loc[2, "lnA0_est"])

    # 5. Verify the wt_theta guess was calculated and assigned
    mock_get_theta.assert_called_once()
    assert "wt_theta" in result_df.columns
    assert result_df.loc[0, "wt_theta"] == 0.8 # From our mock return



# --- Test Fixtures ---

@pytest.fixture
def param_builder_df():
    """
    A fixture to test the different branches of _build_param_df.
    Includes columns for grouping and various guess scenarios.
    """
    data = {
        "genotype": ["wt", "wt", "V30A", "V30A"],
        "library": ["L1", "L2", "L1", "L1"],
        # Guesses with non-zero mean and std dev
        "guess_varying": [10.0, 12.0, 15.0, 15.0],
        # Guesses with non-zero mean and zero std dev
        "guess_constant": [5.0, 5.0, 5.0, 5.0],
         # Guesses with zero mean and zero std dev
        "guess_zero": [0.0, 0.0, 0.0, 0.0],
    }
    return pd.DataFrame(data)


# --- Test Cases ---

def test_build_param_df_no_transform(param_builder_df):
    """
    Tests the basic functionality when no transform is applied.
    """
    df_idx, param_df = _build_param_df(
        df=param_builder_df,
        series_selector=["genotype", "library"],
        base_name="lnA0",
        guess_column="guess_varying",
        transform="none",
        offset=100
    )

    # There are 4 unique genotype/library groups
    assert len(param_df) == 3
    # df_idx should have the same length as the input df
    assert len(df_idx) == len(param_builder_df)

    # Check indices are correctly offset
    assert df_idx.min() == 100
    assert param_df["idx"].min() == 100
    assert np.array_equal(param_df["idx"].values, [100, 101, 102])

    # Check parameter names and class
    assert param_df.loc[0, "name"] == "lnA0_V30A_L1" # Note: default groupby sorts keys
    assert param_df.loc[0, "class"] == "lnA0"

    # Check guess was correctly assigned (V30A/L1's first value is 15.0)
    assert param_df.loc[0, "guess"] == 15.0
    
    # Check that scale parameters have their default values
    assert param_df.loc[0, "transform"] == "none"
    assert param_df.loc[0, "scale_mu"] == 0
    assert param_df.loc[0, "scale_sigma"] == 1


def test_build_param_df_scale_transform_normal(param_builder_df):
    """
    Tests the 'scale' transform logic with guesses that have a non-zero std dev.
    """
    _, param_df = _build_param_df(
        df=param_builder_df,
        series_selector=["genotype"], # Grouping by genotype only
        base_name="dk_geno",
        guess_column="guess_varying",
        transform="scale",
        offset=0
    )
    
    # We group by genotype. The first guess for each is [15.0, 10.0]
    expected_guesses = np.array([15.0, 10.0])
    expected_mu = np.mean(expected_guesses)
    expected_sig = np.std(expected_guesses)

    assert np.isclose(param_df["scale_mu"].iloc[0], expected_mu)
    assert np.isclose(param_df["scale_sigma"].iloc[0], expected_sig)


def test_build_param_df_scale_transform_zero_std(param_builder_df):
    """
    Tests the 'scale' transform fallback when std dev of guesses is zero.
    """
    _, param_df = _build_param_df(
        df=param_builder_df,
        series_selector=["genotype"],
        base_name="dk_geno",
        guess_column="guess_constant", # All guesses are 5.0
        transform="scale",
        offset=0
    )

    # Guesses are all 5.0, so mu=5.0 and sig=0.0.
    # The function should use mu as sigma.
    assert param_df["scale_mu"].iloc[0] == 5.0
    assert param_df["scale_sigma"].iloc[0] == 5.0


def test_build_param_df_scale_transform_all_zero(param_builder_df):
    """
    Tests the 'scale' transform fallback when both mean and std dev are zero.
    """
    _, param_df = _build_param_df(
        df=param_builder_df,
        series_selector=["genotype"],
        base_name="dk_geno",
        guess_column="guess_zero", # All guesses are 0.0
        transform="scale",
        offset=0
    )

    # Guesses are all 0.0, so mu=0.0 and sig=0.0.
    # The function should use the default sigma of 1.0.
    assert param_df["scale_mu"].iloc[0] == 0.0
    assert param_df["scale_sigma"].iloc[0] == 1.0



# --- Test Fixtures ---

@pytest.fixture
def setup_df():
    """
    A fixture simulating a fully prepped DataFrame, ready for _setup_inference.
    """
    data = {
        "genotype": ["wt", "V30A", "wt"],
        "library": ["L1", "L1", "L2"],
        "replicate": [1, 1, 1],
        "titrant_name": ["iptg", "iptg", "none"],
        "titrant_conc": [0.1, 0.1, 0.0],
        "lnA0_est": [10.1, 10.5, 10.2],
        "dk_geno": [0.0, -0.5, 0.0],
        "wt_theta": [0.8, 0.8, 0.0],
        "t_pre": [2, 2, 2],
        "t_sel": [4, 4, 4],
        "dk_m_pre": [0.1, 0.1, 0.2],
        "dk_m_sel": [0.1, 0.1, 0.2],
        "k_bg_b": [1.0, 1.0, 1.1],
        "k_bg_m": [0.5, 0.5, 0.0],
        "dk_b_pre": [2.0, 2.0, 2.1],
        "dk_b_sel": [2.2, 2.2, 2.3],
        "ln_cfu": [15.0, 14.0, 16.0],
        "ln_cfu_std": [0.1, 0.1, 0.1],
    }
    return pd.DataFrame(data)


# --- Test Cases ---

def test_setup_inference_happy_path(mocker, setup_df):
    """
    Tests the main success path of _setup_inference, mocking the helper.
    """
    # --- Mock the _build_param_df helper function ---
    # We expect it to be called 3 times. We'll define the return value
    # for each call.

    # 1. Mock return for lnA0
    lnA0_idx = np.array([0, 1, 2]) # Three unique lnA0 groups
    lnA0_df = pd.DataFrame({"idx": [0, 1, 2], "class": "lnA0"})

    # 2. Mock return for dk_geno
    dk_geno_idx = np.array([3, 4, 3]) # Two unique genotype groups
    dk_geno_df = pd.DataFrame({"idx": [3, 4], "class": "dk_geno"})

    # 3. Mock return for theta
    theta_idx = np.array([5, 5, 6]) # Two unique titrant groups
    theta_df = pd.DataFrame({"idx": [5, 6], "class": "theta"})

    mock_builder = mocker.patch(
        "tfscreen.analysis.independent.cfu_to_theta._build_param_df",
        side_effect=[
            (lnA0_idx, lnA0_df),
            (dk_geno_idx, dk_geno_df),
            (theta_idx, theta_df),
        ]
    )

    # --- Run the function ---
    y_obs, y_std, X, param_df = _setup_inference(setup_df)

    # --- Assertions ---

    # 1. Verify calls to the mock
    assert mock_builder.call_count == 3
    # Check that the offset was updated correctly for the last call
    last_call_args, last_call_kwargs = mock_builder.call_args_list[2]
    # offset = max lnA0 idx (2) + max dk_geno idx (4) = 5
    assert last_call_kwargs["offset"] == 5

    # 2. Verify the final param_df
    # Total params = 3 (lnA0) + 2 (dk_geno) + 2 (theta) = 7
    assert len(param_df) == 7
    assert param_df["class"].tolist() == ["lnA0"]*3 + ["dk_geno"]*2 + ["theta"]*2

    # 3. Verify the design matrix X
    assert X.shape == (3, 7) # 3 observations, 7 total parameters

    # Check row 1 (index 0) of the design matrix
    # This row corresponds to lnA0_idx=0, dk_geno_idx=3, theta_idx=5
    assert X[0, 0] == 1.0 # lnA0 value
    assert X[0, 3] == (setup_df.loc[0, "t_pre"] + setup_df.loc[0, "t_sel"]) # dk_geno value
    assert X[0, 5] == (setup_df.loc[0, "dk_m_pre"] * setup_df.loc[0, "t_pre"] +
                       setup_df.loc[0, "dk_m_sel"] * setup_df.loc[0, "t_sel"]) # theta value
    # Sum should be these three values, as all others should be zero
    assert np.isclose(np.sum(X[0, :]), X[0, 0] + X[0, 3] + X[0, 5])

    # 4. Verify y_obs and y_std
    assert y_obs.shape == (3,)
    assert y_std.shape == (3,)
    
    # Manually calculate y_obs for the first row and check
    row0 = setup_df.iloc[0]
    k_bg = row0["k_bg_b"] + row0["k_bg_m"] * row0["titrant_conc"]
    constant_terms = ((k_bg + row0["dk_b_pre"]) * row0["t_pre"] + 
                      (k_bg + row0["dk_b_sel"]) * row0["t_sel"])
    expected_y_obs_0 = row0["ln_cfu"] - constant_terms
    assert np.isclose(y_obs[0], expected_y_obs_0)



# --- Test Cases ---

def test_run_inference(mocker):
    """
    Tests the main logic of the revised _run_inference, which takes only a
    FitManager object. It verifies guess clipping, correct calls to fitting
    routines, and the structure of the returned dataframes.
    """
    # --- 1. Setup Mocks and Test Data ---

    # Create a mock FitManager object to control its attributes and methods
    mock_fm = MagicMock()
    mock_fm.guesses_transformed = np.array([-1.0, 0.5, 5.0])
    mock_fm.lower_bounds_transformed = np.array([0.0, 0.0, 0.0])
    mock_fm.upper_bounds_transformed = np.array([4.0, 4.0, 4.0])
    mock_fm.y_obs = np.array([10, 11, 12])
    mock_fm.y_std = np.array([0.1, 0.1, 0.1])
    mock_fm.param_df = pd.DataFrame({"name": ["p1", "p2", "p3"]})

    # Mock the transformation methods to perform simple, testable math
    mock_fm.back_transform.side_effect = lambda p: p * 10
    mock_fm.back_transform_std_err.side_effect = lambda p, s: s / 2

    # Mock the external fitting and prediction functions
    mock_lsq = mocker.patch(
        "tfscreen.analysis.independent.cfu_to_theta.run_least_squares",
        return_value=(
            np.array([0.1, 0.5, 3.9]),     # fitted params
            np.array([0.02, 0.05, 0.08]),  # std_errors
            np.eye(3),                     # covariance matrix
            None
        )
    )
    mock_predict = mocker.patch(
        "tfscreen.analysis.independent.cfu_to_theta.predict_with_error",
        return_value=(
            np.array([10.1, 11.0, 11.9]), # y_pred
            np.array([0.11, 0.11, 0.12])  # y_pred_std
        )
    )

    # --- 2. Run the function ---
    param_df, pred_df = _run_inference(fm=mock_fm)

    # --- 3. Assertions ---

    # Verify run_least_squares was called
    mock_lsq.assert_called_once()
    # Get the keyword arguments it was called with
    _, lsq_kwargs = mock_lsq.call_args

    # CRITICAL: Verify that the guesses were clipped before fitting
    expected_clipped_guesses = np.array([0.0, 0.5, 4.0])
    assert np.array_equal(lsq_kwargs["guesses"], expected_clipped_guesses)

    # Verify prediction was called with the fitted params from the lsq result
    mock_predict.assert_called_once()
    fitted_params = mock_lsq.return_value[0]
    assert np.array_equal(mock_predict.call_args[0][1], fitted_params)

    # Verify the final parameter dataframe
    assert "est" in param_df.columns
    assert "std" in param_df.columns
    # Check that the back-transformation methods were called and values are correct
    # est = 0.1 * 10 = 1.0
    assert np.isclose(param_df.loc[0, "est"], 1.0)
    # std = 0.02 / 2 = 0.01
    assert np.isclose(param_df.loc[0, "std"], 0.01)

    # Verify the final prediction dataframe (it's created from scratch)
    assert len(pred_df) == len(mock_fm.y_obs)
    assert "calc_est" in pred_df.columns
    assert "y_obs" in pred_df.columns
    assert np.isclose(pred_df.loc[0, "calc_est"], 10.1) # From mock_predict
    assert np.isclose(pred_df.loc[0, "y_obs"], 10.0)    # From mock_fm



def test_cfu_to_theta_integration(mocker, base_df, calibration_data):
    """
    An integration test for the main cfu_to_theta pipeline.
    
    This test mocks the lowest-level external dependencies (the numerical
    optimizer and the FitManager class) and verifies that the entire pipeline,
    including batching, runs correctly and produces a properly formatted
    final output.
    """
    # --- 1. Mock External Dependencies ---

    # Mock file readers to inject our fixture data
    mocker.patch("tfscreen.analysis.independent.cfu_to_theta.read_dataframe", return_value=base_df.copy())
    mocker.patch("tfscreen.analysis.independent.cfu_to_theta.read_calibration", return_value=calibration_data)
    
    # Mock helpers that are tested elsewhere
    mocker.patch("tfscreen.analysis.independent.cfu_to_theta.get_scaled_cfu", side_effect=lambda df, **kwargs: df)
    mocker.patch("tfscreen.analysis.independent.cfu_to_theta.check_columns")

    # Mock the preliminary fitting in _prep_param_guesses
    # UPDATED: Mock data now uses genotypes from the new base_df
    mock_indiv_params = pd.DataFrame({
        "genotype": ["wt", "V30A"], "library": ["L1", "L1"], "replicate": [1, 1],
        "lnA0_est": [10.5, 11.0], "dk_geno": [0.0, -0.5],
    })
    mocker.patch("tfscreen.analysis.independent.cfu_to_theta.get_indiv_growth", return_value=(mock_indiv_params, None))
    # UPDATED: Mock return value must match the 8-row size of the new base_df
    mocker.patch("tfscreen.analysis.independent.cfu_to_theta.get_wt_theta", return_value=np.array([0.8] * 8))

    # Mock the numerical optimizer in _run_inference
    mock_lsq = mocker.patch(
        "tfscreen.analysis.independent.cfu_to_theta.run_least_squares",
        return_value=(np.array([0.1, 0.2]), np.array([0.01, 0.02]), np.eye(2), None)
    )
    mocker.patch("tfscreen.analysis.independent.cfu_to_theta.predict_with_error", 
                 return_value=(np.array([10.1, 11.1]), np.array([0.15, 0.15])))
    
    # Mock the FitManager class
    mock_fm_instance = MagicMock()
    mock_fm_instance.back_transform.side_effect = lambda p: p * 10
    mock_fm_instance.back_transform_std_err.side_effect = lambda p, s: s / 2
    mock_fm_class = mocker.patch("tfscreen.analysis.independent.cfu_to_theta.FitManager")
    mock_fm_class.return_value = mock_fm_instance

    mock_fm_instance.guesses_transformed = np.array([0.5, 0.6])
    mock_fm_instance.lower_bounds_transformed = np.array([0.0, 0.0])
    mock_fm_instance.upper_bounds_transformed = np.array([1.0, 1.0])
    mock_fm_instance.y_obs = np.array([10, 11])
    mock_fm_instance.y_std = np.array([0.1, 0.1])
    mock_fm_instance.param_df = pd.DataFrame({"name": ["p1", "p2"]})
    mock_fm_instance.predict_from_transformed = MagicMock()

    mocker.patch("tfscreen.analysis.independent.cfu_to_theta.tqdm", side_effect=lambda x: x)

    # --- 2. Run the Full Pipeline ---
    
    # The new base_df has 8 rows and 2 genotypes. max_batch_size is on genotypes.
    # We will test with a batch size that forces one batch.
    param_df, pred_df = cfu_to_theta(
        df=base_df,
        non_sel_conditions=["sel1"],
        calibration_data=calibration_data,
        max_batch_size=4
    )

    # --- 3. Assertions ---

    # UPDATED: The new base_df has 2 genotypes. With max_batch_size=4,
    # they will all fit into two batches.
    assert mock_lsq.call_count == 2
    assert mock_fm_class.call_count == 2

    # Verify the final prediction dataframe
    assert len(pred_df) == len(base_df)
    assert "calc_est" in pred_df.columns
    assert "genotype" in pred_df.columns

    # Verify the final parameter dataframe
    assert "est" in param_df.columns
    assert np.isclose(param_df["est"].iloc[0], 1.0)
    assert np.isclose(param_df["est"].iloc[1], 2.0)