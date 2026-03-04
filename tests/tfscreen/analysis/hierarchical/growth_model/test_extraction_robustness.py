import pytest
import pandas as pd
import numpy as np
import h5py
import os
from unittest.mock import MagicMock, patch
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
from tfscreen.analysis.hierarchical.growth_model.extraction import (
    extract_parameters, 
    extract_theta_curves,
    extract_growth_predictions
)
from tfscreen.analysis.hierarchical.posteriors import get_posterior_samples

@pytest.fixture
def mock_model():
    """Create a ModelClass instance with minimal mocked internals."""
    model = MagicMock(spec=ModelClass)
    model._transformation = "none"
    model._theta = "none"
    model._condition_growth = "none"
    model._dk_geno = "none"
    model._activity = "fixed"
    model._growth_transition = "instant"
    
    # Mock TensorManager
    mock_tm = MagicMock()
    mock_tm.df = pd.DataFrame({
        "genotype": ["wt"],
        "titrant_name": ["iptg"],
        "titrant_conc": [1.0],
        "replicate": ["1"],
        "condition": ["cond1"],
        "condition_pre": ["pre1"],
        "condition_sel": ["sel1"],
        "map_theta": [0],
        "map_theta_group": [0],
        "map_condition": [0],
        "map_ln_cfu0": [0],
        "map_genotype": [0],
        "time_idx": [0],
        "replicate_idx": [0],
        "condition_pre_idx": [0],
        "condition_sel_idx": [0],
        "titrant_name_idx": [0],
        "titrant_conc_idx": [0],
        "genotype_idx": [0]
    })
    mock_tm.map_groups = {
        'condition': pd.DataFrame({"replicate": ["1"], "condition": ["cond1"], "map_condition": [0]})
    }
    mock_tm.tensor_dim_names = ["replicate", "time", "condition_pre", "condition_sel", "titrant_name", "titrant_conc", "genotype"]
    mock_tm.tensor_dim_labels = [["1"], ["1"], ["pre1"], ["sel1"], ["iptg"], [1.0], ["wt"]]
    model.growth_tm = mock_tm
    
    # Mock growth_df
    model.growth_df = mock_tm.df.copy()
    model.growth_df["t_pre"] = 0.0
    model.growth_df["t_sel"] = 1.0
    model.growth_df["ln_cfu"] = 10.0
    model.growth_df["ln_cfu_std"] = 0.1
    
    return model


def test_extract_parameters_all_models(mock_model):
    """Test extract_parameters with various model configurations to hit coverage."""
    # Test 'categorical' theta
    mock_model._theta = "categorical"
    posteriors = {"theta_theta": np.random.rand(10, 1)}
    params = extract_parameters(mock_model, posteriors)
    assert "theta" in params
    
    # Test 'linear_independent' condition growth (same as linear)
    mock_model._theta = "none"
    mock_model._condition_growth = "linear_independent"
    posteriors = {
        "condition_growth_m": np.random.rand(10, 1),
        "condition_growth_k": np.random.rand(10, 1)
    }
    params = extract_parameters(mock_model, posteriors)
    assert "growth_m" in params
    assert "growth_k" in params

    # Test 'independent' condition growth (same as linear)
    mock_model._condition_growth = "independent"
    params = extract_parameters(mock_model, posteriors)
    assert "growth_m" in params

    # Test 'hierarchical' condition growth (same as linear)
    mock_model._condition_growth = "hierarchical"
    params = extract_parameters(mock_model, posteriors)
    assert "growth_m" in params

    # Test 'linear' condition growth
    mock_model._condition_growth = "linear"
    params = extract_parameters(mock_model, posteriors)
    assert "growth_m" in params

    # Test 'power' condition growth
    mock_model._condition_growth = "power"
    posteriors = {
        "condition_growth_k": np.random.rand(10, 1),
        "condition_growth_m": np.random.rand(10, 1),
        "condition_growth_n": np.random.rand(10, 1)
    }
    params = extract_parameters(mock_model, posteriors)
    assert "growth_k" in params
    
    # Test 'saturation' condition growth
    mock_model._condition_growth = "saturation"
    posteriors = {
        "condition_growth_min": np.random.rand(10, 1),
        "condition_growth_max": np.random.rand(10, 1)
    }
    params = extract_parameters(mock_model, posteriors)
    assert "growth_min" in params

    # Test 'memory' growth transition
    mock_model._condition_growth = "none"
    mock_model._growth_transition = "memory"
    posteriors = {
        "growth_transition_tau0": np.random.rand(10, 1),
        "growth_transition_k1": np.random.rand(10, 1),
        "growth_transition_k2": np.random.rand(10, 1)
    }
    params = extract_parameters(mock_model, posteriors)
    assert "growth_transition_tau0" in params

    # Test 'baranyi' growth transition
    mock_model._growth_transition = "baranyi"
    posteriors = {
        "growth_transition_tau_lag": np.random.rand(10, 1),
        "growth_transition_k_sharp": np.random.rand(10, 1)
    }
    params = extract_parameters(mock_model, posteriors)
    assert "growth_transition_tau_lag" in params

    # Test 'hierarchical' dk_geno (includes activity)
    mock_model._growth_transition = "instant"
    mock_model._dk_geno = "hierarchical"
    mock_model._activity = "hierarchical"
    posteriors = {
        "ln_cfu0": np.random.rand(10, 1),
        "dk_geno": np.random.rand(10, 1),
        "activity": np.random.rand(10, 1)
    }
    params = extract_parameters(mock_model, posteriors)
    assert "ln_cfu0" in params
    assert "dk_geno" in params
    assert "activity" in params

    # Test 'horseshoe' activity (ensure ln_cfu0 and dk_geno are present if dk_geno is hierarchical)
    mock_model._activity = "horseshoe"
    posteriors["activity"] = np.random.rand(10, 1) # activity is already there from previous step
    params = extract_parameters(mock_model, posteriors)
    assert "activity" in params

    # Reset and test 'hill' theta
    mock_model._theta = "hill"
    mock_model._condition_growth = "none"
    mock_model._dk_geno = "none"
    mock_model._activity = "fixed"
    mock_model._growth_transition = "instant"
    posteriors = {
        "theta_hill_n": np.random.rand(10, 1),
        "theta_log_hill_K": np.random.rand(10, 1),
        "theta_theta_high": np.random.rand(10, 1),
        "theta_theta_low": np.random.rand(10, 1)
    }
    params = extract_parameters(mock_model, posteriors)
    assert "hill_n" in params

def test_extract_theta_curves_manual_genotype(mock_model):
    """Test extract_theta_curves with manual_titrant_df including genotype."""
    mock_model._theta = "hill"
    manual_df = pd.DataFrame({
        "titrant_name": ["iptg"],
        "titrant_conc": [2.0],
        "genotype": ["wt"]
    })
    posteriors = {
        "theta_hill_n": np.random.rand(10, 1),
        "theta_log_hill_K": np.random.rand(10, 1),
        "theta_theta_high": np.random.rand(10, 1),
        "theta_theta_low": np.random.rand(10, 1)
    }
    df = extract_theta_curves(mock_model, posteriors, manual_titrant_df=manual_df)
    assert "median" in df.columns
    assert len(df) == 1

def test_extract_theta_curves_broadcast(mock_model):
    """Test extract_theta_curves broadcasting across genotypes."""
    mock_model._theta = "hill"
    # Create two genotypes in the model
    mock_model.growth_tm.df = pd.DataFrame({
        "genotype": ["wt", "mut"],
        "titrant_name": ["iptg", "iptg"],
        "titrant_conc": [1.0, 1.0],
        "map_theta_group": [0, 1]
    })
    manual_df = pd.DataFrame({
        "titrant_name": ["iptg"],
        "titrant_conc": [2.0]
    })
    posteriors = {
        "theta_hill_n": np.random.rand(10, 2),
        "theta_log_hill_K": np.random.rand(10, 2),
        "theta_theta_high": np.random.rand(10, 2),
        "theta_theta_low": np.random.rand(10, 2)
    }
    df = extract_theta_curves(mock_model, posteriors, manual_titrant_df=manual_df)
    assert len(df) == 2
    assert set(df["genotype"]) == {"wt", "mut"}

def test_extract_parameters_file_loading(mock_model, tmp_path):
    """Test loading posteriors from file paths."""
    posteriors = {"some_param": np.random.rand(10, 1)}
    
    # .npz
    npz_path = os.path.join(tmp_path, "test.npz")
    np.savez(npz_path, **posteriors)
    params = extract_parameters(mock_model, npz_path)
    assert isinstance(params, dict)
    
    # .h5
    h5_path = os.path.join(tmp_path, "test.h5")
    with h5py.File(h5_path, "w") as f:
        for k, v in posteriors.items():
            f.create_dataset(k, data=v)
    params = extract_parameters(mock_model, h5_path)
    assert isinstance(params, dict)

def test_extract_parameters_errors(mock_model):
    """Test error handling in extract_parameters."""
    with pytest.raises(ValueError, match="q_to_get should be a dictionary"):
        extract_parameters(mock_model, {}, q_to_get=[0.5])

def test_extract_theta_curves_mapping_error(mock_model):
    """Test extract_theta_curves with a mapping error to hit the except branch."""
    mock_model._theta = "hill"
    manual_df = pd.DataFrame({"titrant_name": ["iptg"], "titrant_conc": [1.0]})
    # Patch the map method of the index to raise an exception
    with patch("pandas.Index.map") as mock_map:
        mock_map.side_effect = Exception("Mapping error")
        with pytest.raises(ValueError, match=r"Some \(genotype, titrant_name\) pairs"):
            extract_theta_curves(mock_model, {}, manual_titrant_df=manual_df)

def test_extract_parameters_h5_file(mock_model, tmp_path):
    """Test extract_parameters loading from an HDF5 file path."""
    h5_path = os.path.join(tmp_path, "params.h5")
    
    # We need a model config that actually extracts something
    mock_model._transformation = "congression"
    
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("transformation_lam", data=np.random.rand(10, 1))
        # shape for mu/sigma is (num_samples, num_titrant_name, num_titrant_conc, 1)
        f.create_dataset("transformation_mu", data=np.random.rand(10, 1, 1, 1))
        f.create_dataset("transformation_sigma", data=np.random.rand(10, 1, 1, 1))
    
    params = extract_parameters(mock_model, h5_path)
    assert "lam" in params
    assert "mu" in params

def test_extract_theta_curves_hdf5_path(mock_model, tmp_path):
    """Test extract_theta_curves loading from an HDF5 file path."""
    mock_model._theta = "hill"
    h5_path = os.path.join(tmp_path, "theta.h5")
    posteriors = {
        "theta_hill_n": np.random.rand(10, 1),
        "theta_log_hill_K": np.random.rand(10, 1),
        "theta_theta_high": np.random.rand(10, 1),
        "theta_theta_low": np.random.rand(10, 1)
    }
    with h5py.File(h5_path, "w") as f:
        for k, v in posteriors.items():
            f.create_dataset(k, data=v)
    
    df = extract_theta_curves(mock_model, h5_path)
    assert "median" in df.columns

def test_extract_growth_predictions_hdf5_blocking(mock_model, tmp_path):
    """Test extract_growth_predictions with HDF5 blocking and fallback."""
    # Ensure growth_df is sorted to improve locality for HDF5 coverage
    mock_model.growth_df = mock_model.growth_df.sort_values(by=[
        "replicate_idx", "time_idx", "condition_pre_idx", 
        "condition_sel_idx", "titrant_name_idx", 
        "titrant_conc_idx", "genotype_idx"
    ])

    h5_path = os.path.join(tmp_path, "growth.h5")
    # Shape: (num_samples, replicate, time, condition_pre, condition_sel, titrant_name, titrant_conc, genotype)
    shape = (2, 1, 1, 1, 1, 1, 1, 1)
    data = np.random.rand(*shape)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("growth_pred", data=data)
    
    # Reload from file to ensure it's an HDF5 dataset
    with h5py.File(h5_path, "r") as f:
        # Test normal blocking
        df = extract_growth_predictions(mock_model, f, max_block_elements=1000)
        assert "median" in df.columns
        
        # Test fallback (max_block_elements=0)
        df_fallback = extract_growth_predictions(mock_model, f, max_block_elements=0)
        assert "median" in df_fallback.columns

    # Test load from file path (.h5) to hit the File loading branch
    df_path = extract_growth_predictions(mock_model, h5_path)
    assert "median" in df_path.columns

def test_extract_growth_predictions_hdf5_fallback_sparse(mock_model, tmp_path):
    """Test extract_growth_predictions HDF5 fallback for large spatial volume."""
    mock_model.growth_df = pd.DataFrame({
        "replicate": ["1", "2"],
        "genotype": ["wt", "wt"],
        "condition_pre": ["pre1", "pre2"],
        "condition_sel": ["sel1", "sel2"],
        "titrant_name": ["iptg", "iptg"],
        "titrant_conc": [1.0, 2.0],
        "t_pre": [0.0, 0.0],
        "t_sel": [1.0, 1.0],
        "ln_cfu": [10.0, 10.0],
        "ln_cfu_std": [0.1, 0.1],
        "replicate_idx": [0, 1],
        "time_idx": [0, 0],
        "condition_pre_idx": [0, 1],
        "condition_sel_idx": [0, 1],
        "titrant_name_idx": [0, 0],
        "titrant_conc_idx": [0, 1],
        "genotype_idx": [0, 0]
    })
    
    h5_path = os.path.join(tmp_path, "growth_large.h5")
    # Shape: (num_samples, replicate, time, condition_pre, condition_sel, titrant_name, titrant_conc, genotype)
    shape = (2, 2, 1, 2, 2, 1, 2, 1)
    data = np.random.rand(*shape)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("growth_pred", data=data)
    
    with h5py.File(h5_path, "r") as f:
        # Trigger the "else" branch in spatial volume check by setting max_block_elements VERY small
        df = extract_growth_predictions(mock_model, f, max_block_elements=1)
        assert "median" in df.columns
        assert len(df) == 2

def test_extract_theta_curves_q_to_get_error(mock_model):
    """Test extract_theta_curves with invalid q_to_get."""
    mock_model._theta = "hill"
    with pytest.raises(ValueError, match="q_to_get should be a dictionary"):
        extract_theta_curves(mock_model, {}, q_to_get=[0.5])

def test_extract_growth_predictions_hdf5_is_h5_check(mock_model, tmp_path):
    """Explicitly check that we are hitting the is_h5 branch."""
    h5_path = os.path.join(tmp_path, "growth_check.h5")
    shape = (2, 1, 1, 1, 1, 1, 1, 1)
    data = np.random.rand(*shape)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("growth_pred", data=data)
    
    with h5py.File(h5_path, "r") as f:
        # We pass the dataset object directly
        ds = f["growth_pred"]
        # extract_growth_predictions takes param_posteriors which is the file/group
        df = extract_growth_predictions(mock_model, f)
        assert "median" in df.columns
