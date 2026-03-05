import pytest
import h5py
import numpy as np
import os
from tfscreen.analysis.hierarchical.growth_model.prediction import predict
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
import pandas as pd

def test_predict_h5_indexing(tmp_path):
    """Test that predict works with H5 file indexing (sorting fix)."""
    
    # 1. Create dummy ModelClass
    growth_df = pd.DataFrame({
        "genotype": ["wt"],
        "titrant_name": ["tit1"],
        "titrant_conc": [0.0],
        "condition_pre": ["pre1"],
        "condition_sel": ["sel1"],
        "t_pre": [10.0],
        "t_sel": [0.0],
        "ln_cfu": [0.0],
        "ln_cfu_std": [0.1],
        "replicate": [1]
    })
    binding_df = pd.DataFrame({
        "genotype": ["wt"],
        "titrant_name": ["tit1"],
        "titrant_conc": [0.5],
        "theta_obs": [0.5],
        "theta_std": [0.01]
    })
    mc = ModelClass(growth_df, binding_df)
    
    # 2. Create an H5 file with some "posteriors"
    h5_path = tmp_path / "posteriors.h5"
    with h5py.File(h5_path, "w") as f:
        # Need enough samples to likely get out-of-order random selection if not sorted
        num_avail = 100
        # Just create dummy arrays for some parameters
        # ln_cfu0_offset is one used in the model
        # Shape should match (num_samples, replicate, condition_pre, genotype)
        # Based on dummy growth_df: (1, 1, 1)
        f.create_dataset("ln_cfu0_offset", data=np.random.normal(size=(num_avail, 1, 1, 1)))
        f.create_dataset("ln_cfu0_hyper_loc", data=np.random.normal(size=(num_avail,)))
        f.create_dataset("ln_cfu0_hyper_scale", data=np.random.normal(size=(num_avail,)))
        # Add other necessary params for the model to not crash during trace
        # Actually predict() runs a trace to find sites. 
        # But it also calls get_posterior_samples.
        
    # 3. Call predict with num_samples=10. 
    # Prior to fix, this would have a high chance of failing if indices were [5, 2, ...]
    # We don't need to mock anything else if we just want to see it not fail on indexing.
    # However, predict() will try to run the full predictive.
    # We might need to mock Predictive if we just want to test the slice logic.
    
    from unittest.mock import patch
    with patch("tfscreen.analysis.hierarchical.growth_model.prediction.Predictive") as mock_pred:
        # Mock growth_pred output. 
        # TensorManager has 7 dimensions. Prediction output is (num_samples, *tensor_shape)
        mock_pred.return_value.return_value = {"growth_pred": np.zeros((10, 1, 1, 1, 1, 1, 1, 1))}
        
        # This calls val = val[sample_indices] inside prediction.py
        try:
            results = predict(mc, str(h5_path), num_samples=10)
        except TypeError as e:
            if "Indexing elements must be in increasing order" in str(e):
                pytest.fail("H5 indexing failed: elements not in increasing order")
            raise e

    assert isinstance(results, pd.DataFrame)
