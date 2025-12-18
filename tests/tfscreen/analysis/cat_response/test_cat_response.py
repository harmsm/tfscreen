
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from tfscreen.analysis.cat_response.cat_response import cat_response

@pytest.fixture
def mock_df():
    # Create a DataFrame with 2 genotypes
    data = {
        "genotype": ["WT", "WT", "MUT", "MUT"],
        "titrant_conc": [1.0, 2.0, 1.0, 2.0],
        "theta_est": [10.0, 20.0, 5.0, 15.0],
        "theta_std": [0.1, 0.1, 0.2, 0.2]
    }
    return pd.DataFrame(data)

@patch("tfscreen.analysis.cat_response.cat_response.cat_fit")
@patch("tfscreen.analysis.cat_response.cat_response.MODEL_LIBRARY")
def test_cat_response_integration(mock_library, mock_cat_fit, mock_df):
    """Test standard cat_response execution aggregating multiple genotypes."""
    
    # Setup MODEL_LIBRARY to have one model
    mock_library.keys.return_value = ["linear"]
    mock_library.__getitem__.return_value = {"param_names": ["m", "b"]}
    
    # Setup cat_fit return values for each call (2 genotypes)
    # We need to return (flat_output, pred_df)
    
    # WT Results
    flat_WT = {
        "status": "success",
        "best_model": "linear",
        "best_model_R2": 0.99,
        "best_model_AIC_weight": 0.8,
        "AIC_weight|linear": 0.8,
        "R2|linear": 0.99,
        "linear|m|est": 10.0,
        "linear|m|std": 0.1,
        "linear|b|est": 0.0,
        "linear|b|std": 0.1
    }
    pred_df_WT = pd.DataFrame({
        "model": ["linear"]*2, "x": [1,2], "y": [10,20], "y_std": [0,0], "is_best_model": [True]*2
    })
    
    # MUT Results
    flat_MUT = {
        "status": "success", 
        "best_model": "linear",
        "best_model_R2": 0.95,
        "best_model_AIC_weight": 0.9,
        "AIC_weight|linear": 0.9,
        "R2|linear": 0.95,
        "linear|m|est": 5.0, # Difference
        "linear|m|std": 0.2,
        "linear|b|est": 0.0,
        "linear|b|std": 0.2
    }
    pred_df_MUT = pd.DataFrame({
        "model": ["linear"]*2, "x": [1,2], "y": [5,15], "y_std": [0,0], "is_best_model": [True]*2
    })
    
    mock_cat_fit.side_effect = [(flat_WT, pred_df_WT), (flat_MUT, pred_df_MUT)]

    # Run function
    model_dfs, summary_df, pred_df = cat_response(
        mock_df, 
        models_to_run=["linear"],
        verbose=False
    )
    
    # 1. Verify cat_fit calls
    assert mock_cat_fit.call_count == 2
    
    # 2. Verify summary DataFrame
    assert len(summary_df) == 2
    assert "best_model" in summary_df.columns
    assert "w_linear" in summary_df.columns # Renamed from AIC_weight|linear
    assert summary_df.loc["WT", "best_model_R2"] == 0.99
    
    # 3. Verify model_dataframes
    assert "linear" in model_dfs
    lin_df = model_dfs["linear"]
    assert len(lin_df) == 2
    # Check column renaming: linear|m|est -> m_est
    assert "m_est" in lin_df.columns
    assert lin_df.loc["WT", "m_est"] == 10.0
    assert lin_df.loc["MUT", "m_est"] == 5.0
    
    # 4. Verify pred_df
    assert len(pred_df) == 4 # 2 genotypes * 2 points
    assert "genotype" in pred_df.columns
    assert set(pred_df["genotype"]) == {"WT", "MUT"}

@patch("tfscreen.analysis.cat_response.cat_response.cat_fit")
@patch("tfscreen.analysis.cat_response.cat_response.MODEL_LIBRARY")
def test_cat_response_default_models(mock_library, mock_cat_fit, mock_df):
    """Test behavior when models_to_run is None (defaults to library keys)."""
    
    mock_library.keys.return_value = ["m1"]
    mock_library.__getitem__.return_value = {"param_names": ["p"]}
    
    flat_res = {
        "status": "success", "best_model": "m1", "best_model_R2": 1, "best_model_AIC_weight": 1.0,
        "AIC_weight|m1": 1, "R2|m1": 1,
        "m1|p|est": 0, "m1|p|std": 0
    }
    mock_cat_fit.return_value = (flat_res, pd.DataFrame({"model": ["m1"], "is_best_model":[True]}))
    
    model_dfs, summary_df, pred_df = cat_response(mock_df) # models_to_run=None
    
    assert "m1" in model_dfs
    # Verify cat_fit was called with models_to_run=["m1"]
    _, kwargs = mock_cat_fit.call_args
    assert kwargs["models_to_run"] == ["m1"]

