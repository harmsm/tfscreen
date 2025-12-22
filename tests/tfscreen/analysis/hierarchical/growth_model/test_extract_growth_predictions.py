import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass

@pytest.fixture
def mock_model():
    """Create a ModelClass instance with minimal mocked internals."""
    model = MagicMock(spec=ModelClass)
    
    # Mock growth_df with index columns
    # We need: replicate, time, condition_pre, condition_sel, titrant_name, titrant_conc, genotype
    # Plus their _idx versions
    df = pd.DataFrame({
        "ln_cfu": [10.0, 11.0],
        "replicate_idx": [0, 0],
        "time_idx": [0, 1],
        "condition_pre_idx": [0, 0],
        "condition_sel_idx": [0, 0],
        "titrant_name_idx": [0, 0],
        "titrant_conc_idx": [0, 0],
        "genotype_idx": [0, 0]
    })
    model.growth_df = df
    
    # Bind the method under test
    model.extract_growth_predictions = ModelClass.extract_growth_predictions.__get__(model, ModelClass)
    
    return model

@pytest.fixture
def mock_posteriors():
    """Create a dictionary of mock posterior samples."""
    # (num_samples, replicate, time, condition_pre, condition_sel, titrant_name, titrant_conc, genotype)
    # Shape: (5, 1, 2, 1, 1, 1, 1, 1)
    num_samples = 5
    shape = (num_samples, 1, 2, 1, 1, 1, 1, 1)
    
    # Fill with predictable values
    # row 0 (time 0) -> all 10.5
    # row 1 (time 1) -> all 11.5
    growth_pred = np.zeros(shape)
    growth_pred[:, 0, 0, 0, 0, 0, 0, 0] = 10.5
    growth_pred[:, 0, 1, 0, 0, 0, 0, 0] = 11.5
    
    return {
        "growth_pred": growth_pred
    }

def test_extract_growth_predictions_basic(mock_model, mock_posteriors):
    """Test default behavior for extracting growth predictions."""
    results = mock_model.extract_growth_predictions(mock_posteriors)
    
    # Check output structure
    assert isinstance(results, pd.DataFrame)
    assert "median" in results.columns
    assert len(results) == len(mock_model.growth_df)
    
    # Verify values
    # Row 0 -> median should be 10.5
    assert results.iloc[0]["median"] == 10.5
    # Row 1 -> median should be 11.5
    assert results.iloc[1]["median"] == 11.5

def test_extract_growth_predictions_custom_quantiles(mock_model, mock_posteriors):
    """Test extracting custom quantiles."""
    q_to_get = {"mean": 0.5, "low": 0.1, "high": 0.9}
    results = mock_model.extract_growth_predictions(mock_posteriors, q_to_get=q_to_get)
    
    assert "mean" in results.columns
    assert "low" in results.columns
    assert "high" in results.columns
    assert results.iloc[0]["mean"] == 10.5

def test_extract_growth_predictions_missing_field(mock_model):
    """Test error handling when growth_pred is missing from posteriors."""
    with pytest.raises(ValueError, match="'growth_pred' not found"):
        mock_model.extract_growth_predictions({"something_else": np.array([1])})

def test_extract_growth_predictions_invalid_quantiles(mock_model, mock_posteriors):
    """Test error handling for bad quantile input."""
    with pytest.raises(ValueError, match="q_to_get should be a dictionary"):
        mock_model.extract_growth_predictions(mock_posteriors, q_to_get=[0.5])

def test_extract_growth_predictions_file_loading(mock_model):
    """Test loading posteriors from file."""
    mock_posteriors_data = {
        "growth_pred": np.zeros((2, 1, 1, 1, 1, 1, 1, 1))
    }
    
    # Set up growth_df with one row for the single element in growth_pred
    mock_model.growth_df = pd.DataFrame({
        "replicate_idx": [0],
        "time_idx": [0],
        "condition_pre_idx": [0],
        "condition_sel_idx": [0],
        "titrant_name_idx": [0],
        "titrant_conc_idx": [0],
        "genotype_idx": [0]
    })
    
    with patch("numpy.load", return_value=mock_posteriors_data) as mock_load:
        mock_model.extract_growth_predictions("mock_growth.npz")
        mock_load.assert_called_once_with("mock_growth.npz")
