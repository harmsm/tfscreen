import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass

@pytest.fixture
def mock_model():
    """Create a ModelClass instance with minimal mocked internals."""
    model = MagicMock(spec=ModelClass)
    model._theta = "hill"
    
    # Mock TensorManager and its DataFrame
    mock_tm = MagicMock()
    mock_tm.df = pd.DataFrame({
        "genotype": ["wt", "wt", "mut", "mut"],
        "titrant_name": ["iptg", "iptg", "iptg", "iptg"],
        "titrant_conc": [0.0, 1.0, 0.0, 1.0],
        "map_theta_group": [0, 0, 1, 1]
    })
    model.growth_tm = mock_tm
    
    # Bind the method under test
    model.extract_theta_curves = ModelClass.extract_theta_curves.__get__(model, ModelClass)
    
    return model

@pytest.fixture
def mock_posteriors():
    """Create a dictionary of mock posterior samples."""
    # 10 samples, 2 theta groups
    num_samples = 10
    num_groups = 2
    return {
        "theta_hill_n": np.ones((num_samples, num_groups)) * 2,
        "theta_log_hill_K": np.ones((num_samples, num_groups)) * -1.0, # log(0.36) approx
        "theta_theta_high": np.ones((num_samples, num_groups)) * 0.9,
        "theta_theta_low": np.ones((num_samples, num_groups)) * 0.1
    }

def test_extract_theta_curves_basic(mock_model, mock_posteriors):
    """Test default behavior using data from growth_tm.df."""
    results = mock_model.extract_theta_curves(mock_posteriors)
    
    # Check output structure
    assert isinstance(results, pd.DataFrame)
    assert "genotype" in results.columns
    assert "titrant_name" in results.columns
    assert "titrant_conc" in results.columns
    assert "median" in results.columns
    
    # Should have unique (genotype, titrant_name, titrant_conc)
    # wt: 0.0, 1.0; mut: 0.0, 1.0 -> 4 rows
    assert len(results) == 4
    
    # Verify values for wt at conc=1.0
    # hill_n=2, log_K=-1.0 (K=0.367), high=0.9, low=0.1
    # occupancy = 1 / (1 + exp(-2 * (log(1.0) - (-1.0)))) = 1 / (1 + exp(-2)) = 1 / (1 + 0.135) = 0.88
    # theta = 0.1 + (0.9 - 0.1) * 0.88 = 0.1 + 0.8 * 0.88 = 0.804
    wt_1 = results[(results["genotype"] == "wt") & (results["titrant_conc"] == 1.0)]
    assert np.allclose(wt_1["median"], 0.804, atol=1e-3)

def test_extract_theta_curves_manual_df(mock_model, mock_posteriors):
    """Test providing a manual titrant DataFrame."""
    manual_df = pd.DataFrame({
        "titrant_name": ["iptg", "iptg"],
        "titrant_conc": [0.5, 2.0]
    })
    
    results = mock_model.extract_theta_curves(mock_posteriors, manual_titrant_df=manual_df)
    
    # Should broadcast across 'wt' and 'mut' -> 4 rows
    assert len(results) == 4
    assert set(results["genotype"]) == {"wt", "mut"}
    assert set(results["titrant_conc"]) == {0.5, 2.0}

def test_extract_theta_curves_manual_df_with_genotypes(mock_model, mock_posteriors):
    """Test providing a manual titrant DataFrame with explicit genotypes."""
    manual_df = pd.DataFrame({
        "genotype": ["wt", "mut"],
        "titrant_name": ["iptg", "iptg"],
        "titrant_conc": [0.5, 2.0]
    })
    
    results = mock_model.extract_theta_curves(mock_posteriors, manual_titrant_df=manual_df)
    
    assert len(results) == 2
    assert results.iloc[0]["genotype"] == "wt"
    assert results.iloc[1]["genotype"] == "mut"

def test_extract_theta_curves_wrong_theta_model(mock_model, mock_posteriors):
    """Test that it raises ValueError if theta model is not 'hill'."""
    mock_model._theta = "categorical"
    with pytest.raises(ValueError, match="only available for models where"):
        mock_model.extract_theta_curves(mock_posteriors)

def test_extract_theta_curves_missing_columns(mock_model, mock_posteriors):
    """Test error handling for missing columns in manual_df."""
    manual_df = pd.DataFrame({"titrant_name": ["iptg"]}) # Missing conc
    with pytest.raises(Exception): # check_columns raises an error
         mock_model.extract_theta_curves(mock_posteriors, manual_titrant_df=manual_df)

def test_extract_theta_curves_invalid_genotype(mock_model, mock_posteriors):
    """Test error handling for genotype not in model."""
    manual_df = pd.DataFrame({
        "genotype": ["non_existent"],
        "titrant_name": ["iptg"],
        "titrant_conc": [1.0]
    })
    with pytest.raises(ValueError, match="were not found in the model data"):
        mock_model.extract_theta_curves(mock_posteriors, manual_titrant_df=manual_df)

def test_extract_theta_curves_file_loading(mock_model):
    """Test loading posteriors from file."""
    mock_model._theta = "hill"
    mock_npz = {
        "theta_hill_n": np.ones((5, 2)),
        "theta_log_hill_K": np.ones((5, 2)),
        "theta_theta_high": np.ones((5, 2)),
        "theta_theta_low": np.ones((5, 2))
    }
    
    with patch("numpy.load", return_value=mock_npz) as mock_load:
        mock_model.extract_theta_curves("mock.npz")
        mock_load.assert_called_once_with("mock.npz")
