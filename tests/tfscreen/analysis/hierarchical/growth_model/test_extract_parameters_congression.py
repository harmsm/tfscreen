import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass

@pytest.fixture
def mock_model_congression():
    """Create a ModelClass instance with minimal mocked internals for congression."""
    model = MagicMock(spec=ModelClass)
    model._transformation = "congression"
    model._theta = "none"
    model._condition_growth = "none"
    model._dk_geno = "none"
    model._activity = "fixed"
    
    # Mock TensorManager and its DataFrame
    mock_tm = MagicMock()
    mock_tm.df = pd.DataFrame({
        "titrant_name": ["iptg", "iptg"],
        "titrant_conc": [0.0, 1.0],
        "titrant_name_idx": [0, 0],
        "titrant_conc_idx": [0, 1]
    })
    mock_tm.tensor_dim_names = ["replicate", "time", "condition_pre", "condition_sel", "titrant_name", "titrant_conc", "genotype"]
    mock_tm.tensor_dim_labels = [["1"], ["1"], ["1"], ["1"], ["iptg"], [0.0, 1.0], ["wt"]]
    model.growth_tm = mock_tm
    
    # Bind the method under test
    model.extract_parameters = ModelClass.extract_parameters.__get__(model, ModelClass)
    
    return model

@pytest.fixture
def mock_posteriors_congression():
    """Create mock posterior samples for congression parameters."""
    num_samples = 5
    # lam is (num_samples, 1)
    # mu, sigma are (num_samples, num_titrant_name, num_titrant_conc, 1)
    # flattened: (num_samples, num_titrant_name * num_titrant_conc)
    return {
        "transformation_lam": np.ones((num_samples, 1)) * 1.2,
        "transformation_mu": np.ones((num_samples, 1, 2, 1)) * 0.5,
        "transformation_sigma": np.ones((num_samples, 1, 2, 1)) * 0.1
    }

def test_extract_parameters_congression(mock_model_congression, mock_posteriors_congression):
    """Test extracting lam, mu, and sigma."""
    params = mock_model_congression.extract_parameters(mock_posteriors_congression)
    
    assert "lam" in params
    assert "mu" in params
    assert "sigma" in params
    
    # Check lam
    lam_df = params["lam"]
    assert len(lam_df) == 1
    assert lam_df.iloc[0]["parameter"] == "lam"
    assert lam_df.iloc[0]["median"] == 1.2
    
    # Check mu
    mu_df = params["mu"]
    assert len(mu_df) == 2
    assert set(mu_df["titrant_conc"]) == {0.0, 1.0}
    assert np.allclose(mu_df["median"], 0.5)
    
    # Check sigma
    sigma_df = params["sigma"]
    assert len(sigma_df) == 2
    assert np.allclose(sigma_df["median"], 0.1)

def test_extract_parameters_no_congression(mock_model_congression):
    """Test that congression parameters are NOT extracted when transformation is none."""
    mock_model_congression._transformation = "none"
    params = mock_model_congression.extract_parameters({})
    
    assert "lam" not in params
    assert "mu" not in params
    assert "sigma" not in params
