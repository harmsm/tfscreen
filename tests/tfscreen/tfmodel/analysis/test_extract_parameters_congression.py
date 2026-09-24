import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.analysis.extraction import extract_parameters

@pytest.fixture
def mock_model_congression():
    """Create a ModelOrchestrator instance with minimal mocked internals for congression."""
    model = MagicMock(spec=ModelOrchestrator)
    model._transformation = "mixture"
    model._theta = "none"
    model._condition_growth = "none"
    model._dk_geno = "none"
    model._activity = "fixed"
    model._growth_transition = "instant"
    model._growth_shares_replicates = False
    model.mut_labels = []
    model.pair_labels = []

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
    model.training_tm = mock_tm

    return model

@pytest.fixture
def mock_posteriors_congression():
    """Create mock posterior samples for congression parameters."""
    num_samples = 5
    # lam is (num_samples, 1)
    return {
        "transformation_lam": np.ones((num_samples, 1)) * 1.2,
    }

def test_extract_parameters_congression(mock_model_congression, mock_posteriors_congression):
    """The mixture transformation's only parameter is lambda."""
    params = extract_parameters(mock_model_congression, mock_posteriors_congression)

    assert "lam" in params
    assert "mu" not in params
    assert "sigma" not in params

    lam_df = params["lam"]
    assert len(lam_df) == 1
    assert lam_df.iloc[0]["parameter"] == "lam"
    assert lam_df.iloc[0]["q0.5"] == 1.2

def test_extract_parameters_no_congression(mock_model_congression):
    """Test that congression parameters are NOT extracted when transformation is none."""
    mock_model_congression._transformation = "none"
    params = extract_parameters(mock_model_congression, {})
    
    assert "lam" not in params
    assert "mu" not in params
    assert "sigma" not in params
