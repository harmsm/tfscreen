
import pytest
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from unittest.mock import MagicMock, patch

try:
    import corner
    CORNER_AVAILABLE = True
except ImportError:
    CORNER_AVAILABLE = False

from tfscreen.plot.corner import corner_plot

@pytest.fixture
def mock_corner():
    with patch('corner.corner') as mock:
        mock.return_value = plt.figure()
        yield mock

def test_corner_plot_basic(mock_corner):
    # Fit df requires: "est', "genotype", "class","titrant_conc", and "block"
    fit_df = pd.DataFrame({
        'est': [1.0, 2.0],
        'genotype': ['WT', 'WT'],
        'class': ['A', 'B'],
        'titrant_conc': [0, 10],
        'block': [1, 1]
    }, index=[0, 1])
    
    cov_matrix = np.array([[1.0, 0.1], [0.1, 1.0]])
    
    fig = corner_plot(fit_df, cov_matrix, num_samples=100)
    
    assert fig is not None
    mock_corner.assert_called()
    
    # Check arguments passed to corner.corner
    call_args = mock_corner.call_args
    assert call_args is not None
    # samples should be (100, 2)
    samples = call_args[0][0]
    assert samples.shape == (100, 2)
    
    kwargs = call_args[1]
    assert 'truths' in kwargs
    assert len(kwargs['labels']) == 2
    assert kwargs['labels'][0] == "WT_A_0_1"


def test_corner_plot_too_many_params(mock_corner):
    # max allowed is 10. Let's make 11 rows.
    fit_df = pd.DataFrame({
        'est': np.zeros(11),
        'genotype': ['WT']*11,
        'class': ['A']*11,
        'titrant_conc': [0]*11,
        'block': [1]*11
    }, index=range(11))
    
    cov_matrix = np.eye(11)
    
    with pytest.raises(RuntimeError, match="too many rows"):
        corner_plot(fit_df, cov_matrix)

def test_corner_plot_with_mask(mock_corner):
    fit_df = pd.DataFrame({
        'est': [1.0, 2.0, 3.0],
        'genotype': ['WT', 'WT', 'WT'],
        'class': ['A', 'B', 'C'],
        'titrant_conc': [0, 10, 20],
        'block': [1, 1, 1]
    }, index=[0, 1, 2])
    
    cov_matrix = np.eye(3)
    
    mask = np.array([True, False, True]) # Select 0 and 2
    
    fig = corner_plot(fit_df, cov_matrix, plot_mask=mask, num_samples=100)
    
    assert fig is not None
    # Verify we plotted 2 params
    call_args = mock_corner.call_args
    samples = call_args[0][0]
    assert samples.shape == (100, 2)
