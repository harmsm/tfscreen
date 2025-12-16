
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from tfscreen.plot.heatmap.aa_v_titrant_heatmap import aa_v_titrant_heatmap
from tfscreen.plot.heatmap.aa_v_res_heatmap import aa_v_res_heatmap
from tfscreen.plot.heatmap.epistasis_heatmap import epistasis_heatmap

# Mock tfscreen.plot.heatmap
@pytest.fixture
def mock_heatmap():
    with patch('tfscreen.plot.heatmap') as mock:
        mock.return_value = (MagicMock(), MagicMock())
        yield mock

def test_aa_v_titrant_heatmap(mock_heatmap):
    df = pd.DataFrame({
        'resid': [1, 1, 1, 1],
        'mut_aa': ['A', 'A', 'C', 'C'],
        'titrant': [0, 1, 0, 1],
        'value': [0.1, 0.2, 0.3, 0.4]
    })
    
    fig, ax = aa_v_titrant_heatmap(df, 'titrant', 'value')
    
    assert fig is not None
    assert ax is not None
    mock_heatmap.assert_called()
    
    # Check data passed to heatmap
    args, kwargs = mock_heatmap.call_args
    heatmap_df = args[0]
    # Index should be titrant, columns mut_aa (reversed)
    assert heatmap_df.index.name == 'titrant'
    # columns might include mulitindex or just 'A', 'C'. 
    # Pivot table columns='mut_aa'. 
    # columns are reversed: C, A
    assert list(heatmap_df.columns) == ['C', 'A']
    
    # Check kwargs
    assert kwargs['x_axis_type'] == 'titrant'
    assert kwargs['y_axis_type'] == 'aa'

def test_aa_v_titrant_heatmap_duplicates(mock_heatmap):
    df = pd.DataFrame({
        'resid': [1, 1],
        'mut_aa': ['A', 'A'],
        'titrant': [0, 0],
        'value': [0.1, 0.2]
    })
    with pytest.raises(ValueError, match="must be unique"):
        aa_v_titrant_heatmap(df, 'titrant', 'value')

def test_aa_v_res_heatmap(mock_heatmap):
    df = pd.DataFrame({
        'genotype': ['1A', '2C'],
        'resid': [1, 2],
        'mut_aa': ['A', 'C'],
        'value': [0.1, 0.2]
    })
    
    fig, ax = aa_v_res_heatmap(df, 'value')
    
    heatmap_df = mock_heatmap.call_args[0][0]
    # Index resid, columns mut_aa.
    # Missing 1C and 2A will be NaN.
    # rows should be 1, 2 (continuous)
    assert list(heatmap_df.index) == [1, 2]
    # columns reversed C, A
    assert list(heatmap_df.columns) == ['C', 'A']
    
    assert mock_heatmap.call_args[1]['x_axis_type'] == 'site'

def test_aa_v_res_heatmap_duplicates(mock_heatmap):
    df = pd.DataFrame({
        'genotype': ['1A', '1A'],
        'resid': [1, 1],
        'mut_aa': ['A', 'A'],
        'value': [0.1, 0.2]
    })
    with pytest.raises(ValueError, match="genotypes must be unique"):
        aa_v_res_heatmap(df, 'value')

@patch('tfscreen.analysis.extract_epistasis')
@patch('tfscreen.genetics.expand_genotype_columns')
def test_epistasis_heatmap(mock_expand, mock_extract, mock_heatmap):
    # Setup df
    df = pd.DataFrame({
        'num_muts': [0, 1, 1, 2],
        'resid_1': [None, 10, 20, 10],
        'resid_2': [None, None, None, 20],
        'mut_aa_1': [None, 'A', None, 'A'],
        'mut_aa_2': [None, None, 'C', 'C'],
        'genotype': ['WT', '10A', '20C', '10A,20C'],
        'value': [1.0, 1.2, 1.3, 1.5]
    })
    
    # Mock return of extract/expand
    # extract_epistasis returns a df with epistasis calculated.
    # expand_genotype_columns expands it.
    
    # We need a dataframe that has mut_aa_1, mut_aa_2, ep_obs
    mock_ep_df = pd.DataFrame({
        'mut_aa_1': ['A'],
        'mut_aa_2': ['C'],
        'ep_obs': [0.1]
    })
    mock_expand.return_value = mock_ep_df
    mock_extract.return_value = pd.DataFrame() # passed to expand
    
    fig, ax = epistasis_heatmap(df, 10, 20, 'value')
    
    mock_extract.assert_called()
    # Check what sub_df was passed
    # It should filter for 10 and 20.
    
    heatmap_df = mock_heatmap.call_args[0][0]
    # pivoted on mut_aa_1, mut_aa_2
    assert heatmap_df.shape == (1, 1)
    assert heatmap_df.values[0,0] == 0.1

def test_epistasis_heatmap_duplicates(mock_heatmap):
    # Duplicated WT or something
    df = pd.DataFrame({
        'num_muts': [0, 0],
        'resid_1': [None, None],
        'resid_2': [None, None],
        'mut_aa_1': [None, None],
        'mut_aa_2': [None, None],
        'genotype': ['WT', 'WT'],
        'value': [1.0, 1.0]
    })
    with pytest.raises(ValueError, match="condition_selector must be unique"):
        epistasis_heatmap(df, 1, 2, 'value')

