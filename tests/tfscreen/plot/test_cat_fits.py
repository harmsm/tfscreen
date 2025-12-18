
import pytest
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tfscreen.plot.cat_fits import cat_fits

def test_cat_fits_basic():
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 0.5, 0.8, 0.9])
    y_std = np.array([0.1, 0.1, 0.1, 0.1])
    
    pred_df = pd.DataFrame({
        'x': np.tile(x, 2),
        'y': np.concatenate([y, y*0.9]),
        'y_std': np.tile(y_std, 2),
        'model': ['Model A']*4 + ['Model B']*4,
        'is_best_model': [True]*4 + [False]*4
    })
    
    plt.close('all')
    fig, ax = cat_fits(x, y, y_std, pred_df, title="Test Plot")
    
    assert fig is not None
    assert ax is not None
    
    # Check if title is set
    assert ax.get_figure()._suptitle.get_text() == "Test Plot"
    
    # Check lines (models)
    lines = ax.get_lines()
    # 2 models -> 2 lines. 
    # Also errorbar usually adds lines (vertical lines for errors).
    # But ax.plot is called for models. 
    
    # We should have at least 2 lines for the models.
    # Filter for the model lines? 
    model_lines = [l for l in lines if l.get_label() in ['Model A', 'Model B']]
    assert len(model_lines) == 2

def test_cat_fits_custom_ax():
    fig, ax = plt.subplots()
    x = np.array([1])
    y = np.array([1])
    y_std = np.array([0.1])
    
    pred_df = pd.DataFrame({
        'x': [1],
        'y': [1],
        'y_std': [0.1],
        'model': ['Model A'],
        'is_best_model': [True]
    })
    
    ret_fig, ret_ax = cat_fits(x, y, y_std, pred_df, ax=ax)
    assert ret_ax is ax
    assert ret_fig is fig

def test_cat_fits_nan_handling():
    x = np.array([1, np.nan])
    y = np.array([1, 1])
    y_std = np.array([0.1, 0.1])
    
    # Should handle nans via clean_arrays
    # Pred df
    pred_df = pd.DataFrame({
        'x': [1],
        'y': [1],
        'y_std': [0.1],
        'model': ['Model A'],
        'is_best_model': [True]
    })
    
    fig, ax = cat_fits(x, y, y_std, pred_df)
    # Should not crash
    assert fig is not None

def test_cat_fits_log_scale():
    x = np.array([1, 10])
    y = np.array([1, 1])
    y_std = np.array([0.1, 0.1])
    
    pred_df = pd.DataFrame({
        'x': x,
        'y': y,
        'y_std': y_std,
        'model': ['Model A']*2,
        'is_best_model': [True]*2
    })
    
    fig, ax = cat_fits(x, y, y_std, pred_df, xlog=True, ylog=True)
    assert ax.get_xscale() == 'log'
    assert ax.get_yscale() == 'log'
