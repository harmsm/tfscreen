
import pytest
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tfscreen.plot.hill_fit import hill_fit

def test_hill_fit_basic():
    obs_df = pd.DataFrame({
        'genotype': ['WT', 'WT', 'M1'],
        'titrant_conc': [0, 10, 10],
        'theta_est': [0.1, 0.9, 0.5],
        'theta_std': [0.01, 0.01, 0.01]
    })
    
    pred_df = pd.DataFrame({
        'genotype': ['WT', 'WT'],
        'titrant_conc': [1e-6, 10], # 0 replaced by small value in function for plotting log scale? 
                                    # Actually 'this_obs_df' replacement happens inside. 
                                    # Pred df usually has the range.
        'hill_est': [0.1, 0.9],
        'hill_std': [0.02, 0.02]
    })
    
    plt.close('all')
    ax = hill_fit(obs_df, pred_df, genotype='WT')
    
    assert ax is not None
    assert ax.get_xscale() == 'log'
    
    # Check if data points are plotted (scatter and errorbar)
    # Check collections (scatter) and lines (errorbar + fit line)
    assert len(ax.collections) >= 1
    assert len(ax.lines) >= 1

def test_hill_fit_custom_ax():
    fig, ax = plt.subplots()
    obs_df = pd.DataFrame({
        'genotype': ['WT'],
        'titrant_conc': [1],
        'theta_est': [0.5],
        'theta_std': [0.1]
    })
    pred_df = pd.DataFrame({
        'genotype': ['WT'],
        'titrant_conc': [1],
        'hill_est': [0.5],
        'hill_std': [0.1]
    })
    
    ret_ax = hill_fit(obs_df, pred_df, genotype='WT', ax=ax)
    assert ret_ax is ax
