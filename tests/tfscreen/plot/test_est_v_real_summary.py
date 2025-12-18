
import pytest
import numpy as np
from matplotlib import pyplot as plt
from tfscreen.plot.est_v_real_summary import est_v_real_summary

def test_est_v_real_summary_basic():
    # est, std, real
    # Generate some data
    real = np.random.normal(0, 1, 100)
    std = np.abs(np.random.normal(0, 0.1, 100)) + 0.01
    est = real + np.random.normal(0, std, 100)
    
    plt.close('all')
    fig, axes = est_v_real_summary(est, std, real, suptitle="Test Summary")
    
    assert fig is not None
    assert len(axes) == 3
    assert axes[0] is not None
    assert axes[1] is not None
    assert axes[2] is not None
    
    # Check suptitle
    assert fig._suptitle.get_text() == "Test Summary"
    
    # Check axis labels if prefix is handled
    assert axes[0].get_xlabel() == 'real'
    assert axes[0].get_ylabel() == 'est'

def test_est_v_real_summary_prefix():
    real = np.random.normal(0, 1, 10)
    std = np.random.uniform(0.1, 0.2, 10)
    est = real + np.random.normal(0, std, 10)
    
    fig, axes = est_v_real_summary(est, std, real, axis_prefix="dG")
    assert axes[0].get_xlabel() == 'dG_real'
    assert axes[0].get_ylabel() == 'dG_est'
