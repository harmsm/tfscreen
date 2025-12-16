
import pytest
import numpy as np
from matplotlib import pyplot as plt
from tfscreen.plot.err_zscore import err_zscore
from tfscreen.plot.err_vs_mag import err_vs_mag
from tfscreen.plot.uncertainty_calibration import uncertainty_calibration

def test_err_zscore():
    est = np.array([1, 2, 3])
    real = np.array([1, 2, 2.5])
    std = np.array([0.1, 0.1, 0.5])
    
    plt.close('all')
    ax = err_zscore(est, std, real)
    assert ax is not None
    # Check if histogram patches are created
    # Patches are added to ax.patches or as polygons in ax.collections/artists depending on impl.
    # fill command creates Polygons.
    assert len(ax.patches) > 0 or len(ax.collections) > 0 or len(ax.lines) > 0

def test_err_vs_mag():
    obs = np.array([1, 2, 3])
    pred = np.array([1.1, 1.9, 3.05])
    
    plt.close('all')
    ax = err_vs_mag(obs, pred, axis_name="Test")
    assert ax is not None
    assert ax.get_xlabel() == "Test obs"
    # Expected ylabel: Test predicted - obs
    assert ax.get_ylabel() == "Test predicted - obs"

def test_uncertainty_calibration():
    est = np.array([1, 2, 3])
    real = np.array([1.1, 1.9, 3.2])
    std = np.array([0.2, 0.2, 0.3])
    
    plt.close('all')
    ax = uncertainty_calibration(est, std, real)
    assert ax is not None
    # Check lines (guides)
    assert len(ax.lines) > 0
    # Check scatter
    assert len(ax.collections) > 0
    
    assert ax.get_xlabel() == "est_value - real_value"
    assert ax.get_ylabel() == "est_err"

def test_uncertainty_calibration_custom_kwargs():
    est = np.array([1, 2])
    real = np.array([1, 2])
    std = np.array([0.1, 0.1])
    
    ax = uncertainty_calibration(est, std, real, scatter_kwargs={'s': 50})
    offsets = ax.collections[0].get_offsets()
    assert len(offsets) == 2
