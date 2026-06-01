
import pytest
import numpy as np
from matplotlib import pyplot as plt
from tfscreen.plot.err_zscore import err_zscore
from tfscreen.plot.err_vs_mag import err_vs_mag
from tfscreen.plot.uncertainty_calibration import uncertainty_calibration


# ---------------------------------------------------------------------------
# err_zscore
# ---------------------------------------------------------------------------

def test_err_zscore_basic():
    est = np.array([1, 2, 3])
    real = np.array([1, 2, 2.5])
    std = np.array([0.1, 0.1, 0.5])
    plt.close("all")
    ax = err_zscore(est, std, real)
    assert ax is not None
    assert len(ax.patches) > 0 or len(ax.collections) > 0 or len(ax.lines) > 0
    plt.close("all")


def test_err_zscore_axis_labels():
    est = np.array([1.0, 2.0])
    real = np.array([1.0, 2.0])
    std = np.array([0.1, 0.1])
    plt.close("all")
    ax = err_zscore(est, std, real)
    assert ax.get_xlabel() == "z-score"
    assert "probability" in ax.get_ylabel().lower()
    plt.close("all")


def test_err_zscore_custom_ax():
    est = np.array([1.0, 2.0])
    real = np.array([1.0, 2.0])
    std = np.array([0.1, 0.1])
    _, ax = plt.subplots()
    ret = err_zscore(est, std, real, ax=ax)
    assert ret is ax
    plt.close("all")


def test_err_zscore_nan_removed():
    est = np.array([1.0, np.nan, 3.0])
    real = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.1, 0.1])
    plt.close("all")
    ax = err_zscore(est, std, real)  # must not raise
    assert ax is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# err_vs_mag
# ---------------------------------------------------------------------------

def test_err_vs_mag_basic():
    obs = np.array([1, 2, 3])
    pred = np.array([1.1, 1.9, 3.05])
    plt.close("all")
    ax = err_vs_mag(obs, pred, axis_name="Test")
    assert ax is not None
    assert ax.get_xlabel() == "Test obs"
    assert ax.get_ylabel() == "Test predicted - obs"
    plt.close("all")


def test_err_vs_mag_custom_ax():
    obs = np.array([1.0, 2.0])
    pred = np.array([1.0, 2.0])
    _, ax = plt.subplots()
    ret = err_vs_mag(obs, pred, ax=ax)
    assert ret is ax
    plt.close("all")


def test_err_vs_mag_nan_removed():
    obs = np.array([1.0, np.nan, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    plt.close("all")
    ax = err_vs_mag(obs, pred)
    assert ax is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# uncertainty_calibration
# ---------------------------------------------------------------------------

def test_uncertainty_calibration_basic():
    est = np.array([1, 2, 3])
    real = np.array([1.1, 1.9, 3.2])
    std = np.array([0.2, 0.2, 0.3])
    plt.close("all")
    ax = uncertainty_calibration(est, std, real)
    assert ax is not None
    assert len(ax.lines) > 0
    assert len(ax.collections) > 0
    assert ax.get_xlabel() == "est_value - real_value"
    assert ax.get_ylabel() == "est_err"
    plt.close("all")


def test_uncertainty_calibration_custom_ax():
    est = np.array([1.0, 2.0])
    real = np.array([1.0, 2.0])
    std = np.array([0.1, 0.1])
    _, ax = plt.subplots()
    ret = uncertainty_calibration(est, std, real, ax=ax)
    assert ret is ax
    plt.close("all")


def test_uncertainty_calibration_custom_kwargs():
    est = np.array([1, 2])
    real = np.array([1, 2])
    std = np.array([0.1, 0.1])
    plt.close("all")
    ax = uncertainty_calibration(est, std, real, scatter_kwargs={"s": 50})
    offsets = ax.collections[0].get_offsets()
    assert len(offsets) == 2
    plt.close("all")


def test_uncertainty_calibration_nan_removed():
    est = np.array([1.0, np.nan, 3.0])
    real = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.1, 0.1])
    plt.close("all")
    ax = uncertainty_calibration(est, std, real)
    assert ax is not None
    plt.close("all")
