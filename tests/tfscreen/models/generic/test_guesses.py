
import pytest
import numpy as np
from tfscreen.models.generic.guesses import (
    guess_flat,
    guess_linear,
    guess_repressor,
    guess_inducer,
    guess_hill_repressor,
    guess_hill_inducer,
    guess_bell_peak,
    guess_bell_dip,
    guess_biphasic_peak,
    guess_biphasic_dip,
    guess_poly_2nd
)

def test_guess_flat():
    x = np.array([1, 2, 3])
    y = np.array([5, 5, 5])
    X = guess_flat(x, y)
    # vander(x, N=1). Output columns are x^0. So just 1s.
    # vander order depends on implementation. Default is decreasing powers?
    # np.vander(x, 1) -> column of 1s? No, vander(x, N) returns N columns?
    # np.vander(x, N) returns columns x^(N-1), x^(N-2), ... x^0.
    # guess_flat calls np.vander(x, 1)[:, ::-1].
    # vander(x, 1) -> x^0 -> [1, 1, 1]. Reverse is [1, 1, 1].
    assert X.shape == (3, 1)
    assert np.all(X == 1)

def test_guess_linear():
    x = np.array([1, 2, 3])
    y = np.array([3, 5, 7])
    X = guess_linear(x, y)
    # y = 2x + 1.
    # guess_linear uses vander(x, 2)[:, ::-1].
    # vander(x, 2) -> [x^1, x^0]. Reverse -> [x^0, x^1].
    # Columns: 1, x.
    assert X.shape == (3, 2)
    assert np.allclose(X[:, 0], 1)
    assert np.allclose(X[:, 1], x)

def test_guess_repressor():
    # Repressor: high to low.
    x = np.array([0, 10, 100])
    y = np.array([1.0, 0.5, 0.0])
    
    # Baseline = max(y) = 1.0? 
    # Code: baseline = max(y). amplitude = min(y) - baseline = 0 - 1 = -1.
    # y_half = 1 + 0.5*(-1) = 0.5.
    # K = x[argmin(|y - 0.5|)] -> x=10 matches 0.5 exactly. K=10.
    # lnK = ln(10).
    
    params = guess_repressor(x, y)
    assert np.allclose(params, [1.0, -1.0, np.log(10.0)])

def test_guess_inducer():
    # Inducer: low to high.
    x = np.array([0, 10, 100])
    y = np.array([0.0, 0.5, 1.0])
    
    # Code: baseline = min(y) = 0.
    # amplitude = max(y) - baseline = 1.0.
    # y_half = 0 + 0.5 = 0.5.
    # K = x matching 0.5 = 10.
    
    params = guess_inducer(x, y)
    assert np.allclose(params, [0.0, 1.0, np.log(10.0)])

def test_guess_hill_repressor():
    x = np.array([0, 10, 100])
    y = np.array([1.0, 0.5, 0.0])
    
    params = guess_hill_repressor(x, y)
    # Appends n=2.0 to repressor guess.
    assert np.allclose(params, [1.0, -1.0, np.log(10.0), 2.0])

def test_guess_bell_peak():
    # Peak at x=10.
    x = np.array([1, 10, 20])
    y = np.array([0, 10, 0])
    
    # baseline = min(y) = 0.
    # peak_idx = argmax(y) = 1 (value 10).
    # amplitude = 10 - 0 = 10.
    # x0 = 10. ln_x0 = ln(10).
    # width = (20-1)/4 = 4.75. ln_width = ln(4.75).
    
    params = guess_bell_peak(x, y)
    assert np.allclose(params[0], 0.0)
    assert np.allclose(params[1], 10.0)
    assert np.allclose(params[2], np.log(10.0))
    # width check roughly
    assert np.allclose(params[3], np.log(4.75))

def test_guess_biphasic_peak():
    x = np.array([0.1, 1.0, 10.0])
    y = np.array([0.0, 1.0, 0.0]) # Peak at 1.0
    
    # baseline = y[0] = 0.
    # peak at 1.0. amp = 1.0.
    # x_peak = 1.0.
    # lnK_a = ln(0.5).
    # lnK_i = ln(1.0).
    
    params = guess_biphasic_peak(x, y)
    assert np.allclose(params, [0.0, 1.0, np.log(0.5), np.log(1.0)])

def test_guess_poly_2nd():
    x = np.array([1, 2, 3])
    y = np.array([1, 4, 9])
    X = guess_poly_2nd(x, y)
    # vander(x, 3). [x^0, x^1, x^2]
    assert X.shape == (3, 3)
