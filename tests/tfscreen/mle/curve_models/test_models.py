
import pytest
import numpy as np
from tfscreen.mle.curve_models.models import (
    model_flat,
    model_linear,
    model_linear_logx,
    model_hill_3p,
    model_hill_4p,
    model_bell,
    model_bell_logx,
    model_biphasic_peak,
    model_biphasic_dip,
    model_poly,
    _to_log10_x,
)

def test_model_flat():
    x = np.array([1, 2, 3])
    y = model_flat([5.0], x)
    assert np.allclose(y, [5.0, 5.0, 5.0])

def test_model_linear():
    x = np.array([0, 1, 2])
    # y = 2x + 1
    y = model_linear([2.0, 1.0], x)
    assert np.allclose(y, [1.0, 3.0, 5.0])

def test_model_hill_4p():
    # baseline=0, amp=1, lnK=ln(10), n=2
    # y = x^2 / (x^2 + 10^2)
    params = [0.0, 1.0, np.log(10.0), 2.0]
    
    x = np.array([0.0, 10.0, 100.0])
    y = model_hill_4p(params, x)
    
    # x=0 -> 0 (code clips log(0) but exp(small) is 0. x_to_n / (x_to_n + K_to_n) -> 0/(0+K) = 0)
    # x=10 -> 0.5
    # x=100 -> 10000 / 10100 ~ 0.99
    
    assert np.allclose(y[1], 0.5)
    assert np.allclose(y[2], 100**2 / (100**2 + 10**2))

def test_model_hill_3p():
    # baseline=0, amp=1, lnK=ln(10). n assumed 1.
    params = [0.0, 1.0, np.log(10.0)]
    x = np.array([10.0])
    y = model_hill_3p(params, x)
    # x / (x + K) = 10 / (10 + 10) = 0.5
    assert np.allclose(y, 0.5)

def test_model_bell():
    # baseline=0, amp=1, ln_x0=0, ln_width=0 (width=1)
    # x0 = 1. width = 1.
    # y = exp(-0.5 * ((x-1)/1)^2)
    params = [0.0, 1.0, 0.0, 0.0]
    
    x = np.array([1.0, 1.0 + np.sqrt(2*np.log(2))]) 
    # At x=1, exp(0) = 1. y=1.
    # Half width at half max? checking standard normal width.
    # At x=1+1 = 2. exp(-0.5 * 1^2) = exp(-0.5) ~ 0.606
    
    y = model_bell(params, x)
    assert np.allclose(y[0], 1.0)
    assert np.allclose(y[1], np.exp(-0.5 * ((x[1]-1)**2)))

def test_model_poly():
    # c0 + c1*x + c2*x^2
    # 1 + 2x + 3x^2
    params = np.array([1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0])
    
    y = model_poly(params, x)
    
    assert np.allclose(y, [1.0, 6.0, 17.0])

def test_model_biphasic_peak():
    # baseline=0, amp=1, lnKa=ln(1), lnKi=ln(100)
    # y = 1 * (x / (1+x)) * (1 / (1 + x/100))
    params = [0.0, 1.0, np.log(1.0), np.log(100.0)]
    x = np.array([1.0, 100.0])
    
    # x=1: (1/2) * (1 / 1.01) ~ 0.5
    # x=100: (100/101) * (1 / 2) ~ 0.5
    
    y = model_biphasic_peak(params, x)
    
    assert np.allclose(y[0], 0.5 / 1.01)
    assert np.allclose(y[1], (100/101) * 0.5)

def test_model_biphasic_dip():
    # baseline=1, amp=1, lnK_dip=ln(1), lnK_rise=ln(100)
    # term1: 1 / (1 + x/1). (Standard repressor, baseline=1, amp=1, but form 1/(1+x/K))
    # term2: 1 * (x / (100+x)). (Standard activator)
    
    params = [1.0, 1.0, np.log(1.0), np.log(100.0)]
    x = np.array([1.0, 100.0])
    
    # x=1: term1 = 1 / (1+1) = 0.5. term2 = 1/101 ~ 0.01. Sum ~ 0.51
    # x=100: term1 = 1 / 101 ~ 0.01. term2 = 100/200 = 0.5. Sum ~ 0.51
    
    y = model_biphasic_dip(params, x)
    
    assert np.allclose(y[0], 0.5 + 1.0/101.0)
    assert np.allclose(y[1], 1.0/101.0 + 0.5)


# --- log-concentration models ------------------------------------------------

def test_to_log10_x_floor_and_passthrough():
    # x == 0 -> log10(min positive / 100). min positive = 1e-3 -> floor 1e-5.
    x = np.array([0.0, 1e-3, 1e-2, 1e-1])
    z = _to_log10_x(x)
    assert np.isclose(z[0], np.log10(1e-5))
    assert np.allclose(z[1:], np.log10([1e-3, 1e-2, 1e-1]))


def test_to_log10_x_nan_preserved():
    z = _to_log10_x(np.array([np.nan, 1e-2, 0.0]))
    assert np.isnan(z[0])
    assert np.isclose(z[1], -2.0)
    # 0 -> floor = 1e-2/100 = 1e-4 -> log10 = -4
    assert np.isclose(z[2], -4.0)


def test_to_log10_x_all_nonpositive_is_nan():
    z = _to_log10_x(np.array([0.0, 0.0, -1.0]))
    assert np.all(np.isnan(z))


def test_model_linear_logx():
    # y = m*log10(x) + b with m=2, b=1.
    x = np.array([1e-2, 1e-1, 1.0])  # log10 -> [-2, -1, 0]
    y = model_linear_logx([2.0, 1.0], x)
    assert np.allclose(y, [2 * -2 + 1, 2 * -1 + 1, 2 * 0 + 1])


def test_model_bell_logx_peak_centered_in_log_space():
    # Peak at center = -3 (i.e. x = 1e-3), baseline 0, amplitude 1.
    params = [0.0, 1.0, -3.0, np.log(1.0)]
    x = np.array([1e-4, 1e-3, 1e-2])  # log10 -> [-4, -3, -2]
    y = model_bell_logx(params, x)
    # Max at the center concentration.
    assert np.argmax(y) == 1
    assert np.isclose(y[1], 1.0)
    # Symmetric one log-unit either side of center.
    assert np.isclose(y[0], y[2])


def test_model_bell_logx_negative_amplitude_is_dip():
    params = [1.0, -1.0, -3.0, np.log(1.0)]
    x = np.array([1e-4, 1e-3, 1e-2])
    y = model_bell_logx(params, x)
    # Minimum at the center concentration.
    assert np.argmin(y) == 1
    assert np.isclose(y[1], 0.0)


def test_model_bell_logx_handles_zero_x():
    # x == 0 must not blow up (it is floored inside _to_log10_x).
    params = [0.0, 1.0, -3.0, np.log(1.0)]
    y = model_bell_logx(params, np.array([0.0, 1e-3, 1e-2]))
    assert np.all(np.isfinite(y))
