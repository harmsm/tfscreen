"""Tests for tfscreen.simulate.od600 (simulated OD600 readings)."""
import numpy as np
import pytest

from tfscreen.simulate.od600 import (
    cfu_per_mL_to_od600,
    od600_to_cfu_per_mL,
    read_od600_calibration,
    simulate_od600,
)

CAL = {"A_CFU": 1e6, "B_CFU": 8e7, "C_CFU": 2e7, "OD600_PCT_STD": 0.02,
       "OD600_MEAS_THRESHOLD": 0.08, "OD600_MAX": 0.8}


@pytest.fixture
def cal():
    return read_od600_calibration(dict(CAL))


def test_read_fills_optional(cal):
    assert cal["P_JCJT_CFU"] is None
    assert cal["OD600_MAX"] == 0.8


def test_read_needs_required():
    bad = dict(CAL)
    del bad["B_CFU"]
    with pytest.raises(ValueError, match="B_CFU"):
        read_od600_calibration(bad)


def test_read_refuses_decreasing_curve():
    bad = dict(CAL, C_CFU=-1e9)
    with pytest.raises(ValueError, match="not increasing"):
        read_od600_calibration(bad)


def test_round_trip(cal):
    od = np.array([0.1, 0.3, 0.7])
    cfu, _, detectable = od600_to_cfu_per_mL(od, cal)
    assert detectable.all()
    assert np.allclose(cfu_per_mL_to_od600(cfu, cal), od)


def test_below_threshold_is_an_upper_bound(cal):
    cfu, _, detectable = od600_to_cfu_per_mL(np.array([0.01]), cal)
    at_threshold, _, _ = od600_to_cfu_per_mL(np.array([0.08]), cal)
    assert not detectable[0]
    assert cfu[0] == pytest.approx(at_threshold[0])


def test_forward_sd_is_reading_noise_through_slope(cal):
    od = 0.4
    _, sd, _ = od600_to_cfu_per_mL(np.array([od]), cal)
    slope = CAL["B_CFU"] + 2 * CAL["C_CFU"] * od
    assert sd[0] == pytest.approx(slope * 0.02 * od)


def test_forward_sd_includes_curve_error():
    c = read_od600_calibration(dict(CAL, P_JCJT_CFU=1e5, Q_JCJT_CFU=0.0,
                                    R_JCJT_CFU=0.0))
    _, sd, _ = od600_to_cfu_per_mL(np.array([0.4]), c)
    slope = CAL["B_CFU"] + 2 * CAL["C_CFU"] * 0.4
    assert sd[0] == pytest.approx(np.hypot(slope * 0.02 * 0.4, 1e5))


def test_simulate_od600(cal):
    rng = np.random.default_rng(0)
    volume = 5.0
    true_od = 0.4
    cfu, _, _ = od600_to_cfu_per_mL(np.array([true_od]), cal)
    total = np.full(20_000, cfu[0] * volume)
    r = simulate_od600(total, volume, cal, rng)
    assert r["od600"].mean() == pytest.approx(true_od, rel=0.005)
    assert r["od600"].std() / true_od == pytest.approx(0.02, rel=0.05)
    assert r["od600_detectable"].all()
    assert r["od600_in_range"].all()


def test_simulate_flags(cal):
    rng = np.random.default_rng(0)
    # Evaluate the curve directly: the forward calibration clamps readings
    # below the threshold to the threshold.
    def curve(od):
        return CAL["A_CFU"] + CAL["B_CFU"] * od + CAL["C_CFU"] * od ** 2
    low, high = curve(0.02), curve(1.5)
    r = simulate_od600(np.array([low, high]), 1.0, cal, rng)
    assert list(r["od600_detectable"]) == [False, True]
    assert list(r["od600_in_range"]) == [True, False]
