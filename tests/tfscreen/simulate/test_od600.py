"""Tests for tfscreen.simulate.od600 (simulated OD600 readings)."""
import numpy as np
import pytest

from tfscreen.simulate.od600 import (
    cfu_per_mL_to_od600,
    od600_to_cfu_per_mL,
    read_od600_calibration,
    simulate_od600,
)

CAL = {"kind": "od600_to_cfu_per_mL", "degree": 2,
       "coefficients": [1e6, 8e8, 2e8],
       "covariance": [[1e10, 0, 0], [0, 1e12, 0], [0, 0, 1e12]],
       "reading_rel_sd": 0.02, "detection_threshold": 0.08,
       "calibrated_od600_range": [0.1, 0.8]}


@pytest.fixture
def cal():
    return read_od600_calibration(dict(CAL))


def test_round_trip(cal):
    od = np.array([0.1, 0.3, 0.7])
    cfu, _, detectable = od600_to_cfu_per_mL(od, cal)
    assert detectable.all()
    assert np.allclose(cfu_per_mL_to_od600(cfu, cal), od)


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


def test_simulate_flags_blind_and_saturated_tubes(cal):
    rng = np.random.default_rng(1)
    low, _, _ = od600_to_cfu_per_mL(np.array([0.02]), dict(cal, detection_threshold=0.0))
    high, _, _ = od600_to_cfu_per_mL(np.array([1.2]), cal)
    r = simulate_od600(np.array([low[0], high[0]]) * 5.0, 5.0, cal, rng)
    assert not r["od600_detectable"][0]
    assert not r["od600_in_range"][1]


def test_accepts_raw_dict():
    r = simulate_od600(np.array([1e9]), 5.0, dict(CAL), np.random.default_rng(0))
    assert r["od600"].shape == (1,)
