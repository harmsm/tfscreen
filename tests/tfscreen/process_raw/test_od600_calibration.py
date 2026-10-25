"""Tests for tfscreen.process_raw.od600 (OD600-to-CFU calibration)."""
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.optimize import curve_fit

from tfscreen.process_raw import od600 as O

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                       "examples", "od600")


def _plates(od, colonies, volume=0.1, dilution=1e5):
    n = len(od)
    return pd.DataFrame({"od600": od, "colonies": colonies,
                         "dilution": dilution, "plated_volume_mL": volume,
                         "num_dilutions": 5, "plating_steps": 1})


def _replicates():
    rows = []
    for d, m in [(1.0, 0.6), (1 / 3, 0.27), (1 / 9, 0.15), (1 / 27, 0.10),
                 (1 / 81, 0.092)]:
        for k in (-1, 1):
            rows.append({"dilution": d, "od600": m * (1 + 0.01 * k)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def example():
    rep = pd.read_csv(os.path.join(EXAMPLE, "replicates.csv"))
    plate = pd.read_csv(os.path.join(EXAMPLE, "plate_counts.csv"))
    return O.calibrate(rep, plate)


# ---------------------------------------------------------------------------
# the two experiments
# ---------------------------------------------------------------------------

def test_reading_noise_table():
    t = O.reading_noise(_replicates())
    assert list(t["dilution"]) == sorted(t["dilution"], reverse=True)
    assert np.allclose(t["rel_sd"], 0.01)
    assert (t["n"] == 2).all()


def test_detection_threshold_is_midway_between_two_most_dilute():
    t = O.reading_noise(_replicates())
    assert O.detection_threshold(t) == pytest.approx((0.10 + 0.092) / 2)


def test_reading_noise_needs_columns():
    with pytest.raises(ValueError, match="dilution"):
        O.reading_noise(pd.DataFrame({"od600": [0.1, 0.2]}))


def test_plate_counts_divide_by_plated_volume():
    df = O.plate_counts_to_cfu(_plates([0.5], [400]), pipette_rel_error=0.02)
    assert df["cfu_per_mL"].iloc[0] == pytest.approx(400 * 1e5 / 0.1)
    rel = np.sqrt(1 / 400 + 6 * 0.02 ** 2)
    assert df["cfu_per_mL_std"].iloc[0] == pytest.approx(4e8 * rel)


def test_plate_counts_refuse_empty_plates():
    with pytest.raises(ValueError, match="no colonies"):
        O.plate_counts_to_cfu(_plates([0.5, 0.3], [400, 0]))


def test_fit_recovers_exact_polynomial():
    od = np.linspace(0.1, 0.6, 8)
    true = np.array([1e6, 8e8, 2e8])
    y = true[0] + true[1] * od + true[2] * od ** 2
    coef, cov, chi2 = O.fit_polynomial(od, y, 0.05 * y)
    assert np.allclose(coef, true, rtol=1e-8)


def test_fit_covariance_matches_curve_fit():
    rng = np.random.default_rng(1)
    od = np.linspace(0.1, 0.6, 12)
    y = 8e8 * od + 2e8 * od ** 2
    sd = 0.08 * y
    y_obs = y + sd * rng.normal(size=y.size)
    coef, cov, chi2 = O.fit_polynomial(od, y_obs, sd)
    p, pcov = curve_fit(lambda x, a, b, c: a + b * x + c * x ** 2, od, y_obs,
                        sigma=sd, absolute_sigma=False)
    assert np.allclose(coef, p, rtol=1e-6)
    assert np.allclose(cov, pcov, rtol=1e-5)
    assert np.allclose(cov, cov.T)


def test_fit_needs_enough_points():
    with pytest.raises(ValueError, match="at least"):
        O.fit_polynomial([0.1, 0.2, 0.3], [1, 2, 3], [1, 1, 1], degree=2)


# ---------------------------------------------------------------------------
# calibrate and the file
# ---------------------------------------------------------------------------

def test_calibrate_example(example):
    cal, noise, plates = example
    assert cal["kind"] == O.CALIBRATION_KIND
    assert cal["degree"] == 2 and len(cal["coefficients"]) == 3
    # example data: cfu/mL = 8e8 OD + 2e8 OD^2
    od = np.array([0.2, 0.4, 0.6])
    cfu, _, _ = O.od600_to_cfu_per_mL(od, cal)
    assert np.allclose(cfu, 8e8 * od + 2e8 * od ** 2, rtol=0.1)
    assert 0.01 < cal["reading_rel_sd"] < 0.03
    assert 0.08 < cal["detection_threshold"] < 0.1
    assert {"cfu_per_mL_fit", "z"} <= set(plates.columns)
    assert cal["num_plate_counts"] == len(plates)


def test_threshold_override():
    rep = pd.read_csv(os.path.join(EXAMPLE, "replicates.csv"))
    plate = pd.read_csv(os.path.join(EXAMPLE, "plate_counts.csv"))
    cal, _, _ = O.calibrate(rep, plate, threshold=0.11)
    assert cal["detection_threshold"] == 0.11


def test_write_read_round_trip(example, tmp_path):
    cal = example[0]
    path = tmp_path / "cal.yaml"
    O.write_calibration(cal, path, source={"x": "y"})
    back = O.read_calibration(str(path))
    assert np.allclose(back["coefficients"], cal["coefficients"])
    assert np.allclose(back["covariance"], cal["covariance"])
    assert back["source"] == {"x": "y"}
    assert open(path).read().startswith("# OD600-to-CFU calibration")


def test_example_file_is_current(example):
    """examples/od600/od600_calibration.yaml is what the CLI makes from the
    example data."""
    shipped = O.read_calibration(os.path.join(EXAMPLE, "od600_calibration.yaml"))
    assert np.allclose(shipped["coefficients"], example[0]["coefficients"])


def test_legacy_format_refused():
    legacy = {"A_CFU": 0.0, "B_CFU": 8e7, "C_CFU": 2e7, "OD600_PCT_STD": 0.02,
              "OD600_MEAS_THRESHOLD": 0.08}
    with pytest.raises(ValueError, match="tfs-calibrate-od600"):
        O.read_calibration(legacy)


def test_decreasing_curve_refused(example):
    bad = dict(example[0], coefficients=[0.0, 1e8, -1e9])
    with pytest.raises(ValueError, match="not increasing"):
        O.check_calibration(bad)


def test_shape_mismatch_refused(example):
    bad = dict(example[0], covariance=[[1.0]])
    with pytest.raises(ValueError, match="covariance"):
        O.check_calibration(bad)


# ---------------------------------------------------------------------------
# applying it
# ---------------------------------------------------------------------------

def test_error_components(example):
    cal = O.read_calibration(example[0])
    od = np.array([0.15, 0.4])
    curve, reading = O.cfu_per_mL_error_components(od, cal)
    for i, x in enumerate(od):
        J = np.array([1.0, x, x ** 2])
        assert curve[i] == pytest.approx(np.sqrt(J @ cal["covariance"] @ J))
        c = cal["coefficients"]
        slope = c[1] + 2 * c[2] * x
        assert reading[i] == pytest.approx(slope * cal["reading_rel_sd"] * x)
    cfu, sd, _ = O.od600_to_cfu_per_mL(od, cal)
    assert np.allclose(sd, np.hypot(curve, reading))


def test_below_threshold_is_an_upper_bound(example):
    cal = example[0]
    cfu, _, detectable = O.od600_to_cfu_per_mL([0.01], cal)
    at, _, _ = O.od600_to_cfu_per_mL([cal["detection_threshold"]], cal)
    assert not detectable[0]
    assert cfu[0] == pytest.approx(at[0])


def test_inverse_round_trip(example):
    cal = example[0]
    od = np.array([0.1, 0.3, 0.59])
    cfu, _, _ = O.od600_to_cfu_per_mL(od, cal)
    assert np.allclose(O.cfu_per_mL_to_od600(cfu, cal), od, rtol=1e-9)
    # totals below the curve at OD 0 map to 0
    assert O.cfu_per_mL_to_od600([cal["coefficients"][0] - 1e6], cal)[0] == 0.0


def test_in_calibrated_range(example):
    lo, hi = example[0]["calibrated_od600_range"]
    got = O.in_calibrated_range([lo - 0.01, (lo + hi) / 2, hi + 0.01], example[0])
    assert list(got) == [False, True, False]


def test_accepts_path_or_dict(example, tmp_path):
    path = tmp_path / "c.yaml"
    O.write_calibration(example[0], path)
    a = O.od600_to_cfu_per_mL([0.3], str(path))[0]
    b = O.od600_to_cfu_per_mL([0.3], example[0])[0]
    assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# input checks
# ---------------------------------------------------------------------------

def test_reading_noise_needs_two_dilutions():
    rep = pd.DataFrame({"dilution": [1.0, 1.0], "od600": [0.5, 0.51]})
    with pytest.raises(ValueError, match="at least two dilutions"):
        O.reading_noise(rep)


def test_check_calibration_missing_keys(example):
    bad = dict(example[0])
    del bad["covariance"]
    with pytest.raises(ValueError, match=r"missing \['covariance'\]"):
        O.check_calibration(bad)


def test_check_calibration_wrong_kind(example):
    bad = dict(example[0], kind="something_else")
    with pytest.raises(ValueError, match="Not an OD600 calibration"):
        O.check_calibration(bad)


@pytest.mark.parametrize("volume", [None, 0, -1.0])
def test_tube_totals_refuses_bad_volume(example, volume):
    od = pd.DataFrame({"sample": ["a"], "od600": [0.3]})
    with pytest.raises(ValueError, match="tube_volume_mL"):
        O.tube_totals_from_od600(od, example[0], volume)


def test_tube_totals_refuses_non_numeric_reading(example):
    od = pd.DataFrame({"sample": ["a", "b"], "od600": ["0.3", "oops"]})
    with pytest.raises(ValueError, match=r"no numeric reading for: \['b'\]"):
        O.tube_totals_from_od600(od, example[0], 5.0)
