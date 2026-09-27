"""
Simulated OD600 readings of each tube's total population.

The experiment estimates each tube's total cells from one OD600 reading
through a lab-specific quadratic calibration, ``cfu/mL = A + B OD + C OD^2``
(see ``docs/source/process-raw.rst``, "One tube per time-point"). The
simulator knows each tube's true total, so it runs the calibration backwards
to get the true OD600, adds the reader's noise, and applies the reader's
detection threshold, then (optionally) runs the calibration forwards again,
as the lab does, to get the estimated total the pipeline would ingest.

The calibration constants are read in the format the lab's calibration
notebook writes (``A_CFU``, ``B_CFU``, ``C_CFU``, ``OD600_PCT_STD``,
``OD600_MEAS_THRESHOLD`` and, optionally, ``P_JCJT_CFU``, ``Q_JCJT_CFU``,
``R_JCJT_CFU`` for the calibration curve's own error and ``OD600_MAX`` for
the top of the calibrated range). Roadmap step 3 replaces this with a
generic calibration file (planning/analysis-roadmap.md).
"""

import numpy as np

from tfscreen.util import read_yaml

_REQUIRED = ("A_CFU", "B_CFU", "C_CFU", "OD600_PCT_STD", "OD600_MEAS_THRESHOLD")
_OPTIONAL = ("P_JCJT_CFU", "Q_JCJT_CFU", "R_JCJT_CFU", "OD600_MAX")


def read_od600_calibration(calibration):
    """
    Read OD600-to-CFU calibration constants.

    Parameters
    ----------
    calibration : str or dict
        Path to a calibration YAML, or the dict itself.

    Returns
    -------
    dict
        The constants as floats; optional ones missing from the input are
        None.

    Raises
    ------
    ValueError
        If a required constant is missing, or the calibration curve is not
        increasing over the calibrated range.
    """
    cal = read_yaml(calibration)
    missing = [k for k in _REQUIRED if k not in cal]
    if missing:
        raise ValueError(f"OD600 calibration is missing {missing}.")
    out = {k: float(cal[k]) for k in _REQUIRED}
    for k in _OPTIONAL:
        out[k] = None if cal.get(k) is None else float(cal[k])

    top = out["OD600_MAX"] if out["OD600_MAX"] is not None else 1.0
    for od in (out["OD600_MEAS_THRESHOLD"], top):
        if out["B_CFU"] + 2 * out["C_CFU"] * od <= 0:
            raise ValueError(
                f"OD600 calibration curve is not increasing at OD600 {od}; "
                f"it cannot be inverted.")
    return out


def od600_to_cfu_per_mL(od600, cal):
    """
    Forward calibration: estimated cfu/mL and its SD from OD600.

    Readings below the detection threshold are evaluated at the threshold
    (the lab's convention), so they give an upper bound. The SD combines
    the reading noise through the curve's slope and, when the calibration
    supplies it, the curve's own error.

    Returns
    -------
    cfu_per_mL, cfu_per_mL_std, detectable : numpy.ndarray
    """
    od = np.atleast_1d(np.asarray(od600, dtype=float)).copy()
    detectable = od >= cal["OD600_MEAS_THRESHOLD"]
    od[~detectable] = cal["OD600_MEAS_THRESHOLD"]

    cfu = cal["A_CFU"] + cal["B_CFU"] * od + cal["C_CFU"] * od ** 2
    slope = cal["B_CFU"] + 2 * cal["C_CFU"] * od
    var = (slope * cal["OD600_PCT_STD"] * od) ** 2
    if cal["P_JCJT_CFU"] is not None:
        curve_sd = (cal["P_JCJT_CFU"] + cal["Q_JCJT_CFU"] * od
                    + cal["R_JCJT_CFU"] * od ** 2)
        var = var + curve_sd ** 2
    return cfu, np.sqrt(var), detectable


def cfu_per_mL_to_od600(cfu_per_mL, cal):
    """
    Inverse calibration: the OD600 whose calibrated value is ``cfu_per_mL``
    (the root on the increasing branch of the quadratic). Totals below the
    curve's value at OD 0 map to OD 0.
    """
    y = np.atleast_1d(np.asarray(cfu_per_mL, dtype=float))
    a, b, c = cal["A_CFU"], cal["B_CFU"], cal["C_CFU"]
    if c == 0:
        od = (y - a) / b
    else:
        disc = np.maximum(b ** 2 - 4 * c * (a - y), 0.0)
        od = (-b + np.sqrt(disc)) / (2 * c)
    return np.maximum(od, 0.0)


def simulate_od600(total_cfu, volume_mL, cal, rng):
    """
    One OD600 reading per tube from its true total population.

    Parameters
    ----------
    total_cfu : array_like
        True total cells in each tube.
    volume_mL : float
        Tube volume (the calibration is per mL).
    cal : dict
        Constants from :func:`read_od600_calibration`.
    rng : numpy.random.Generator

    Returns
    -------
    dict of numpy.ndarray
        ``od600`` (the reading, with multiplicative noise of SD
        ``OD600_PCT_STD``), ``od600_detectable`` (reading at or above the
        detection threshold) and ``od600_in_range`` (reading at or below
        ``OD600_MAX``; always True when the calibration gives no maximum).
    """
    true_od = cfu_per_mL_to_od600(np.asarray(total_cfu, dtype=float) / volume_mL, cal)
    od = true_od * (1.0 + cal["OD600_PCT_STD"] * rng.standard_normal(true_od.shape))
    od = np.maximum(od, 0.0)
    detectable = od >= cal["OD600_MEAS_THRESHOLD"]
    if cal["OD600_MAX"] is None:
        in_range = np.ones(od.shape, dtype=bool)
    else:
        in_range = od <= cal["OD600_MAX"]
    return {"od600": od, "od600_detectable": detectable,
            "od600_in_range": in_range}
