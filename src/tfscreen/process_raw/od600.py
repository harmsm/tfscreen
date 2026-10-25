"""
OD600-to-CFU calibration: fit, read, write and apply.

Roadmap step 3 (planning/analysis-roadmap.md; constraints C10, C11). A tube's
total population is estimated from its OD600 reading through a calibration
that depends on the plate reader, plate, volume and strain, so every lab
measures its own. This module holds the method; the calibration data stay
with the lab. ``tfs-calibrate-od600`` (``scripts/calibrate_od600_cli.py``)
runs it.

Two experiments feed a calibration:

- **Replicate readings of a dilution series** (``replicate_df``: columns
  ``dilution`` and ``od600``, one row per reading). Each reading repeats the
  whole handling step used in production (swirl, pipette into the plate,
  read). The spread of repeated readings gives the reading noise, as a
  fraction of the reading; the flattening of the series at high dilution
  gives the detection threshold.
- **Plate counts of cultures whose OD600 was read** (``plate_df``: columns
  ``od600``, ``colonies``, ``dilution`` (total dilution factor before
  plating), ``plated_volume_mL``, ``num_dilutions`` and ``plating_steps``).
  CFU/mL is ``colonies * dilution / plated_volume_mL``; its variance
  combines Poisson counting error (``1 / colonies``, relative) and one
  pipetting error per dilution and plating step.

A polynomial of CFU/mL in OD600 (the better-measured variable) is fit by
weighted least squares. Its full parameter covariance is kept: the curve's
error is one error shared by every tube calibrated with it (C11), so it
cannot be treated as independent per tube, and callers get it separately
from the reading noise (``cfu_per_mL_error_components``).

The method follows the lab notebook it replaces, with one change: the
notebook's CFU/mL omitted the division by the plated volume.
"""

import datetime

import numpy as np
import pandas as pd
import yaml

from tfscreen.util import read_yaml

CALIBRATION_KIND = "od600_to_cfu_per_mL"

REPLICATE_COLUMNS = ("dilution", "od600")
PLATE_COLUMNS = ("od600", "colonies", "dilution", "plated_volume_mL",
                 "num_dilutions", "plating_steps")

# Keys of the lab notebook's format, refused with a pointer to the new one.
_LEGACY_KEYS = ("A_CFU", "B_CFU", "C_CFU", "OD600_PCT_STD",
                "OD600_MEAS_THRESHOLD")


def _require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing column(s) {missing}; it needs "
                         f"{list(columns)}.")


# ---------------------------------------------------------------------------
# The two experiments
# ---------------------------------------------------------------------------

def reading_noise(replicate_df):
    """
    Per-dilution summary of repeated OD600 readings.

    Parameters
    ----------
    replicate_df : pandas.DataFrame
        Columns ``dilution`` and ``od600``, one row per reading.

    Returns
    -------
    pandas.DataFrame
        One row per dilution, most concentrated first: ``dilution``, ``n``,
        ``mean``, ``sd`` (population SD, as the lab notebook) and ``rel_sd``.
    """
    _require_columns(replicate_df, REPLICATE_COLUMNS, "replicate_df")
    df = replicate_df[list(REPLICATE_COLUMNS)].dropna()
    table = (df.groupby("dilution")["od600"]
               .agg(n="size", mean="mean", sd=lambda x: np.std(x, ddof=0))
               .reset_index()
               .sort_values("dilution", ascending=False)
               .reset_index(drop=True))
    if len(table) < 2:
        raise ValueError("replicate_df needs at least two dilutions.")
    table["rel_sd"] = table["sd"] / table["mean"]
    return table


def detection_threshold(noise_table):
    """
    Midway between the mean readings of the two most dilute samples, where
    the series has flattened onto the reader's floor (the lab notebook's
    rule). ``noise_table`` is the output of :func:`reading_noise`.
    """
    lowest = noise_table.sort_values("dilution")["mean"].to_numpy()[:2]
    return float(lowest.mean())


def plate_counts_to_cfu(plate_df, pipette_rel_error=0.02):
    """
    CFU/mL and its SD for each plated culture.

    ``cfu/mL = colonies * dilution / plated_volume_mL``, with relative
    variance ``1 / colonies + (num_dilutions + plating_steps) *
    pipette_rel_error^2`` (Poisson counting plus independent pipetting
    errors of equal relative size).

    Returns
    -------
    pandas.DataFrame
        ``plate_df`` with ``cfu_per_mL`` and ``cfu_per_mL_std`` added.
    """
    _require_columns(plate_df, PLATE_COLUMNS, "plate_df")
    df = plate_df.copy()
    colonies = df["colonies"].to_numpy(dtype=float)
    if np.any(colonies <= 0):
        raise ValueError("plate_df has plates with no colonies; they carry "
                         "no count and cannot be weighted.")
    cfu = colonies * df["dilution"].to_numpy(dtype=float) \
        / df["plated_volume_mL"].to_numpy(dtype=float)
    steps = (df["num_dilutions"].to_numpy(dtype=float)
             + df["plating_steps"].to_numpy(dtype=float))
    rel_var = 1.0 / colonies + steps * pipette_rel_error ** 2
    df["cfu_per_mL"] = cfu
    df["cfu_per_mL_std"] = cfu * np.sqrt(rel_var)
    return df


def fit_polynomial(od600, cfu, cfu_std, degree=2):
    """
    Weighted least-squares polynomial of CFU/mL in OD600.

    The covariance is scaled by the reduced chi-square, so misfit beyond the
    stated errors widens it (the convention of ``scipy.optimize.curve_fit``
    and of the lab notebook).

    Returns
    -------
    coefficients : numpy.ndarray
        ``c_0 .. c_degree``; ``cfu/mL = sum_i c_i OD600^i``.
    covariance : numpy.ndarray
    chi2_reduced : float
    """
    od = np.asarray(od600, dtype=float)
    y = np.asarray(cfu, dtype=float)
    w = 1.0 / np.asarray(cfu_std, dtype=float)
    dof = len(y) - (degree + 1)
    if dof < 1:
        raise ValueError(f"A degree-{degree} polynomial needs at least "
                         f"{degree + 2} plate counts; got {len(y)}.")
    X = np.vander(od, degree + 1, increasing=True)
    Xw = X * w[:, None]
    yw = y * w
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ coef
    chi2_red = float(resid @ resid / dof)
    covariance = np.linalg.inv(Xw.T @ Xw) * chi2_red
    covariance = 0.5 * (covariance + covariance.T)
    return coef, covariance, chi2_red


def calibrate(replicate_df, plate_df, degree=2, pipette_rel_error=0.02,
              threshold=None):
    """
    Build a calibration from the two experiments.

    Parameters
    ----------
    replicate_df, plate_df : pandas.DataFrame
        See the module docstring.
    degree : int, optional
        Polynomial degree (default 2).
    pipette_rel_error : float, optional
        Relative error of one pipetting step (default 0.02).
    threshold : float, optional
        Detection threshold; default from the dilution series
        (:func:`detection_threshold`).

    Returns
    -------
    calibration : dict
        The calibration, as written by :func:`write_calibration`.
    noise_table : pandas.DataFrame
        :func:`reading_noise` output.
    plate_table : pandas.DataFrame
        The plate counts with CFU/mL, its SD, the fitted curve and the
        standardized residual.
    """
    noise_table = reading_noise(replicate_df)
    if threshold is None:
        threshold = detection_threshold(noise_table)
    plate_table = plate_counts_to_cfu(plate_df, pipette_rel_error)
    od = plate_table["od600"].to_numpy(dtype=float)
    coef, cov, chi2_red = fit_polynomial(od, plate_table["cfu_per_mL"],
                                         plate_table["cfu_per_mL_std"],
                                         degree)

    calibration = {
        "kind": CALIBRATION_KIND,
        "degree": int(degree),
        "coefficients": [float(c) for c in coef],
        "covariance": [[float(v) for v in row] for row in cov],
        "chi2_reduced": chi2_red,
        "reading_rel_sd": float(noise_table["rel_sd"].max()),
        "detection_threshold": float(threshold),
        "calibrated_od600_range": [float(od.min()), float(od.max())],
        "num_plate_counts": int(len(od)),
        "pipette_rel_error": float(pipette_rel_error),
    }
    check_calibration(calibration)

    fitted = np.vander(od, degree + 1, increasing=True) @ coef
    plate_table["cfu_per_mL_fit"] = fitted
    plate_table["z"] = (plate_table["cfu_per_mL"] - fitted) \
        / plate_table["cfu_per_mL_std"]
    return calibration, noise_table, plate_table


# ---------------------------------------------------------------------------
# The calibration file
# ---------------------------------------------------------------------------

def check_calibration(cal):
    """
    Validate a calibration dict; raise ValueError on a problem.

    The curve must increase across ``[detection_threshold, top of the
    calibrated range]`` so every reading maps to one total.
    """
    if any(k in cal for k in _LEGACY_KEYS) and "coefficients" not in cal:
        raise ValueError(
            "This is the lab notebook's calibration format (A_CFU, B_CFU, "
            "...). Regenerate it with tfs-calibrate-od600, which keeps the "
            "full parameter covariance and fixes the notebook's CFU/mL, "
            "which did not divide by the plated volume.")
    required = ("kind", "degree", "coefficients", "covariance",
                "reading_rel_sd", "detection_threshold",
                "calibrated_od600_range")
    missing = [k for k in required if k not in cal]
    if missing:
        raise ValueError(f"OD600 calibration is missing {missing}.")
    if cal["kind"] != CALIBRATION_KIND:
        raise ValueError(f"Not an OD600 calibration (kind {cal['kind']!r}).")
    n = int(cal["degree"]) + 1
    coef = np.asarray(cal["coefficients"], dtype=float)
    cov = np.asarray(cal["covariance"], dtype=float)
    if coef.shape != (n,) or cov.shape != (n, n):
        raise ValueError(f"A degree-{n - 1} calibration needs {n} "
                         f"coefficients and an {n}x{n} covariance.")
    lo, hi = (float(v) for v in cal["calibrated_od600_range"])
    grid = np.linspace(min(float(cal["detection_threshold"]), lo), hi, 200)
    if np.any(_poly_slope(coef, grid) <= 0):
        raise ValueError("The calibration curve is not increasing between "
                         "the detection threshold and the top of the "
                         "calibrated range; it cannot be inverted.")


def read_calibration(calibration):
    """
    Read and validate an OD600 calibration (a YAML path or the dict).

    Returns
    -------
    dict
        With ``coefficients`` and ``covariance`` as numpy arrays and the
        other values as floats.
    """
    cal = dict(read_yaml(calibration))
    check_calibration(cal)
    cal["coefficients"] = np.asarray(cal["coefficients"], dtype=float)
    cal["covariance"] = np.asarray(cal["covariance"], dtype=float)
    cal["reading_rel_sd"] = float(cal["reading_rel_sd"])
    cal["detection_threshold"] = float(cal["detection_threshold"])
    cal["calibrated_od600_range"] = [float(v) for v in
                                     cal["calibrated_od600_range"]]
    return cal


def write_calibration(calibration, path, source=None):
    """
    Write a calibration YAML, with a header saying what it is and how it was
    made (``source``: optional dict recorded under ``source``).
    """
    cal = dict(calibration)
    if source is not None:
        cal["source"] = dict(source)
    cal["written"] = datetime.date.today().isoformat()
    header = (
        "# OD600-to-CFU calibration, written by tfs-calibrate-od600.\n"
        "# cfu/mL = sum_i coefficients[i] * OD600^i. covariance is the\n"
        "# coefficients' (shared by every tube calibrated with this file);\n"
        "# reading_rel_sd is one reading's SD as a fraction of the reading.\n"
        "# Specific to one plate reader, plate, volume and strain.\n")
    with open(path, "w") as fh:
        fh.write(header)
        yaml.safe_dump(cal, fh, sort_keys=False)


# ---------------------------------------------------------------------------
# Applying it
# ---------------------------------------------------------------------------

def as_calibration(cal):
    """A calibration as read by :func:`read_calibration` (read it if not)."""
    if isinstance(cal, dict) and isinstance(cal.get("coefficients"), np.ndarray):
        return cal
    return read_calibration(cal)


def _poly(coef, od):
    return np.vander(np.ravel(od), len(coef), increasing=True) @ coef


def _poly_slope(coef, od):
    powers = np.arange(1, len(coef))
    od = np.ravel(od)
    return (od[:, None] ** (powers - 1)) @ (coef[1:] * powers)


def cfu_per_mL_error_components(od600, cal):
    """
    The two parts of the SD of a calibrated CFU/mL, kept apart.

    Returns
    -------
    curve_sd : numpy.ndarray
        From the coefficients' covariance, ``sqrt(J C J^T)`` with
        ``J = [1, OD, OD^2, ...]``: one error shared by every tube
        calibrated with this curve (C11).
    reading_sd : numpy.ndarray
        One reading's noise through the curve's slope, independent per
        tube.
    """
    cal = as_calibration(cal)
    od = np.atleast_1d(np.asarray(od600, dtype=float))
    J = np.vander(od, len(cal["coefficients"]), increasing=True)
    curve_var = np.einsum("ij,jk,ik->i", J, cal["covariance"], J)
    reading_sd = np.abs(_poly_slope(cal["coefficients"], od)) \
        * cal["reading_rel_sd"] * od
    return np.sqrt(np.maximum(curve_var, 0.0)), reading_sd


def od600_to_cfu_per_mL(od600, cal):
    """
    Forward calibration.

    Readings below the detection threshold are evaluated at the threshold,
    so they give an upper bound.

    Returns
    -------
    cfu_per_mL, cfu_per_mL_std, detectable : numpy.ndarray
        The SD combines the curve and reading parts
        (:func:`cfu_per_mL_error_components`).
    """
    cal = as_calibration(cal)
    od = np.atleast_1d(np.asarray(od600, dtype=float)).copy()
    detectable = od >= cal["detection_threshold"]
    od[~detectable] = cal["detection_threshold"]
    cfu = _poly(cal["coefficients"], od)
    curve_sd, reading_sd = cfu_per_mL_error_components(od, cal)
    return cfu, np.sqrt(curve_sd ** 2 + reading_sd ** 2), detectable


def in_calibrated_range(od600, cal):
    """True where a reading is inside the plate counts' OD600 range."""
    lo, hi = as_calibration(cal)["calibrated_od600_range"]
    od = np.atleast_1d(np.asarray(od600, dtype=float))
    return (od >= lo) & (od <= hi)


def cfu_per_mL_to_od600(cfu_per_mL, cal, od_max=None):
    """
    Inverse calibration: the OD600 whose calibrated CFU/mL is the given one,
    by bisection on ``[0, od_max]`` (default twice the top of the calibrated
    range), where the curve must be increasing. Totals below the curve at
    OD 0 map to 0; totals above it at ``od_max`` map to ``od_max``.
    """
    cal = as_calibration(cal)
    coef = cal["coefficients"]
    if od_max is None:
        od_max = 2.0 * cal["calibrated_od600_range"][1]
    y = np.atleast_1d(np.asarray(cfu_per_mL, dtype=float))
    lo = np.zeros_like(y)
    hi = np.full_like(y, float(od_max))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = _poly(coef, mid) < y
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    od = 0.5 * (lo + hi)
    od[y <= _poly(coef, np.zeros(1))[0]] = 0.0
    return od


TUBE_OD600_COLUMNS = ("sample", "od600")


def tube_totals_from_od600(od600_df, cal, tube_volume_mL):
    """
    Each tube's total CFU from its OD600 reading.

    Parameters
    ----------
    od600_df : pandas.DataFrame
        One row per sequenced tube, with columns ``sample`` and ``od600``.
    cal : dict or str
        A ``tfs-calibrate-od600`` calibration, or the path to one.
    tube_volume_mL : float
        Culture volume of a tube in mL. The calibration gives CFU/mL; the
        model's ``sample_cfu`` is the cells in the whole tube.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``sample``: ``od600``, ``sample_cfu``, ``sample_cfu_std``
        (curve and reading errors combined), ``sample_cfu_curve_std`` (the
        curve's error, shared by every tube read through this calibration;
        C11), ``sample_cfu_reading_std`` (the reading's own noise, independent
        per tube) and ``od600_in_calibrated_range``.

    Raises
    ------
    ValueError
        On missing columns, a duplicated or missing sample, a non-positive
        volume, or a reading below the calibration's detection threshold.
    """
    _require_columns(od600_df, TUBE_OD600_COLUMNS, "OD600 table")
    if not tube_volume_mL or tube_volume_mL <= 0:
        raise ValueError("tube_volume_mL must be a positive volume in mL.")

    df = od600_df[list(TUBE_OD600_COLUMNS)].copy()
    df["sample"] = df["sample"].astype(str)
    dups = df["sample"][df["sample"].duplicated()].unique().tolist()
    if dups:
        raise ValueError(f"OD600 table has more than one reading for: {dups}")
    od = pd.to_numeric(df["od600"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(od)):
        bad = df["sample"][~np.isfinite(od)].tolist()
        raise ValueError(f"OD600 table has no numeric reading for: {bad}")

    cal = as_calibration(cal)
    cfu, cfu_sd, detectable = od600_to_cfu_per_mL(od, cal)
    if not np.all(detectable):
        bad = df["sample"][~detectable].tolist()
        raise ValueError(
            f"OD600 below the calibration's detection threshold "
            f"({cal['detection_threshold']:g}) for {bad}. Their totals are "
            "only upper bounds; drop these tubes from the tube table."
        )
    curve_sd, reading_sd = cfu_per_mL_error_components(od, cal)

    out = pd.DataFrame({
        "od600": od,
        "sample_cfu": cfu * tube_volume_mL,
        "sample_cfu_std": cfu_sd * tube_volume_mL,
        "sample_cfu_curve_std": curve_sd * tube_volume_mL,
        "sample_cfu_reading_std": reading_sd * tube_volume_mL,
        "od600_in_calibrated_range": in_calibrated_range(od, cal),
    }, index=pd.Index(df["sample"].to_numpy(), name="sample"))
    return out
