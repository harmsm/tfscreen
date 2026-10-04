"""
Simulated OD600 readings of each tube's total population.

The experiment estimates each tube's total cells from one OD600 reading
through a lab-specific calibration (see ``docs/source/process-raw.rst``,
"One tube per time-point", and ``tfs-calibrate-od600``). The simulator knows
each tube's true total, so it runs the calibration backwards to get the true
OD600, adds the reader's noise and applies its detection threshold, then
(optionally) runs the calibration forwards again, as the lab does, to get
the estimated total the pipeline would ingest.

The calibration file is the one ``tfs-calibrate-od600`` writes; reading,
inverting and applying it live in ``tfscreen.process_raw.od600``.
"""

import numpy as np

from tfscreen.process_raw.od600 import (
    as_calibration,
    cfu_per_mL_to_od600,
    od600_to_cfu_per_mL,
    read_calibration,
)

# The simulator's names for the shared functions.
read_od600_calibration = read_calibration

__all__ = ["read_od600_calibration", "od600_to_cfu_per_mL",
           "cfu_per_mL_to_od600", "simulate_od600"]


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
        Calibration from :func:`read_od600_calibration`.
    rng : numpy.random.Generator

    Returns
    -------
    dict of numpy.ndarray
        ``od600`` (the reading, with multiplicative noise of SD
        ``reading_rel_sd``), ``od600_detectable`` (reading at or above the
        detection threshold) and ``od600_in_range`` (reading at or below
        the top of the calibrated range).
    """
    cal = as_calibration(cal)
    true_od = cfu_per_mL_to_od600(np.asarray(total_cfu, dtype=float) / volume_mL,
                                  cal)
    od = true_od * (1.0 + cal["reading_rel_sd"]
                    * rng.standard_normal(true_od.shape))
    od = np.maximum(od, 0.0)
    detectable = od >= cal["detection_threshold"]
    in_range = od <= cal["calibrated_od600_range"][1]
    return {"od600": od, "od600_detectable": detectable,
            "od600_in_range": in_range}
