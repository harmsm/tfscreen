"""
Rules for combining a co-transformed cell's plasmids into cell-level physics.

A cell that carries more than one plasmid has one growth rate. The simulator
builds that rate from cell-level physical quantities, never by combining the
plasmids' whole growth rates:

    k_cell = growth_model(theta_cell, activity_cell) + dk_cell

Each rule takes per-slot values for a block of cells and each slot's share of
its cell (``1/n`` for a cell with ``n`` plasmids, 0 for a masked slot; see
``selection_experiment._plasmid_shares``).

Theta rules
-----------
``fcn(theta, activity, shares) -> (theta_cell, activity_cell)``

- ``theta``, ``activity``: ``(num_cells, max_plasmids, num_conditions)``
- ``shares``: ``(num_cells, max_plasmids)``
- returns two ``(num_cells, num_conditions)`` arrays.

TF activity multiplies theta in the growth model, so it is combined by the
theta rule: it comes from whatever sets the cell's theta.

dk rules
--------
``fcn(dk_geno, shares) -> dk_cell``

- ``dk_geno``: ``(num_cells, max_plasmids)``
- returns ``(num_cells,)``.

See ``planning/congression-physics-plan.md`` for the physics and the planned
rules (partition-function theta, soft-min dk).
"""

import numpy as np


def theta_max(theta, activity, shares):
    """
    The plasmid with the highest theta sets the cell's theta and activity.

    Evaluated separately at each condition (the highest-theta plasmid can
    change with titrant concentration). Masked slots are ignored. If any
    valid slot's theta is NaN, the cell's theta is NaN.

    Parameters
    ----------
    theta : numpy.ndarray
        ``(num_cells, max_plasmids, num_conditions)``.
    activity : numpy.ndarray
        Same shape as ``theta``.
    shares : numpy.ndarray
        ``(num_cells, max_plasmids)``; 0 marks a masked slot.

    Returns
    -------
    theta_cell, activity_cell : numpy.ndarray
        ``(num_cells, num_conditions)`` each.
    """
    valid = (shares > 0)[:, :, np.newaxis]
    masked_theta = np.where(valid, theta, -np.inf)

    # np.argmax picks the first NaN if one is present, so a NaN theta in any
    # valid slot propagates to the cell.
    best = np.argmax(masked_theta, axis=1)[:, np.newaxis, :]
    theta_cell = np.take_along_axis(masked_theta, best, axis=1)[:, 0, :]
    activity_cell = np.take_along_axis(activity, best, axis=1)[:, 0, :]

    return theta_cell, activity_cell


def dk_dilution(dk_geno, shares):
    """
    Share-weighted mean dk_geno (each variant's burden diluted by its share).

    The ``alpha -> 0`` limit of the soft-min family in the congression plan.

    Parameters
    ----------
    dk_geno : numpy.ndarray
        ``(num_cells, max_plasmids)``.
    shares : numpy.ndarray
        ``(num_cells, max_plasmids)``; rows sum to 1.

    Returns
    -------
    numpy.ndarray
        ``(num_cells,)``.
    """
    return np.sum(np.where(shares > 0, dk_geno, 0.0) * shares, axis=1)


THETA_RULES = {
    "max": theta_max,
}

DK_RULES = {
    "dilution": dk_dilution,
}
