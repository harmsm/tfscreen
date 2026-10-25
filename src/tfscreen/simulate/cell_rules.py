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
theta rule. Under ``max`` it comes from the plasmid that sets the cell's
theta. The partition-function rules (``homodimer``, ``heterodimer``) mix the
variants' occupancies and require activity 1 (the lac repressor is taken as
fully active; see the congression plan, step 4).

dk rules
--------
``fcn(dk_geno, shares, alpha=None) -> dk_cell``

- ``dk_geno``: ``(num_cells, max_plasmids)``
- returns ``(num_cells,)``.

The soft-min family ``dk_cell = -(1/alpha) log sum_g x_g exp(-alpha dk_g)``
(congression plan, step 5): ``dilution`` (``alpha -> 0``, the default: the
share-weighted mean), ``softmin`` (finite ``alpha > 0``, in units of 1/dk)
and ``min`` (``alpha -> inf``: the worst variant sets the cell's cost). Only
``softmin`` takes ``alpha``.

See ``planning/congression-physics-plan.md`` for the physics.
"""

import numpy as np
from scipy.special import logsumexp

# Theta is clipped to [THETA_EPS, 1 - THETA_EPS] before taking its logit in
# the partition-function rules (the fit uses the same value).
THETA_EPS = 1e-6


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


def _theta_partition(theta, activity, shares, power):
    """
    Partition-function cell theta, in logit space.

    ``logit(theta_cell) = (1 / power) * log sum_g x_g exp(power * l_g)``,
    with ``l_g = logit(theta_g)`` and ``x_g`` the slot's share. ``power`` 1
    is homodimers, 0.5 random heterodimers with additive half-site energies.
    A cell with one plasmid keeps that plasmid's theta. Requires activity 1
    in every valid slot. Masked slots are ignored; a NaN theta in a valid
    slot makes the cell's theta NaN.
    """
    valid = shares > 0
    if np.any(np.abs(activity[valid] - 1.0) > 1e-9):
        raise ValueError(
            "The partition-function congression theta rules ('homodimer', "
            "'heterodimer') require TF activity 1 for every genotype (set "
            "activity_mut_scale to 0 and activity_wt to 1), or use "
            "congression_theta_rule 'max'. See the congression physics plan, "
            "step 4.")

    clipped = np.clip(theta, THETA_EPS, 1.0 - THETA_EPS)
    logit = np.log(clipped) - np.log1p(-clipped)
    valid3 = valid[:, :, np.newaxis]
    logit = np.where(valid3, logit, 0.0)
    weights = np.broadcast_to(shares[:, :, np.newaxis], logit.shape)

    cell_logit = logsumexp(power * logit, axis=1, b=weights) / power
    theta_cell = 1.0 / (1.0 + np.exp(-cell_logit))

    activity_cell = np.ones(theta_cell.shape)
    return theta_cell, activity_cell


def theta_homodimer(theta, activity, shares):
    """
    Homodimers only: the cell's odds are the share-weighted mean of its
    variants' odds, ``logit(theta_cell) = log sum_g x_g exp(l_g)``.

    A soft maximum between the share-weighted mean logit and the max; a
    strong binder in a 1:1 cell loses ``log 2`` logit units against the
    ``max`` rule. Requires activity 1. Arguments and returns as in
    :func:`theta_max`.
    """
    return _theta_partition(theta, activity, shares, 1.0)


def theta_heterodimer(theta, activity, shares):
    """
    Random heterodimers with additive half-site energies:
    ``logit(theta_cell) = 2 log sum_g x_g exp(l_g / 2)``.

    Closer to the share-weighted mean logit than :func:`theta_homodimer`;
    a strong binder in a 1:1 cell loses ``log 4``. Exact per titrant
    concentration only if subunits bind ligand independently. Requires
    activity 1. Arguments and returns as in :func:`theta_max`.
    """
    return _theta_partition(theta, activity, shares, 0.5)


def _check_no_alpha(name, alpha):
    if alpha is not None:
        raise ValueError(f"congression_dk_rule '{name}' takes no alpha "
                         f"(got {alpha}); only 'softmin' does.")


def dk_dilution(dk_geno, shares, alpha=None):
    """
    Share-weighted mean dk_geno (each variant's burden diluted by its share).

    The ``alpha -> 0`` limit of the soft-min family in the congression plan.

    Parameters
    ----------
    dk_geno : numpy.ndarray
        ``(num_cells, max_plasmids)``.
    shares : numpy.ndarray
        ``(num_cells, max_plasmids)``; rows sum to 1.
    alpha : None
        Not used; must be None.

    Returns
    -------
    numpy.ndarray
        ``(num_cells,)``.
    """
    _check_no_alpha("dilution", alpha)
    return np.sum(np.where(shares > 0, dk_geno, 0.0) * shares, axis=1)


def dk_softmin(dk_geno, shares, alpha=None):
    """
    Soft minimum: ``dk_cell = -(1/alpha) log sum_g x_g exp(-alpha dk_g)``.

    Between the share-weighted mean (``alpha -> 0``) and the minimum
    (``alpha -> inf``); ``alpha`` is in units of 1/dk (minutes when dk is
    per minute), so it matters once ``alpha`` times the spread of a cell's
    dk values reaches ~1. Masked slots are ignored; a NaN dk_geno in a
    valid slot makes the cell's dk NaN. Arguments and returns as in
    :func:`dk_dilution`, with ``alpha`` a positive float.
    """
    if alpha is None or not alpha > 0 or not np.isfinite(alpha):
        raise ValueError(f"congression_dk_rule 'softmin' needs a finite "
                         f"alpha > 0, not {alpha}.")
    valid = shares > 0
    dk = np.where(valid, dk_geno, 0.0)
    return -logsumexp(-alpha * dk, axis=1, b=shares) / alpha


def dk_min(dk_geno, shares, alpha=None):
    """
    The worst variant (lowest dk_geno) sets the cell's cost: the
    ``alpha -> inf`` limit of the soft-min family. Masked slots are ignored;
    a NaN dk_geno in a valid slot makes the cell's dk NaN. Arguments and
    returns as in :func:`dk_dilution`.
    """
    _check_no_alpha("min", alpha)
    return np.min(np.where(shares > 0, dk_geno, np.inf), axis=1)


THETA_RULES = {
    "max": theta_max,
    "homodimer": theta_homodimer,
    "heterodimer": theta_heterodimer,
}

DK_RULES = {
    "dilution": dk_dilution,
    "softmin": dk_softmin,
    "min": dk_min,
}
