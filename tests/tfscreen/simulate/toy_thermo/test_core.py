"""Tests for tfscreen.simulate.toy_thermo.core."""

import numpy as np
import pytest

from tfscreen.simulate.toy_thermo.core import (
    fraction_bound, solve_species, _solve_free_L,
)

# A representative wild-type system: strong DNA binding, moderate effector.
KC, KD, KE = np.exp(0.0), np.exp(18.0), np.exp(14.0)
P_TOT, D_TOT = 1e-6, 1e-9


def test_mass_balance_closes():
    """Solved species satisfy all three conservation laws exactly."""
    for E in np.logspace(-8, -2, 7):
        s = solve_species(KC, KD, KE, P_TOT, D_TOT, E)
        assert s["L"] + s["H"] + s["HD"] + s["LE"] == pytest.approx(P_TOT, rel=1e-9)
        assert s["D"] + s["HD"] == pytest.approx(D_TOT, rel=1e-9)
        assert s["E"] + s["LE"] == pytest.approx(E, rel=1e-9)


def test_observable_matches_species():
    """fraction_bound equals HD/(HD+D) from the explicit species solve."""
    for E in np.logspace(-8, -2, 7):
        s = solve_species(KC, KD, KE, P_TOT, D_TOT, E)
        expected = s["HD"] / (s["HD"] + s["D"])
        assert fraction_bound(KC, KD, KE, P_TOT, D_TOT, E) == pytest.approx(expected, rel=1e-9)


def test_observable_bounds():
    """Observable always lies in [0, 1]."""
    y = fraction_bound(KC, KD, KE, P_TOT, D_TOT, np.logspace(-9, -1, 25))
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


def test_effector_induces_derepression():
    """More effector sequesters protein as LE -> observable decreases."""
    grid = np.logspace(-8, -2, 13)
    y = fraction_bound(KC, KD, KE, P_TOT, D_TOT, grid)
    assert np.all(np.diff(y) <= 1e-12)          # monotonically non-increasing
    assert y[0] > y[-1]                          # and actually drops


def test_scalar_in_scalar_out():
    """Scalar effector returns a Python float; array returns an ndarray."""
    scalar = fraction_bound(KC, KD, KE, P_TOT, D_TOT, 1e-5)
    assert isinstance(scalar, float)
    arr = fraction_bound(KC, KD, KE, P_TOT, D_TOT, np.array([1e-6, 1e-5]))
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)


def test_zero_effector_is_valid():
    """Zero effector -> no LE, fully DNA-limited repression, no NaN."""
    s = solve_species(KC, KD, KE, P_TOT, D_TOT, 0.0)
    assert s["LE"] == 0.0
    y = fraction_bound(KC, KD, KE, P_TOT, D_TOT, 0.0)
    assert np.isfinite(y) and 0.0 <= y <= 1.0


def test_free_L_within_protein_total():
    """Free [L] is bracketed by (0, protein_total]."""
    L = _solve_free_L(KC, KD, KE, P_TOT, D_TOT, 1e-5)
    assert 0.0 < L <= P_TOT


def test_weak_dna_binding_gives_low_occupancy():
    """Dropping K_dna far below the DNA scale leaves DNA mostly unbound."""
    y = fraction_bound(KC, np.exp(4.0), KE, P_TOT, D_TOT, 0.0)
    assert y < 0.5
