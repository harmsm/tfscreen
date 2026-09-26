"""Tests for the co-transformed cell rules (simulate/cell_rules.py)."""

import numpy as np
import pytest

from tfscreen.simulate.cell_rules import (
    DK_RULES,
    THETA_EPS,
    THETA_RULES,
    dk_dilution,
    dk_min,
    dk_softmin,
    theta_heterodimer,
    theta_homodimer,
    theta_max,
)


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _cells():
    """Two cells over two conditions: a 2-plasmid cell and a 3-plasmid cell
    stored with one masked slot in the first."""
    theta = np.array([[[0.9, 0.2], [0.3, 0.6], [0.5, 0.5]],
                      [[0.1, 0.8], [0.7, 0.4], [0.95, 0.05]]])
    shares = np.array([[0.5, 0.5, 0.0],
                       [1 / 3, 1 / 3, 1 / 3]])
    activity = np.ones_like(theta)
    return theta, activity, shares


def test_registry():
    assert set(THETA_RULES) == {"max", "homodimer", "heterodimer"}
    assert set(DK_RULES) == {"dilution", "softmin", "min"}


def test_max_takes_highest_theta_and_its_activity():
    theta, _, shares = _cells()
    activity = np.arange(theta.size, dtype=float).reshape(theta.shape) + 1
    cell, act = theta_max(theta, activity, shares)
    np.testing.assert_allclose(cell, [[0.9, 0.6], [0.95, 0.8]])
    np.testing.assert_allclose(act, [[1, 4], [11, 8]])


@pytest.mark.parametrize("rule,power", [(theta_homodimer, 1.0),
                                        (theta_heterodimer, 0.5)])
def test_partition_rules_match_hand_calculation(rule, power):
    theta, activity, shares = _cells()
    cell, act = rule(theta, activity, shares)
    for c in range(2):
        n = int(np.sum(shares[c] > 0))
        for j in range(2):
            logits = _logit(theta[c, :n, j])
            expected = _expit(np.log(np.mean(np.exp(power * logits))) / power)
            assert cell[c, j] == pytest.approx(expected, rel=1e-10)
    np.testing.assert_array_equal(act, np.ones_like(cell))


def test_partition_rules_ordering():
    """heterodimer <= homodimer <= max; all at least the mean logit."""
    theta, activity, shares = _cells()
    het, _ = theta_heterodimer(theta, activity, shares)
    hom, _ = theta_homodimer(theta, activity, shares)
    mx, _ = theta_max(theta, activity, shares)
    assert np.all(het <= hom + 1e-12)
    assert np.all(hom <= mx + 1e-12)
    for c in range(2):
        n = int(np.sum(shares[c] > 0))
        mean_logit = np.mean(_logit(theta[c, :n]), axis=0)
        assert np.all(_logit(het[c]) >= mean_logit - 1e-10)


def test_strong_binder_loses_log2_and_log4():
    """A strong binder with a dead partner in a 1:1 cell loses log 2
    (homodimer) or log 4 (heterodimer) logit units against max."""
    strong, dead = 0.999, THETA_EPS
    theta = np.array([[[strong], [dead]]])
    shares = np.array([[0.5, 0.5]])
    activity = np.ones_like(theta)
    hom, _ = theta_homodimer(theta, activity, shares)
    het, _ = theta_heterodimer(theta, activity, shares)
    assert _logit(strong) - _logit(hom[0, 0]) == pytest.approx(np.log(2), abs=1e-3)
    assert _logit(strong) - _logit(het[0, 0]) == pytest.approx(np.log(4), abs=1e-2)


@pytest.mark.parametrize("rule", [theta_homodimer, theta_heterodimer, theta_max])
def test_single_plasmid_keeps_its_theta(rule):
    theta = np.array([[[0.3, 0.8], [0.9, 0.9]]])
    shares = np.array([[1.0, 0.0]])
    cell, _ = rule(theta, np.ones_like(theta), shares)
    np.testing.assert_allclose(cell, [[0.3, 0.8]], rtol=1e-9)


@pytest.mark.parametrize("rule", [theta_homodimer, theta_heterodimer])
def test_partition_rules_require_activity_one(rule):
    theta, activity, shares = _cells()
    activity[0, 1, 0] = 0.5
    with pytest.raises(ValueError, match="activity 1"):
        rule(theta, activity, shares)


@pytest.mark.parametrize("rule", [theta_homodimer, theta_heterodimer])
def test_partition_rules_ignore_masked_slot_activity(rule):
    """A masked slot's activity is not checked."""
    theta, activity, shares = _cells()
    activity[0, 2, :] = 0.5          # slot 2 of cell 0 is masked
    rule(theta, activity, shares)


@pytest.mark.parametrize("rule", [theta_homodimer, theta_heterodimer, theta_max])
def test_nan_in_valid_slot_propagates(rule):
    theta, activity, shares = _cells()
    theta[1, 0, 1] = np.nan
    cell, _ = rule(theta, activity, shares)
    assert np.isnan(cell[1, 1])
    assert np.all(np.isfinite(cell[0]))


def test_dk_dilution_is_share_weighted_mean():
    dk = np.array([[0.0, -0.02, 5.0], [-0.03, 0.0, 0.03]])
    shares = np.array([[0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    np.testing.assert_allclose(dk_dilution(dk, shares), [-0.01, 0.0])


def _dk_cells():
    dk = np.array([[0.0, -0.03, 5.0], [-0.03, 0.0, 0.03]])
    shares = np.array([[0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    return dk, shares


def test_dk_min_takes_worst_valid_slot():
    dk, shares = _dk_cells()
    np.testing.assert_allclose(dk_min(dk, shares), [-0.03, -0.03])


def test_dk_softmin_matches_hand_calculation():
    dk, shares = _dk_cells()
    alpha = 100.0
    expected = [-np.log(0.5 * np.exp(0.0) + 0.5 * np.exp(3.0)) / alpha,
                -np.log(np.mean(np.exp(-alpha * dk[1]))) / alpha]
    np.testing.assert_allclose(dk_softmin(dk, shares, alpha=alpha), expected)
    # The worked example in the plan: 1:1 cell, dk 0 and -0.03 -> -0.024
    assert dk_softmin(dk, shares, alpha=alpha)[0] == pytest.approx(-0.0236, abs=1e-4)


def test_dk_softmin_limits_and_order():
    dk, shares = _dk_cells()
    dil, mn = dk_dilution(dk, shares), dk_min(dk, shares)
    np.testing.assert_allclose(dk_softmin(dk, shares, alpha=1e-6), dil, atol=1e-9)
    np.testing.assert_allclose(dk_softmin(dk, shares, alpha=1e6), mn, atol=1e-5)
    soft = dk_softmin(dk, shares, alpha=50.0)
    assert np.all(mn <= soft) and np.all(soft <= dil)


@pytest.mark.parametrize("rule", [dk_dilution, dk_min])
def test_dk_rules_without_alpha_refuse_one(rule):
    dk, shares = _dk_cells()
    with pytest.raises(ValueError, match="takes no alpha"):
        rule(dk, shares, alpha=10.0)


@pytest.mark.parametrize("alpha", [None, 0.0, -1.0, np.inf])
def test_dk_softmin_needs_positive_finite_alpha(alpha):
    dk, shares = _dk_cells()
    with pytest.raises(ValueError, match="alpha > 0"):
        dk_softmin(dk, shares, alpha=alpha)


@pytest.mark.parametrize("rule,kwargs", [(dk_dilution, {}), (dk_min, {}),
                                         (dk_softmin, {"alpha": 100.0})])
def test_dk_rules_single_plasmid_and_nan(rule, kwargs):
    single = np.array([[-0.02, 7.0]])
    np.testing.assert_allclose(rule(single, np.array([[1.0, 0.0]]), **kwargs),
                               [-0.02], atol=1e-12)
    dk, shares = _dk_cells()
    dk[1, 0] = np.nan
    out = rule(dk, shares, **kwargs)
    assert np.isnan(out[1]) and np.isfinite(out[0])
