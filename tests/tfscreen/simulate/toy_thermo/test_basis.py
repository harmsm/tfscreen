"""Tests for the logit-epistasis basis-curve decomposition."""

import numpy as np
import pytest

from tfscreen.simulate.toy_thermo import ThermoModel
from tfscreen.simulate.toy_thermo.basis import (
    free_ensemble,
    basis_curves,
    epistasis_coeffs,
    predict_epistasis,
    exact_epistasis,
    classify_shape,
    resolvable_logit,
    logit_ci,
    measurement_window,
    FREE_STATES,
    _STATE_PAIRS,
)


@pytest.fixture
def model():
    # Notebook default: mixed L/H apo state, tight operator, trace DNA.
    return ThermoModel(ln_K_conf=0.0, ln_K_dna=18.0, ln_K_eff=14.0,
                       protein_total=1e-6, dna_total=1e-9)


@pytest.fixture
def effector():
    return np.logspace(-9, -1, 60)


# --------------------------------------------------------------------------- #
# Occupancies / basis curves
# --------------------------------------------------------------------------- #

def test_occupancies_sum_to_one(model, effector):
    p = free_ensemble(model, effector)
    total = sum(p[s] for s in FREE_STATES)
    assert np.allclose(total, 1.0)


def test_occupancies_limits(model, effector):
    # LE turns on with effector: ~0 at the low end, dominant at the high end.
    p = free_ensemble(model, effector)
    assert p["LE"][0] < 1e-3
    assert p["LE"][-1] > 0.9


def test_basis_products_are_products(model, effector):
    b = basis_curves(model, effector)
    for a, c in _STATE_PAIRS:
        assert np.allclose(b[f"{a}*{c}"], b[a] * b[c])


def test_le_products_peak_interior(model, effector):
    # p_L*p_LE and p_H*p_LE are zero at both ends and peak in the interior;
    # p_L*p_H is monotone (extremum at an edge).
    b = basis_curves(model, effector)
    for key in ("L*LE", "H*LE"):
        curve = b[key]
        assert curve[0] < curve.max() * 0.1
        assert curve[-1] < curve.max() * 0.1
        i = int(np.argmax(curve))
        assert 3 < i < len(curve) - 4
    assert int(np.argmax(b["L*H"])) < 3  # monotone decay from the low-E end


# --------------------------------------------------------------------------- #
# Coefficient bookkeeping
# --------------------------------------------------------------------------- #

def test_coeffs_cross_state_le():
    # gH x gLE -> single nonzero coefficient on the H*LE product, sign +4.
    c = epistasis_coeffs({"H": 2.0}, {"LE": 2.0})
    assert c["cross"][("H", "LE")] == pytest.approx(4.0)
    assert c["cross"][("L", "H")] == 0.0
    assert c["cross"][("L", "LE")] == 0.0


def test_coeffs_hd_is_offset_only():
    # Additive HD and pair epistasis on HD never touch the cross/direct basis.
    c = epistasis_coeffs({"HD": 2.0}, {"HD": 2.0}, epi={"HD": 3.0})
    assert all(v == 0.0 for v in c["cross"].values())
    assert all(v == 0.0 for v in c["direct"].values())
    assert c["offset"] == pytest.approx(-3.0)


def test_coeffs_direct_free_state():
    c = epistasis_coeffs({"H": 1.0}, {"H": 1.0}, epi={"LE": 2.0})
    assert c["direct"]["LE"] == pytest.approx(2.0)
    assert c["offset"] == 0.0


# --------------------------------------------------------------------------- #
# Prediction vs exact
# --------------------------------------------------------------------------- #

def test_prediction_converges_without_depletion(effector):
    # The closed form assumes effector is an external field. In the
    # low-depletion limit (Ke*[L] << 1) the second-order prediction converges
    # to exact as ddG shrinks.
    m = ThermoModel(ln_K_conf=0.0, ln_K_dna=18.0, ln_K_eff=14.0,
                    protein_total=1e-10, dna_total=1e-12)
    a, b = {"H": 0.1}, {"LE": 0.1}
    pred = predict_epistasis(m, effector, a, b)["total"]
    exact = exact_epistasis(m, effector, a, b)
    i = int(np.argmax(np.abs(exact)))
    assert pred[i] / exact[i] == pytest.approx(1.0, abs=0.1)


def test_prediction_recovers_shape_in_depleted_regime(model, effector):
    # With meaningful depletion the magnitude drifts (~15% high) and the tails
    # skew, but the qualitative signature is preserved: same peak sign and the
    # peak sits at the same location.
    a, b = {"H": 2.0}, {"LE": 2.0}
    pred = predict_epistasis(model, effector, a, b)["total"]
    exact = exact_epistasis(model, effector, a, b)
    assert classify_shape(pred) == "peak" and classify_shape(exact) == "peak"
    ip, ie = int(np.argmax(np.abs(pred))), int(np.argmax(np.abs(exact)))
    assert np.sign(pred[ip]) == np.sign(exact[ie])
    assert abs(ip - ie) <= 3


def test_prediction_breakdown_sums_to_total(model, effector):
    parts = predict_epistasis(model, effector, {"H": 2.0}, {"LE": 2.0},
                              epi={"LE": 1.0})
    recon = parts["offset"].copy()
    for v in parts["cross"].values():
        recon = recon + v
    for v in parts["direct"].values():
        recon = recon + v
    assert np.allclose(recon, parts["total"])


def test_cross_state_le_pair_peaks(model, effector):
    # The headline result: a mutation on an apo state x a mutation on LE peaks.
    exact = exact_epistasis(model, effector, {"H": 2.0}, {"LE": 2.0})
    assert classify_shape(exact) == "peak"


def test_pure_hd_additive_never_peaks(model, effector):
    exact = exact_epistasis(model, effector, {"HD": 2.0}, {"HD": 2.0})
    assert classify_shape(exact) in ("flat", "step")


def test_cross_state_no_le_is_step(model, effector):
    # gH x gL has an L*H coefficient only -> monotone, never a peak.
    exact = exact_epistasis(model, effector, {"H": 2.0}, {"L": 2.0})
    assert classify_shape(exact) == "step"


def test_direct_le_interaction_is_step(model, effector):
    # A within-state interaction is occupancy-weighted (monotone), not a peak.
    exact = exact_epistasis(model, effector, {"LE": 0.0}, {"LE": 0.0},
                            epi={"LE": 2.0})
    assert classify_shape(exact) == "step"


def test_peak_location_tracks_ligand_transition(model, effector):
    # The epistasis peak sits at the ligand transition (where p_LE ~ 0.5).
    exact = exact_epistasis(model, effector, {"H": 2.0}, {"LE": 2.0})
    p = free_ensemble(model, effector)
    i_peak = int(np.argmax(np.abs(exact)))
    i_trans = int(np.argmin(np.abs(p["LE"] - 0.5)))
    assert abs(i_peak - i_trans) <= 5


# --------------------------------------------------------------------------- #
# Shape classifier
# --------------------------------------------------------------------------- #

def test_classify_shape_flat():
    assert classify_shape(np.zeros(30)) == "flat"


def test_classify_shape_peak():
    x = np.linspace(-3, 3, 41)
    assert classify_shape(np.exp(-x**2)) == "peak"


def test_classify_shape_step():
    assert classify_shape(1 / (1 + np.exp(-np.linspace(-6, 6, 41)))) == "step"


# --------------------------------------------------------------------------- #
# Measurement range / logit error propagation
# --------------------------------------------------------------------------- #

def test_resolvable_logit_value_and_monotonicity():
    assert resolvable_logit(0.01) == pytest.approx(4.5951, abs=1e-3)
    # tighter resolution -> wider usable band
    assert resolvable_logit(0.001) > resolvable_logit(0.01) > resolvable_logit(0.1)


def test_logit_ci_center_matches_logit():
    theta = np.array([0.2, 0.5, 0.8])
    center, _, _ = logit_ci(theta, 0.01)
    assert np.allclose(center, np.log(theta / (1 - theta)))


def test_logit_ci_symmetric_in_the_middle():
    center, lo, hi = logit_ci(np.array([0.5]), 0.01, z=1.0)
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    assert (hi - center) == pytest.approx(center - lo, abs=1e-6)  # ~symmetric at 0.5


def test_logit_ci_one_sided_blowup_near_one():
    # theta + z*sigma reaches 1 -> upper tail is +inf, lower stays finite.
    center, lo, hi = logit_ci(np.array([0.995]), 0.01, z=1.0)
    assert np.isposinf(hi).all()
    assert np.isfinite(lo).all()
    assert np.isfinite(center).all()


def test_logit_ci_one_sided_blowup_near_zero():
    center, lo, hi = logit_ci(np.array([0.005]), 0.01, z=1.0)
    assert np.isneginf(lo).all()
    assert np.isfinite(hi).all()


def test_measurement_window_is_interval_at_edges(model, effector):
    fine = np.logspace(-9, -1, 400)
    win = measurement_window(model, fine, {"wt": {}}, eps=0.01)
    lo, hi = win["wt"]
    assert lo < hi
    # theta leaves the [eps, 1-eps] band just outside the reported edges
    th = model.observable(np.array([lo, hi]))
    assert (th <= 1 - 0.01 + 1e-6).all() and (th >= 0.01 - 1e-6).all()


def test_measurement_window_joint_is_intersection(model):
    fine = np.logspace(-9, -1, 400)
    genos = {"wt": {}, "B": {"LE": 2.0}}
    win = measurement_window(model, fine, genos, eps=0.01)
    lo = max(win["wt"][0], win["B"][0])
    hi = min(win["wt"][1], win["B"][1])
    assert win["joint"] == pytest.approx((lo, hi))


def test_measurement_window_none_when_unresolvable():
    # Very tight DNA binding keeps theta pinned at 1 across a low-effector grid.
    m = ThermoModel(0.0, 30.0, 14.0, 1e-6, 1e-9)
    grid = np.logspace(-9, -6, 50)
    win = measurement_window(m, grid, {"wt": {}}, eps=0.01)
    assert win["wt"] is None
    assert win["joint"] is None
