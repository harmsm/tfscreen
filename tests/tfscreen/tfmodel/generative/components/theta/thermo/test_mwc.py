"""
The standard MWC dimer shared by the O2_C4 (M2) and O2_C12 (M3) models.

Covers the two-site effector polynomial (against a brute-force sum over
explicit species), the free-effector cubic and its Newton solve, the exact
nesting of M2 in M3 (K_h_e = K_l_o = 0), the 80 µM default wt midpoints and
the structural projection rows.
"""
import importlib
import itertools
import math
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from tfscreen.tfmodel.generative.components.theta.thermo import mwc

_PKG = "tfscreen.tfmodel.generative.components.theta.thermo"
_TF = mwc.DEFAULT_TF_TOTAL_M

Data = namedtuple("Data", ["titrant_conc", "batch_idx", "geno_theta_idx",
                           "scatter_theta"])


def _module(family, kind):
    return importlib.import_module(f"{_PKG}.{family}.{kind}")


# ---------------------------------------------------------------------------
# Brute force: explicit species, numerical mass balance
# ---------------------------------------------------------------------------

def _brute_force(K_h_l, K_h_o, K_h_e, K_l_o, K_l_e, K_u, e_total, tf_total):
    """
    θ and free effector from an explicit list of dimer species.

    Each conformation X in {H, L} has two distinguishable effector sites, so
    four microstates (s1, s2) with weight c_X·(K_x_e·e)^(s1+s2); U has one.
    TF not bound to operator obeys r = h·Σw, effector obeys
    e_total = e + h·Σ w·(s1+s2). Operator-bound species carry an extra
    K_x_o and give W = h·Σ w·K_x_o, θ = W/(1+W). The polynomial is never
    written down, so this checks the statistical factor of 2 independently.
    """
    r = tf_total / 2.0
    confs = [(1.0, K_h_e, K_h_o), (K_h_l, K_l_e, K_l_o)]

    def sums(e):
        tot = K_u
        bound = 0.0
        op = 0.0
        for c, Ke, Ko in confs:
            for s1, s2 in itertools.product((0, 1), repeat=2):
                n = s1 + s2
                w = c * (Ke * e) ** n
                tot += w
                bound += w * n
                op += w * Ko
        return tot, bound, op

    def balance(e):
        tot, bound, _ = sums(e)
        return e + r * bound / tot - e_total

    e = 0.0 if e_total == 0 else brentq(balance, 0.0, e_total,
                                        xtol=1e-300, rtol=1e-14)
    tot, _, op = sums(e)
    W = r / tot * op
    return W / (1.0 + W), e


_CASES = [
    # K_h_l, K_h_o, K_h_e, K_l_o, K_l_e, K_u
    (6.3, 4.2e8, 2.1e4, 0.1, 2.9e5, 0.0),
    (0.5, 1e7, 3e5, 1e5, 2e4, 0.0),
    (6.3, 4.2e8, 0.0, 0.0, 2.5e5, 0.0),      # M2
    (20.0, 1e9, 1e3, 1e2, 1e6, 3.0),         # with U
]
_E = [0.0, 1e-7, 3e-6, 1e-5, 8e-5, 1e-3, 0.1]


@pytest.mark.parametrize("ks", _CASES)
@pytest.mark.parametrize("tf_total", [_TF, 1e-9])
def test_theta_matches_brute_force_species_sum(ks, tf_total):
    for e_total in _E:
        want_theta, want_e = _brute_force(*ks, e_total, tf_total)
        got = mwc.mwc_theta(*ks, np.array(e_total), tf_total, xp=np)
        assert float(got) == pytest.approx(want_theta, rel=1e-10, abs=1e-14)
        B0, B1, B2 = mwc._poly_coeffs(ks[0], ks[2], ks[4], ks[5])
        e = mwc.free_effector(B0, B1, B2, np.array(e_total), tf_total / 2,
                              xp=np)
        assert float(e) == pytest.approx(want_e, rel=1e-10, abs=1e-20)


def test_two_independent_sites_give_macroscopic_ratio_one_quarter():
    """P(e) = (1 + K·e)² has K1 = 2K and K2 = K/2, so K2/K1 = 1/4."""
    K = 3.7e4
    B0, B1, B2 = mwc._poly_coeffs(0.0, K, 0.0, 0.0)     # H only
    K1 = B1 / B0
    K2 = B2 / B1
    assert K1 == pytest.approx(2 * K)
    assert K2 / K1 == pytest.approx(0.25)


def test_jax_and_numpy_paths_agree():
    rng = np.random.default_rng(0)
    ks = [np.exp(rng.normal(m, 1.5, 50)) for m in
          (1.8, 19.9, 10.0, -2.3, 12.6, -3.0)]
    e = np.logspace(-8, -1, 50)
    got_np = mwc.mwc_theta(*ks, e, _TF, xp=np)
    got_jax = np.asarray(mwc.mwc_theta(*[jnp.asarray(k) for k in ks],
                                       jnp.asarray(e), _TF))
    np.testing.assert_allclose(got_jax, got_np, rtol=2e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Free-effector cubic
# ---------------------------------------------------------------------------

def test_newton_solve_matches_bracketing_root_finder():
    """NEWTON_ITERATIONS steps reach float64 precision over broad draws."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(2000):
        K_h_l = math.exp(rng.normal(1.84, 2.0))
        K_h_e = math.exp(rng.normal(10.0, 2.0))
        K_l_e = math.exp(rng.normal(12.6, 2.0))
        tf = 10 ** rng.uniform(-7, -4.5)
        e_total = 10 ** rng.uniform(-8, -1)
        B0, B1, B2 = mwc._poly_coeffs(K_h_l, K_h_e, K_l_e, 0.0)
        r = tf / 2

        def g(x):
            return (e_total - x) * (B0 + B1 * x + B2 * x * x) \
                - r * (B1 * x + 2 * B2 * x * x)

        want = brentq(g, 0.0, e_total, xtol=1e-300, rtol=1e-15)
        got = float(mwc.free_effector(B0, B1, B2, np.array(e_total), r,
                                      xp=np))
        worst = max(worst, abs(got - want) / e_total)
    assert worst < 1e-12


def test_free_effector_bounds_and_depletion():
    e_total = np.array([0.0, 1e-6, 1e-5, 1e-4])
    B0, B1, B2 = mwc._poly_coeffs(6.3, 2e4, 3e5, 0.0)
    e = mwc.free_effector(B0, B1, B2, e_total, _TF / 2, xp=np)
    assert np.all(e >= 0) and np.all(e <= e_total)
    assert e[0] == 0.0
    # At tf_total = 1e-5 M a micromolar effector is mostly bound.
    assert e[1] < 0.5 * e_total[1]
    e_weak = mwc.free_effector(B0, B1, B2, e_total, 1e-15, xp=np)
    np.testing.assert_allclose(e_weak, e_total, rtol=1e-8)


# ---------------------------------------------------------------------------
# M2 is M3 with K_h_e = K_l_o = 0
# ---------------------------------------------------------------------------

def _random_m2(S, rng):
    return (rng.normal(mwc.DEFAULT_LN_K_H_O, 2.0, S),
            rng.normal(mwc.DEFAULT_LN_K_H_L, 2.0, S),
            rng.normal(mwc.DEFAULT_LN_K_E_C4, 2.0, S),
            rng.normal(-3.0, 2.0, S))


@pytest.mark.parametrize("unfold", [False, True])
def test_c4_is_exactly_c12_without_h_effector_and_l_operator(unfold):
    """theta_c4 equals theta_c12 with ln K_h_e, ln K_l_o = -inf (numpy)."""
    rng = np.random.default_rng(2)
    S = 300
    ln_op, ln_hl, ln_e, ln_u = _random_m2(S, rng)
    ln_u = ln_u if unfold else None
    conc = np.logspace(-8, -1, 40)
    a = mwc.theta_c4(ln_op[:, None], ln_hl[:, None], ln_e[:, None],
                     None if ln_u is None else ln_u[:, None],
                     conc[None, :], _TF, xp=np)
    b = mwc.theta_c12(ln_hl[:, None], ln_op[:, None], np.full((S, 1), -np.inf),
                      np.full((S, 1), -np.inf), ln_e[:, None],
                      None if ln_u is None else ln_u[:, None],
                      conc[None, :], _TF, xp=np)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("u", ["U0", "U1"])
def test_c4_components_equal_c12_components_with_depletion(u):
    """
    The families' own compute_theta_samples and run_model agree to float
    tolerance when C12 has ln K_h_e, ln K_l_o -> -inf, at tf_total = 1e-5 M
    where effector depletion matters below about 10 µM.
    """
    c4 = _module(f"O2_C4_K3_{u}_a", "thermo")
    c12 = _module(f"O2_C12_K5_{u}_a", "thermo")
    rng = np.random.default_rng(3)
    S = 200
    ln_op, ln_hl, ln_e, ln_u = _random_m2(S, rng)
    conc_mM = np.concatenate([[0.0], np.logspace(-5, 1, 50)])
    calc_df = pd.DataFrame({"titrant_conc": conc_mM, "genotype_idx": 0,
                            "titrant_name_idx": 0})
    post4 = {"theta_ln_K_op": ln_op[:, None], "theta_ln_K_HL": ln_hl[:, None],
             "theta_ln_K_E": ln_e[:, None, None]}
    post12 = {"theta_ln_K_h_l": ln_hl[:, None], "theta_ln_K_h_o": ln_op[:, None],
              "theta_ln_K_h_e": np.full((S, 1, 1), -np.inf),
              "theta_ln_K_l_o": np.full((S, 1), -np.inf),
              "theta_ln_K_l_e": ln_e[:, None, None]}
    if u == "U1":
        post4["theta_ln_K_U"] = ln_u[:, None]
        post12["theta_ln_K_u"] = ln_u[:, None]
    t4 = c4.compute_theta_samples(calc_df, post4, tf_total=_TF,
                                  conc_unit_scale=1e-3)
    t12 = c12.compute_theta_samples(calc_df, post12, tf_total=_TF,
                                    conc_unit_scale=1e-3)
    np.testing.assert_allclose(t4, t12, rtol=1e-12, atol=1e-15)
    # Depletion is not negligible here: a near-zero tf_total changes theta.
    t4_nodep = c4.compute_theta_samples(calc_df, post4, tf_total=1e-15,
                                        conc_unit_scale=1e-3)
    assert np.max(np.abs(t4 - t4_nodep)) > 1e-3

    # JAX path (run_model), float32.
    G = 5
    k = dict(ln_op=jnp.asarray(ln_op[:G]), ln_hl=jnp.asarray(ln_hl[:G]),
             ln_e=jnp.asarray(ln_e[:G])[None, :], ln_u=jnp.asarray(ln_u[:G]))
    data = Data(titrant_conc=jnp.asarray(conc_mM), batch_idx=jnp.arange(G),
                geno_theta_idx=jnp.arange(G), scatter_theta=0)
    mom = dict(mu=jnp.zeros((1, conc_mM.size, 1)),
               sigma=jnp.ones((1, conc_mM.size, 1)),
               tf_total=_TF, conc_unit_scale=1e-3)
    extra4 = {"ln_K_U": k["ln_u"]} if u == "U1" else {}
    extra12 = {"ln_K_u": k["ln_u"]} if u == "U1" else {}
    p4 = c4.ThetaParam(ln_K_op=k["ln_op"], ln_K_HL=k["ln_hl"],
                       ln_K_E=k["ln_e"], **extra4, **mom)
    p12 = c12.ThetaParam(ln_K_h_l=k["ln_hl"], ln_K_h_o=k["ln_op"],
                         ln_K_h_e=jnp.full((1, G), -jnp.inf),
                         ln_K_l_o=jnp.full(G, -jnp.inf),
                         ln_K_l_e=k["ln_e"], **extra12, **mom)
    np.testing.assert_allclose(np.asarray(c4.run_model(p4, data)),
                               np.asarray(c12.run_model(p12, data)),
                               rtol=1e-6, atol=1e-7)


def test_m2_apo_and_saturation_limits():
    """θ(0) = W/(1+W), W = r·K_op/(1 + K_HL); θ → 0 at saturating effector."""
    K_op, K_HL, K_E = 4.2e8, 6.3, 2.5e5
    r = _TF / 2
    W = r * K_op / (1 + K_HL)
    got = mwc.theta_c4(math.log(K_op), math.log(K_HL), math.log(K_E), None,
                       np.array([0.0, 10.0]), _TF, xp=np)
    assert got[0] == pytest.approx(W / (1 + W), rel=1e-12)
    assert got[1] < 1e-6


# ---------------------------------------------------------------------------
# Default wild type: midpoint at 80 µM IPTG
# ---------------------------------------------------------------------------

def _wt_curve(family):
    """Default wt θ(e) and its saturating limit."""
    hp = _module(family, "PK").get_hyperparameters()
    assert hp["theta_tf_total_M"] == pytest.approx(1e-5)
    if family.startswith("O2_C4"):
        t_sat = 0.0
        def f(e):
            return mwc.theta_c4(hp["theta_ln_K_op_wt_loc"],
                                hp["theta_ln_K_HL_wt_loc"],
                                hp["theta_ln_K_E_wt_loc"], None,
                                np.asarray(e, float),
                                hp["theta_tf_total_M"], xp=np)
    else:
        ks = [math.exp(hp[f"theta_ln_K_{k}_wt_loc"])
              for k in ("h_l", "h_o", "h_e", "l_o", "l_e")]
        t_sat = mwc.theta_saturated(*ks, 0.0, hp["theta_tf_total_M"], xp=np)

        def f(e):
            return mwc.theta_c12(hp["theta_ln_K_h_l_wt_loc"],
                                 hp["theta_ln_K_h_o_wt_loc"],
                                 hp["theta_ln_K_h_e_wt_loc"],
                                 hp["theta_ln_K_l_o_wt_loc"],
                                 hp["theta_ln_K_l_e_wt_loc"], None,
                                 np.asarray(e, float),
                                 hp["theta_tf_total_M"], xp=np)
    return f, float(t_sat)


@pytest.mark.parametrize("family", ["O2_C4_K3_U0_a", "O2_C12_K5_U0_a"])
def test_default_wt_midpoint_is_80_uM(family):
    f, t_sat = _wt_curve(family)
    t0 = float(f(0.0))
    # Saturating effector at 10 M is the analytic limit to 1e-3.
    assert float(f(10.0)) == pytest.approx(t_sat, abs=1e-3)
    half = 0.5 * (t0 + t_sat)
    mid = math.exp(brentq(lambda le: float(f(math.exp(le))) - half,
                          math.log(1e-9), math.log(1.0)))
    assert mid == pytest.approx(80e-6, rel=1e-6)
    assert t0 > 0.99


def _calibrate_c12_scale():
    """Common factor on Sochor's K_h_e and K_l_e that puts the midpoint at 80 µM."""
    def mid_err(ln_f):
        f = math.exp(ln_f)
        args = (mwc.SOCHOR_K_H_L, mwc.SOCHOR_K_H_O, mwc.SOCHOR_K_H_E * f,
                mwc.SOCHOR_K_L_O, mwc.SOCHOR_K_L_E * f, 0.0)
        t0 = float(mwc.mwc_theta(*args, np.array(0.0), _TF, xp=np))
        ts = float(mwc.theta_saturated(*args, _TF, xp=np))
        tm = float(mwc.mwc_theta(*args, np.array(80e-6), _TF, xp=np))
        return tm - 0.5 * (t0 + ts)
    return math.exp(brentq(mid_err, -10.0, 5.0, xtol=1e-14))


def _calibrate_c4_ln_K_E():
    def mid_err(ln_k):
        th = [float(mwc.theta_c4(mwc.DEFAULT_LN_K_H_O, mwc.DEFAULT_LN_K_H_L,
                                 ln_k, None, np.array(e), _TF, xp=np))
              for e in (0.0, 80e-6)]
        return th[1] - 0.5 * th[0]        # θ_sat = 0 for M2
    return brentq(mid_err, 0.0, 25.0, xtol=1e-14)


def test_default_constants_are_the_calibrated_ones():
    assert mwc.MWC_EFFECTOR_SCALE == pytest.approx(_calibrate_c12_scale(),
                                                   rel=1e-8)
    assert mwc.DEFAULT_LN_K_E_C4 == pytest.approx(_calibrate_c4_ln_K_E(),
                                                  abs=1e-8)
    # C12 keeps Sochor's K_h_l, K_h_o, K_l_o and K_l_e/K_h_e.
    assert math.exp(mwc.DEFAULT_LN_K_H_L) == pytest.approx(6.3)
    assert math.exp(mwc.DEFAULT_LN_K_H_O) == pytest.approx(4.2e8)
    assert math.exp(mwc.DEFAULT_LN_K_L_O) == pytest.approx(0.1)
    assert math.exp(mwc.DEFAULT_LN_K_L_E - mwc.DEFAULT_LN_K_H_E) == \
        pytest.approx(7.6e5 / 5.6e4)


@pytest.mark.parametrize("kind", ["PK", "PddG", "PnnC"])
@pytest.mark.parametrize("u", ["U0", "U1"])
def test_every_component_uses_the_shared_defaults(kind, u):
    c4 = _module(f"O2_C4_K3_{u}_a", kind).get_hyperparameters()
    c12 = _module(f"O2_C12_K5_{u}_a", kind).get_hyperparameters()
    assert c4["theta_ln_K_op_wt_loc"] == mwc.DEFAULT_LN_K_H_O
    assert c4["theta_ln_K_HL_wt_loc"] == mwc.DEFAULT_LN_K_H_L
    assert c4["theta_ln_K_E_wt_loc"] == mwc.DEFAULT_LN_K_E_C4
    assert c12["theta_ln_K_h_l_wt_loc"] == mwc.DEFAULT_LN_K_H_L
    assert c12["theta_ln_K_h_o_wt_loc"] == mwc.DEFAULT_LN_K_H_O
    assert c12["theta_ln_K_l_o_wt_loc"] == mwc.DEFAULT_LN_K_L_O
    assert c12["theta_ln_K_h_e_wt_loc"] == mwc.DEFAULT_LN_K_H_E
    assert c12["theta_ln_K_l_e_wt_loc"] == mwc.DEFAULT_LN_K_L_E
    for hp in (c4, c12):
        assert hp["theta_tf_total_M"] == mwc.DEFAULT_TF_TOTAL_M == 1e-5
        assert "theta_op_total_M" not in hp


# ---------------------------------------------------------------------------
# Projection rows
# ---------------------------------------------------------------------------

def _structure_ddG_from_dlnK(names, d):
    """
    Structure free-energy shifts that produce the constant shifts ``d``.

    ΔΔG_X adds to state X's free energy and Δln K_{A→B} = ΔΔG_A − ΔΔG_B.
    Take ΔΔG_H = a (arbitrary); then ΔΔG_L = a − Δln K_h_l,
    ΔΔG_HO = a − Δln K_h_o, ΔΔG_LO = ΔΔG_L − Δln K_l_o and, with two
    per-site constants, ΔΔG_XE2 = ΔΔG_X − 2·Δln K_x_e.
    """
    a = 0.7
    G = {"H": a, "L": a - d["h_l"]}
    G["HO"] = G["HD"] = a - d["h_o"]
    G["LO"] = G["L"] - d.get("l_o", 0.0)
    G["HE2"] = a - 2 * d.get("h_e", 0.0)
    G["LE2"] = G["L"] - 2 * d["l_e"]
    return np.array([G[n] for n in names])


def test_c12_projection_recovers_per_site_constants():
    rng = np.random.default_rng(4)
    for _ in range(20):
        d = dict(zip(["h_l", "h_o", "h_e", "l_o", "l_e"], rng.normal(0, 2, 5)))
        ddG = _structure_ddG_from_dlnK(mwc.STRUCTURE_NAMES_C12, d)
        got = mwc.PROJ_C12 @ ddG
        np.testing.assert_allclose(
            got, [d["h_l"], d["h_o"], d["h_e"], d["l_o"], d["l_e"]], atol=1e-12)
        assert mwc.PROJ_U_C12 @ ddG == pytest.approx(ddG[0])   # Δln K_u = ΔΔG_H


def test_c4_projection_recovers_per_site_constants():
    rng = np.random.default_rng(5)
    for _ in range(20):
        d = dict(zip(["h_o", "h_l", "l_e"], rng.normal(0, 2, 3)))
        ddG = _structure_ddG_from_dlnK(mwc.STRUCTURE_NAMES_C4, d)
        got = mwc.PROJ_C4 @ ddG        # (K_op, K_HL, K_E)
        np.testing.assert_allclose(got, [d["h_o"], d["h_l"], d["l_e"]],
                                   atol=1e-12)
        assert mwc.PROJ_U_C4 @ ddG == pytest.approx(ddG[0])


def test_c4_rows_are_the_c12_rows_of_the_same_constants():
    """K_op, K_HL, K_E rows equal C12's K_h_o, K_h_l, K_l_e rows (HD ≡ HO)."""
    c12 = dict(zip(mwc.STRUCTURE_NAMES_C12, mwc.PROJ_C12.T))
    as_c12 = np.stack([c12[{"HD": "HO"}.get(n, n)] for n in
                       mwc.STRUCTURE_NAMES_C4], axis=1)   # (5, 4)
    np.testing.assert_array_equal(mwc.PROJ_C4[0], as_c12[1])   # K_op = K_h_o
    np.testing.assert_array_equal(mwc.PROJ_C4[1], as_c12[0])   # K_HL = K_h_l
    np.testing.assert_array_equal(mwc.PROJ_C4[2], as_c12[4])   # K_E = K_l_e


def test_projected_shift_reproduces_the_doubly_liganded_weight():
    """
    A ΔΔG on LE2 alone, projected, changes θ exactly as raising LE2's free
    energy in the brute-force species sum would: LE2's weight K_l_e²·e² is
    multiplied by exp(−ΔΔG_LE2), and LE's by exp(−ΔΔG_LE2/2) per site.
    """
    ddG_LE2 = 1.3
    ddG = np.zeros(4)
    ddG[3] = ddG_LE2
    d = mwc.PROJ_C4 @ ddG
    assert d[2] == pytest.approx(-ddG_LE2 / 2)
    K_l_e = 2.5e5
    for e in (1e-6, 8e-5, 1e-3):
        got = mwc.theta_c4(mwc.DEFAULT_LN_K_H_O, mwc.DEFAULT_LN_K_H_L,
                           math.log(K_l_e) + d[2], None, np.array(e), _TF,
                           xp=np)
        want, _ = _brute_force(6.3, 4.2e8, 0.0, 0.0,
                               K_l_e * math.exp(-ddG_LE2 / 2), 0.0, e, _TF)
        assert float(got) == pytest.approx(want, rel=1e-10)


@pytest.mark.parametrize("family, proj", [("O2_C4_K3_U0_a", mwc.PROJ_C4),
                                          ("O2_C4_K3_U1_a", mwc.PROJ_C4),
                                          ("O2_C12_K5_U0_a", mwc.PROJ_C12),
                                          ("O2_C12_K5_U1_a", mwc.PROJ_C12)])
@pytest.mark.parametrize("kind", ["PddG", "PnnC"])
def test_components_use_the_shared_projection(family, proj, kind):
    module = _module(family, kind)
    np.testing.assert_array_equal(np.asarray(module._PROJ), proj)
    if family.endswith("U1_a"):
        u = mwc.PROJ_U_C4 if "C4" in family else mwc.PROJ_U_C12
        np.testing.assert_array_equal(np.asarray(module._PROJ_U), u)
