"""
Standard MWC dimer equations shared by every O2_C4 and O2_C12 theta model.

Both families are one physical model, the Monod-Wyman-Changeux (MWC)
two-conformation lac repressor dimer. The dimer is in a high operator
affinity conformation H or a low affinity conformation L, optionally with a
globally unfolded state U. Each dimer has two identical, independent
effector sites and one operator site. The constants are

    K_h_l  [L]/[H], dimensionless
    K_h_o  H + O → HO, M⁻¹
    K_l_o  L + O → LO, M⁻¹
    K_h_e  per-site (microscopic) effector association constant of H, M⁻¹
    K_l_e  per-site (microscopic) effector association constant of L, M⁻¹
    K_u    [U]/[H], dimensionless (U1 models only; 0 in U0 models)

The O2_C12 models (M3) fit all five. The O2_C4 models (M2) are the exact
restriction K_h_e = 0 (H does not bind effector) and K_l_o = 0 (L does not
bind operator). Their three constants are K_op = K_h_o, K_HL = K_h_l and
K_E = K_l_e, and they call :func:`mwc_theta` with the two restricted constants
fixed at 0, so M2 is M3 with those constants removed by construction.

Effector polynomial
-------------------
With two independent, equivalent sites the effector binding polynomial of a
conformation is

    P_H(e) = (1 + K_h_e·e)² = 1 + 2·K_h_e·e + K_h_e²·e²
    P_L(e) = (1 + K_l_e·e)² = 1 + 2·K_l_e·e + K_l_e²·e²

The 2 counts the two ways to fill one site. Within a conformation the two
sites are independent, so the macroscopic constants are K1 = 2·K and
K2 = K/2, with K2/K1 = 1/4. Cooperativity comes only from the conformational
equilibrium: effector pulls H toward L when K_l_e > K_h_e and K_h_l > 1.

Before 2026-10-11 these models used Sochor's (2014, PeerJ 2:e498) polynomial
P = 1 + K·e + K²·e², which has no factor of 2 on the singly bound term. That
gives K1 = K and K2 = K, so K2/K1 = 1, a fixed fourfold excess over the
independent-site value 1/4. It builds a within-conformation cooperativity
into every genotype that no constant can turn off. The O2_C4 models used a
third form, L + 2E → LE2 in one step with an M⁻² constant and no singly bound
state. Both forms are replaced by the standard one here.

Bookkeeping
-----------
TF is in large excess over operator (tf_total = 1e-5 M monomer against about
2.5e-8 M operator, about 400-fold), so operator-bound TF is left out of the
TF mass balance and operator total does not enter the model. With
r = tf_total/2 the dimer concentration (tf_total is in MONOMER units) and
h the free apo H dimer, the TF mass balance over the states that are not
bound to operator is

    r = h·F(e),    F(e) = P_H(e) + K_h_l·P_L(e) + K_u = B0 + B1·e + B2·e²

    B0 = 1 + K_h_l + K_u
    B1 = 2·(K_h_e + K_h_l·K_l_e)
    B2 = K_h_e² + K_h_l·K_l_e²

Operator occupancy is the Langmuir isotherm of the operator for the TF
species that bind it (operator binding does not change effector binding):

    W = h·(K_h_o·P_H(e) + K_h_l·K_l_o·P_L(e))
    θ = W / (1 + W)

With K_h_e = K_l_o = 0 this is W = h·K_op, h = r / (1 + K_HL·(1 + K_E·e)² + K_U).

Free effector
-------------
Bound effector per dimer is e·dF/de / F, summed over the weighted states
(e·dP/de counts the occupied sites of each species). The effector mass
balance e_total = e + r·e·F'(e)/F(e) with e·F'(e) = B1·e + 2·B2·e² gives,
after multiplying through by F(e),

    (e_total − e)·(B0 + B1·e + B2·e²) = r·(B1·e + 2·B2·e²)

    B2·e³ + (B1 + B2·(2r − e_total))·e² + (B0 + B1·(r − e_total))·e
          − B0·e_total = 0

The cubic is negative at e = 0 and positive at e = e_total. Newton steps
start at e_total and the result is clipped to [0, e_total].

Projection of structural ΔΔG
-----------------------------
ΔΔG_X is the shift a mutation makes to the free energy of state X (kT,
positive destabilizes X). For K_{A→B} = [B]/[A], Δln K_{A→B} = ΔΔG_A − ΔΔG_B.
A doubly liganded structure XE2 has weight K_x_e²·e² relative to X, so
ΔΔG_X − ΔΔG_XE2 = 2·Δln K_x_e and the per-site row carries 1/2. Mutations
do not shift U (ΔΔG_U = 0), so Δln K_u = ΔΔG_H. The projection rows for each
family are :data:`PROJ_C4`, :data:`PROJ_U_C4`, :data:`PROJ_C12` and
:data:`PROJ_U_C12`.

Units
-----
Data concentrations (``titrant_conc``) are in mM. The components multiply
them by ``theta_conc_unit_scale`` (1e-3) so that every concentration here is
in M and every effector and operator constant in M⁻¹.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

# Newton iterations for the free-effector cubic. From e_total, 8 steps left
# relative errors up to 4e-3 in free effector over broad random constants,
# TF and effector concentrations; 16 steps reach float64 precision against a
# bracketing root finder (test_mwc.py).
NEWTON_ITERATIONS = 16

# Data concentrations are mM; this converts them to M.
DEFAULT_CONC_UNIT_SCALE = 1e-3

# Total TF, MONOMER units (M); r = tf_total/2 = 5e-6 M dimer. The PI's
# current in vivo estimate.
DEFAULT_TF_TOTAL_M = 1e-5

# Sochor 2014 (PeerJ 2:e498) Table 1, in M.
SOCHOR_K_H_L = 6.3
SOCHOR_K_H_O = 4.2e8
SOCHOR_K_L_O = 0.1
SOCHOR_K_H_E = 5.6e4
SOCHOR_K_L_E = 7.6e5

# Calibrated wt defaults: the default wt theta(IPTG) curve has its midpoint
# (theta halfway between theta(0) and the e → ∞ limit, theta_saturated, with
# effector depletion at DEFAULT_TF_TOTAL_M) at 80 µM in both families.
# O2_C12 keeps Sochor's K_h_l, K_h_o, K_l_o and K_l_e/K_h_e and scales K_h_e
# and K_l_e by one common factor, MWC_EFFECTOR_SCALE. O2_C4 uses the same
# K_h_l and K_h_o with K_h_e = K_l_o = 0 and calibrates K_l_e alone.
# Regenerate with test_mwc.py::_calibrate_* if the equations change.
CALIBRATION_MIDPOINT_M = 80e-6
MWC_EFFECTOR_SCALE = 0.3794956281766634
DEFAULT_LN_K_H_L = math.log(SOCHOR_K_H_L)
DEFAULT_LN_K_H_O = math.log(SOCHOR_K_H_O)
DEFAULT_LN_K_L_O = math.log(SOCHOR_K_L_O)
DEFAULT_LN_K_H_E = math.log(SOCHOR_K_H_E * MWC_EFFECTOR_SCALE)
DEFAULT_LN_K_L_E = math.log(SOCHOR_K_L_E * MWC_EFFECTOR_SCALE)
DEFAULT_LN_K_E_C4 = 12.410839287464412

# Structure bases and projection rows (see module docstring).
STRUCTURE_NAMES_C4 = ('H', 'HD', 'L', 'LE2')
# rows = (K_op, K_HL, K_E), cols = (H, HD, L, LE2)
PROJ_C4 = np.array([[1.0, -1.0, 0.0,  0.0],    # Δln K_op = ΔΔG_H − ΔΔG_HD
                    [1.0,  0.0, -1.0, 0.0],    # Δln K_HL = ΔΔG_H − ΔΔG_L
                    [0.0,  0.0, 0.5, -0.5]])   # Δln K_E  = (ΔΔG_L − ΔΔG_LE2)/2
PROJ_U_C4 = np.array([1.0, 0.0, 0.0, 0.0])     # Δln K_U = ΔΔG_H

STRUCTURE_NAMES_C12 = ('H', 'HO', 'L', 'LO', 'HE2', 'LE2')
# rows = (K_h_l, K_h_o, K_h_e, K_l_o, K_l_e), cols = (H, HO, L, LO, HE2, LE2)
PROJ_C12 = np.array([
    [1.0,  0.0, -1.0,  0.0,  0.0,  0.0],   # Δln K_h_l = ΔΔG_H − ΔΔG_L
    [1.0, -1.0,  0.0,  0.0,  0.0,  0.0],   # Δln K_h_o = ΔΔG_H − ΔΔG_HO
    [0.5,  0.0,  0.0,  0.0, -0.5,  0.0],   # Δln K_h_e = (ΔΔG_H − ΔΔG_HE2)/2
    [0.0,  0.0,  1.0, -1.0,  0.0,  0.0],   # Δln K_l_o = ΔΔG_L − ΔΔG_LO
    [0.0,  0.0,  0.5,  0.0,  0.0, -0.5],   # Δln K_l_e = (ΔΔG_L − ΔΔG_LE2)/2
])
PROJ_U_C12 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # Δln K_u = ΔΔG_H


def _poly_coeffs(K_h_l, K_h_e, K_l_e, K_u):
    """F(e) = B0 + B1·e + B2·e² over the states not bound to operator."""
    B0 = 1.0 + K_h_l + K_u
    B1 = 2.0 * (K_h_e + K_h_l * K_l_e)
    B2 = K_h_e ** 2 + K_h_l * K_l_e ** 2
    return B0, B1, B2


def free_effector(B0, B1, B2, e_total, r, xp=jnp):
    """
    Free effector from the cubic mass balance (see module docstring).

    Parameters
    ----------
    B0, B1, B2 : array-like
        Coefficients of F(e), broadcastable against ``e_total``.
    e_total : array-like
        Total effector (M).
    r : float
        Total TF dimer (M), tf_total/2.
    xp : module
        ``jax.numpy`` (default, Newton steps in a ``fori_loop``) or ``numpy``.

    Returns
    -------
    array
        Free effector, clipped to [0, e_total], in the broadcast shape.
    """
    c3 = B2
    c2 = B1 + B2 * (2.0 * r - e_total)
    c1 = B0 + B1 * (r - e_total)
    c0 = -B0 * e_total

    def step(_, x):
        f = ((c3 * x + c2) * x + c1) * x + c0
        df = (3.0 * c3 * x + 2.0 * c2) * x + c1
        return x - f / xp.where(xp.abs(df) < 1e-30, 1e-30, df)

    shape = xp.broadcast_shapes(xp.shape(c0), xp.shape(c2), xp.shape(c3))
    x = xp.broadcast_to(e_total, shape) * xp.ones(shape)
    if xp is jnp:
        x = jax.lax.fori_loop(0, NEWTON_ITERATIONS, step, x)
    else:
        for i in range(NEWTON_ITERATIONS):
            x = step(i, x)
    return xp.clip(x, 0.0, e_total)


def mwc_theta(K_h_l, K_h_o, K_h_e, K_l_o, K_l_e, K_u, e_total, tf_total,
              xp=jnp):
    """
    Operator occupancy of the standard MWC dimer (see module docstring).

    Every argument broadcasts against the others. Pass 0 for a constant the
    model does not have (K_h_e and K_l_o in O2_C4, K_u in U0 models).

    Parameters
    ----------
    K_h_l, K_h_o, K_h_e, K_l_o, K_l_e, K_u : array-like
        Equilibrium constants (linear, not log); effector constants per site.
    e_total : array-like
        Total effector (M).
    tf_total : float
        Total TF in MONOMER units (M).
    xp : module
        ``jax.numpy`` (default) or ``numpy``.

    Returns
    -------
    array
        θ in the broadcast shape.
    """
    r = tf_total / 2.0
    B0, B1, B2 = _poly_coeffs(K_h_l, K_h_e, K_l_e, K_u)
    e = free_effector(B0, B1, B2, e_total, r, xp=xp)

    P_H = (1.0 + K_h_e * e) ** 2
    P_L = (1.0 + K_l_e * e) ** 2
    F = B0 + B1 * e + B2 * e ** 2
    h = r / xp.where(F < 1e-30, 1e-30, F)
    W = h * (K_h_o * P_H + K_h_l * K_l_o * P_L)
    return W / (1.0 + W)


def theta_saturated(K_h_l, K_h_o, K_h_e, K_l_o, K_l_e, K_u, tf_total,
                    xp=jnp):
    """
    θ in the limit of saturating effector (e → ∞, so depletion vanishes).

    F(e), P_H and P_L are dominated by their e² terms, so
    W → r·(K_h_o·K_h_e² + K_h_l·K_l_o·K_l_e²) / (K_h_e² + K_h_l·K_l_e²).
    It is 0 for M2 (K_h_e = K_l_o = 0).
    """
    r = tf_total / 2.0
    B2 = K_h_e ** 2 + K_h_l * K_l_e ** 2
    W = r * (K_h_o * K_h_e ** 2 + K_h_l * K_l_o * K_l_e ** 2) / B2
    return W / (1.0 + W)


def theta_c12(ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e, ln_K_u,
              e_total, tf_total, xp=jnp):
    """M3 (O2_C12) θ from log constants; ``ln_K_u=None`` for U0."""
    K_u = 0.0 if ln_K_u is None else xp.exp(ln_K_u)
    return mwc_theta(xp.exp(ln_K_h_l), xp.exp(ln_K_h_o), xp.exp(ln_K_h_e),
                     xp.exp(ln_K_l_o), xp.exp(ln_K_l_e), K_u,
                     e_total, tf_total, xp=xp)


def theta_c4(ln_K_op, ln_K_HL, ln_K_E, ln_K_U, e_total, tf_total, xp=jnp):
    """
    M2 (O2_C4) θ from log constants; ``ln_K_U=None`` for U0.

    M3 with K_h_o = K_op, K_h_l = K_HL, K_l_e = K_E (per site, M⁻¹) and
    K_h_e = K_l_o = 0.
    """
    K_U = 0.0 if ln_K_U is None else xp.exp(ln_K_U)
    return mwc_theta(xp.exp(ln_K_HL), xp.exp(ln_K_op), 0.0, 0.0,
                     xp.exp(ln_K_E), K_U, e_total, tf_total, xp=xp)
