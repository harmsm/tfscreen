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

The cubic is negative at e = 0 and positive at e = e_total.

Log space
---------
Every quantity is computed from the log constants (``theta_c4``,
``theta_c12``, ``log_W``): partition-function sums are log-sum-exps,
log P = 2·softplus(ln K + log e), and θ = sigmoid(log W). No constant is
exponentiated, so any finite ln K gives a finite θ in [0, 1] with the right
limits, in float32. Before 2026-10-11 θ was computed from exp(ln K): Laplace
draws of the PddG pilot fits with ln K beyond ~44 overflowed K² in float32
and wrote NaN θ for whole genotypes. The free effector is solved for
x = log e by safeguarded Newton (``_solve_log_free_effector``) on the mass
balance written so that it stays well conditioned on both sides of the
equivalence point e_total = 2r (``_mass_balance``). Its derivatives come
from the implicit function theorem (a custom JVP), not from the iterations,
so gradients and Hessians are exact and finite. ``mwc_theta`` and
``free_effector`` keep their linear-constant signatures and log their
inputs.

Projection of structural ΔΔG
-----------------------------
ΔΔG_X is the shift a mutation makes to the free energy of state X (kT,
positive destabilizes X). For K_{A→B} = [B]/[A], Δln K_{A→B} = ΔΔG_A − ΔΔG_B.
A doubly liganded structure XE2 has weight K_x_e²·e² relative to X, so
ΔΔG_X − ΔΔG_XE2 = 2·Δln K_x_e and the per-site row carries 1/2. Mutations
do not shift U (ΔΔG_U = 0), so Δln K_u = ΔΔG_H. The projection rows for each
family are :data:`PROJ_C4`, :data:`PROJ_U_C4`, :data:`PROJ_C12` and
:data:`PROJ_U_C12`.

Default wild type
-----------------
The default wt constants put strong repression and near-complete induction
at the PI's TF estimate, tf_total = 1e-5 M monomer (r = 5e-6 M dimer). The
O2_C12 defaults keep Sochor's K_h_l = 6.3, K_h_o = 4.2e8 M⁻¹ and
K_l_o = 0.1 M⁻¹ and set the effector selectivity ρ = K_l_e/K_h_e from the
saturating limit. As e → ∞ only the doubly liganded HE2 and LE2 remain, so

    W_sat = r·(K_h_o + K_h_l·K_l_o·ρ²) / (1 + K_h_l·ρ²)

and θ_sat = W_sat/(1 + W_sat) depends on ρ alone. Asking for θ_sat = 0.05
(W_sat = 1/19) and solving for ρ gives

    ρ² = (r·K_h_o − W_sat) / (K_h_l·(W_sat − r·K_l_o))

so ρ ≈ 79.6 (:data:`MWC_EFFECTOR_SELECTIVITY`). This departs from Sochor's
fitted ρ = 7.6e5/5.6e4 ≈ 13.6. With that ρ the HE2 state still repressed:
W_sat = r·K_h_o/(1 + K_h_l·ρ²) ≈ 1.8 and θ_sat ≈ 0.64 at 1e-5 M TF, which is
implausible for lac. Sochor fitted ρ at a far lower TF level, where the
residual HE2 repression did not show.

The two effector constants then share one scale. K_h_e and K_l_e start at
(SOCHOR_K_L_E/ρ, SOCHOR_K_L_E) and are multiplied by a common factor,
:data:`MWC_EFFECTOR_SCALE` ≈ 0.393, chosen so that the midpoint is 80 µM
IPTG. The midpoint is the total effector at which θ is halfway between θ(0)
and θ_sat, with effector depletion at 1e-5 M TF. That gives
K_h_e ≈ 3.7e3 M⁻¹ and K_l_e ≈ 3.0e5 M⁻¹ per site. Scaling both leaves ρ and
so θ_sat unchanged.

The O2_C4 defaults use the same K_h_l and K_h_o with K_h_e = K_l_o = 0, so
θ_sat = 0, and calibrate K_E = K_l_e alone to the same 80 µM midpoint
(:data:`DEFAULT_LN_K_E_C4`).

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

# Iterations of the safeguarded Newton solve for log free effector
# (_solve_log_free_effector). Over 20,000 draws each of broad constants
# (ln K SD 2 about the defaults), wide ones (SD 15) and extreme ones (every
# ln K uniform on [-150, 150]), with total effector 1e-12 to 1 M (30% of them
# within 0.1% of the equivalence point 2r) and TF 1e-12 to 1e-3 M, 12
# iterations matched a 400-iteration solve exactly in float64, and float32
# matched it to 7e-5 in log e. 16 leaves a margin. The old linear-space
# Newton solve also took 16 steps, but overflowed in float32 and needed more
# than 30 at large K_E.
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

# Calibrated wt defaults (see the module docstring). O2_C12 keeps Sochor's
# K_h_l, K_h_o and K_l_o, widens the effector selectivity rho = K_l_e/K_h_e so
# that theta_saturated = CALIBRATION_THETA_SATURATED at DEFAULT_TF_TOTAL_M, and
# scales K_h_e and K_l_e by one common factor, MWC_EFFECTOR_SCALE, so that the
# midpoint (theta halfway between theta(0) and theta_saturated, with effector
# depletion at DEFAULT_TF_TOTAL_M) is at 80 µM. O2_C4 uses the same K_h_l and
# K_h_o with K_h_e = K_l_o = 0 and calibrates K_l_e alone to the same
# midpoint. Regenerate with test_mwc.py::_calibrate_* if the equations change.
CALIBRATION_MIDPOINT_M = 80e-6
CALIBRATION_THETA_SATURATED = 0.05


def effector_selectivity_for_theta_saturated(theta_sat, K_h_l, K_h_o, K_l_o,
                                             tf_total):
    """
    ρ = K_l_e/K_h_e that gives θ_sat at saturating effector.

    Solves W_sat = r·(K_h_o + K_h_l·K_l_o·ρ²)/(1 + K_h_l·ρ²) for ρ, with
    W_sat = θ_sat/(1 − θ_sat) and r = tf_total/2 (see module docstring).
    Requires r·K_l_o < W_sat < r·K_h_o.
    """
    r = tf_total / 2.0
    W = theta_sat / (1.0 - theta_sat)
    return math.sqrt((r * K_h_o - W) / (K_h_l * (W - r * K_l_o)))


MWC_EFFECTOR_SELECTIVITY = effector_selectivity_for_theta_saturated(
    CALIBRATION_THETA_SATURATED, SOCHOR_K_H_L, SOCHOR_K_H_O, SOCHOR_K_L_O,
    DEFAULT_TF_TOTAL_M)                                # ≈ 79.6
MWC_EFFECTOR_SCALE = 0.39263690389004985
DEFAULT_LN_K_H_L = math.log(SOCHOR_K_H_L)
DEFAULT_LN_K_H_O = math.log(SOCHOR_K_H_O)
DEFAULT_LN_K_L_O = math.log(SOCHOR_K_L_O)
DEFAULT_LN_K_L_E = math.log(SOCHOR_K_L_E * MWC_EFFECTOR_SCALE)   # ≈ 3.0e5 M⁻¹
DEFAULT_LN_K_H_E = DEFAULT_LN_K_L_E - math.log(MWC_EFFECTOR_SELECTIVITY)  # ≈ 3.7e3
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


_LOG2 = math.log(2.0)
_LOG4 = math.log(4.0)
_LOG8 = math.log(8.0)


def _lae3(a, b, c, xp):
    """log(exp(a) + exp(b) + exp(c))."""
    return xp.logaddexp(xp.logaddexp(a, b), c)


def _log(x, xp):
    """log, with log(0) = -inf and no numpy warning."""
    if xp is np:
        with np.errstate(divide="ignore"):
            return np.log(x)
    return xp.log(x)


def _poly_coeffs(K_h_l, K_h_e, K_l_e, K_u):
    """F(e) = B0 + B1·e + B2·e² over the states not bound to operator."""
    B0 = 1.0 + K_h_l + K_u
    B1 = 2.0 * (K_h_e + K_h_l * K_l_e)
    B2 = K_h_e ** 2 + K_h_l * K_l_e ** 2
    return B0, B1, B2


def _log_poly_coeffs(ln_K_h_l, ln_K_h_e, ln_K_l_e, ln_K_u, xp=jnp):
    """log B0, log B1, log B2 of F(e) from log constants (-inf for a 0)."""
    lB0 = _lae3(0.0, ln_K_h_l, ln_K_u, xp)
    lB1 = _LOG2 + xp.logaddexp(ln_K_h_e, ln_K_h_l + ln_K_l_e)
    lB2 = xp.logaddexp(2.0 * ln_K_h_e, ln_K_h_l + 2.0 * ln_K_l_e)
    return lB0, lB1, lB2


def _effector_moments(lB0, lB1, lB2, x, xp):
    """
    Moments of n ∈ {0, 1, 2}, the number of effector sites filled on a dimer
    not bound to operator, at x = log e. The weights are B0, B1·e and B2·e²
    (n̄ = e·F'/F, and d n̄/d log e = Var(n)).

    Returns log F(e), log n̄, log Var(n) and log m with m = 2 − n̄ (the empty
    sites per dimer), each from log weights only.
    """
    t1 = lB1 + x
    t2 = lB2 + 2.0 * x
    lF = _lae3(lB0, t1, t2, xp)
    lw0, lw1, lw2 = lB0 - lF, t1 - lF, t2 - lF
    log_nbar = xp.logaddexp(lw1, _LOG2 + lw2)
    log_m = xp.logaddexp(_LOG2 + lw0, lw1)
    # Var(n) = Σ_{i<j} w_i w_j (i − j)², a sum of positive terms
    log_var = _lae3(lw0 + lw1, _LOG4 + lw0 + lw2, lw1 + lw2, xp)
    return lF, log_nbar, log_var, log_m


def _log_excess(e_total, r, xp):
    """
    The pieces of the mass balance that depend only on the totals.

    Returns log(max(c, 0)), log(max(−c, 0)) with c = e_total − 2r (the
    effector left over once every TF site is filled; ``-inf`` for a zero),
    and ``free_form``, 1.0 where e_total ≤ r and 0.0 elsewhere (which
    residual :func:`_mass_balance` uses). c comes from the linear inputs, so
    e_total at the equivalence point 2r gives exactly 0.
    """
    c = e_total - 2.0 * r

    def log_pos(v):
        ok = v > 0
        return xp.where(ok, _log(xp.where(ok, v, 1.0), xp), -np.inf)

    free_form = xp.where(e_total <= r, 1.0, 0.0)
    return log_pos(c), log_pos(-c), free_form


def _mass_balance(x, lB0, lB1, lB2, lt, lr, lc_pos, lc_neg, free_form, xp):
    """
    The effector mass balance e_total = e + r·n̄(e) as a residual φ(x) at
    x = log e, and dφ/dx. Both forms rise monotonically through the same
    root, the free effector.

    - e_total ≤ r (TF sites well in excess, so n̄ ≤ 1 at the root):
      φ = log(e + r·n̄) − log e_total, slope (e + r·Var)/(e + r·n̄).
    - e_total > r: φ = log(e + c⁻) − log(c⁺ + r·m), m = 2 − n̄ the empty
      sites per dimer, c = e_total − 2r, c⁺ = max(c, 0), c⁻ = max(−c, 0).
      This is e = e_total − r·n̄ with e_total − r·n̄ = c + r·m written
      without a difference of large terms. Slope
      e/(e + c⁻) + r·Var/(c⁺ + r·m). The first form fails here: near
      saturation e is a vanishing part of e + r·n̄, so φ is flat in x to
      float precision and the root is lost (float32 put it 0.05-50 in
      log e off at e_total ≈ 2r).
    """
    _, log_nbar, log_var, log_m = _effector_moments(lB0, lB1, lB2, x, xp)

    log_bal = xp.logaddexp(x, lr + log_nbar)
    phi_a = log_bal - lt
    dphi_a = xp.exp(xp.logaddexp(x, lr + log_var) - log_bal)

    log_left = xp.logaddexp(x, lc_neg)               # log(e + c⁻)
    log_right = xp.logaddexp(lc_pos, lr + log_m)     # log(c⁺ + r·m)
    phi_b = log_left - log_right
    dphi_b = (xp.exp(x - log_left)
              + xp.exp(lr + log_var - log_right))

    use_a = free_form > 0
    return (xp.where(use_a, phi_a, phi_b),
            xp.where(use_a, dphi_a, dphi_b))


def _solve_log_free_effector(lB0, lB1, lB2, lt, lr, lc_pos, lc_neg, free_form,
                             xp):
    """
    Safeguarded Newton solve of φ(x) = 0 (see :func:`_mass_balance`).

    The root is bracketed by [x_lo, log e_total]: φ(log e_total) ≥ 0, and
    at e_lo = min(e_t/2, e_t·B0/(8r·B1), sqrt(e_t·B0/(8r·B2))) the bound
    n̄ ≤ 2·(B1·e + B2·e²)/B0 gives e + r·n̄ ≤ e_total. Newton steps start at
    log e_total (as the old linear solve did). A step that leaves the
    current bracket, or does not at least halve the previous step, is
    replaced by bisection (Numerical Recipes' rtsafe), so the bracket
    shrinks at least as fast as bisection.
    """
    L = lt - _LOG8 - lr + lB0
    x_lo = xp.minimum(xp.minimum(lt - _LOG2, L - lB1), 0.5 * (L - lB2))
    lo = x_lo
    hi = lt
    x = hi
    dx_old = 2.0 * (hi - lo)

    def step(_, carry):
        x, lo, hi, dx_old = carry
        phi, dphi = _mass_balance(x, lB0, lB1, lB2, lt, lr, lc_pos, lc_neg,
                                  free_form, xp)
        lo = xp.where(phi < 0, x, lo)
        hi = xp.where(phi > 0, x, hi)
        dx = phi / dphi
        x_newton = x - dx
        ok = ((x_newton >= lo) & (x_newton <= hi)
              & (xp.abs(dx) <= 0.5 * xp.abs(dx_old)))
        x_bisect = 0.5 * (lo + hi)
        x_next = xp.where(ok, x_newton, x_bisect)
        dx_next = xp.where(ok, dx, x - x_bisect)
        # Converged to float precision: stay. A Newton step there is
        # rounding noise and need not halve, and bisecting would leave the
        # root. φ is a difference of logs of size about |log e_total|.
        done = ((xp.abs(phi) <= 4.0 * tol * xp.maximum(1.0, xp.abs(lt)))
                | (xp.abs(dx) <= tol * xp.maximum(1.0, xp.abs(x))))
        return (xp.where(done, x, x_next), lo, hi,
                xp.where(done, 0.0, dx_next))

    tol = 4.0 * float(xp.finfo(xp.result_type(x)).eps)
    carry = (x, lo, hi, dx_old)
    if xp is jnp:
        carry = jax.lax.fori_loop(0, NEWTON_ITERATIONS, step, carry)
    else:
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            for i in range(NEWTON_ITERATIONS):
                carry = step(i, carry)
    return carry[0]


@jax.custom_jvp
def _log_free_effector_jax(lB0, lB1, lB2, lt, lr, lc_pos, lc_neg, free_form):
    return _solve_log_free_effector(lB0, lB1, lB2, lt, lr, lc_pos, lc_neg,
                                    free_form, jnp)


@_log_free_effector_jax.defjvp
def _log_free_effector_jvp(primals, tangents):
    """
    Implicit derivative of the root, dx = −(∂φ/∂p·dp)/(∂φ/∂x), in place of
    differentiating through the Newton iterations. It is written in JAX ops,
    so it differentiates again (the Laplace Hessian).
    """
    x = _log_free_effector_jax(*primals)

    def phi(x_, *p):
        return _mass_balance(x_, *p, jnp)[0]

    _, dphi_p = jax.jvp(lambda *p: phi(x, *p), primals, tangents)
    _, dphi_dx = jax.jvp(lambda x_: phi(x_, *primals), (x,),
                         (jnp.ones_like(x),))
    return x, -dphi_p / dphi_dx


def log_free_effector(lB0, lB1, lB2, e_total, r, xp=jnp):
    """
    log of the free effector from the mass balance, in log space.

    Parameters
    ----------
    lB0, lB1, lB2 : array-like
        log of the coefficients of F(e) (``_log_poly_coeffs``).
    e_total : array-like
        Total effector (M), ≥ 0.
    r : float or array-like
        Total TF dimer (M), tf_total/2.
    xp : module
        ``jax.numpy`` (default) or ``numpy``.

    Returns
    -------
    x : array
        log free effector in the broadcast shape; ``-inf`` where
        ``e_total`` is 0.
    pos : array of bool
        ``e_total > 0``.
    """
    e_total = xp.asarray(e_total)
    r = xp.asarray(r)
    pos = e_total > 0
    lt = _log(xp.where(pos, e_total, 1.0), xp)
    lr = _log(r, xp)
    lc_pos, lc_neg, free_form = _log_excess(xp.where(pos, e_total, 1.0), r,
                                            xp)
    args = xp.broadcast_arrays(*[xp.asarray(v, dtype=float) for v in
                                 (lB0, lB1, lB2, lt, lr, lc_pos, lc_neg,
                                  free_form)])
    if xp is jnp:
        x = _log_free_effector_jax(*args)
    else:
        x = _solve_log_free_effector(*args, np)
    return xp.where(pos, x, -np.inf), pos


def free_effector(B0, B1, B2, e_total, r, xp=jnp):
    """
    Free effector from the cubic mass balance (see module docstring).

    Solved in log space by :func:`log_free_effector`, so the coefficients
    can span any range.

    Parameters
    ----------
    B0, B1, B2 : array-like
        Coefficients of F(e), broadcastable against ``e_total``.
    e_total : array-like
        Total effector (M).
    r : float
        Total TF dimer (M), tf_total/2.
    xp : module
        ``jax.numpy`` (default) or ``numpy``.

    Returns
    -------
    array
        Free effector, in [0, e_total], in the broadcast shape.
    """
    x, pos = log_free_effector(_log(B0, xp), _log(B1, xp), _log(B2, xp),
                               e_total, r, xp=xp)
    return xp.where(pos, xp.exp(xp.where(pos, x, 0.0)), 0.0)


def log_W(ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e, ln_K_u, e_total,
          tf_total, xp=jnp):
    """
    log W, the operator's log odds of being bound (θ = sigmoid(log W)),
    computed in log space from log constants.

    ``-inf`` stands for a constant the model does not have (K_h_e and K_l_o
    in O2_C4, K_u in U0 models). Every sum of partition-function terms is a
    log-sum-exp, so no constant is ever exponentiated and any finite log
    constant gives a finite log W.
    """
    lr = _log(tf_total / 2.0, xp)
    lB0, lB1, lB2 = _log_poly_coeffs(ln_K_h_l, ln_K_h_e, ln_K_l_e, ln_K_u,
                                     xp=xp)
    x, pos = log_free_effector(lB0, lB1, lB2, e_total, tf_total / 2.0, xp=xp)
    x = xp.where(pos, x, 0.0)

    # e > 0: W = r·(K_h_o·P_H + K_h_l·K_l_o·P_L)/F, log P = 2·softplus(ln K + x)
    lP_H = 2.0 * xp.logaddexp(0.0, ln_K_h_e + x)
    lP_L = 2.0 * xp.logaddexp(0.0, ln_K_l_e + x)
    lF = _lae3(lB0, lB1 + x, lB2 + 2.0 * x, xp)
    lW_e = lr + xp.logaddexp(ln_K_h_o + lP_H, ln_K_h_l + ln_K_l_o + lP_L) - lF

    # e = 0: P_H = P_L = 1, F = B0
    lW_0 = lr + xp.logaddexp(ln_K_h_o, ln_K_h_l + ln_K_l_o) - lB0
    return xp.where(pos, lW_e, lW_0)


def _sigmoid(z, xp):
    """1/(1 + e^-z) without overflow; exactly 0 or 1 only past float range."""
    return xp.exp(-xp.logaddexp(0.0, -z))


def mwc_theta_log(ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e, ln_K_u,
                  e_total, tf_total, xp=jnp):
    """θ = sigmoid(log W) from log constants (see :func:`log_W`)."""
    return _sigmoid(log_W(ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e,
                          ln_K_u, e_total, tf_total, xp=xp), xp)


def mwc_theta(K_h_l, K_h_o, K_h_e, K_l_o, K_l_e, K_u, e_total, tf_total,
              xp=jnp):
    """
    Operator occupancy of the standard MWC dimer (see module docstring).

    Every argument broadcasts against the others. Pass 0 for a constant the
    model does not have (K_h_e and K_l_o in O2_C4, K_u in U0 models). The
    constants are logged and θ computed by :func:`mwc_theta_log`.

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
    ln = [_log(xp.asarray(k, dtype=float), xp)
          for k in (K_h_l, K_h_o, K_h_e, K_l_o, K_l_e, K_u)]
    return mwc_theta_log(*ln, e_total, tf_total, xp=xp)


def theta_saturated(K_h_l, K_h_o, K_h_e, K_l_o, K_l_e, K_u, tf_total,
                    xp=jnp):
    """
    θ in the limit of saturating effector (e → ∞, so depletion vanishes).

    F(e), P_H and P_L are dominated by their e² terms, so
    W → r·(K_h_o·K_h_e² + K_h_l·K_l_o·K_l_e²) / (K_h_e² + K_h_l·K_l_e²).
    It is 0 for M2 (K_h_e = K_l_o = 0). Computed from the log constants.
    """
    hl, ho, he, lo, le = [_log(xp.asarray(k, dtype=float), xp)
                          for k in (K_h_l, K_h_o, K_h_e, K_l_o, K_l_e)]
    num = xp.logaddexp(ho + 2.0 * he, hl + lo + 2.0 * le)
    den = xp.logaddexp(2.0 * he, hl + 2.0 * le)
    lW = _log(tf_total / 2.0, xp) + num - den
    return _sigmoid(xp.where(num == -np.inf, -np.inf, lW), xp)


def theta_c12(ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e, ln_K_u,
              e_total, tf_total, xp=jnp):
    """M3 (O2_C12) θ from log constants; ``ln_K_u=None`` for U0."""
    ln_K_u = -np.inf if ln_K_u is None else ln_K_u
    return mwc_theta_log(ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e,
                         ln_K_u, e_total, tf_total, xp=xp)


def theta_c4(ln_K_op, ln_K_HL, ln_K_E, ln_K_U, e_total, tf_total, xp=jnp):
    """
    M2 (O2_C4) θ from log constants; ``ln_K_U=None`` for U0.

    M3 with K_h_o = K_op, K_h_l = K_HL, K_l_e = K_E (per site, M⁻¹) and
    K_h_e = K_l_o = 0 (log constants -inf).
    """
    ln_K_U = -np.inf if ln_K_U is None else ln_K_U
    return mwc_theta_log(ln_K_HL, ln_K_op, -np.inf, -np.inf, ln_K_E, ln_K_U,
                         e_total, tf_total, xp=xp)
