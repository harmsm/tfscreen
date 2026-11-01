"""
Species table and equilibrium solver for the general dimeric-TF model.

A model is a set of species, each with a log formation constant built
linearly from named energy terms. The operator is a probe: an integrated
single-copy reporter, fixed and far below the TF, so it leaves the TF and
effector mass balances and enters only through its binding polynomial
(``planning/studies/thermo-identifiability``, which this ports from
``species_model.py``).

Terms
-----
All constants are natural logs; association constants are in M⁻¹, the
rest are unitless. The basis is referenced to the dimer, so with every
optional term off it is tfscreen's O2_C12 U1 model with the same names:

    ln_K_h_l     [L2]/[H2], conformational equilibrium of the dimer
    ln_K_h_o     first operator half-site engagement of H2 (M⁻¹)
    ln_K_l_o     the same for L2
    ln_K_h_e     per-site effector binding of H (M⁻¹), per titrant
    ln_K_l_e     the same for L, per titrant

and, by variant (:class:`Variant`):

    ln_K_u       [U2]/[H2], unfolded dimer             (unfolded_dimer)
    ln_K_h_oii   second, intramolecular engagement      (second_engagement)
    ln_K_l_oii     of H2 and L2; a dimer binds with
                   K_o (1 + K_oii)
    ln_K_h_ee    coupling of the second effector site   (effector_coupling)
    ln_K_l_ee      to the first, per titrant
    ln_K_h_eo    coupling of each bound effector to     (operator_coupling)
    ln_K_l_eo      operator binding, per titrant; 0 is MWC independence
    ln_K_d_h     2 H ⇌ H2 (M⁻¹)                         (monomers)
    ln_K_m_h_l   [L]/[H] of the monomer
    ln_K_m_h_o   monomer half-site binding of H and L (M⁻¹)
    ln_K_m_l_o
    ln_K_m_u     [U]/[H], unfolded monomer              (unfolded_monomer)

This is an invertible linear reparameterization of the study's 16 terms
(for example its per-protomer L1 and dL become ln_K_m_h_l and
ln_K_h_l = 2 L1 + dL − dH), so the study's identifiability results hold
for it.

Species
-------
Log formation constants from ``a`` reference protomers and ``n``
effectors. With monomers the reference is the free H monomer; without
them it is half an H2 (ln_K_d_h ≡ 0):

    H     a1 n0   0                    L     a1 n0   m_h_l
    H.E   a1 n1   h_e                  L.E   a1 n1   m_h_l + l_e
    H2    a2 n0   d_h                  L2    a2 n0   d_h + h_l
    H2.E  a2 n1   d_h + ln2 + h_e      L2.E  a2 n1   d_h + h_l + ln2 + l_e
    H2.E2 a2 n2   d_h + 2 h_e + h_ee   L2.E2 a2 n2   d_h + h_l + 2 l_e + l_ee
    U2    a2 n0   d_h + u              U     a1 n0   m_u

ln2 is the statistical factor of two equivalent sites. Operator binding:
a monomer X with n effectors binds with ln K = m_x_o + n·x_eo, a dimer
with ln K = x_o + n·x_eo + softplus(x_oii) (the softplus term only with
``second_engagement``). Unfolded species bind nothing. θ = B/(1 + B),
B = Σ_s [s]·K_s^O.

Solve
-----
For a fixed free effector e the protein balance Σ a_s [s] = P_t is
A1·x + 2·A2·x² = P_t in the free reference protomer x, solved in closed
form in log space (x = 2 P_t / (A1 + sqrt(A1² + 8 A2 P_t)); without
monomers x² = P_t / (2 A2)). The effector balance is then a monotone 1D
root in log e, solved by safeguarded Newton (rtsafe) with exact bracket
[log e_lo, log E_t]: every bound effector sits on a species whose parent
(one effector fewer) has concentration at most P_t, so
bound ≤ 2·K_max·e·P_t with K_max the largest step constant, and
e_lo = E_t / (1 + 2·K_max·P_t). The residual takes the two forms of
``thermo/mwc.py``, generalized: with m_s = a_s − n_s the empty slots of a
species (an unfolded protomer counts as one), E_t − bound = c + Σ m_s [s]
with c = E_t − P_t, so

    E_t ≤ P_t/2:  φ = log(e + bound) − log E_t
    E_t >  P_t/2: φ = log(e + c⁻) − log(c⁺ + Σ m_s [s])

The second form keeps the root resolved in float32 near saturation, where
e is a vanishing part of e + bound. Derivatives of the root come from the
implicit function theorem (``jax.custom_jvp``), not from the iterations,
so gradients and the Laplace Hessian are exact. At E_t ≈ P_t with very
tight effector binding the root is ill-posed at float precision, as in
``mwc.py``.

E_t = 0 is exact: log e = −inf and every liganded species drops out.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

LN2 = math.log(2.0)
LN8 = math.log(8.0)

# Safeguarded Newton iterations. Each one at least halves the bracket, so
# this covers a bracket of 2^24 times float32 resolution; Newton converges
# in a handful on every case tested.
SOLVER_ITERATIONS = 24


# ---------------------------------------------------------------------------
# Variant and terms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    """
    Which species and terms a model has.

    The defaults are O2_C12 U1 plus the second operator engagement, the
    study's recommended starting model (step 2 found folding it costly near
    K_oii ≈ 1).
    """

    monomers: bool = False
    unfolded_dimer: bool = True
    unfolded_monomer: bool = False
    second_engagement: bool = True
    effector_coupling: bool = False
    operator_coupling: bool = False

    def __post_init__(self):
        if self.unfolded_monomer and not self.monomers:
            raise ValueError("unfolded_monomer needs monomers (theta_monomers).")


# Every term in canonical order; genotype-level terms first, then the
# per-titrant ones.
GENOTYPE_TERMS = ("ln_K_h_l", "ln_K_h_o", "ln_K_l_o", "ln_K_u",
                  "ln_K_h_oii", "ln_K_l_oii",
                  "ln_K_d_h", "ln_K_m_h_l", "ln_K_m_h_o", "ln_K_m_l_o",
                  "ln_K_m_u")
TITRANT_TERMS = ("ln_K_h_e", "ln_K_l_e", "ln_K_h_ee", "ln_K_l_ee",
                 "ln_K_h_eo", "ln_K_l_eo")
ALL_TERMS = GENOTYPE_TERMS + TITRANT_TERMS

# Terms a variant switches on (always-present terms are not listed).
_TERM_FLAGS = {
    "ln_K_u": "unfolded_dimer",
    "ln_K_h_oii": "second_engagement",
    "ln_K_l_oii": "second_engagement",
    "ln_K_d_h": "monomers",
    "ln_K_m_h_l": "monomers",
    "ln_K_m_h_o": "monomers",
    "ln_K_m_l_o": "monomers",
    "ln_K_m_u": "unfolded_monomer",
    "ln_K_h_ee": "effector_coupling",
    "ln_K_l_ee": "effector_coupling",
    "ln_K_h_eo": "operator_coupling",
    "ln_K_l_eo": "operator_coupling",
}


def term_active(term, variant):
    """True when ``variant`` has ``term``."""
    flag = _TERM_FLAGS.get(term)
    return flag is None or bool(getattr(variant, flag))


def genotype_terms(variant):
    """The variant's titrant-independent terms, in canonical order."""
    return tuple(t for t in GENOTYPE_TERMS if term_active(t, variant))


def titrant_terms(variant):
    """The variant's per-titrant (effector) terms, in canonical order."""
    return tuple(t for t in TITRANT_TERMS if term_active(t, variant))


def active_terms(variant):
    """Every term the variant has: genotype terms, then titrant terms."""
    return genotype_terms(variant) + titrant_terms(variant)


# ---------------------------------------------------------------------------
# Species table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeciesRow:
    """
    One species.

    ``lnK`` and ``op`` are tuples of (term, coefficient); ``op`` is None
    for a species that does not bind operator. ``oii`` names the
    second-engagement term added through softplus, or None. ``parent`` is
    the row index of the species with one effector fewer, or -1.
    """

    name: str
    a: int
    n: int
    lnK: tuple
    const: float
    op: tuple
    oii: str
    parent: int


def species_table(variant):
    """
    The species of ``variant`` as a tuple of :class:`SpeciesRow`.

    The table is static (a function of the variant only), so it is built
    in Python at trace time.
    """
    v = variant
    oii_h = "ln_K_h_oii" if v.second_engagement else None
    oii_l = "ln_K_l_oii" if v.second_engagement else None
    dh = (("ln_K_d_h", 1),) if v.monomers else ()

    def eo(x, n):
        if not v.operator_coupling or n == 0:
            return ()
        return ((f"ln_K_{x}_eo", n),)

    rows = []

    def add(name, a, n, lnK, const, op, oii, parent_name):
        parent = -1
        if parent_name is not None:
            parent = [r.name for r in rows].index(parent_name)
        rows.append(SpeciesRow(name, a, n, tuple(lnK), float(const),
                               None if op is None else tuple(op), oii,
                               parent))

    if v.monomers:
        add("H", 1, 0, (), 0.0, (("ln_K_m_h_o", 1),), None, None)
        add("H.E", 1, 1, (("ln_K_h_e", 1),), 0.0,
            (("ln_K_m_h_o", 1),) + eo("h", 1), None, "H")
        add("L", 1, 0, (("ln_K_m_h_l", 1),), 0.0, (("ln_K_m_l_o", 1),),
            None, None)
        add("L.E", 1, 1, (("ln_K_m_h_l", 1), ("ln_K_l_e", 1)), 0.0,
            (("ln_K_m_l_o", 1),) + eo("l", 1), None, "L")

    ee_h = (("ln_K_h_ee", 1),) if v.effector_coupling else ()
    ee_l = (("ln_K_l_ee", 1),) if v.effector_coupling else ()
    hl = (("ln_K_h_l", 1),)
    add("H2", 2, 0, dh, 0.0, (("ln_K_h_o", 1),), oii_h, None)
    add("H2.E", 2, 1, dh + (("ln_K_h_e", 1),), LN2,
        (("ln_K_h_o", 1),) + eo("h", 1), oii_h, "H2")
    add("H2.E2", 2, 2, dh + (("ln_K_h_e", 2),) + ee_h, 0.0,
        (("ln_K_h_o", 1),) + eo("h", 2), oii_h, "H2.E")
    add("L2", 2, 0, dh + hl, 0.0, (("ln_K_l_o", 1),), oii_l, None)
    add("L2.E", 2, 1, dh + hl + (("ln_K_l_e", 1),), LN2,
        (("ln_K_l_o", 1),) + eo("l", 1), oii_l, "L2")
    add("L2.E2", 2, 2, dh + hl + (("ln_K_l_e", 2),) + ee_l, 0.0,
        (("ln_K_l_o", 1),) + eo("l", 2), oii_l, "L2.E")
    if v.unfolded_dimer:
        add("U2", 2, 0, dh + (("ln_K_u", 1),), 0.0, None, None, None)
    if v.unfolded_monomer:
        add("U", 1, 0, (("ln_K_m_u", 1),), 0.0, None, None, None)
    return tuple(rows)


@dataclass(frozen=True)
class _Spec:
    """Hashable static part of a table, passed to the custom_jvp solve."""

    a: tuple
    n: tuple
    parent: tuple

    @classmethod
    def from_table(cls, table):
        return cls(tuple(r.a for r in table), tuple(r.n for r in table),
                   tuple(r.parent for r in table))


def species_log_constants(table, terms):
    """
    Stack the species' log formation constants, shape (..., S).

    ``terms`` maps term name to an array; the arrays broadcast together.
    """
    vals = []
    for row in table:
        v = row.const
        for t, c in row.lnK:
            v = v + c * terms[t]
        vals.append(jnp.asarray(v))
    return jnp.stack(jnp.broadcast_arrays(*vals), axis=-1)


def species_log_operator(table, terms):
    """
    Log operator-binding constant of each species, shape (..., S).

    ``-inf`` for species that do not bind (unfolded).
    """
    vals = []
    ref = None
    for row in table:
        if row.op is None:
            vals.append(None)
            continue
        v = 0.0
        for t, c in row.op:
            v = v + c * terms[t]
        if row.oii is not None:
            v = v + jax.nn.softplus(terms[row.oii])
        v = jnp.asarray(v)
        ref = v if ref is None else ref
        vals.append(v)
    vals = [jnp.full_like(ref, -jnp.inf) if v is None else v for v in vals]
    return jnp.stack(jnp.broadcast_arrays(*vals), axis=-1)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _lse(parts):
    return logsumexp(jnp.stack(jnp.broadcast_arrays(*parts)), axis=0)


def _log_weights(spec, x, lnK, lnPt):
    """
    log [s] for every species at log free effector x, as a list.

    The free reference protomer comes from the protein quadratic.
    """
    S = len(spec.a)
    lwn = [lnK[..., i] + spec.n[i] * x if spec.n[i] else lnK[..., i]
           for i in range(S)]
    la2 = _lse([lwn[i] for i in range(S) if spec.a[i] == 2])
    mono = [lwn[i] for i in range(S) if spec.a[i] == 1]
    if mono:
        la1 = _lse(mono)
        lq = 0.5 * jnp.logaddexp(2.0 * la1, LN8 + la2 + lnPt)
        u = LN2 + lnPt - jnp.logaddexp(la1, lq)
    else:
        u = 0.5 * (lnPt - LN2 - la2)
    return [lwn[i] + spec.a[i] * u for i in range(S)]


def _residual(spec, x, lnK, lnPt, lt, lc_pos, lc_neg, free_form):
    """The effector mass balance as a residual in x = log e (module doc)."""
    lw = _log_weights(spec, x, lnK, lnPt)
    S = len(spec.a)
    lbound = _lse([lw[i] + math.log(spec.n[i])
                   for i in range(S) if spec.n[i] > 0])
    lempty = _lse([lw[i] + math.log(spec.a[i] - spec.n[i])
                   for i in range(S) if spec.a[i] > spec.n[i]])
    phi_a = jnp.logaddexp(x, lbound) - lt
    phi_b = jnp.logaddexp(x, lc_neg) - jnp.logaddexp(lc_pos, lempty)
    return jnp.where(free_form > 0, phi_a, phi_b)


def _solve(spec, lnK, lnPt, lt, lc_pos, lc_neg, free_form):
    """Safeguarded Newton (rtsafe) for the root of :func:`_residual`."""
    S = len(spec.a)
    steps = [lnK[..., i] - lnK[..., spec.parent[i]]
             for i in range(S) if spec.parent[i] >= 0]
    ln_k_max = jnp.max(jnp.stack(jnp.broadcast_arrays(*steps)), axis=0)
    shape = jnp.broadcast_shapes(lnK.shape[:-1], jnp.shape(lnPt),
                                 jnp.shape(lt), jnp.shape(lc_pos),
                                 jnp.shape(free_form))
    hi = jnp.broadcast_to(lt, shape).astype(lnK.dtype)
    lo = jnp.broadcast_to(
        lt - jax.nn.softplus(LN2 + ln_k_max + lnPt), shape).astype(lnK.dtype)
    x = hi
    dx_old = 2.0 * (hi - lo)
    tol = 4.0 * float(jnp.finfo(lnK.dtype).eps)

    def phi_fn(x_):
        return _residual(spec, x_, lnK, lnPt, lt, lc_pos, lc_neg, free_form)

    def step(_, carry):
        x, lo, hi, dx_old = carry
        phi, dphi = jax.jvp(phi_fn, (x,), (jnp.ones_like(x),))
        lo = jnp.where(phi < 0, x, lo)
        hi = jnp.where(phi > 0, x, hi)
        dx = phi / dphi
        x_newton = x - dx
        ok = ((x_newton >= lo) & (x_newton <= hi)
              & (jnp.abs(dx) <= 0.5 * jnp.abs(dx_old)))
        x_bisect = 0.5 * (lo + hi)
        x_next = jnp.where(ok, x_newton, x_bisect)
        dx_next = jnp.where(ok, dx, x - x_bisect)
        done = ((jnp.abs(phi) <= 4.0 * tol * jnp.maximum(1.0, jnp.abs(lt)))
                | (jnp.abs(dx) <= tol * jnp.maximum(1.0, jnp.abs(x))))
        return (jnp.where(done, x, x_next), lo, hi,
                jnp.where(done, 0.0, dx_next))

    carry = jax.lax.fori_loop(0, SOLVER_ITERATIONS, step,
                              (x, lo, hi, dx_old))
    return carry[0]


def _log_free_effector_impl(spec, lnK, lnPt, lt, lc_pos, lc_neg, free_form):
    return _solve(spec, lnK, lnPt, lt, lc_pos, lc_neg, free_form)


_log_free_effector = jax.custom_jvp(_log_free_effector_impl,
                                    nondiff_argnums=(0,))


@_log_free_effector.defjvp
def _log_free_effector_jvp(spec, primals, tangents):
    """
    Implicit derivative of the root, dx = −(∂φ/∂p·dp)/(∂φ/∂x). Written in
    JAX ops, so it differentiates again (the Laplace Hessian).
    """
    x = _log_free_effector(spec, *primals)
    _, dphi_p = jax.jvp(lambda *p: _residual(spec, x, *p), primals, tangents)
    _, dphi_dx = jax.jvp(lambda x_: _residual(spec, x_, *primals), (x,),
                         (jnp.ones_like(x),))
    return x, -dphi_p / dphi_dx


def _log_pos(v):
    ok = v > 0
    return jnp.where(ok, jnp.log(jnp.where(ok, v, 1.0)), -jnp.inf)


def log_free_effector(table, lnK, e_total, tf_total):
    """
    log free effector at equilibrium.

    Parameters
    ----------
    table : tuple of SpeciesRow
    lnK : array, shape (..., S)
        :func:`species_log_constants`.
    e_total : array
        Total effector (M), ≥ 0; broadcasts against ``lnK[..., 0]``.
    tf_total : float or array
        Total TF in protomer (monomer) units (M).

    Returns
    -------
    array
        log e, ``-inf`` where ``e_total`` is 0.
    """
    spec = _Spec.from_table(table)
    dtype = lnK.dtype
    e_total = jnp.asarray(e_total, dtype=dtype)
    tf_total = jnp.asarray(tf_total, dtype=dtype)
    pos = e_total > 0
    et = jnp.where(pos, e_total, 1.0)
    lt = jnp.log(et)
    lnPt = jnp.log(tf_total)
    c = et - tf_total
    free_form = jnp.where(et <= 0.5 * tf_total, 1.0, 0.0).astype(dtype)
    x = _log_free_effector(spec, lnK, lnPt, lt, _log_pos(c), _log_pos(-c),
                           free_form)
    return jnp.where(pos, x, -jnp.inf)


def log_species(terms, e_total, tf_total, variant):
    """
    Equilibrium log concentrations of every species.

    Returns
    -------
    names : tuple of str
    log_conc : array, shape (..., S)
    log_e : array
        log free effector.
    """
    table = species_table(variant)
    lnK = species_log_constants(table, terms)
    x = log_free_effector(table, lnK, e_total, tf_total)
    spec = _Spec.from_table(table)
    lw = _log_weights(spec, x, lnK, jnp.log(jnp.asarray(tf_total,
                                                        dtype=lnK.dtype)))
    return (tuple(r.name for r in table),
            jnp.stack(jnp.broadcast_arrays(*lw), axis=-1), x)


def log_operator_polynomial(terms, e_total, tf_total, variant):
    """log B, the operator binding polynomial minus one; θ = sigmoid(log B)."""
    table = species_table(variant)
    _, log_conc, _ = log_species(terms, e_total, tf_total, variant)
    lnKO = species_log_operator(table, terms)
    return logsumexp(log_conc + lnKO, axis=-1)


def theta(terms, e_total, tf_total, variant):
    """
    Operator occupancy θ.

    Parameters
    ----------
    terms : dict
        Term name → array (natural log constants; association constants in
        M⁻¹). Every term of ``variant`` must be present; the arrays and
        ``e_total`` broadcast together.
    e_total : array
        Total effector (M).
    tf_total : float
        Total TF, protomer units (M).
    variant : Variant

    Returns
    -------
    array
        θ in the broadcast shape.
    """
    missing = [t for t in active_terms(variant) if t not in terms]
    if missing:
        raise ValueError(f"terms lacks {missing} for variant {variant}.")
    return jax.nn.sigmoid(
        log_operator_polynomial(terms, e_total, tf_total, variant))


_theta_jit = jax.jit(theta, static_argnums=(3,))


def theta_np(terms, e_total, tf_total, variant, chunk=2_000_000):
    """
    :func:`theta` on numpy inputs, evaluated in chunks along the last axis.

    For posterior extraction, where the arrays are (samples, rows) and can
    be large. Returns a numpy array.
    """
    arrs = {k: np.asarray(v) for k, v in terms.items()}
    e_total = np.asarray(e_total)
    shape = np.broadcast_shapes(e_total.shape,
                                *[a.shape for a in arrs.values()])
    if len(shape) == 0:
        return np.asarray(_theta_jit(arrs, e_total, tf_total, variant))
    n = shape[-1]
    per_col = max(1, int(np.prod(shape[:-1])))
    step = max(1, chunk // per_col)
    out = np.empty(shape, dtype=float)

    def cut(a, lo, hi):
        if a.ndim == 0 or a.shape[-1] == 1:
            return a
        return a[..., lo:hi]

    for lo in range(0, n, step):
        hi = min(n, lo + step)
        part = _theta_jit({k: jnp.asarray(cut(a, lo, hi))
                           for k, a in arrs.items()},
                          jnp.asarray(cut(e_total, lo, hi)), tf_total, variant)
        out[..., lo:hi] = np.asarray(part)
    return out
