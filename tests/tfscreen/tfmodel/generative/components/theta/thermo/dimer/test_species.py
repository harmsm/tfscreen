"""
Numerical tests of the thermo.dimer solver (species.py).

The reference is an independent float64 solve: nested scipy brentq on the
two mass balances over the same species table, with no closed form and no
two-form residual. Also: mass balances, exact nesting into O2_C12 U1
(``mwc.theta_c12``), implicit-function gradients and Hessians against
finite differences, finite values at extreme constants, and float32
against float64 near the equivalence point.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.optimize import brentq

from tfscreen.tfmodel.generative.components.theta.thermo import mwc
from tfscreen.tfmodel.generative.components.theta.thermo.dimer import species as sp

E_GRID = np.concatenate([[0.0], np.logspace(-8, -2, 13)])

BASE = dict(
    ln_K_h_l=np.log(6.3), ln_K_h_o=np.log(4.2e8) - 1.3, ln_K_l_o=np.log(0.1),
    ln_K_u=-3.0, ln_K_h_oii=1.0, ln_K_l_oii=0.5, ln_K_d_h=np.log(1e7),
    ln_K_m_h_l=0.9, ln_K_m_h_o=np.log(1e6), ln_K_m_l_o=np.log(1e3),
    ln_K_m_u=-2.0, ln_K_h_e=np.log(3.7e3), ln_K_l_e=np.log(3e5),
    ln_K_h_ee=0.4, ln_K_l_ee=-0.3, ln_K_h_eo=-0.5, ln_K_l_eo=0.3)

FULL = sp.Variant(monomers=True, unfolded_monomer=True, effector_coupling=True,
                  operator_coupling=True)
VARIANTS = [
    sp.Variant(),
    sp.Variant(monomers=True),
    FULL,
    sp.Variant(second_engagement=False, unfolded_dimer=False),
    sp.Variant(monomers=True, unfolded_dimer=False, unfolded_monomer=True),
]


def _reference_theta(terms, e_total, tf_total, variant):
    """θ by nested brentq on the species table (float64 numpy)."""
    table = sp.species_table(variant)
    lnK = np.array([r.const + sum(c * terms[t] for t, c in r.lnK)
                    for r in table])
    a = np.array([r.a for r in table], float)
    n = np.array([r.n for r in table], float)
    lko = np.array([
        -np.inf if r.op is None else
        sum(c * terms[t] for t, c in r.op)
        + (np.logaddexp(0.0, terms[r.oii]) if r.oii else 0.0)
        for r in table])

    def u_of(le, keep):
        def f(u):
            return (np.log(np.sum((a * np.exp(lnK + a * u + n * le))[keep]))
                    - np.log(tf_total))
        return brentq(f, -200, 60, xtol=1e-14, rtol=1e-15, maxiter=500)

    if e_total == 0:
        keep = n == 0
        u = u_of(0.0, keep)
        lw = np.where(keep, lnK + a * u, -np.inf)
    else:
        keep = np.ones_like(n, bool)

        def g(le):
            u = u_of(le, keep)
            return (np.log(np.exp(le) + np.sum(n * np.exp(lnK + a * u + n * le)))
                    - np.log(e_total))
        le = brentq(g, np.log(e_total) - 150, np.log(e_total), xtol=1e-14,
                    rtol=1e-15, maxiter=500)
        lw = lnK + a * u_of(le, keep) + n * le
    return 1.0 / (1.0 + np.exp(-np.logaddexp.reduce(lw + lko)))


def _draw_terms(rng, sd=1.0):
    return {k: v + rng.normal(0, sd) for k, v in BASE.items()}


def _jterms(t, dtype=None):
    return {k: jnp.asarray(v, dtype=dtype) for k, v in t.items()}


@pytest.mark.parametrize("variant", VARIANTS, ids=str)
def test_theta_matches_nested_brentq(variant):
    rng = np.random.default_rng(1)
    with jax.enable_x64(True):
        for _ in range(4):
            t = _draw_terms(rng)
            pt = 1e-5 * np.exp(rng.normal(0, 1))
            ours = np.asarray(sp.theta(_jterms(t), jnp.asarray(E_GRID), pt,
                                       variant))
            ref = np.array([_reference_theta(t, e, pt, variant) for e in E_GRID])
            np.testing.assert_allclose(ours, ref, atol=1e-11)


@pytest.mark.parametrize("variant", VARIANTS, ids=str)
def test_mass_balances_close(variant):
    rng = np.random.default_rng(2)
    table = sp.species_table(variant)
    a = np.array([r.a for r in table], float)
    n = np.array([r.n for r in table], float)
    with jax.enable_x64(True):
        t = _draw_terms(rng)
        pt = 2e-5
        e = jnp.asarray(E_GRID)
        _, logc, loge = sp.log_species(_jterms(t), e, pt, variant)
        conc = np.exp(np.asarray(logc))
        np.testing.assert_allclose(conc @ a, pt, rtol=1e-12)
        bound = np.exp(np.asarray(loge)) + conc @ n
        np.testing.assert_allclose(bound[1:], E_GRID[1:], rtol=1e-12)
        assert np.all(conc[0][n > 0] == 0.0)     # no effector, no liganded species


def test_nests_into_mwc_c12():
    """No monomers, no coupling, no second engagement: exactly O2_C12 U1."""
    rng = np.random.default_rng(3)
    variant = sp.Variant(second_engagement=False)
    with jax.enable_x64(True):
        for _ in range(20):
            t = _draw_terms(rng, sd=2.0)
            ours = np.asarray(sp.theta(_jterms(t), jnp.asarray(E_GRID), 1e-5,
                                       variant))
            ref = np.asarray(mwc.theta_c12(
                t["ln_K_h_l"], t["ln_K_h_o"], t["ln_K_h_e"], t["ln_K_l_o"],
                t["ln_K_l_e"], t["ln_K_u"], E_GRID, 1e-5, xp=np))
            np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_second_engagement_is_a_factor_on_the_operator_constant():
    """K_o (1 + K_oii) with ln K_oii = x equals K_o' = K_o (1 + e^x)."""
    rng = np.random.default_rng(4)
    t = _draw_terms(rng)
    folded = dict(t)
    folded["ln_K_h_o"] = t["ln_K_h_o"] + np.logaddexp(0, t["ln_K_h_oii"])
    folded["ln_K_l_o"] = t["ln_K_l_o"] + np.logaddexp(0, t["ln_K_l_oii"])
    with jax.enable_x64(True):
        a = sp.theta(_jterms(t), jnp.asarray(E_GRID), 1e-5, sp.Variant())
        b = sp.theta(_jterms(folded), jnp.asarray(E_GRID), 1e-5,
                     sp.Variant(second_engagement=False))
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-13)


def _objective(variant, e):
    names = sorted(sp.active_terms(variant))

    def f(p):
        th = sp.theta(dict(zip(names, p)), e, 1e-5, variant)
        return jnp.sum(jnp.sin(3.0 * th))
    return names, f


def test_gradient_and_hessian_match_finite_differences():
    with jax.enable_x64(True):
        e = jnp.asarray(E_GRID)
        names, f = _objective(FULL, e)
        p0 = jnp.array([BASE[k] for k in names])
        h = 1e-6
        f = jax.jit(f)
        grad = jax.jit(jax.grad(f))
        g = np.asarray(grad(p0))
        fd = np.array([(f(p0.at[i].add(h)) - f(p0.at[i].add(-h))) / (2 * h)
                       for i in range(len(p0))])
        np.testing.assert_allclose(g, fd, atol=1e-7)
        H = np.asarray(jax.jit(jax.hessian(f))(p0))
        fdH = np.array([(grad(p0.at[i].add(h)) - grad(p0.at[i].add(-h))) / (2 * h)
                        for i in range(len(p0))])
        np.testing.assert_allclose(H, fdH, atol=1e-6)


def test_extreme_constants_give_finite_theta_and_gradients():
    names = sorted(sp.active_terms(FULL))
    e = jnp.asarray(E_GRID, dtype=jnp.float32)

    @jax.jit
    def both(p):
        f = lambda q: jnp.sum(sp.theta(dict(zip(names, q)), e, 1e-5, FULL))
        return sp.theta(dict(zip(names, p)), e, 1e-5, FULL), jax.grad(f)(p)

    rng = np.random.default_rng(5)
    for _ in range(50):
        p = jnp.asarray(rng.uniform(-60, 60, len(names)), dtype=jnp.float32)
        th, g = both(p)
        assert np.all(np.isfinite(np.asarray(th)))
        assert np.all((np.asarray(th) >= 0) & (np.asarray(th) <= 1))
        assert np.all(np.isfinite(np.asarray(g)))


def test_float32_agrees_with_float64_near_equivalence():
    """Tight effector binding with E_t around P_t, where one residual form
    alone loses the root in float32 (mwc.py)."""
    rng = np.random.default_rng(6)
    e = 1e-5 * np.linspace(0.3, 3.0, 40)
    theta = jax.jit(sp.theta, static_argnums=(3,))
    for variant in (sp.Variant(), FULL):
        for _ in range(5):
            t = _draw_terms(rng)
            t["ln_K_h_e"] = t["ln_K_l_e"] = 20.0 + rng.normal(0, 2)
            with jax.enable_x64(True):
                r64 = np.asarray(theta(_jterms(t), jnp.asarray(e), 1e-5,
                                       variant))
            r32 = np.asarray(theta(_jterms(t, jnp.float32),
                                      jnp.asarray(e, dtype=jnp.float32), 1e-5,
                                      variant))
            np.testing.assert_allclose(r32, r64, atol=2e-5)


def test_theta_np_chunks_match_theta():
    rng = np.random.default_rng(7)
    variant = sp.Variant(monomers=True)
    terms = {k: rng.normal(BASE[k], 1.0, (3, 50)) for k in sp.active_terms(variant)}
    conc = np.tile(np.logspace(-7, -3, 50), (3, 1))
    whole = sp.theta_np(terms, conc, 1e-5, variant)
    chunked = sp.theta_np(terms, conc, 1e-5, variant, chunk=20)
    np.testing.assert_allclose(chunked, whole, atol=1e-7)
    assert whole.shape == (3, 50)


def test_variant_terms_and_validation():
    assert sp.active_terms(sp.Variant(second_engagement=False)) == (
        "ln_K_h_l", "ln_K_h_o", "ln_K_l_o", "ln_K_u", "ln_K_h_e", "ln_K_l_e")
    assert set(sp.active_terms(FULL)) == set(sp.ALL_TERMS)
    assert "ln_K_d_h" not in sp.active_terms(sp.Variant())
    with pytest.raises(ValueError, match="needs monomers"):
        sp.Variant(unfolded_monomer=True)
    with pytest.raises(ValueError, match="lacks"):
        sp.theta({"ln_K_h_l": 0.0}, 1e-3, 1e-5, sp.Variant())


def test_species_table_bookkeeping():
    table = sp.species_table(FULL)
    names = [r.name for r in table]
    assert names == ["H", "H.E", "L", "L.E", "H2", "H2.E", "H2.E2", "L2",
                     "L2.E", "L2.E2", "U2", "U"]
    for r in table:
        if r.parent >= 0:
            assert table[r.parent].n == r.n - 1
            assert table[r.parent].a == r.a
        if r.name.startswith("U"):
            assert r.op is None and r.n == 0


def test_theta_np_scalar_and_broadcast_columns():
    variant = sp.Variant()
    scalar = {k: BASE[k] for k in sp.active_terms(variant)}
    th = sp.theta_np(scalar, 1e-4, 1e-5, variant)
    assert th.shape == () and 0.0 <= float(th) <= 1.0
    # a term with a size-1 last axis broadcasts against the chunked one
    terms = dict(scalar)
    terms["ln_K_h_l"] = np.array([[BASE["ln_K_h_l"]]])
    conc = np.logspace(-7, -3, 30)[None, :]
    np.testing.assert_allclose(sp.theta_np(terms, conc, 1e-5, variant, chunk=7),
                               sp.theta_np(terms, conc, 1e-5, variant),
                               atol=1e-7)
