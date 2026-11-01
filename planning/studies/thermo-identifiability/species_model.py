"""
Species/energy model of a dimeric transcription factor, for identifiability.

The model is a set of species with a log formation constant each, built
linearly from a small vector of energy terms. The operator is a probe: it is
fixed and far below the TF (integrated single-copy reporter), so it leaves
the TF and effector mass balances, and every operator-bound species enters
only through the operator's binding polynomial.

Reference species: the free H monomer (concentration x). Free effector e.

Protein-side species, log formation constant from a H monomers + n E:

    H      a=1 n=0   0                      L      a=1 n=0   L1
    H.E    a=1 n=1   eH                     L.E    a=1 n=1   L1 + eL
    H2     a=2 n=0   dH                     L2     a=2 n=0   2 L1 + dL
    H2.E   a=2 n=1   dH + ln2 + eH          L2.E   a=2 n=1   2 L1 + dL + ln2 + eL
    H2.E2  a=2 n=2   dH + 2 eH + cH         L2.E2  a=2 n=2   2 L1 + dL + 2 eL + cL
    U      a=1 n=0   U          (u_form="monomer": U <-> H)
    U2     a=2 n=0   dH + U     (u_form="dimer": U2 <-> H2, tfscreen's U1)

L1 is per protomer, so a mutation that shifts it by d shifts L2/H2 by 2d (the
homodimer counts the mutation twice). eH/eL are per-site constants; ln2 and
the absence of a statistical factor on the doubly bound state make the two
sites independent unless cH/cL (second-site coupling) are nonzero.

Operator binding (ln K, M^-1) of each protein species:

    monomer X with n effectors:  oXm + n xX
    dimer X2 with n effectors:   oX + n xX + softplus(oiiX)

oX is the first (bimolecular) half-site engagement, oiiX the unitless second
(intramolecular) one, so a dimer contributes K_oi (1 + K_oii). xX couples
bound effector to operator binding; xX = 0 is MWC independence. The
repressed fraction is theta = B / (1 + B), B = sum_s [s] K_s^O.

Solve: for fixed free effector the protein balance A1 x + 2 A2 x^2 = P_t is a
quadratic; the effector balance is a monotone 1D root in ln e (brentq). The
Jacobian of theta in the terms and ln P_t comes from the implicit function
theorem on the log-space residuals, so it is exact at the solve's precision.
"""

import numpy as np
from scipy.optimize import brentq

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp

TERMS = ["L1", "U", "dH", "dL", "eH", "eL", "cH", "cL", "xH", "xL",
         "oHm", "oLm", "oH", "oL", "oiiH", "oiiL"]
TI = {t: i for i, t in enumerate(TERMS)}
LN2 = float(np.log(2.0))


class Species:
    """Static species table for one model variant."""

    def __init__(self, monomers=True, u_form="dimer"):
        rows = []   # (name, a, n, {term: coeff}, const, op_kind)
        if monomers:
            rows += [("H", 1, 0, {}, 0.0, "Hm"),
                     ("L", 1, 0, {"L1": 1}, 0.0, "Lm"),
                     ("H.E", 1, 1, {"eH": 1}, 0.0, "Hm"),
                     ("L.E", 1, 1, {"L1": 1, "eL": 1}, 0.0, "Lm")]
        rows += [("H2", 2, 0, {"dH": 1}, 0.0, "H2"),
                 ("H2.E", 2, 1, {"dH": 1, "eH": 1}, LN2, "H2"),
                 ("H2.E2", 2, 2, {"dH": 1, "eH": 2, "cH": 1}, 0.0, "H2"),
                 ("L2", 2, 0, {"L1": 2, "dL": 1}, 0.0, "L2"),
                 ("L2.E", 2, 1, {"L1": 2, "dL": 1, "eL": 1}, LN2, "L2"),
                 ("L2.E2", 2, 2, {"L1": 2, "dL": 1, "eL": 2, "cL": 1}, 0.0,
                  "L2")]
        if u_form == "monomer":
            if not monomers:
                raise ValueError("u_form='monomer' needs monomers")
            rows.append(("U", 1, 0, {"U": 1}, 0.0, None))
        elif u_form == "dimer":
            rows.append(("U2", 2, 0, {"dH": 1, "U": 1}, 0.0, None))
        elif u_form != "none":
            raise ValueError(u_form)

        self.names = [r[0] for r in rows]
        self.a = np.array([r[1] for r in rows], float)
        self.n = np.array([r[2] for r in rows], float)
        self.C = np.zeros((len(rows), len(TERMS)))
        self.c0 = np.array([r[4] for r in rows], float)
        # operator ln K = Co @ T + softplus(oii) * has_oii; -inf where unbound
        self.Co = np.zeros((len(rows), len(TERMS)))
        self.oii = np.full(len(rows), -1, int)
        self.binds = np.zeros(len(rows), bool)
        for i, (_, a, n, coef, _, kind) in enumerate(rows):
            for t, v in coef.items():
                self.C[i, TI[t]] = v
            if kind is None:
                continue
            self.binds[i] = True
            X = kind[0]
            if kind.endswith("m"):
                self.Co[i, TI[f"o{X}m"]] = 1
            else:
                self.Co[i, TI[f"o{X}"]] = 1
                self.oii[i] = TI[f"oii{X}"]
            self.Co[i, TI[f"x{X}"]] = n
        self.monomers = monomers
        self.u_form = u_form

    # ---- numpy solve -------------------------------------------------------

    def _lnK(self, T):
        return self.C @ T + self.c0

    def _ln_x(self, lnK, v, lnPt):
        """Free H monomer (log) from the protein quadratic at ln e = v."""
        lw = lnK + self.n * v
        A1 = np.exp(lw[self.a == 1]).sum() if (self.a == 1).any() else 0.0
        A2 = np.exp(lw[self.a == 2]).sum()
        Pt = np.exp(lnPt)
        if A1 == 0.0:
            return 0.5 * (np.log(Pt / 2) - np.log(A2))
        # x = 2 Pt / (A1 + sqrt(A1^2 + 8 A2 Pt)), stable for small A2
        return np.log(2 * Pt) - np.log(A1 + np.sqrt(A1 ** 2 + 8 * A2 * Pt))

    def solve(self, T, lnPt, Et):
        """Return (ln x, ln e); ln e is None at Et == 0."""
        T = np.asarray(T, float)
        lnK = self._lnK(T)
        if Et == 0:
            keep = self.n == 0
            lw = lnK[keep]
            a = self.a[keep]
            A1 = np.exp(lw[a == 1]).sum() if (a == 1).any() else 0.0
            A2 = np.exp(lw[a == 2]).sum()
            Pt = np.exp(lnPt)
            if A1 == 0.0:
                return 0.5 * (np.log(Pt / 2) - np.log(A2)), None
            return (np.log(2 * Pt)
                    - np.log(A1 + np.sqrt(A1 ** 2 + 8 * A2 * Pt))), None
        lnEt = np.log(Et)

        def g(v):
            u = self._ln_x(lnK, v, lnPt)
            lw = lnK + self.a * u + self.n * v
            m = self.n > 0
            bound = np.sum(self.n[m] * np.exp(lw[m]))
            return np.log(np.exp(v) + bound) - lnEt

        v = brentq(g, lnEt - 120.0, lnEt, xtol=1e-14, rtol=1e-15,
                   maxiter=500)
        return self._ln_x(lnK, v, lnPt), v

    # ---- jax pieces for the implicit Jacobian -----------------------------

    def _jlnK(self, T):
        return jnp.asarray(self.C) @ T + jnp.asarray(self.c0)

    def _jlnKO(self, T):
        lo = jnp.asarray(self.Co) @ T
        sp = jnp.where(self.oii >= 0,
                       jax.nn.softplus(T[jnp.asarray(np.maximum(self.oii,
                                                                 0))]),
                       0.0)
        return jnp.where(jnp.asarray(self.binds), lo + sp, -jnp.inf)

    def _residual(self, uv, T, lnPt, lnEt, has_e):
        lnK = self._jlnK(T)
        a = jnp.asarray(self.a)
        n = jnp.asarray(self.n)
        if has_e:
            u, v = uv[0], uv[1]
            lw = lnK + a * u + n * v
            r1 = logsumexp(lw + jnp.log(a)) - lnPt
            mask = self.n > 0
            r2 = jnp.logaddexp(v, logsumexp(lw[mask] + jnp.log(n[mask]))) \
                - lnEt
            return jnp.stack([r1, r2])
        keep = self.n == 0
        u = uv[0]
        lw = lnK[keep] + a[keep] * u
        return jnp.stack([logsumexp(lw + jnp.log(a[keep])) - lnPt])

    def _ln_B(self, uv, T, has_e):
        lnK = self._jlnK(T)
        lnKO = self._jlnKO(T)
        a = jnp.asarray(self.a)
        n = jnp.asarray(self.n)
        if has_e:
            lw = lnK + a * uv[0] + n * uv[1]
            terms = lw + lnKO
        else:
            keep = self.n == 0
            terms = (lnK + a * uv[0] + lnKO)[keep]
        return logsumexp(jnp.where(jnp.isfinite(terms), terms, -jnp.inf))

    def theta_and_grad(self, T, lnPt, Et):
        """theta and d theta / d(T, lnPt) (length len(TERMS) + 1)."""
        u, v = self.solve(T, lnPt, Et)
        has_e = v is not None
        uv = jnp.array([u, v]) if has_e else jnp.array([u])
        lnEt = jnp.log(Et) if has_e else 0.0
        T = jnp.asarray(T, float)

        def R(uv_, p):
            return self._residual(uv_, p[:-1], p[-1], lnEt, has_e)

        def lnB(uv_, p):
            return self._ln_B(uv_, p[:-1], has_e)

        p = jnp.concatenate([T, jnp.array([lnPt])])
        Ru = jax.jacfwd(R, 0)(uv, p)
        Rp = jax.jacfwd(R, 1)(uv, p)
        Bu = jax.grad(lnB, 0)(uv, p)
        Bp = jax.grad(lnB, 1)(uv, p)
        dlnB = Bp - Bu @ jnp.linalg.solve(Ru, Rp)
        lb = lnB(uv, p)
        th = jax.nn.sigmoid(lb)
        return float(th), np.asarray(th * (1 - th) * dlnB)

    def _grad_fn(self, has_e):
        """Jitted, vmapped (uv, p, lnEt) -> (theta, d theta / dp)."""
        key = "_g1" if has_e else "_g0"
        if not hasattr(self, key):
            def one(uv, p, lnEt):
                R = lambda uv_, p_: self._residual(uv_, p_[:-1], p_[-1],
                                                   lnEt, has_e)
                lnB = lambda uv_, p_: self._ln_B(uv_, p_[:-1], has_e)
                Ru = jax.jacfwd(R, 0)(uv, p)
                Rp = jax.jacfwd(R, 1)(uv, p)
                Bu = jax.grad(lnB, 0)(uv, p)
                Bp = jax.grad(lnB, 1)(uv, p)
                dlnB = Bp - Bu @ jnp.linalg.solve(Ru, Rp)
                th = jax.nn.sigmoid(lnB(uv, p))
                return th, th * (1 - th) * dlnB
            setattr(self, key, jax.jit(jax.vmap(one)))
        return getattr(self, key)

    def solve_batch(self, T, lnPt, Et, iters=110):
        """Vectorized solve over rows: (ln x, ln e), ln e NaN at Et == 0.

        Bisection in ln e on [ln Et - 120, ln Et] (the effector residual is
        monotone), with the protein quadratic solved exactly inside.
        """
        T = np.asarray(T, float)
        lnPt = np.asarray(lnPt, float)
        Et = np.asarray(Et, float)
        lnK = T @ self.C.T + self.c0
        Pt = np.exp(lnPt)
        m1 = self.a == 1
        m2 = self.a == 2

        def ln_x(lw, Pt):
            A1 = np.exp(lw[:, m1]).sum(1) if m1.any() else np.zeros(len(lw))
            A2 = np.exp(lw[:, m2]).sum(1)
            q = np.sqrt(A1 ** 2 + 8 * A2 * Pt)
            with np.errstate(divide="ignore"):
                u = np.log(2 * Pt) - np.log(A1 + q)
                u0 = 0.5 * (np.log(Pt / 2) - np.log(A2))
            return np.where(A1 > 0, u, u0)

        u = np.empty(len(Et))
        v = np.full(len(Et), np.nan)
        zero = Et == 0
        if zero.any():
            lw = np.where(self.n[None, :] == 0, lnK[zero], -np.inf)
            u[zero] = ln_x(lw, Pt[zero])
        pos = ~zero
        if pos.any():
            lnEt = np.log(Et[pos])
            lo, hi = lnEt - 120.0, lnEt.copy()
            K = lnK[pos]
            Pp = Pt[pos]
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                lw = K + self.n[None, :] * mid[:, None]
                uu = ln_x(lw, Pp)
                lw = lw + self.a[None, :] * uu[:, None]
                bound = np.sum(self.n[None, :] * np.exp(lw), axis=1)
                g = np.log(np.exp(mid) + bound) - lnEt
                hi = np.where(g > 0, mid, hi)
                lo = np.where(g > 0, lo, mid)
            vv = 0.5 * (lo + hi)
            v[pos] = vv
            u[pos] = ln_x(K + self.n[None, :] * vv[:, None], Pp)
        return u, v

    def batch_theta_grad(self, T, lnPt, Et):
        """theta (R,) and d theta / d(T, lnPt) (R, len(TERMS) + 1)."""
        T = np.asarray(T, float)
        lnPt = np.asarray(lnPt, float)
        Et = np.asarray(Et, float)
        R = len(Et)
        th = np.empty(R)
        g = np.empty((R, len(TERMS) + 1))
        uu, vv = self.solve_batch(T, lnPt, Et)
        sol = [(uu[i], None if np.isnan(vv[i]) else vv[i]) for i in range(R)]
        p = np.concatenate([T, lnPt[:, None]], axis=1)
        for has_e in (True, False):
            idx = np.array([i for i in range(R)
                            if (sol[i][1] is not None) == has_e], int)
            if len(idx) == 0:
                continue
            if has_e:
                uv = np.array([[sol[i][0], sol[i][1]] for i in idx])
                lnEt = np.log(Et[idx])
            else:
                uv = np.array([[sol[i][0]] for i in idx])
                lnEt = np.zeros(len(idx))
            t_, g_ = self._grad_fn(has_e)(jnp.asarray(uv),
                                          jnp.asarray(p[idx]),
                                          jnp.asarray(lnEt))
            th[idx] = np.asarray(t_)
            g[idx] = np.asarray(g_)
        return th, g

    def theta(self, T, lnPt, Et):
        """theta by numpy (no jax), for loops over many solves."""
        T = np.asarray(T, float)
        u, v = self.solve(T, lnPt, Et)
        lw = self._lnK(T) + self.a * u + (self.n * v if v is not None
                                          else 0.0)
        keep = self.binds & ((self.n == 0) if v is None else True)
        lo = self.Co @ T
        oii = np.where(self.oii >= 0,
                       np.logaddexp(0.0, T[np.maximum(self.oii, 0)]), 0.0)
        z = lw[keep] + lo[keep] + oii[keep]
        lnB = np.logaddexp.reduce(z)
        return float(1.0 / (1.0 + np.exp(-lnB)))

    def mass_balance_error(self, T, lnPt, Et):
        """Relative error of both mass balances at the solution."""
        u, v = self.solve(T, lnPt, Et)
        lnK = self._lnK(np.asarray(T, float))
        if v is None:
            keep = self.n == 0
            lw = lnK[keep] + self.a[keep] * u
            P = np.sum(self.a[keep] * np.exp(lw))
            return abs(P / np.exp(lnPt) - 1), 0.0
        lw = lnK + self.a * u + self.n * v
        P = np.sum(self.a * np.exp(lw))
        E = np.exp(v) + np.sum(self.n * np.exp(lw))
        return abs(P / np.exp(lnPt) - 1), abs(E / Et - 1)
