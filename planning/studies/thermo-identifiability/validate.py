"""
Checks of species_model.py before any identifiability result rests on it.

1. Mass balances close at every solve.
2. The implicit-function Jacobian matches central finite differences.
3. With no monomers, a dimer-level U, MWC independence (x = 0) and
   statistical effector sites (c = 0), theta equals tfscreen's mwc_theta
   (O2_C12 U1 physics) with K_h_o = K_oi (1 + K_oii).
4. Optional (--prototype DIR): with monomers and a monomer U, theta equals
   the prototype reaction-tree model (thermo_model/lac_model.py) solved by
   nested brentq with explicit operator at O_t = 1e-14 M.

Usage: python validate.py [--prototype /path/to/thermo_model]
"""

import argparse
import sys

import numpy as np

from species_model import Species, TERMS, TI

E_GRID = np.concatenate([[0.0], np.logspace(-7, -2, 11)])


def generic_terms(rng, scale=0.5):
    base = dict(L1=0.5, U=np.log(0.1), dH=np.log(2e5), dL=np.log(1e5),
                eH=np.log(1e4), eL=np.log(1e5), cH=0.3, cL=-0.3, xH=-0.5,
                xL=0.5, oHm=np.log(1e6), oLm=np.log(1e5), oH=np.log(1e6),
                oL=np.log(1e5), oiiH=0.0, oiiL=0.5)
    return np.array([base[t] + rng.normal(0, scale) for t in TERMS])


def check_balance_and_grad(rng):
    worst_mb, worst_fd = 0.0, 0.0
    for monomers, u_form in [(True, "dimer"), (True, "monomer"),
                             (False, "dimer")]:
        sp = Species(monomers, u_form)
        for _ in range(5):
            T = generic_terms(rng)
            lnPt = np.log(1e-5) + rng.normal(0, 0.5)
            for Et in E_GRID:
                worst_mb = max(worst_mb, *sp.mass_balance_error(T, lnPt, Et))
                th, g = sp.theta_and_grad(T, lnPt, Et)
                p = np.concatenate([T, [lnPt]])
                h = 1e-5
                fd = np.empty_like(p)
                for i in range(len(p)):
                    pp, pm = p.copy(), p.copy()
                    pp[i] += h
                    pm[i] -= h
                    fd[i] = (sp.theta(pp[:-1], pp[-1], Et)
                             - sp.theta(pm[:-1], pm[-1], Et)) / (2 * h)
                worst_fd = max(worst_fd, np.max(np.abs(fd - g)))
    print(f"1. worst relative mass-balance error: {worst_mb:.1e}")
    print(f"2. worst |IFT - finite difference| in d theta: {worst_fd:.1e}")
    return worst_mb < 1e-9 and worst_fd < 1e-7


def check_mwc(rng):
    sys.path.insert(0, "../../../src")
    from tfscreen.tfmodel.generative.components.theta.thermo import mwc
    sp = Species(monomers=False, u_form="dimer")
    worst = 0.0
    for _ in range(20):
        T = generic_terms(rng, 2.0)
        T[TI["cH"]] = T[TI["cL"]] = T[TI["xH"]] = T[TI["xL"]] = 0.0
        Pt = 1e-5 * np.exp(rng.normal(0, 1))
        K_h_l = np.exp(2 * T[TI["L1"]] + T[TI["dL"]] - T[TI["dH"]])
        K_h_o = np.exp(T[TI["oH"]]) * (1 + np.exp(T[TI["oiiH"]]))
        K_l_o = np.exp(T[TI["oL"]]) * (1 + np.exp(T[TI["oiiL"]]))
        ref = np.asarray(mwc.mwc_theta(K_h_l, K_h_o, np.exp(T[TI["eH"]]),
                                       K_l_o, np.exp(T[TI["eL"]]),
                                       np.exp(T[TI["U"]]), E_GRID, Pt,
                                       xp=np))
        ours = np.array([sp.theta(T, np.log(Pt), e) for e in E_GRID])
        worst = max(worst, np.max(np.abs(ref - ours)))
    print(f"3. worst |theta - mwc_theta| (no monomers, U2, MWC): {worst:.1e}")
    return worst < 1e-8


def check_prototype(rng, path):
    sys.path.insert(0, path)
    from scipy.optimize import brentq
    from lac_model import build_graph, OPERATOR_BOUND_SPECIES
    g = build_graph("U")
    sp = Species(monomers=True, u_form="monomer")
    O_t = 1e-14
    worst = 0.0
    for _ in range(5):
        T = generic_terms(rng)
        t = dict(zip(TERMS, T))
        lnK = dict(
            K_u_h=-t["U"], K_h_h2=t["dH"], K_l_l2=t["dL"],
            K_h2_l2=2 * t["L1"] + t["dL"] - t["dH"],
            K_h_e=t["eH"], K_l_e=t["eL"],
            K_h_oi=t["oHm"], K_l_oi=t["oLm"],
            K_he_oi=t["oHm"] + t["xH"], K_le_oi=t["oLm"] + t["xL"],
            K_h2_e=np.log(2) + t["eH"], K_l2_e=np.log(2) + t["eL"],
            K_h2e_e=t["eH"] + t["cH"] - np.log(2),
            K_l2e_e=t["eL"] + t["cL"] - np.log(2),
            K_h2_oi=t["oH"], K_h2e_oi=t["oH"] + t["xH"],
            K_h2e2_oi=t["oH"] + 2 * t["xH"],
            K_l2_oi=t["oL"], K_l2e_oi=t["oL"] + t["xL"],
            K_l2e2_oi=t["oL"] + 2 * t["xL"],
            K_h2_oii=t["oiiH"], K_h2e_oii=t["oiiH"], K_h2e2_oii=t["oiiH"],
            K_l2_oii=t["oiiL"], K_l2e_oii=t["oiiL"], K_l2e2_oii=t["oiiL"])
        names = g.species
        pref = g.ln_pref(lnK)
        eu = np.array([g.exp_U[s] for s in names])
        ee = np.array([g.exp_E[s] for s in names])
        eo = np.array([g.exp_O[s] for s in names])
        lp = np.array([pref[s] for s in names])
        bound = np.array([s in OPERATOR_BOUND_SPECIES for s in names])
        Pt = 1e-5

        def conc(lr, le, lo):
            return np.exp(lp + eu * lr + ee * le + eo * lo)

        def root(le, lo):
            return brentq(lambda lr: np.log(np.sum(eu * conc(lr, le, lo)))
                          - np.log(Pt), -600, 100, xtol=1e-14)

        def opr(le):
            f = (lambda lo: np.log(np.exp(lo) + np.sum(
                eo * conc(root(le, lo), le, lo))) - np.log(O_t))
            lo = brentq(f, -600, np.log(O_t) + 1e-12, xtol=1e-14)
            return lo, root(le, lo)

        for Et in E_GRID:
            if Et == 0:
                le = -1e3
            else:
                def fE(le):
                    lo, lr = opr(le)
                    return np.log(np.exp(le) + np.sum(
                        ee * conc(lr, le, lo))) - np.log(Et)
                le = brentq(fE, -600, np.log(Et) + 1e-12, xtol=1e-14)
            lo, lr = opr(le)
            th_ref = np.sum(conc(lr, le, lo)[bound]) / O_t
            worst = max(worst, abs(th_ref - sp.theta(T, np.log(Pt), Et)))
    print(f"4. worst |theta - prototype| (monomers, U monomer): {worst:.1e}")
    return worst < 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototype", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(20261031)
    ok = check_balance_and_grad(rng) and check_mwc(rng)
    if args.prototype:
        ok = check_prototype(rng, args.prototype) and ok
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
