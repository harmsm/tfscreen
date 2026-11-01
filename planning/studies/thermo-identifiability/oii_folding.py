"""
What folding Oii into one operator constant costs in spurious epistasis.

Truth: a dimer conformation X binds operator with K_oi (1 + K_oii), and a
mutation shifts the first engagement by d1 and the second by d2, additively.
Folded model: one constant K_eff per conformation, with mutations additive
in ln K_eff. Each single then has some K_eff exactly, so singles fit
perfectly, but a double predicted additively in ln K_eff misses by

    eps = sp(x + a2 + b2) - sp(x + a2) - sp(x + b2) + sp(x),   sp = softplus

with x = wt ln K_oii. The same eps is the apparent coupling between a
mutation (a2) and a background that shifts the second engagement (b2).

The script draws operator-touching mutations (d1, d2 iid Normal(0, sigma)
kT), forms random doubles, and puts the ln K_eff error through the full
MWC dimer (tfscreen O2_C12 U1 defaults: K_h_l 6.3, K_h_o 4.2e8 M^-1 total,
K_l_o 0.1, per-site K_h_e 3.7e3 and K_l_e 3.0e5 M^-1, K_u 0.01,
1e-5 M TF) to get the error in theta over an IPTG curve. The wt K_eff is
held at 4.2e8 for every x, so only the split between the two engagements
changes.

Usage: python oii_folding.py [--num_pairs 2000] [--out_prefix oii]
"""

import argparse

import numpy as np
import pandas as pd

from species_model import Species, TERMS, TI

E_GRID = np.concatenate([[0.0], np.logspace(-7, -2, 21)])
LN_K_EFF_H = np.log(4.2e8)
LN_K_EFF_L = np.log(0.1)


def softplus(z):
    return np.logaddexp(0.0, z)


def wt_terms(x):
    T = np.zeros(len(TERMS))
    T[TI["dH"]] = np.log(1e8)
    T[TI["dL"]] = np.log(1e8)
    T[TI["L1"]] = 0.5 * np.log(6.3)
    T[TI["U"]] = np.log(0.01)
    T[TI["eH"]] = np.log(3.7e3)
    T[TI["eL"]] = np.log(3.0e5)
    T[TI["oiiH"]] = x
    T[TI["oiiL"]] = x
    T[TI["oH"]] = LN_K_EFF_H - softplus(x)
    T[TI["oL"]] = LN_K_EFF_L - softplus(x)
    return T


def curve(sp, T):
    return np.array([sp.theta(T, np.log(1e-5), e) for e in E_GRID])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_pairs", type=int, default=2000)
    ap.add_argument("--num_theta_pairs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--out_prefix", default="oii")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    sp = Species(monomers=False, u_form="dimer")

    rows = []
    for sigma in (0.5, 1.0, 2.0):
        for x in (-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0):
            d = rng.normal(0, sigma, size=(args.num_pairs, 2, 2))
            a1, a2 = d[:, 0, 0], d[:, 0, 1]
            b1, b2 = d[:, 1, 0], d[:, 1, 1]
            eps = (softplus(x + a2 + b2) - softplus(x + a2)
                   - softplus(x + b2) + softplus(x))
            # theta error on a subset of pairs, mutations on H only (L binds
            # operator 1e9-fold more weakly, so its split cannot matter)
            dth = []
            T0 = wt_terms(x)
            for i in range(args.num_theta_pairs):
                Tt = T0.copy()
                Tt[TI["oH"]] += a1[i] + b1[i]
                Tt[TI["oiiH"]] += a2[i] + b2[i]
                Tf = T0.copy()
                # folded: ln K_eff additive; put the whole shift on oH
                ke = lambda s1, s2: s1 + softplus(x + s2) - softplus(x)
                Tf[TI["oH"]] += ke(a1[i], a2[i]) + ke(b1[i], b2[i])
                dth.append(np.max(np.abs(curve(sp, Tt) - curve(sp, Tf))))
            dth = np.array(dth)
            rows.append(dict(
                sigma_kT=sigma, wt_ln_K_oii=x,
                eps_median=np.median(np.abs(eps)),
                eps_q95=np.quantile(np.abs(eps), 0.95),
                eps_max=np.max(np.abs(eps)),
                dtheta_median=np.median(dth),
                dtheta_q95=np.quantile(dth, 0.95),
                frac_dtheta_gt_0p02=np.mean(dth > 0.02)))
            print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_prefix}_folding.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(df)


if __name__ == "__main__":
    main()
