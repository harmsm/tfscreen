"""
Step 4: what a misspecified fit does to predictions of unmeasured doubles.

Noise-free data from a richer truth are fit by MAP with a reduced model
(the MWC species model of step 3). With no noise the MAP is the pseudo-true
point a misspecified fit converges to, so its prediction error for a double
is the bias that real data would carry. Each double's error is scored
against the Laplace SD the fit itself reports at that point, and the fit's
misfit to the measured data is reported as the chi-square it adds over
noise, in SDs of the noise-only chi-square (would we notice?).

Cases (truth -> fit):
    correct          truth is the fit's model (MWC, Oii far from 1); the
                     only error is prior shrinkage
    oii0_folded      wt ln K_oii = 0; the fit folds Oii into one constant
    oii4_folded      wt ln K_oii = 4; the fit folds Oii
    x_coupling       effector-operator coupling: wt xH -0.7, xL +0.7 kT
                     per bound effector, pocket and allosteric mutations
                     shift x by SD 0.7; the fit assumes MWC independence
    c_coupling       second-site effector coupling: wt cH +0.7, cL -0.7,
                     pocket mutations shift c by SD 0.7; the fit assumes
                     independent sites
    bg_offtarget     backgrounds' off-target shifts (SD 0.2, as in every
                     case) held at 0 by the fit
    mut_x_bg         specific mutation x background coupling: 30% of
                     (mutation, background) pairs get a Normal(0, 0.5) shift
                     on one of the mutation's main terms; the fit has none
    ligand_specific  a mutation's effector-site shifts differ under effector
                     2 by Normal(0, 0.5); the fit shares them
    pair_coupling    10% of doubles carry a Normal(0, 1) specific coupling
                     on one term; affects only the doubles being predicted

Truth, priors, designs and noise otherwise follow design_power.py (20
mutations, fixed budget of 12 conditions' reads at theta SD 0.03 each).
Doubles are all 190 pairs in the wt background at the base knobs.

Usage: python misspec.py --case correct [--seed 1] [--out_prefix misspec]
"""

import argparse
import itertools
import json

import numpy as np
from scipy.optimize import least_squares

from species_model import TERMS, TI
from rank_check import (Problem, ALL, DESIGNS, BG_TARGETS, E_GRID,
                        TF_LEVELS, EFF_TERMS, OP_TERMS)
from design_power import CLASSES, lac_terms

CASES = ["correct", "oii0_folded", "oii4_folded", "x_coupling", "c_coupling",
         "bg_offtarget", "mut_x_bg", "ligand_specific", "pair_coupling"]
REC6 = [(0, 0, 0, 0), (0, 0, 0, 1), (1, 0, 0, 1), (5, 0, 0, 1),
        (0, 1, 0, 1), (1, 1, 0, 1)]
DESIGN_SET = {"wt_only": DESIGNS["wt_only"], "rec6": REC6,
              "star": DESIGNS["star"], "full": DESIGNS["full"]}
COUPLING = ("cH", "cL", "xH", "xL")
SLOT_P = len(TERMS)
FOLD = -30.0


def softplus(z):
    return np.logaddexp(0.0, z)


class Setup:
    """Registry, truth, fit mask and priors for one case and seed."""

    def __init__(self, case, seed, num_mut=20):
        self.case = case
        self.num_mut = num_mut
        rng = np.random.default_rng(seed)
        pr = Problem("coupled", ALL, num_mut, bg_terms="all", tf="ratio",
                     rng=np.random.default_rng(seed + 1000))
        self.pr = pr
        oii = 0.0 if case == "oii0_folded" else 4.0
        pr.T0 = lac_terms(oii)
        names = pr.names
        P = len(names)
        true = np.zeros(P)
        mu = np.zeros(P)
        sd = np.zeros(P)
        cls = {}
        for m in range(num_mut):
            keys = list(CLASSES)
            p = np.array([CLASSES[k][0] for k in keys])
            cls[m] = keys[rng.choice(len(keys), p=p / p.sum())]
        self.cls = cls
        for i, name in enumerate(names):
            head, _, term = name.partition(":")
            if head == "wt":
                true[i] = pr.T0[TI[term]]
                if case == "x_coupling" and term in ("xH", "xL"):
                    true[i] = -0.7 if term == "xH" else 0.7
                if case == "c_coupling" and term in ("cH", "cL"):
                    true[i] = 0.7 if term == "cH" else -0.7
                mu[i], sd[i] = true[i], 3.0
            elif head.startswith("bg"):
                b = int(head[2:])
                if term in BG_TARGETS[b]:
                    true[i] = rng.choice([-1, 1]) * rng.uniform(1.5, 2.5)
                    sd[i] = 3.0
                else:
                    true[i] = rng.normal(0, 0.2)
                    sd[i] = 0.5
            elif head in ("eff2", "op2"):
                true[i] = rng.normal(0, 1.0)
                sd[i] = 3.0
            elif head.startswith("tf"):
                t = int(head[2:])
                mu[i] = TF_LEVELS[t]
                true[i] = TF_LEVELS[t] + rng.normal(0, 0.1)
                sd[i] = 0.3
            elif head.startswith("mut"):
                m = int(head[3:])
                mean, s = CLASSES[cls[m]][1].get(term, (0.0, 0.2))
                if case == "x_coupling" and term in ("xH", "xL") and \
                        cls[m] in ("pocket", "allosteric"):
                    mean, s = 0.0, 0.7
                if case == "c_coupling" and term in ("cH", "cL") and \
                        cls[m] == "pocket":
                    mean, s = 0.0, 0.7
                true[i] = rng.normal(mean, s)
                sd[i] = 1.5
            if term in COUPLING and not (
                    (case == "x_coupling" and term in ("xH", "xL")) or
                    (case == "c_coupling" and term in ("cH", "cL"))):
                if head != "wt":
                    true[i] = 0.0
        self.true = true

        # fit: MWC (couplings held at 0); folded Oii; off-target held at 0
        fixed = {}
        for i, name in enumerate(names):
            head, _, term = name.partition(":")
            if term in COUPLING:
                fixed[i] = 0.0
            if case in ("oii0_folded", "oii4_folded") and term in (
                    "oiiH", "oiiL"):
                fixed[i] = FOLD if head == "wt" else 0.0
            if case == "bg_offtarget" and head.startswith("bg") and \
                    term not in BG_TARGETS[int(head[2:])]:
                fixed[i] = 0.0
        self.fixed = fixed
        self.free = np.array([i not in fixed for i in range(P)])
        start = true.copy()
        if case in ("oii0_folded", "oii4_folded"):
            for X in ("H", "L"):
                i = names.index(f"wt:o{X}")
                start[i] += softplus(true[names.index(f"wt:oii{X}")]) \
                    - softplus(FOLD)
                mu[i] = start[i]
        for i, v in fixed.items():
            start[i] = v
        self.start = start
        self.mu = mu
        self.sd = sd

        # specific effects the fit cannot represent
        self.extra_bg = {}
        self.extra_eff2 = {}
        self.extra_pair = {}
        if case == "mut_x_bg":
            for m in range(num_mut):
                main = list(CLASSES[cls[m]][1])
                for b in range(1, 6):
                    if rng.random() < 0.3:
                        d = np.zeros(SLOT_P)
                        d[TI[rng.choice(main)]] = rng.normal(0, 0.5)
                        self.extra_bg[(m, b)] = d
        if case == "ligand_specific":
            for m in range(num_mut):
                d = np.zeros(SLOT_P)
                d[TI["eH"]], d[TI["eL"]] = rng.normal(0, 0.5, 2)
                self.extra_eff2[m] = d
        self.pairs = list(itertools.combinations(range(num_mut), 2))
        if case == "pair_coupling":
            for pr_ in self.pairs:
                if rng.random() < 0.1:
                    d = np.zeros(SLOT_P)
                    d[TI[rng.choice(["L1", "eH", "eL", "oH", "U"])]] = \
                        rng.normal(0, 1.0)
                    self.extra_pair[pr_] = d


class Rows:
    """Index arrays mapping parameters to the energy terms of each row."""

    def __init__(self, setup, design=None, pairs=None):
        pr = setup.pr
        self.sp = pr.sp
        cms, links = [], []
        if design is not None:
            pr.design = design
            for c, m, lk in pr.rows():
                cms.append((c, m))
                links.append(lk)
        else:
            for (m1, m2) in pairs:
                lk = []
                for term in pr.free:
                    s = TI[term]
                    lk += [(s, pr.wt[term]), (s, pr.mut[(m1, term)]),
                           (s, pr.mut[(m2, term)])]
                cms.append(((0, 0, 0, 0), (m1, m2)))
                links.append(lk)
        nE = len(E_GRID)
        self.cms = cms
        n = len(cms)
        base = np.repeat(pr.T0[None, :].copy(), n, 0)
        for term in pr.free:
            base[:, TI[term]] = 0.0
        base_p = np.array([TF_LEVELS[c[3]] if c[3] not in pr.tf else 0.0
                           for c, _ in cms])
        extra = np.zeros((n, SLOT_P))
        for k, (c, m) in enumerate(cms):
            b, f = c[0], c[1]
            if isinstance(m, tuple):
                extra[k] += setup.extra_pair.get(m, 0.0)
            elif m is not None:
                if (m, b) in setup.extra_bg:
                    extra[k] += setup.extra_bg[(m, b)]
                if f == 1 and m in setup.extra_eff2:
                    extra[k] += setup.extra_eff2[m]
        self.base = base
        self.base_p = base_p
        self.extra = extra
        cm_i, sl, pi = [], [], []
        for k, lk in enumerate(links):
            for s, p in lk:
                cm_i.append(k)
                sl.append(s)
                pi.append(p)
        self.cm_i = np.array(cm_i)
        self.sl = np.array(sl)
        self.pi = np.array(pi)
        self.n = n
        self.nE = nE
        self.E = np.tile(E_GRID, n)

    def terms(self, p, with_extra):
        T = self.base.copy()
        lnP = self.base_p.copy()
        tmask = self.sl < SLOT_P
        np.add.at(T, (self.cm_i[tmask], self.sl[tmask]), p[self.pi[tmask]])
        np.add.at(lnP, self.cm_i[~tmask], p[self.pi[~tmask]])
        if with_extra:
            T = T + self.extra
        return (np.repeat(T, self.nE, 0), np.repeat(lnP, self.nE))

    def theta(self, p, with_extra=False):
        T, lnP = self.terms(p, with_extra)
        th, _ = self._theta_grad(T, lnP, need_grad=False)
        return th

    def theta_jac(self, p):
        T, lnP = self.terms(p, False)
        th, g = self._theta_grad(T, lnP, need_grad=True)
        J = np.zeros((len(th), len(p)))
        rows = (self.cm_i[:, None] * self.nE
                + np.arange(self.nE)[None, :]).ravel()
        cols = np.repeat(self.pi, self.nE)
        slots = np.repeat(self.sl, self.nE)
        np.add.at(J, (rows, cols), g[rows, slots])
        return th, J

    def _theta_grad(self, T, lnP, need_grad):
        if need_grad:
            return self.sp.batch_theta_grad(T, lnP, self.E)
        sp = self.sp
        u, v = sp.solve_batch(T, lnP, self.E)
        lnK = T @ sp.C.T + sp.c0
        vv = np.where(np.isnan(v), 0.0, v)
        lw = lnK + sp.a[None] * u[:, None] + sp.n[None] * vv[:, None]
        lw = np.where(np.isnan(v)[:, None] & (sp.n[None] > 0), -np.inf, lw)
        lo = T @ sp.Co.T
        oii = np.where(sp.oii[None] >= 0,
                       softplus(T[:, np.maximum(sp.oii, 0)]), 0.0)
        z = np.where(sp.binds[None], lw + lo + oii, -np.inf)
        lnB = np.logaddexp.reduce(z, axis=1)
        return 1.0 / (1.0 + np.exp(-lnB)), None


def fit(setup, rows, y, sigma, max_nfev=200):
    free = setup.free
    p0 = setup.start.copy()
    mu, sd = setup.mu[free], setup.sd[free]

    def full(x):
        p = p0.copy()
        p[free] = x
        return p

    def resid(x):
        return np.concatenate([(rows.theta(full(x)) - y) / sigma,
                               (x - mu) / sd])

    def jac(x):
        _, J = rows.theta_jac(full(x))
        return np.vstack([J[:, free] / np.reshape(sigma, (-1, 1)),
                          np.diag(1.0 / sd)])

    res = least_squares(resid, p0[free], jac=jac, method="trf",
                        x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-10,
                        max_nfev=max_nfev)
    return full(res.x), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=CASES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma_ref", type=float, default=0.03)
    ap.add_argument("--budget_conditions", type=int, default=12)
    ap.add_argument("--max_nfev", type=int, default=2000)
    ap.add_argument("--out_prefix", default="misspec")
    args = ap.parse_args()

    st = Setup(args.case, args.seed)
    dbl = Rows(st, pairs=st.pairs)
    y_dbl = dbl.theta(st.true, with_extra=True).reshape(len(st.pairs), -1)
    out = []
    for dname, design in DESIGN_SET.items():
        K = len(design)
        sigma = args.sigma_ref * np.sqrt(K / args.budget_conditions)
        rows = Rows(st, design=design)
        y = rows.theta(st.true, with_extra=True)
        p_hat, res = fit(st, rows, y, sigma, args.max_nfev)
        r = (rows.theta(p_hat) - y) / sigma
        misfit = float(np.sum(r ** 2))
        detect_z = misfit / np.sqrt(2 * len(y))
        # Laplace at the fitted point, free parameters only
        _, J = rows.theta_jac(p_hat)
        Jf = J[:, st.free]
        Lam = Jf.T @ Jf / sigma ** 2 + np.diag(1.0 / st.sd[st.free] ** 2)
        Sigma = np.linalg.inv(Lam)
        th_d, G = dbl.theta_jac(p_hat)
        Gf = G[:, st.free]
        sd_d = np.sqrt(np.einsum("ij,jk,ik->i", Gf, Sigma, Gf))
        err = th_d.reshape(len(st.pairs), -1) - y_dbl
        sd_d = sd_d.reshape(len(st.pairs), -1)
        z = err / sd_d
        maxerr = np.max(np.abs(err), axis=1)
        maxz = np.max(np.abs(z), axis=1)
        rec = dict(case=args.case, seed=args.seed, design=dname, K=K,
                   sigma=sigma, nfev=int(res.nfev), status=int(res.status),
                   misfit_chi2=misfit, n_obs=len(y), detect_z=detect_z,
                   err_max_median=float(np.median(maxerr)),
                   err_max_q90=float(np.quantile(maxerr, 0.9)),
                   sd_max_median=float(np.median(sd_d.max(1))),
                   coverage95=float(np.mean(np.abs(z) < 1.96)),
                   frac_double_maxz_gt3=float(np.mean(maxz > 3)),
                   rms_z=float(np.sqrt(np.mean(z ** 2))))
        out.append(rec)
        print(json.dumps(rec))
    with open(f"{args.out_prefix}_{args.case}_s{args.seed}.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
