"""
Step 3: how well does a design predict double mutants it never measured?

The goal is predicting combined effects, not resolving every term. So the
criterion is the posterior predictive SD of theta for unmeasured doubles
(m1 + m2, wt background, base effector/operator/TF), from a Laplace
(linear-Gaussian) posterior at lac-like true constants:

    Sigma = (sum_c F_c / sigma_c^2 + P)^-1,   var(theta_d) = g_d Sigma g_d^T

F_c = J_c^T J_c is condition c's Fisher information at unit noise, P the
prior precision and g_d the gradient of the double's theta. A term the data
never resolve costs nothing unless a double's theta depends on it; when it
does, its prior variance shows up in the double's predictive SD. That is
the "hidden change that bites when combined" this study asks about.

Truth and priors:
- wt: tfscreen O2_C12 U1 defaults (L2/H2 6.3, per-site K_h_e 3.7e3 and
  K_l_e 3.0e5 M^-1, effective K_h_o 4.2e8 and K_l_o 0.1 M^-1, U2/H2 0.01)
  plus a tight dimer (K 1e10 M^-1 for H2 and L2), monomer half-site binding
  1e6 (H) and 1e3 (L) M^-1, and the second engagement at ln K_oii = 4
  (--ln_k_oii to change). Prior: Normal(truth, 3) per term.
- Mutations (--num_mut, default 20): each has one class (allosteric L1;
  stability U, mean +1; pocket eH and eL; DNA-binding oH, oL, oHm, oLm,
  oiiH, oiiL; interface dH, dL, mean -1) with main effects SD 1.5 kT
  (oii 1.0) and every other term SD 0.2. Prior: Normal(0, 1.5) on every
  term (the fit does not know the class).
- Backgrounds: their target term shifted by +-1.5 to 2.5, every other term
  by Normal(0, 0.2). Prior: SD 3 on targets, 0.5 off target.
- Effector 2, operator 2: offsets Normal(0, 1). Prior SD 3.
- TF levels 1 and 2 (0.3x and 3x): latent, prior SD 0.3 around nominal
  (a measured expression ratio), truth nominal + Normal(0, 0.1).

Noise: theta measured with SD sigma per point. --sigma_ref is the SD at one
condition's depth; at a fixed total budget of --budget_conditions
conditions' reads, a design of K conditions has sigma_ref*sqrt(K/budget).

Usage: python design_power.py [--out_prefix power] [--ln_k_oii 4]
                              [--variant mwc] [--seed 1]
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from species_model import TERMS, TI, LN2
from rank_check import Problem, DESIGNS, ALL, BG_TARGETS, E_GRID, TF_LEVELS

CLASSES = {
    "allosteric": (0.30, {"L1": (0.0, 1.5)}),
    "stability": (0.25, {"U": (1.0, 1.0)}),
    "pocket": (0.15, {"eH": (0.0, 1.5), "eL": (0.0, 1.5)}),
    "dna": (0.15, {"oH": (0.0, 1.5), "oL": (0.0, 1.5), "oHm": (0.0, 1.5),
                   "oLm": (0.0, 1.5), "oiiH": (0.0, 1.0),
                   "oiiL": (0.0, 1.0)}),
    "interface": (0.15, {"dH": (-1.0, 1.0), "dL": (-1.0, 1.0)}),
}


def lac_terms(ln_k_oii):
    sp = lambda z: np.logaddexp(0.0, z)
    t = dict(L1=0.5 * np.log(6.3), U=np.log(0.01), dH=np.log(1e10),
             dL=np.log(1e10), eH=np.log(3.7e3), eL=np.log(3.0e5),
             cH=0.0, cL=0.0, xH=0.0, xL=0.0,
             oHm=np.log(1e6), oLm=np.log(1e3),
             oH=np.log(4.2e8) - sp(ln_k_oii), oL=np.log(0.1) - sp(ln_k_oii),
             oiiH=ln_k_oii, oiiL=ln_k_oii)
    return np.array([t[k] for k in TERMS])


def build(variant, num_mut, ln_k_oii, seed):
    """Problem over every condition with lac-like truth and priors."""
    rng = np.random.default_rng(seed)
    pr = Problem(variant, ALL, num_mut, bg_terms="all", tf="ratio",
                 rng=np.random.default_rng(seed + 1000))
    pr.T0 = lac_terms(ln_k_oii)
    prior_sd = np.zeros(len(pr.names))
    true = np.zeros(len(pr.names))
    mut_class = {}
    for m in range(num_mut):
        names = list(CLASSES)
        p = np.array([CLASSES[k][0] for k in names])
        mut_class[m] = names[rng.choice(len(names), p=p / p.sum())]
    for i, name in enumerate(pr.names):
        head, _, term = name.partition(":")
        if head == "wt":
            true[i], prior_sd[i] = pr.T0[TI[term]], 3.0
        elif head.startswith("bg"):
            b = int(head[2:])
            if term in BG_TARGETS[b]:
                true[i] = rng.choice([-1, 1]) * rng.uniform(1.5, 2.5)
                prior_sd[i] = 3.0
            else:
                true[i], prior_sd[i] = rng.normal(0, 0.2), 0.5
        elif head in ("eff2", "op2"):
            true[i], prior_sd[i] = rng.normal(0, 1.0), 3.0
        elif head.startswith("tf"):
            t = int(head[2:])
            true[i], prior_sd[i] = TF_LEVELS[t] + rng.normal(0, 0.1), 0.3
        elif head.startswith("mut"):
            m = int(head[3:])
            mu, sd = CLASSES[mut_class[m]][1].get(term, (0.0, 0.2))
            true[i], prior_sd[i] = rng.normal(mu, sd), 1.5
        else:
            raise ValueError(name)
    pr.true = true
    return pr, prior_sd, mut_class


def condition_fishers(pr):
    """Per-condition unit-noise Fisher information over every condition."""
    J, _ = pr.jacobian()
    rows = (len({k[0] for k in pr.mut}) + 1) * len(E_GRID)
    F = {}
    for i, c in enumerate(pr.design):
        Jc = J[i * rows:(i + 1) * rows]
        F[c] = Jc.T @ Jc
    return F


def double_gradients(pr, pairs):
    """theta and its gradient for each double at the base condition."""
    sp = pr.sp
    Ts, metas = [], []
    for (m1, m2) in pairs:
        T = pr.T0.copy()
        links = []
        for term in pr.free:
            T[TI[term]] = 0.0
            s = TI[term]
            links.append((s, pr.wt[term]))
            links.append((s, pr.mut[(m1, term)]))
            links.append((s, pr.mut[(m2, term)]))
        for s, pi in links:
            T[s] += pr.true[pi]
        for e in E_GRID:
            Ts.append(T)
            metas.append(links)
    th, g = sp.batch_theta_grad(np.array(Ts),
                                np.full(len(Ts), TF_LEVELS[0]),
                                np.tile(E_GRID, len(pairs)))
    G = np.zeros((len(Ts), len(pr.names)))
    for r, links in enumerate(metas):
        for s, pi in links:
            G[r, pi] += g[r, s]
    return th, G


def single_thetas(pr, num_mut):
    out = {}
    for m in [None] + list(range(num_mut)):
        T = pr.T0.copy()
        for term in pr.free:
            T[TI[term]] = pr.true[pr.wt[term]]
            if m is not None:
                T[TI[term]] += pr.true[pr.mut[(m, term)]]
        out[m] = np.array([pr.sp.theta(T, TF_LEVELS[0], e) for e in E_GRID])
    return out


def predictive(Fsum, P, G):
    Sigma = np.linalg.inv(Fsum + P)
    return np.sqrt(np.einsum("ij,jk,ik->i", G, Sigma, G))


def summarize_sd(sd, n_pairs):
    sd = sd.reshape(n_pairs, len(E_GRID))
    mx = sd.max(axis=1)
    return dict(sd_median=float(np.median(sd)),
                sd_q90=float(np.quantile(sd, 0.9)),
                maxsd_median=float(np.median(mx)),
                maxsd_q90=float(np.quantile(mx, 0.9)),
                frac_max_lt_0p02=float(np.mean(mx < 0.02)),
                frac_max_lt_0p05=float(np.mean(mx < 0.05)))


def greedy(F, P, G, K, sigma, candidates):
    """Forward selection minimizing the mean predictive variance of the
    doubles, mean_d g_d Sigma g_d^T = tr(Sigma G^T G) / n."""
    M = G.T @ G / len(G)
    chosen, Fsum = [], np.zeros_like(P)
    for _ in range(K):
        best = None
        for c in candidates:
            if c in chosen:
                continue
            v = np.sum(np.linalg.inv(Fsum + F[c] / sigma ** 2 + P) * M)
            if best is None or v < best[0]:
                best = (v, c)
        chosen.append(best[1])
        Fsum = Fsum + F[best[1]] / sigma ** 2
    return chosen


def group_of(name):
    head, _, term = name.partition(":")
    if head.startswith("mut"):
        return f"mut:{term}"
    if head.startswith("bg"):
        return "background shifts"
    if head in ("eff2", "op2") or head.startswith("tf"):
        return "knob offsets"
    return f"wt:{term}"


def decompose(Fsum, P, G, names):
    """Predictive variance of the doubles explained by each parameter group:
    full variance minus the variance with that group known."""
    Lam = Fsum + P
    Sigma = np.linalg.inv(Lam)
    full = np.einsum("ij,jk,ik->i", G, Sigma, G)
    groups = sorted({group_of(n) for n in names})
    out = {}
    for gname in groups:
        keep = np.array([group_of(n) != gname for n in names])
        Sc = np.linalg.inv(Lam[np.ix_(keep, keep)])
        Gk = G[:, keep]
        cond = np.einsum("ij,jk,ik->i", Gk, Sc, Gk)
        out[gname] = float(np.mean(full - cond) / np.mean(full))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mwc")
    ap.add_argument("--num_mut", type=int, default=20)
    ap.add_argument("--ln_k_oii", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma_ref", type=float, default=0.03)
    ap.add_argument("--budget_conditions", type=int, default=12)
    ap.add_argument("--out_prefix", default="power")
    args = ap.parse_args()

    pr, prior_sd, mut_class = build(args.variant, args.num_mut,
                                    args.ln_k_oii, args.seed)
    P = np.diag(1.0 / prior_sd ** 2)
    print("mutation classes:", {k: list(mut_class.values()).count(k)
                                for k in CLASSES})
    F = condition_fishers(pr)
    pairs = list(itertools.combinations(range(args.num_mut), 2))
    th2, G = double_gradients(pr, pairs)
    singles = single_thetas(pr, args.num_mut)
    th2 = th2.reshape(len(pairs), len(E_GRID))
    epi = np.array([np.max(np.abs(th2[i] - singles[a] - singles[b]
                                  + singles[None]))
                    for i, (a, b) in enumerate(pairs)])
    context = dict(
        epistasis_theta_max_median=float(np.median(epi)),
        epistasis_theta_max_q90=float(np.quantile(epi, 0.9)),
        frac_epistasis_gt_0p05=float(np.mean(epi > 0.05)))
    print("true epistasis (max |theta| over curve):", context)

    rows = []
    prior_only = summarize_sd(predictive(np.zeros_like(P), P, G), len(pairs))
    rows.append(dict(design="prior_only", K=0, regime="-", sigma=np.nan,
                     **prior_only))
    named = {k: v for k, v in DESIGNS.items()}
    for dname, design in named.items():
        K = len(design)
        for regime in ("fixed_depth", "fixed_budget"):
            sigma = args.sigma_ref * (1.0 if regime == "fixed_depth"
                                      else np.sqrt(K / args.budget_conditions))
            Fsum = sum(F[c] for c in design) / sigma ** 2
            res = summarize_sd(predictive(Fsum, P, G), len(pairs))
            rows.append(dict(design=dname, K=K, regime=regime, sigma=sigma,
                             **res))
            print(dname, K, regime, f"sigma {sigma:.3f}", res)

    greedy_sets = {}
    for K in (1, 3, 6, 12, 17, 30):
        sigma = args.sigma_ref * np.sqrt(K / args.budget_conditions)
        chosen = greedy(F, P, G, K, sigma, ALL)
        Fsum = sum(F[c] for c in chosen) / sigma ** 2
        res = summarize_sd(predictive(Fsum, P, G), len(pairs))
        rows.append(dict(design=f"greedy_{K}", K=K, regime="fixed_budget",
                         sigma=sigma, **res))
        greedy_sets[K] = [list(map(int, c)) for c in chosen]
        print(f"greedy K={K}", res, chosen)

    decomp = {}
    for dname in ("wt_only", "base", "knobs_only", "star", "full"):
        design = named[dname]
        Fsum = sum(F[c] for c in design) / args.sigma_ref ** 2
        decomp[dname] = decompose(Fsum, P, G, pr.names)
        top = sorted(decomp[dname].items(), key=lambda kv: -kv[1])[:6]
        print("variance share", dname, [(k, round(v, 3)) for k, v in top])

    pd.DataFrame(rows).to_csv(f"{args.out_prefix}_designs.csv", index=False)
    with open(f"{args.out_prefix}_details.json", "w") as fh:
        json.dump(dict(args=vars(args), mutation_classes=mut_class,
                       true_epistasis=context, greedy=greedy_sets,
                       variance_share_fixed_depth=decomp), fh, indent=1)


if __name__ == "__main__":
    main()
