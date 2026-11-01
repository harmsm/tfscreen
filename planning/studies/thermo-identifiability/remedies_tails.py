"""
Step 5b: a heavy-tailed pair prior, and how large the doubles library has
to be.

Truth, geometry, the rec6 singles screen and the targets follow
remedies.py. The fit is remedies.py's `pair` model, with one change: its
pair terms start at 0 rather than at the truth, as a real fit would. Two
things vary:

    prior    the prior on each pair term, at scale s = the matched SD of
             remedies.py, pair_sd*sqrt(exp(-d/length)/n_terms):
             gauss      Normal(0, s)
             gauss_x3   Normal(0, 3s)
             t3         Student-t, 3 dof, scale s (SD sqrt(3) s)
             cauchy     Cauchy, scale s
    close    the doubles library: every site pair closer than this many A
             (library pairs and mutation + background-site pairs), measured
             in the base condition (remedies.py's dbl1); or none

Intervals are sampled, not linearized. The data-informed parameters come
from the Laplace at the MAP, with a Gauss-Newton prior curvature for the
pair terms so that it stays positive. A pair term that no measured row
touches is drawn from its own prior instead, so an unmeasured pair keeps
its prior's tails. Every draw goes through the full model. Widths are
the largest 95% width over the curve, the median over the curve, and the
largest with every pair term held at the MAP: the part of the width that
the singles leave. Calibration is
scored over every target. The uhit, mhit and clean subsets are selected
by the truth, so their coverage is diagnostic only: an honest sparse
prior must under-cover the targets that turned out to hold a coupled
pair.

Usage: python remedies_tails.py [--seed 1] [--out_prefix tails]
"""

import argparse
import itertools
import json

import numpy as np
from scipy.optimize import least_squares

from remedies import Model, Rows, FitSpec, DESIGN_SET, BASE, target_groups

PRIORS = {"gauss": (None, 1.0), "gauss_x3": (None, 3.0),
          "t3": (3.0, 1.0), "cauchy": (1.0, 1.0)}


def prior_resid(x, s, nu):
    """Residual r with r^2/2 = -log prior (+ const), and dr/dx."""
    if nu is None:
        return x / s, 1.0 / s
    u = x / s
    q = np.log1p(u * u / nu)
    r = np.sign(x) * np.sqrt((nu + 1) * q)
    small = np.abs(u) < 1e-6
    safe = np.where(small, 1.0, r)
    dr = np.where(small, np.sqrt((nu + 1) / nu) / s,
                  (nu + 1) * u / (s * (nu + u * u) * safe))
    return np.where(small, np.sqrt((nu + 1) / nu) * u, r), dr


def fit(spec, rows, y, sigma, pair_mask, nu, max_nfev):
    free = spec.free
    p0 = spec.start.copy()
    mu, sd = spec.mu[free], spec.sd[free]
    pm = pair_mask[free]
    sig = np.reshape(sigma, (-1, 1))

    def full(x):
        p = p0.copy()
        p[free] = x
        return p

    def priors(x):
        r = (x - mu) / sd
        dr = 1.0 / sd
        if nu is not None:
            rp, drp = prior_resid(x[pm], sd[pm], nu)
            r = r.copy()
            dr = dr.copy()
            r[pm], dr[pm] = rp, drp
        return r, dr

    def resid(x):
        return np.concatenate([(rows.theta(full(x)) - y) / sigma,
                               priors(x)[0]])

    def jac(x):
        _, J = rows.theta_jac(full(x))
        return np.vstack([J[:, free] / sig, np.diag(priors(x)[1])])

    res = least_squares(resid, p0[free], jac=jac, method="trf",
                        x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-10,
                        max_nfev=max_nfev)
    return full(res.x), res, priors


def sample(p_hat, spec, J, sigma, priors, pair_mask, nu, n, rng):
    """Draws of the full parameter vector (n x P)."""
    free = spec.free
    Jf = J[:, free] / np.reshape(sigma, (-1, 1))
    _, dr = priors(p_hat[free])
    untouched = (np.abs(Jf).sum(0) == 0) & pair_mask[free]
    keep = ~untouched
    Lam = Jf[:, keep].T @ Jf[:, keep] + np.diag(dr[keep] ** 2)
    L = np.linalg.cholesky(np.linalg.inv(Lam))
    draws = np.repeat(p_hat[None], n, 0)
    idx = np.flatnonzero(free)
    draws[:, idx[keep]] += rng.standard_normal((n, keep.sum())) @ L.T
    s = spec.sd[idx[untouched]]
    z = (rng.standard_normal((n, untouched.sum())) if nu is None
         else rng.standard_t(nu, (n, untouched.sum())))
    draws[:, idx[untouched]] = z * s
    return draws, int(untouched.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num_mut", type=int, default=15)
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--length", type=float, default=6.0)
    ap.add_argument("--pair_sd", type=float, default=1.0)
    ap.add_argument("--sigma_ref", type=float, default=0.03)
    ap.add_argument("--budget_conditions", type=int, default=12)
    ap.add_argument("--dbl_budget", type=int, default=3)
    ap.add_argument("--num_targets", type=int, default=200)
    ap.add_argument("--num_draws", type=int, default=1000)
    ap.add_argument("--max_nfev", type=int, default=400)
    ap.add_argument("--closes", default="0,10,15,20")
    ap.add_argument("--priors", default="gauss,gauss_x3,t3,cauchy")
    ap.add_argument("--out_prefix", default="tails")
    args = ap.parse_args()

    md = Model(args.seed, args.num_mut, args.radius, args.length,
               args.pair_sd)
    nm = args.num_mut
    rng = np.random.default_rng(args.seed + 77)
    coupled = {p for p, d in md.pair_par.items()
               if np.any(md.true[list(d.values())] != 0)}
    targets = {
        "dbl": list(itertools.combinations(range(nm), 2)),
        "tri": [tuple(sorted(rng.choice(nm, 3, replace=False)))
                for _ in range(args.num_targets)],
        "quint": [tuple(sorted(rng.choice(nm, 5, replace=False)))
                  for _ in range(args.num_targets)]}
    trows = {k: Rows(md, [(BASE, s) for s in v]) for k, v in targets.items()}
    ttrue = {k: r.theta(md.true).reshape(len(targets[k]), -1)
             for k, r in trows.items()}
    pair_mask = md.kind == "pair"
    design = DESIGN_SET["rec6"]
    K = len(design)
    sig1 = args.sigma_ref * np.sqrt(K / args.budget_conditions)
    sig2 = args.sigma_ref * np.sqrt(1 / args.dbl_budget)
    single = [(c, ()) for c in design] + \
        [(c, (m,)) for c in design for m in range(nm)]
    out = []
    for close in [float(c) for c in args.closes.split(",")]:
        cl = [p for p in md.pair_par if md.dist[p] < close]
        dent = [(BASE, p) if p[1] < nm else
                ((p[1] - nm + 1, 0, 0, 0), (p[0],)) for p in cl]
        rows = Rows(md, single + dent)
        sigma = np.full(rows.n * rows.nE, sig1)
        sigma[len(single) * rows.nE:] = sig2
        y = rows.theta(md.true)
        meas = {p for p in cl if p[1] < nm}
        for pname in args.priors.split(","):
            nu, scale = PRIORS[pname]
            spec = FitSpec(md, "pair", scale)
            spec.start = np.where(pair_mask, 0.0, spec.start)
            p_hat, res, priors = fit(spec, rows, y, sigma, pair_mask, nu,
                                     args.max_nfev)
            r = (rows.theta(p_hat) - y) / sigma
            _, J = rows.theta_jac(p_hat)
            draws, n_untouched = sample(
                p_hat, spec, J, sigma, priors, pair_mask, nu,
                args.num_draws, np.random.default_rng(args.seed + 5))
            rec = dict(seed=args.seed, close=close, prior=pname,
                       n_dbl_rows=len(dent),
                       frac_coupled_measured=len(coupled & set(cl)) /
                       max(len(coupled), 1),
                       n_coupled=len(coupled), nfev=int(res.nfev),
                       status=int(res.status),
                       detect_z=float(np.sum(r ** 2) /
                                      np.sqrt(2 * len(y))),
                       n_untouched_pair_params=n_untouched)
            for k, tr in trows.items():
                n = len(targets[k])
                th_hat = tr.theta(p_hat).reshape(n, -1)
                sims = np.stack([tr.theta(d).reshape(n, -1)
                                 for d in draws])
                lo95, hi95, lo99, hi99 = np.quantile(
                    sims, [0.025, 0.975, 0.005, 0.995], axis=0)
                d0 = draws.copy()
                d0[:, pair_mask] = p_hat[pair_mask]
                sims0 = np.stack([tr.theta(d).reshape(n, -1) for d in d0])
                q0 = np.quantile(sims0, [0.025, 0.975], axis=0)
                width_np = np.max(q0[1] - q0[0], axis=1)
                tt = ttrue[k]
                in95 = (tt >= lo95) & (tt <= hi95)
                in99 = (tt >= lo99) & (tt <= hi99)
                err = np.max(np.abs(th_hat - tt), axis=1)
                width = np.max(hi95 - lo95, axis=1)
                mask = np.array([p not in meas for p in targets[k]]) \
                    if k == "dbl" else np.ones(n, bool)
                groups = target_groups(md, targets[k], meas)
                for g, gm in groups.items():
                    gm = gm & mask
                    if not gm.any():
                        continue
                    pre = f"{k}_{g}"
                    rec.update({
                        f"{pre}_n": int(gm.sum()),
                        f"{pre}_err_q90": float(np.quantile(err[gm], 0.9)),
                        f"{pre}_width95_median": float(
                            np.median(width[gm])),
                        f"{pre}_width95_nopair_median": float(
                            np.median(width_np[gm])),
                        f"{pre}_width95_curve_median": float(
                            np.median(np.median(hi95 - lo95, 1)[gm])),
                        f"{pre}_cov95": float(in95[gm].mean()),
                        f"{pre}_cov99": float(in99[gm].mean()),
                        f"{pre}_frac_out99": float(
                            np.mean(~in99[gm].all(1)))})
            out.append(rec)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float)
                                  else v) for k, v in rec.items()}),
                  flush=True)
    with open(f"{args.out_prefix}_s{args.seed}.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
