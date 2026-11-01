"""
Step 5c: are step 5b's sampled intervals the posterior's, or the Laplace's?

Same problem as remedies_tails.py (rec6 singles, doubles closer than
--close A, Gaussian pair prior at --scale). The posterior over every
parameter a measured row touches is sampled by HMC, in coordinates
whitened by the Laplace covariance at the MAP, with the step size tuned by
dual averaging in warm-up. Pair terms no row touches do not enter the
likelihood, so they are drawn from their prior, as in remedies_tails.py.
One chain per call; remedies_hmc_summary.py pools chains, checks them and
compares five-mutant intervals with the Laplace's.

Saved per chain: the draws of the touched parameters, the step size, the
acceptance rate and -2 log p relative to the MAP for every draw (a
Gaussian posterior gives a chi-square with as many degrees of freedom as
touched parameters).

Usage: python remedies_hmc.py --seed 1 --chain 0 [--close 15]
"""

import argparse
import time

import numpy as np

from remedies import Model, Rows, FitSpec, DESIGN_SET, BASE
from remedies_tails import fit


def build(seed, close, scale, num_mut=15, sigma_ref=0.03,
          budget_conditions=12, dbl_budget=3):
    """The step 5b problem: model, rows, data, noise SD and fit spec."""
    md = Model(seed, num_mut, 20.0, 6.0, 1.0)
    design = DESIGN_SET["rec6"]
    sig1 = sigma_ref * np.sqrt(len(design) / budget_conditions)
    sig2 = sigma_ref * np.sqrt(1 / dbl_budget)
    single = [(c, ()) for c in design] + \
        [(c, (m,)) for c in design for m in range(num_mut)]
    cl = [p for p in md.pair_par if md.dist[p] < close]
    dent = [(BASE, p) if p[1] < num_mut else
            ((p[1] - num_mut + 1, 0, 0, 0), (p[0],)) for p in cl]
    rows = Rows(md, single + dent)
    sigma = np.full(rows.n * rows.nE, sig1)
    sigma[len(single) * rows.nE:] = sig2
    y = rows.theta(md.true)
    spec = FitSpec(md, "pair", scale)
    spec.start = np.where(md.kind == "pair", 0.0, spec.start)
    return md, rows, y, sigma, spec


class Posterior:
    """-log p over the touched parameters, with its gradient."""

    def __init__(self, md, rows, y, sigma, spec, p_hat):
        self.rows, self.y, self.sigma, self.p_hat = rows, y, sigma, p_hat
        _, J = rows.theta_jac(p_hat)
        free = np.flatnonzero(spec.free)
        Jf = J[:, free] / sigma[:, None]
        untouched = (np.abs(Jf).sum(0) == 0) & (md.kind[free] == "pair")
        self.idx = free[~untouched]
        self.untouched = free[untouched]
        self.mu, self.sd = spec.mu[self.idx], spec.sd[self.idx]
        Jt = Jf[:, ~untouched]
        lam = Jt.T @ Jt + np.diag(1 / self.sd ** 2)
        self.L = np.linalg.cholesky(np.linalg.inv(lam))

    def x_of(self, z):
        p = self.p_hat.copy()
        p[self.idx] += self.L @ z
        return p

    def nlp(self, z, grad=True):
        p = self.x_of(z)
        if grad:
            th, J = self.rows.theta_jac(p)
        else:
            th = self.rows.theta(p)
        r = (th - self.y) / self.sigma
        pr = (p[self.idx] - self.mu) / self.sd
        f = 0.5 * (r @ r + pr @ pr)
        if not grad:
            return f
        g = (r / self.sigma) @ J[:, self.idx] + pr / self.sd
        return f, self.L.T @ g


def hmc(post, n_warm, n_draw, length, rng, target=0.8, max_leap=400):
    d = post.L.shape[0]
    z = np.zeros(d)
    f, g = post.nlp(z)
    log_eps, log_eps_bar, h_bar = np.log(0.1), 0.0, 0.0
    mu_da = np.log(1.0)
    out, accs, fs = [], [], []
    for it in range(n_warm + n_draw):
        eps = np.exp(log_eps) if it < n_warm else np.exp(log_eps_bar)
        n_leap = int(min(max_leap, max(1, np.ceil(
            length * rng.uniform(0.5, 1.0) / eps))))
        mom = rng.standard_normal(d)
        zn, gn, pn = z.copy(), g.copy(), mom - 0.5 * eps * g
        ok = True
        for i in range(n_leap):
            zn = zn + eps * pn
            fn, gn = post.nlp(zn)
            if not np.isfinite(fn):
                ok = False
                break
            if i < n_leap - 1:
                pn = pn - eps * gn
        if ok:
            pn = pn - 0.5 * eps * gn
            dh = (fn + 0.5 * pn @ pn) - (f + 0.5 * mom @ mom)
            acc = float(np.exp(min(0.0, -dh))) if np.isfinite(dh) else 0.0
        else:
            acc = 0.0
        if rng.uniform() < acc:
            z, f, g = zn, fn, gn
        if it < n_warm:
            m = it + 1
            h_bar = (1 - 1 / (m + 10)) * h_bar + (target - acc) / (m + 10)
            log_eps = mu_da - np.sqrt(m) / 0.05 * h_bar
            w = m ** -0.75
            log_eps_bar = w * log_eps + (1 - w) * log_eps_bar
        else:
            out.append(post.x_of(z)[post.idx])
            accs.append(acc)
            fs.append(f)
        if it % 50 == 0:
            print(f"it {it} eps {eps:.3g} leap {n_leap} acc {acc:.2f} "
                  f"f {f:.1f}", flush=True)
    return np.array(out), np.exp(log_eps_bar), np.mean(accs), np.array(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--close", type=float, default=15.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--num_warmup", type=int, default=400)
    ap.add_argument("--num_draws", type=int, default=600)
    ap.add_argument("--length", type=float, default=3.0)
    ap.add_argument("--out_prefix", default="hmc")
    args = ap.parse_args()
    md, rows, y, sigma, spec = build(args.seed, args.close, args.scale)
    p_hat, _, _ = fit(spec, rows, y, sigma, md.kind == "pair", None, 400)
    post = Posterior(md, rows, y, sigma, spec, p_hat)
    f0 = post.nlp(np.zeros(post.L.shape[0]), grad=False)
    t = time.time()
    draws, eps, acc, fs = hmc(post, args.num_warmup, args.num_draws,
                              args.length,
                              np.random.default_rng(1000 * args.seed +
                                                    args.chain))
    tag = f"{args.out_prefix}_s{args.seed}_c{args.close:g}_k{args.chain}"
    np.savez(f"{tag}.npz", draws=draws, idx=post.idx, p_hat=p_hat,
             eps=eps, acc=acc, m2dlogp=2 * (fs - f0),
             seconds=time.time() - t)
    print(f"done eps {eps:.3g} acc {acc:.2f} touched {len(post.idx)} "
          f"median -2dlogp {np.median(2 * (fs - f0)):.1f}", flush=True)


if __name__ == "__main__":
    main()
