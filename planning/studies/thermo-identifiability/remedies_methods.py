"""
Compare posterior approximations with HMC on step 5's five-mutant targets.

Methods, all on the posterior of remedies_hmc.py (seed 1):

    hmc          remedies_hmc.py chains, the reference
    laplace      MAP + Laplace, sampled through the model (step 5b); the
                 production arrowhead Laplace is the same Gaussian
    linearized   MAP + Laplace, propagated linearly (step 5)
    mean_field, low_rank, full_rank
                 Gaussian variational fits from remedies_vi.py

Untouched pair terms are drawn from their prior in every sampled method
and enter the linearized one through their prior variance. Scores per
method: coverage of the truth, 95% widths, and the per-point ratio of the
method's 95% width to HMC's. For the Laplace, the effective sample size
of importance weights toward the true posterior.

Usage: python remedies_methods.py [--seed 1] [--closes 15,0]
Writes methods_results.csv.
"""

import argparse
import glob

import numpy as np
import pandas as pd

from remedies import Rows, BASE
from remedies_hmc import build, Posterior
from remedies_hmc_summary import quint_targets

METHODS = ["hmc", "laplace", "linearized", "mean_field", "low_rank",
           "full_rank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--closes", default="15,0")
    ap.add_argument("--num_laplace", type=int, default=2400)
    args = ap.parse_args()
    out = []
    for close in [float(c) for c in args.closes.split(",")]:
        md, rows, y, sigma, spec = build(args.seed, close, 1.0)
        hm = [np.load(f) for f in
              sorted(glob.glob(f"hmc_s{args.seed}_c{close:g}_k*.npz"))]
        p_hat, idx = hm[0]["p_hat"], hm[0]["idx"]
        post = Posterior(md, rows, y, sigma, spec, p_hat)
        assert np.array_equal(post.idx, idx)
        f0 = post.nlp(np.zeros(len(idx)), grad=False)
        untouched = post.untouched
        quint = quint_targets(args.seed)
        tr = Rows(md, [(BASE, s) for s in quint])
        nq = len(quint)
        tt = tr.theta(md.true).reshape(nq, -1)
        rng = np.random.default_rng(args.seed + 21)

        def full(X):
            P = np.repeat(p_hat[None], len(X), 0)
            P[:, idx] = X
            P[:, untouched] = rng.standard_normal(
                (len(X), len(untouched))) * spec.sd[untouched]
            return P

        def bounds(X):
            sims = np.stack([tr.theta(p).reshape(nq, -1) for p in full(X)])
            return np.quantile(sims, [0.025, 0.975], axis=0)

        draws = {"hmc": np.concatenate([h["draws"] for h in hm])}
        zl = np.random.default_rng(args.seed + 5).standard_normal(
            (args.num_laplace, len(idx)))
        draws["laplace"] = p_hat[idx] + zl @ post.L.T
        for fam in ("mean_field", "low_rank", "full_rank"):
            f = f"vi_s{args.seed}_c{close:g}_{fam}.npz"
            if glob.glob(f):
                draws[fam] = np.load(f)["draws"]
        iv = {k: bounds(v) for k, v in draws.items()}
        th_hat, Jt = tr.theta_jac(p_hat)
        Jz = Jt[:, idx] @ post.L
        var = (Jz ** 2).sum(1) + (Jt[:, untouched] ** 2) @ \
            spec.sd[untouched] ** 2
        sd = np.sqrt(var).reshape(nq, -1)
        th_hat = th_hat.reshape(nq, -1)
        iv["linearized"] = np.stack([th_hat - 1.96 * sd, th_hat + 1.96 * sd])
        lw = np.array([-(post.nlp(z, grad=False) - f0) + 0.5 * z @ z
                       for z in zl])
        w = np.exp(lw - lw.max())
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        wh = iv["hmc"][1] - iv["hmc"][0]
        for k in METHODS:
            if k not in iv:
                continue
            lo, hi = iv[k]
            wd = hi - lo
            ratio = wd / np.maximum(wh, 1e-6)
            rec = dict(seed=args.seed, close=close, method=k,
                       cov95=float(((tt >= lo) & (tt <= hi)).mean()),
                       width95_median=float(np.median(wd.max(1))),
                       width95_curve_median=float(
                           np.median(np.median(wd, 1))),
                       ratio_median=float(np.median(ratio)),
                       ratio_q10=float(np.quantile(ratio, 0.1)),
                       ratio_q90=float(np.quantile(ratio, 0.9)))
            if k in draws:
                X = draws[k]
                zz = np.linalg.solve(post.L, (X - p_hat[idx]).T).T
                sub = zz[np.linspace(0, len(zz) - 1, 300).astype(int)]
                rec["m2dlogp_median"] = float(np.median(
                    [2 * (post.nlp(z, grad=False) - f0) for z in sub]))
            if k == "laplace":
                rec["importance_ess"] = ess
            out.append(rec)
        print(pd.DataFrame([r for r in out if r["close"] == close])
              .round(3).to_string(index=False), flush=True)
    pd.DataFrame(out).to_csv("methods_results.csv", index=False)


if __name__ == "__main__":
    main()
