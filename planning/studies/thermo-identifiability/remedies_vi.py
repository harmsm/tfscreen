"""
Step 5d: which posterior approximation gets five-mutant intervals right?

Same posterior as remedies_hmc.py. Fits a Gaussian variational
approximation by stochastic ELBO (reparameterized gradients, Adam), in
the coordinates whitened by the Laplace covariance at the MAP, with one
of three covariance families, the ones the production fit offers:

    mean_field   diagonal (the component guide, AutoNormal)
    low_rank     diagonal + rank-k (AutoLowRankMultivariateNormal)
    full_rank    dense Cholesky (AutoMultivariateNormal)

Untouched pair terms are drawn from their prior, as everywhere in step 5.
Saved per call: draws of the touched parameters from the fitted q, the
ELBO trace and -2 log p of the draws relative to the MAP.
remedies_methods.py compares them, the Laplace and HMC.

Usage: python remedies_vi.py --seed 1 --close 15 --family low_rank
"""

import argparse
import time

import numpy as np

from remedies_hmc import build, Posterior
from remedies_tails import fit


class Adam:
    def __init__(self, shapes, lr):
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t, self.lr = 0, lr

    def step(self, params, grads):
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            mh = m / (1 - 0.9 ** self.t)
            vh = v / (1 - 0.999 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class Family:
    """q = N(mu, S S^T) in whitened coordinates."""

    def __init__(self, kind, d, rank, init_scale, rng):
        self.kind, self.d, self.rank = kind, d, rank
        self.mu = np.zeros(d)
        self.logs = np.full(d, np.log(init_scale))
        if kind == "low_rank":
            self.W = 0.01 * rng.standard_normal((d, rank))
        elif kind == "full_rank":
            self.Loff = np.zeros((d, d))
        self.params = [self.mu, self.logs] + (
            [self.W] if kind == "low_rank" else
            [self.Loff] if kind == "full_rank" else [])

    def noise(self, rng, n):
        k = self.rank if self.kind == "low_rank" else 0
        return rng.standard_normal((n, self.d)), rng.standard_normal((n, k))

    def z(self, e1, e2):
        s = np.exp(self.logs)
        if self.kind == "mean_field":
            return self.mu + e1 * s
        if self.kind == "low_rank":
            return self.mu + e1 * s + e2 @ self.W.T
        L = np.tril(self.Loff, -1) + np.diag(s)
        return self.mu + e1 @ L.T

    def entropy(self):
        s = np.exp(self.logs)
        if self.kind == "low_rank":
            S = np.diag(s ** 2) + self.W @ self.W.T
            return 0.5 * np.linalg.slogdet(S)[1]
        return float(self.logs.sum())

    def grads(self, G, e1, e2):
        """Gradients of E f - H, given f's gradients G at each draw."""
        n = len(G)
        s = np.exp(self.logs)
        gmu = G.mean(0)
        if self.kind == "mean_field":
            return [gmu, (G * e1).mean(0) * s - 1.0]
        if self.kind == "low_rank":
            S = np.diag(s ** 2) + self.W @ self.W.T
            Si = np.linalg.inv(S)
            gls = (G * e1).mean(0) * s - np.diag(Si) * s ** 2
            gW = G.T @ e2 / n - Si @ self.W
            return [gmu, gls, gW]
        M = G.T @ e1 / n
        gls = np.diag(M) * s - 1.0
        return [gmu, gls, np.tril(M, -1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--close", type=float, default=15.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--family", default="low_rank",
                    choices=["mean_field", "low_rank", "full_rank"])
    ap.add_argument("--rank", type=int, default=20)
    ap.add_argument("--num_steps", type=int, default=3000)
    ap.add_argument("--num_particles", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--init_scale", type=float, default=0.1)
    ap.add_argument("--num_draws", type=int, default=2400)
    ap.add_argument("--out_prefix", default="vi")
    args = ap.parse_args()
    md, rows, y, sigma, spec = build(args.seed, args.close, args.scale)
    p_hat, _, _ = fit(spec, rows, y, sigma, md.kind == "pair", None, 400)
    post = Posterior(md, rows, y, sigma, spec, p_hat)
    d = post.L.shape[0]
    f0 = post.nlp(np.zeros(d), grad=False)
    rng = np.random.default_rng(100 * args.seed + 7)
    q = Family(args.family, d, args.rank, args.init_scale, rng)
    opt = Adam([p.shape for p in q.params], args.lr)
    trace = []
    t = time.time()
    for it in range(args.num_steps):
        if it == 2 * args.num_steps // 3:
            opt.lr = args.lr / 10
        e1, e2 = q.noise(rng, args.num_particles)
        Z = q.z(e1, e2)
        out = [post.nlp(z) for z in Z]
        F = np.array([o[0] for o in out])
        G = np.array([o[1] for o in out])
        if not np.all(np.isfinite(F)):
            continue
        opt.step(q.params, q.grads(G, e1, e2))
        trace.append(float(F.mean() - f0 - q.entropy()))
        if it % 100 == 0:
            print(f"it {it} -elbo {np.mean(trace[-100:]):.1f}", flush=True)
    e1, e2 = q.noise(rng, args.num_draws)
    Z = q.z(e1, e2)
    m2 = np.array([2 * (post.nlp(z, grad=False) - f0) for z in Z])
    X = np.array([post.x_of(z)[post.idx] for z in Z])
    tag = f"{args.out_prefix}_s{args.seed}_c{args.close:g}_{args.family}"
    np.savez(f"{tag}.npz", draws=X, idx=post.idx, p_hat=p_hat,
             trace=np.array(trace), m2dlogp=m2, seconds=time.time() - t)
    print(f"done {tag} median -2dlogp {np.median(m2):.1f} "
          f"final -elbo {np.mean(trace[-200:]):.1f}", flush=True)


if __name__ == "__main__":
    main()
