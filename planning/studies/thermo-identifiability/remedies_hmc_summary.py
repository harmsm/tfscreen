"""
Pool remedies_hmc.py chains and compare them with the Laplace draws of
remedies_tails.py on the same five-mutant targets.

Usage: python remedies_hmc_summary.py [--seed 1] [--closes 15,0]
Writes hmc_results.csv (one row per close and method) and
hmc_param_sd.csv (posterior SD per parameter, HMC and Laplace).
"""

import argparse
import glob

import numpy as np
import pandas as pd

from remedies import Rows, BASE
from remedies_hmc import build
from remedies_tails import fit, sample


def split_rhat(x):
    """Split R-hat over chains for each column of x (chains x draws x k)."""
    c, n = x.shape[:2]
    h = n // 2
    s = np.concatenate([x[:, :h], x[:, h:2 * h]], 0)
    m = s.mean(1)
    w = s.var(1, ddof=1).mean(0)
    b = h * m.var(0, ddof=1)
    return np.sqrt(((h - 1) / h * w + b / h) / w)


def quint_targets(seed, nm=15, n=200):
    rng = np.random.default_rng(seed + 77)
    [tuple(sorted(rng.choice(nm, 3, replace=False))) for _ in range(n)]
    return [tuple(sorted(rng.choice(nm, 5, replace=False)))
            for _ in range(n)]


def interval_stats(sims, tt, label):
    lo, hi = np.quantile(sims, [0.025, 0.975], axis=0)
    w = hi - lo
    return {f"{label}_cov95": float(((tt >= lo) & (tt <= hi)).mean()),
            f"{label}_width95_median": float(np.median(w.max(1))),
            f"{label}_width95_curve_median": float(
                np.median(np.median(w, 1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--closes", default="15,0")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    rows_out, sd_out = [], []
    for close in [float(c) for c in args.closes.split(",")]:
        files = sorted(glob.glob(f"hmc_s{args.seed}_c{close:g}_k*.npz"))
        if not files:
            continue
        ch = [np.load(f) for f in files]
        md, rows, y, sigma, spec = build(args.seed, close, args.scale)
        pm = md.kind == "pair"
        p_hat = ch[0]["p_hat"]
        idx = ch[0]["idx"]
        n = min(len(c["draws"]) for c in ch)
        X = np.stack([c["draws"][:n] for c in ch])
        rh = split_rhat(X)
        quint = quint_targets(args.seed)
        tr = Rows(md, [(BASE, s) for s in quint])
        nq = len(quint)
        tt = tr.theta(md.true).reshape(nq, -1)
        free = np.flatnonzero(spec.free)
        untouched = np.setdiff1d(free, idx)
        rng = np.random.default_rng(args.seed + 9)
        flat = X.reshape(-1, X.shape[-1])
        hd = np.repeat(p_hat[None], len(flat), 0)
        hd[:, idx] = flat
        hd[:, untouched] = rng.standard_normal(
            (len(flat), len(untouched))) * spec.sd[untouched]
        hsims = np.stack([tr.theta(d).reshape(nq, -1) for d in hd])
        hpair = hd.copy()
        hpair[:, pm] = p_hat[pm]
        hsims0 = np.stack([tr.theta(d).reshape(nq, -1) for d in hpair])
        rq = split_rhat(hsims.reshape(len(ch), n, -1))
        _, J = rows.theta_jac(p_hat)
        _, _, priors = fit(spec, rows, y, sigma, pm, None, 1)
        ld, _ = sample(p_hat, spec, J, sigma, priors, pm, None, len(flat),
                       np.random.default_rng(args.seed + 5))
        lsims = np.stack([tr.theta(d).reshape(nq, -1) for d in ld])
        th_hat = tr.theta(p_hat).reshape(nq, -1)
        hmean = hsims.mean(0)
        rec = dict(seed=args.seed, close=close, chains=len(ch), draws=n,
                   touched=len(idx),
                   acc=float(np.mean([c["acc"] for c in ch])),
                   eps=float(np.mean([c["eps"] for c in ch])),
                   m2dlogp_median=float(np.median(
                       np.concatenate([c["m2dlogp"] for c in ch]))),
                   rhat_param_max=float(rh.max()),
                   rhat_param_q99=float(np.quantile(rh, 0.99)),
                   rhat_quint_max=float(rq.max()),
                   map_err_q90=float(np.quantile(
                       np.abs(th_hat - tt).max(1), 0.9)),
                   hmc_mean_err_q90=float(np.quantile(
                       np.abs(hmean - tt).max(1), 0.9)))
        rec.update(interval_stats(hsims, tt, "hmc"))
        rec.update(interval_stats(hsims0, tt, "hmc_pairs_held"))
        rec.update(interval_stats(lsims, tt, "laplace"))
        rows_out.append(rec)
        hsd = hd[:, idx].std(0)
        lsd = ld[:, idx].std(0)
        for j, i in enumerate(idx):
            sd_out.append(dict(seed=args.seed, close=close,
                               name=md.names[i], kind=md.kind[i],
                               prior_sd=spec.sd[i], hmc_sd=hsd[j],
                               laplace_sd=lsd[j], rhat=rh[j]))
        print(pd.Series(rec).to_string(), flush=True)
    pd.DataFrame(rows_out).to_csv("hmc_results.csv", index=False)
    sd = pd.DataFrame(sd_out)
    sd.to_csv("hmc_param_sd.csv", index=False)
    sd["ratio"] = sd.hmc_sd / sd.laplace_sd
    for close, g in sd.groupby("close"):
        print(f"\nclose {close:g}: widest by Laplace SD")
        print(g.sort_values("laplace_sd", ascending=False).head(12)
              [["name", "prior_sd", "laplace_sd", "hmc_sd", "rhat"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
