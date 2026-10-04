"""
How far an SVI guide is from the posterior it approximates (PSIS).

Draws S samples from the fitted guide, computes the importance weights
w = p(theta, y) / q(theta) on the full data set, and reports:

- khat: the Pareto shape of the largest weights (Vehtari et al. 2024,
  "Pareto smoothed importance sampling"). Below 0.5 the guide is close to
  the posterior; 0.5-0.7 importance sampling still works with more draws;
  above 0.7 the guide misses mass the posterior has, and reweighted
  estimates are unreliable.
- ess: the effective sample size of the smoothed weights.
- per condition, growth_k and growth_m under the guide and reweighted by
  the smoothed weights (mean and SD), plus the simulated truth when the run
  directory has it. Where khat is small the reweighted numbers are a
  reference for the guide's own.

Writes {out_prefix}.json and {out_prefix}_growth.csv.

Usage, from a run directory holding tfs_configure_config.yaml and an SVI
checkpoint:
    python guide_diagnostics.py tfs_fit_model_checkpoint.pkl --num_draws 2000
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.handlers import seed, substitute, trace
from numpyro.infer.util import log_density

from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.run_inference import RunInference


# ---------------------------------------------------------------------------
# PSIS (after Vehtari, Simpson, Gelman, Yao and Gabry, the reference code)
# ---------------------------------------------------------------------------

def _gpd_fit(x):
    """Generalized Pareto (k, sigma) of exceedances x > 0 (Zhang and Stephens 2009)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    prior_bs, prior_k = 3, 10
    m = 30 + int(np.sqrt(n))
    bs = 1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs /= prior_bs * x[int(n / 4 + 0.5) - 1]
    bs += 1 / x[-1]
    ks = np.mean(np.log1p(-bs[:, None] * x), axis=1)
    L = n * (np.log(-bs / ks) - ks - 1)
    w = 1 / np.sum(np.exp(L[None, :] - L[:, None]), axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, bs = w[keep], bs[keep]
    w /= w.sum()
    b = np.sum(bs * w)
    k = np.mean(np.log1p(-b * x))
    sigma = -k / b
    k = (n * k + prior_k * 0.5) / (n + prior_k)   # weak prior toward 0.5
    return k, sigma


def psis(log_w):
    """Smoothed log weights, Pareto k-hat and ESS for one set of log weights."""
    log_w = np.asarray(log_w, dtype=float)
    log_w = log_w - log_w.max()
    S = log_w.size
    M = int(min(0.2 * S, 3 * np.sqrt(S)))
    order = np.argsort(log_w)
    tail_idx = order[-M:]
    cutoff = log_w[order[-M - 1]]
    exceed = np.exp(log_w[tail_idx]) - np.exp(cutoff)
    if np.all(exceed <= 0):
        k = -np.inf
        smoothed = log_w.copy()
    else:
        k, sigma = _gpd_fit(exceed)
        p = (np.arange(1, M + 1) - 0.5) / M
        # generalized Pareto quantiles, in units of sigma
        if abs(k) < 1e-8:
            q = -np.log1p(-p)
        else:
            q = ((1 - p) ** (-k) - 1) / k
        smoothed_tail = np.log(np.exp(cutoff) + sigma * q)
        smoothed = log_w.copy()
        smoothed[tail_idx] = np.minimum(smoothed_tail, 0.0)
    w = np.exp(smoothed - smoothed.max())
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    return np.log(w), float(k), float(ess)


# ---------------------------------------------------------------------------

def _weighted_mean_sd(x, w):
    mean = np.sum(w[:, None] * x, axis=0)
    sd = np.sqrt(np.sum(w[:, None] * (x - mean) ** 2, axis=0))
    return mean, sd


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint_file")
    parser.add_argument("--config_file", default="tfs_configure_config.yaml")
    parser.add_argument("--num_draws", type=int, default=2000)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_prefix", default="tfs_guide_diagnostics")
    args = parser.parse_args()

    orchestrator, init_params = read_configuration(args.config_file)
    ri = RunInference(orchestrator, args.seed)
    svi, state = ri.restore_svi_from_checkpoint(args.checkpoint_file,
                                                init_params=init_params)
    params = jax.tree_util.tree_map(jnp.asarray, svi.get_params(state))
    data = orchestrator.get_batch(orchestrator.data,
                                  jnp.arange(orchestrator.data.num_genotype))
    kwargs = dict(priors=orchestrator.priors, data=data)
    guide = svi.guide
    model = orchestrator.jax_model

    def one(key):
        g = substitute(seed(guide, key), data=params)
        log_q, g_trace = log_density(g, (), kwargs, {})
        latent = {n: s["value"] for n, s in g_trace.items()
                  if s["type"] == "sample" and not s.get("is_observed", False)
                  and not n.startswith("_")}
        log_p, m_trace = log_density(model, (), kwargs, latent)
        return (log_p - log_q,
                m_trace["condition_growth_k"]["value"],
                m_trace["condition_growth_m"]["value"])

    one = jax.jit(jax.vmap(one))
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_draws)
    log_w, ks, ms = [], [], []
    for i in range(0, args.num_draws, args.chunk):
        lw, k, m = one(keys[i:i + args.chunk])
        log_w.append(np.asarray(lw)); ks.append(np.asarray(k)); ms.append(np.asarray(m))
    log_w = np.concatenate(log_w)
    ks = np.concatenate(ks).reshape(args.num_draws, -1)
    ms = np.concatenate(ms).reshape(args.num_draws, -1)

    finite = np.isfinite(log_w)
    smoothed, khat, ess = psis(log_w[finite])
    w = np.exp(smoothed)

    # growth_k/growth_m are indexed in map_condition_rep order
    cond_df = orchestrator.growth_tm.map_groups["condition_rep"]
    conditions = list(cond_df.sort_values("map_condition_rep")["condition_rep"])
    rows = []
    for name, draws in (("growth_k", ks), ("growth_m", ms)):
        d = draws[finite]
        g_mean, g_sd = d.mean(0), d.std(0)
        w_mean, w_sd = _weighted_mean_sd(d, w)
        for j in range(d.shape[1]):
            rows.append(dict(parameter=name, index=j,
                             condition_rep=(conditions[j]
                                            if j < len(conditions) else None),
                             guide_mean=g_mean[j], guide_sd=g_sd[j],
                             reweighted_mean=w_mean[j], reweighted_sd=w_sd[j]))
    table = pd.DataFrame(rows)

    summary_files = {"growth_k": "summary/tfs_summarize_params_growth_k.csv",
                     "growth_m": "summary/tfs_summarize_params_growth_m.csv"}
    for name, f in summary_files.items():
        # the summary's ref column is truth on the fit's scale (X for relative)
        if os.path.exists(f):
            ref = pd.read_csv(f)[["condition_rep", "ref"]]
            sel = table.parameter == name
            table.loc[sel, "ref"] = table.loc[sel, "condition_rep"].map(
                ref.set_index("condition_rep")["ref"])

    out = dict(checkpoint=args.checkpoint_file, num_draws=args.num_draws,
               num_finite=int(finite.sum()), khat=khat, ess=ess,
               log_w_sd=float(np.std(log_w[finite])))
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(out, f, indent=2)
    table.to_csv(f"{args.out_prefix}_growth.csv", index=False)
    print(json.dumps(out, indent=2))
    with pd.option_context("display.width", 200):
        print(table.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
