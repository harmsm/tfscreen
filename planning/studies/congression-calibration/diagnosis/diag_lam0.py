"""
Diagnose why mixture fits with lambda ~ 0 do worse than single on the same data.

Runs 0001 (single) and 0002 (mixture, matched prior, lambda -> ~0.004) were
fit to identical simulated data. Evaluate both models' log-likelihood and
predictions at both fits' posterior-median latents.
"""

import glob
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, substitute, trace

from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.run_inference import RunInference

GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "congression_calibration")
RUNS = sys.argv[1:] or ["run_0001", "run_0002"]


def load(run):
    d = glob.glob(os.path.join(GRID, run + "_*"))[0]
    cwd = os.getcwd()
    os.chdir(d)
    try:
        orch, init_params = read_configuration("tfs_configure_config.yaml")
        ri = RunInference(orch, 0)
        svi, state = ri.restore_svi_from_checkpoint(
            "tfs_fit_model_checkpoint.pkl", init_params=init_params)
    finally:
        os.chdir(cwd)
    return d, orch, svi, state


def full_batch(orch):
    return orch.get_batch(orch.data, jnp.arange(orch.data.num_genotype))


def median_latents(orch, svi, state, n=60):
    params = svi.get_params(state)
    data = full_batch(orch)
    guide = substitute(svi.guide, data=params)
    draws = {}
    for i in range(n):
        tr = trace(seed(guide, i)).get_trace(data=data, priors=orch.priors)
        for name, site in tr.items():
            if site["type"] == "sample" and not name.startswith("_"):
                draws.setdefault(name, []).append(np.asarray(site["value"]))
    return {k: jnp.asarray(np.median(np.stack(v), axis=0)) for k, v in draws.items()}, params


def evaluate(orch, latents):
    data = full_batch(orch)
    model = substitute(seed(orch.jax_model, 0), data=latents)
    tr = trace(model).get_trace(data=data, priors=orch.priors)
    obs = {name: float(jnp.sum(site["fn"].log_prob(site["value"])
                               * (site["mask"] if site.get("mask") is not None else 1.0)))
           for name, site in tr.items()
           if site["type"] == "sample" and site.get("is_observed", False)}
    return obs, np.asarray(tr["growth_pred"]["value"])


if __name__ == "__main__":
    (d1, o1, s1, st1), (d2, o2, s2, st2) = load(RUNS[0]), load(RUNS[1])
    print("single run :", os.path.basename(d1))
    print("mixture run:", os.path.basename(d2))

    lat1, _ = median_latents(o1, s1, st1)
    lat2, p2 = median_latents(o2, s2, st2)
    print("\nmixture guide lam params:",
          {k: np.asarray(v).round(4).tolist() for k, v in p2.items() if k.startswith("transformation")})
    print("mixture median lam:", float(lat2["transformation_lam"]))

    only1 = sorted(set(lat1) - set(lat2)); only2 = sorted(set(lat2) - set(lat1))
    print("latents only in single:", only1, " only in mixture:", only2)

    rows = []
    obs, g_s_s = evaluate(o1, lat1)
    rows.append(("single model @ single fit", obs))
    obs, g_m_m = evaluate(o2, lat2)
    rows.append(("mixture model @ mixture fit", obs))

    for lam in (float(lat2["transformation_lam"]), 1e-8):
        l = dict(lat1); l["transformation_lam"] = jnp.array(lam)
        obs, g = evaluate(o2, l)
        rows.append((f"mixture model @ single fit, lam={lam:.3g}", obs))
        diff = g - g_s_s
        print(f"\ngrowth_pred mixture@single-fit(lam={lam:.3g}) - single@single-fit: "
              f"max|d| {np.nanmax(np.abs(diff)):.4g}, rms {np.sqrt(np.nanmean(diff**2)):.4g}")

    l = dict(lat2); l.pop("transformation_lam")
    obs, _ = evaluate(o1, l)
    rows.append(("single model @ mixture fit", obs))

    print("\nobserved-site log-likelihoods (sum):")
    sites = sorted({k for _, o in rows for k in o})
    print(f"{'':48s}" + "".join(f"{s:>18s}" for s in sites) + f"{'total':>18s}")
    for name, o in rows:
        print(f"{name:48s}" + "".join(f"{o.get(s, np.nan):18.1f}" for s in sites)
              + f"{sum(o.values()):18.1f}")


def per_genotype(orch, latents, off_latents):
    """Growth log-lik per genotype with and without the congressed classes."""
    data = full_batch(orch)
    out = []
    for lat in (latents, off_latents):
        tr = trace(substitute(seed(orch.jax_model, 0), data=lat)).get_trace(
            data=data, priors=orch.priors)
        site = tr["growth_obs"]
        lp = site["fn"].log_prob(site["value"])
        if site.get("mask") is not None:
            lp = lp * site["mask"]
        out.append(np.asarray(lp))
    return out
