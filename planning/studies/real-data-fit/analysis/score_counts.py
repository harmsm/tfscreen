"""
Count log-likelihood of a MAP run, by tube, from the model itself.

    python score_counts.py <run_dir> [genotype_chunk]

Traces the run's model at its MAP (tfs_fit_model_params.npz, constrained
values) over genotype chunks of the full batch with the mini-batch scale
removed, and sums the log-probability of every observed count site. Unlike
residual_compare.py, which rebuilds the single-transformation prediction by
hand, this works for any model, the congression mixture included.

Writes <run_dir>/score_tubes.csv: one row per tube (replicate x time x
condition_pre x condition_sel x titrant_conc index) with its summed ll, and
prints the total. Compare two runs with compare_scores.py.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro import handlers

from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.run_inference import RunInference


def main():
    run = sys.argv[1]
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    os.chdir(run)  # config paths are relative to the run directory
    run = "."
    orch, _ = read_configuration(f"{run}/tfs_configure_config.yaml")
    ri = RunInference(orch, 0)
    p = np.load(f"{run}/tfs_fit_model_params.npz")
    values = {k[:-len("_auto_loc")]: jnp.asarray(p[k]) for k in p.files
              if k.endswith("_auto_loc")}
    data = jax.device_put(orch.data)
    G = orch.data.num_genotype

    @jax.jit
    def tube_ll(batch):
        model = handlers.substitute(orch.jax_model, data=values)
        tr = handlers.trace(handlers.seed(model, 0)).get_trace(
            data=batch, priors=orch.priors)
        out = {}
        for name, site in tr.items():
            if site["type"] != "sample" or not site["is_observed"]:
                continue
            if not name.startswith("growth"):
                continue
            lp = site["fn"].log_prob(site["value"])
            mask = site.get("mask")
            if mask is not None:
                lp = jnp.where(jnp.broadcast_to(mask, lp.shape), lp, 0.0)
            out[name] = lp.sum(axis=-1)
        return out

    total = None
    for start in range(0, G, chunk):
        idx = jnp.arange(start, min(start + chunk, G))
        res = tube_ll(ri._unscaled_batch(data, idx))
        res = {k: np.asarray(v, dtype=float) for k, v in res.items()}
        total = res if total is None else {k: total[k] + res[k] for k in total}

    for name, arr in total.items():
        print(f"{run}: {name} sum ll {arr.sum():.6e}  (tube array {arr.shape})")
    arr = total[[k for k in total if k.endswith("_obs")][0]]
    idx = np.indices(arr.shape).reshape(arr.ndim, -1).T
    df = pd.DataFrame(idx, columns=["replicate", "time", "condition_pre",
                                    "condition_sel", "titrant_name",
                                    "titrant_conc"][:arr.ndim])
    df["ll"] = arr.ravel()
    df = df[df.ll != 0]
    df.to_csv(f"{run}/score_tubes.csv", index=False)
    print(f"wrote {run}/score_tubes.csv ({len(df)} tubes)")


if __name__ == "__main__":
    main()
