"""
Exact full-batch scores of MAP runs, for comparing their optima.

    python score_map.py <run_dir> [<run_dir> ...] [--params <name>] [--chunk N]

Run from planning/dev-data/real_fit. For each run directory, reads its
tfs_configure_config.yaml and a MAP params file (default
tfs_fit_model_params.npz; --params tfs_fit_model_stage1_params.npz, say)
and evaluates, with every genotype once and no mini-batch scale:

- count_ll: the log-likelihood of the growth observations;
- other_obs_ll: any other observed sites (traced once);
- log_prior: the log density of every latent at its MAP value (constrained
  space, no Jacobian; latents are library-sized, so traced once);
- log_joint: their sum.

The losses in tfs_fit_model_losses.txt are block medians of mini-batch
losses, scaled up from a batch: noisy at the 1e4-nat level on the dev data,
so two runs' losses cannot rank optima that close. These numbers can.

A latent the params file does not name (a site held fixed in that stage)
is drawn from the prior, and the run is flagged; score a stage's params
under its own held values only through tfs-fit-model's stage files.

Prints one line per run and writes score_map.csv in the current directory.
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro import handlers  # noqa: E402

from tfscreen.tfmodel.configuration_io import read_configuration  # noqa: E402
from tfscreen.tfmodel.inference.run_inference import RunInference  # noqa: E402


def _masked_sum(site):
    lp = site["fn"].log_prob(site["value"])
    mask = site.get("mask")
    if mask is not None:
        lp = jnp.where(jnp.broadcast_to(mask, lp.shape), lp, 0.0)
    return lp.sum()


def score_run(run, params_name, chunk):
    cwd = os.getcwd()
    os.chdir(run)  # config paths are relative to the run directory
    try:
        orch, _ = read_configuration("tfs_configure_config.yaml")
        ri = RunInference(orch, 0)
        with np.load(params_name) as p:
            values = {k[:-len("_auto_loc")]: jnp.asarray(p[k]) for k in p.files
                      if k.endswith("_auto_loc")}
        data = jax.device_put(orch.data)
        G = orch.data.num_genotype

        def trace_at(batch):
            model = handlers.substitute(orch.jax_model, data=values)
            return handlers.trace(handlers.seed(model, 0)).get_trace(
                data=batch, priors=orch.priors)

        @jax.jit
        def per_chunk(batch):
            tr = trace_at(batch)
            return sum((_masked_sum(s) for n, s in tr.items()
                        if s["type"] == "sample" and s["is_observed"]
                        and n.startswith("growth")), jnp.array(0.0))

        count_ll = 0.0
        for start in range(0, G, chunk):
            idx = jnp.arange(start, min(start + chunk, G))
            count_ll += float(per_chunk(ri._unscaled_batch(data, idx)))

        # latents (library-sized) and any non-growth observations, once
        tr = trace_at(ri._unscaled_batch(data, jnp.arange(min(chunk, G))))
        log_prior, other_obs, missing = 0.0, 0.0, []
        for name, s in tr.items():
            if s["type"] != "sample":
                continue
            if s["is_observed"]:
                if not name.startswith("growth"):
                    other_obs += float(_masked_sum(s))
            else:
                if name not in values:
                    missing.append(name)
                log_prior += float(_masked_sum(s))
        if G > chunk and other_obs != 0.0:
            print(f"  {run}: non-growth observations scored on the first "
                  f"{chunk} genotypes only")
        return dict(run=run, params=params_name, count_ll=count_ll,
                    other_obs_ll=other_obs, log_prior=log_prior,
                    log_joint=count_ll + other_obs + log_prior,
                    unset_latents=";".join(missing))
    finally:
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--params", default="tfs_fit_model_params.npz")
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        r = score_run(run, args.params, args.chunk)
        flag = f"  UNSET {r['unset_latents']}" if r["unset_latents"] else ""
        print(f"{run}: log joint {r['log_joint']:.8g}  count ll "
              f"{r['count_ll']:.8g}  log prior {r['log_prior']:.6g}{flag}",
              flush=True)
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df) > 1:
        df["log_joint_vs_first"] = df["log_joint"] - df["log_joint"].iloc[0]
        df["count_ll_vs_first"] = df["count_ll"] - df["count_ll"].iloc[0]
    df.to_csv("score_map.csv", index=False)
    print(df.drop(columns=["params"]).to_string(index=False))


if __name__ == "__main__":
    main()
