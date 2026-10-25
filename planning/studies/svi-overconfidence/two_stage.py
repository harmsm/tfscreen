"""
Two-stage fit: carry the per-condition k and m uncertainty into every
genotype's posterior.

The SVI guides collapse the uncertainty on the shared growth parameters
(k, m sit 5-125 of their own SDs off truth), and genotypes whose own counts
are precise then undercover. A Laplace posterior at the MAP covers k and m.
So:

1. Stage 1 (done before this script): MAP, then a Laplace posterior
   (`tfs-sample-posterior` on the MAP checkpoint), written to
   ``--laplace_file``.
2. Stage 2 (this script): take ``--num_draws`` draws of (k, m) from it; for
   each, refit everything else by SVI with k and m pinned at the draw
   (`linear`'s ``k_pinned``/``m_pinned``), and sample that conditional
   posterior.
3. Pool the conditional posteriors with equal weight into one posterior
   file (``--out_prefix``.h5, the layout `tfs-sample-posterior` writes), so
   `tfs-extract-params`, `tfs-predict-theta` and `tfs-summarize-fit` run on
   it unchanged. With the draws from p(k, m | y), the pool approximates the
   marginal posterior of everything else.

Run from a run directory holding ``tfs_configure_config.yaml``:

    python two_stage.py --laplace_file tfs_laplace.h5 --num_draws 10
"""

import argparse
import os
import shutil

import h5py
import numpy as np
import pandas as pd
import yaml

from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.scripts.fit_model_cli import fit_model
from tfscreen.tfmodel.scripts.prefit_calibration_cli import (
    _apply_priors_updates,
    _condition_rep_labels,
    _csv_row_name,
)
from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior


def draw_config(config_file, draw_dir, k, m, cond_labels):
    """Config + priors CSV with k and m pinned at one draw; returns the config path."""
    os.makedirs(draw_dir, exist_ok=True)
    with open(config_file) as f:
        config = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(config_file))

    priors_src = os.path.join(base, config["priors_file"])
    priors_dst = os.path.join(draw_dir, "priors.csv")
    shutil.copy(priors_src, priors_dst)
    _apply_priors_updates(priors_dst, {
        _csv_row_name("condition_growth", "k_loc"): np.asarray(k, dtype=float),
        _csv_row_name("condition_growth", "m_loc"): np.asarray(m, dtype=float),
        _csv_row_name("condition_growth", "k_pinned"): 1.0,
        _csv_row_name("condition_growth", "m_pinned"): 1.0,
    }, cond_rep_labels=cond_labels)

    # the other sibling files stay where they are
    config["priors_file"] = os.path.relpath(priors_dst, draw_dir)
    for key in ("guesses_file", "library_file"):
        if config.get(key):
            config[key] = os.path.relpath(os.path.join(base, config[key]), draw_dir)
    out = os.path.join(draw_dir, "config.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return out


def final_loss(prefix):
    """Last logged loss (block median) of a fit, or NaN."""
    try:
        lines = open(f"{prefix}_losses.txt").read().strip().splitlines()
        return float(lines[-1].split(",")[1])
    except (OSError, IndexError, ValueError):
        return float("nan")


def pool(files, out_file):
    """Concatenate posterior files draw-wise (same keys and trailing shapes)."""
    handles = [h5py.File(f, "r") for f in files]
    try:
        keys = set(handles[0].keys())
        for h in handles[1:]:
            keys &= set(h.keys())
        dropped = sorted(set().union(*[set(h.keys()) for h in handles]) - keys)
        if dropped:
            print(f"pool: sites missing from some draws, left out: {dropped}")
        with h5py.File(out_file, "w") as out:
            total = 0
            for k in sorted(keys):
                arr = np.concatenate([np.asarray(h[k]) for h in handles], axis=0)
                out.create_dataset(k, data=arr)
                total = arr.shape[0]
            out.attrs["num_samples"] = total
            out.attrs["two_stage_draws"] = len(files)
    finally:
        for h in handles:
            h.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config_file", default="tfs_configure_config.yaml")
    parser.add_argument("--laplace_file", default="tfs_laplace.h5")
    parser.add_argument("--num_draws", type=int, default=10)
    parser.add_argument("--samples_per_draw", type=int, default=1000)
    parser.add_argument("--guide_type", default="component")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work_dir", default="two_stage")
    parser.add_argument("--out_prefix", default="tfs_posterior")
    parser.add_argument("--max_num_epochs", type=int, default=None,
                        help="cap on each conditional fit (default: "
                             "tfs-fit-model's)")
    args = parser.parse_args()

    with h5py.File(args.laplace_file, "r") as f:
        k_all = np.asarray(f["condition_growth_k"])
        m_all = np.asarray(f["condition_growth_m"])
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(k_all), size=args.num_draws, replace=False)

    orchestrator, _ = read_configuration(args.config_file)
    cond_labels = _condition_rep_labels(orchestrator)
    del orchestrator

    records, posteriors = [], []
    for d, i in enumerate(idx):
        draw_dir = os.path.join(args.work_dir, f"draw{d:02d}")
        k, m = k_all[i], m_all[i]
        cfg = draw_config(args.config_file, draw_dir, k, m, cond_labels)
        prefix = os.path.join(draw_dir, "fit")
        print(f">>> two-stage draw {d + 1}/{args.num_draws}: k={np.round(k, 5)} "
              f"m={np.round(m, 5)}", flush=True)
        fit_kwargs = {}
        if args.max_num_epochs is not None:
            fit_kwargs["max_num_epochs"] = args.max_num_epochs
        fit_model(cfg, seed=args.seed + 1 + d, out_prefix=prefix,
                  guide_type=args.guide_type, epoch_checkpoint_interval=None,
                  **fit_kwargs)
        post = os.path.join(draw_dir, "posterior")
        sample_posterior(cfg, f"{prefix}_checkpoint.pkl", out_prefix=post,
                         seed=args.seed + 1 + d,
                         num_posterior_samples=args.samples_per_draw)
        posteriors.append(f"{post}.h5")
        records.append(dict(draw=d, laplace_index=int(i),
                            final_loss=final_loss(prefix),
                            **{f"k_{j}": float(v) for j, v in enumerate(k)},
                            **{f"m_{j}": float(v) for j, v in enumerate(m)}))

    draws = pd.DataFrame(records)
    draws.to_csv(os.path.join(args.work_dir, "draws.csv"), index=False)
    spread = draws["final_loss"].max() - draws["final_loss"].min()
    print(f"Conditional fits' final losses span {spread:.0f} nats "
          f"(min {draws['final_loss'].min():.0f}). A span of hundreds to "
          f"thousands of nats flags Laplace draws of k, m far outside their "
          f"posterior (two-stage grid, 2026-09-28).", flush=True)
    pool(posteriors, f"{args.out_prefix}.h5")
    print(f"Pooled {len(posteriors)} conditional posteriors into "
          f"{args.out_prefix}.h5", flush=True)


if __name__ == "__main__":
    main()
