"""
Why the step-3.5 baseline/profile fits never converged and "blew up": the
per-step loss is 41k + n * 243k (n = draws crossing one binding step), and
ClippedAdam's elementwise clip makes Adam settle where gradient *signs*
balance, holding a large share of draws across the step.

Three checks at one checkpoint, from inside a run directory:

    cd congression_calibration_baseline/run_0005_lam1_0_seed1_single_off
    python ../../diagnosis/diag_clip.py checkpoints/0040000_checkpoint.pkl

1. events: per-draw binding log-likelihood; which observations carry the
   penalty, and the predicted theta there.
2. gradients: mean raw vs mean clipped (+/-1) gradient of the loss for the
   parameters that differ most between penalized and clean draws.
3. resume: --steps optimizer steps from the checkpoint with and without the
   clip (--lr, default 1e-3); share of penalized steps and mean loss per
   2000 steps.

Needs this worktree's src on PYTHONPATH (the editable install may point at
another checkout).
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.run_inference import (
    RunInference,
    adam_optimizer,
)

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--draws", type=int, default=40)
parser.add_argument("--steps", type=int, default=8000)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

orch, init_params = read_configuration("tfs_configure_config.yaml")
ri = RunInference(orch, seed=1)
svi, state = ri.restore_svi_from_checkpoint(args.checkpoint,
                                            init_params=init_params)
params = svi.get_params(state)
data = jax.device_put(orch.data)
# The training index (binding genotypes first, the rest shuffled); the
# library-ordered data.batch_idx pairs genotypes with the wrong rows.
batch = orch.get_batch(data, jnp.asarray(ri.model.get_random_idx()))


def penalized(losses):
    """Steps carrying at least one step-crossing penalty (~2.4e5)."""
    return losses > np.percentile(losses, 1) + 1.2e5


# --- 1. events -------------------------------------------------------------

lp, loc = [], []
for i in range(args.draws):
    key = jax.random.PRNGKey(i)
    g = handlers.trace(handlers.seed(
        handlers.substitute(orch.jax_model_guide, data=params), key)
    ).get_trace(priors=orch.priors, data=batch)
    m = handlers.trace(handlers.replay(handlers.seed(orch.jax_model, key), g)
                       ).get_trace(priors=orch.priors, data=batch)
    site = m["binding_obs"]
    fn = site["fn"]
    lp.append(np.asarray(fn.log_prob(site["value"])).ravel())  # masked
    loc.append(np.broadcast_to(np.asarray(fn.base_dist.loc),
                               site["value"].shape).ravel())
lp, loc = np.array(lp), np.array(loc)
obs = np.asarray(batch.binding.theta_obs).ravel()
shape = batch.binding.theta_obs.shape
print("\n1. binding log-likelihood per draw:",
      np.round(lp.sum(1)).astype(int).tolist())
for j in np.where(lp.min(0) < -1e4)[0]:
    idx = tuple(int(i) for i in np.unravel_index(j, shape))
    hit = lp[:, j] < -1e4
    print(f"   obs {idx} theta_obs {obs[j]:.4f}: penalized in {hit.mean():.0%} "
          f"of draws, predicted theta {np.round(loc[hit, j], 3)[:4]} "
          f"(clean draws ~{np.median(loc[~hit, j]):.4f})")

# --- 2. gradients ----------------------------------------------------------

u = svi.optim.get_params(state.optim_state)


def loss(p, key):
    return svi.loss.loss(key, svi.constrain_fn(p), orch.jax_model,
                         orch.jax_model_guide, priors=orch.priors, data=batch)


vg = jax.jit(jax.value_and_grad(loss))
L, G = [], []
for i in range(4 * args.draws):
    v, g = vg(u, jax.random.PRNGKey(1000 + i))
    L.append(float(v))
    G.append(g)
L = np.array(L)
ev = penalized(L)
print(f"\n2. {ev.mean():.0%} of {L.size} draws penalized; mean loss "
      f"{L.mean():.4g}, median {np.median(L):.4g}")
rows = []
for k in u:
    g = np.stack([np.asarray(x[k]).ravel() for x in G])
    if ev.all() or not ev.any():
        continue
    diff = np.abs(g[ev].mean(0) - g[~ev].mean(0))
    j = int(np.argmax(diff))
    rows.append((diff[j], k, j, g[:, j].mean(),
                 g[:, j].std() / np.sqrt(len(g)),
                 np.clip(g[:, j], -1, 1).mean(),
                 np.mean(np.abs(g) > 1)))
rows.sort(reverse=True)
print(f"   {'parameter[element]':44s} {'raw mean':>10s} {'+/- se':>8s} "
      f"{'clipped':>8s}   array's |grad| > 1 (clipped)")
for _, k, j, raw, se, clipped, frac in rows[:10]:
    print(f"   {k + f'[{j}]':44s} {raw:10.3g} {se:8.2g} {clipped:8.3f}   "
          f"{frac:.0%} of elements x draws")

# --- 3. resume with and without the clip -------------------------------------

for clip in (1.0, None):
    _, st = ri.restore_svi_from_checkpoint(args.checkpoint,
                                           init_params=init_params)
    svi.optim = adam_optimizer(args.lr, clip)

    def step(s, idx):
        return svi.update(s, priors=orch.priors,
                          data=orch.get_batch(data, idx))

    scan = jax.jit(lambda s, idx: jax.lax.scan(step, s, idx))
    losses = []
    for _ in range(args.steps // 250):
        st, lb = scan(st, jnp.asarray(ri.model.get_random_idx(num_batches=250)))
        losses.append(np.asarray(lb))
    losses = np.concatenate(losses)
    ev = penalized(losses)
    w = 2000
    print(f"\n3. resume, clip {clip}, step size {args.lr:g}:")
    print("   penalized share per 2000:",
          [round(float(ev[a:a + w].mean()), 2) for a in range(0, ev.size, w)])
    print("   mean loss per 2000:      ",
          [int(losses[a:a + w].mean()) for a in range(0, ev.size, w)])
