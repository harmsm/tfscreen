"""
Predictive fall-off: how well does each fit predict θ for combinations of
k library mutations, k = 1, 2, 3, ...?

Per-term errors are the wrong criterion: the model has ridges the data
never see, and a term left unresolved along one costs nothing unless it
changes θ. What matters is whether the fitted constants predict genotypes
nobody measured, and how fast that degrades as mutations are combined.

The truth has no pair epistasis (the pair horseshoe's tau is held at
1e-9), so it is additive in ln K: a combination's truth is wt plus the sum
of its mutations' true constants. Each fit's prediction is the same sum
over its posterior draws (wt and per-mutation constants drawn jointly; the
fit's pair terms are left out, since almost no combination's pairs were
measured). Everything is scored in the reference background at TF mid
(held at 10 uM in every fit), at the screen's eight IPTG concentrations.

Combinations: every single at k = 1, else NUM_COMBOS random sets of k library singles at distinct
residues, from the mutations every fit shares (backgrounds' own mutations
excluded), the same sets for every fit. Most k = 1 sets were measured;
almost no k >= 2 set was.

Per fit and k: rmse of the posterior median against the truth over every
concentration; r2 = 1 - mse / var(truth θ); cov95, the fraction of points
inside the 95% interval; ok, the fraction of combinations predicted within
0.1 at every concentration. Two baselines per table slope: `table`, the
ΔΔG table alone (c = 1 projection, no data: what the structural prior
says), and `wt`, predicting the wt curve for everything (no effect).

    python predictability.py --draws DIR --sims DIR

DIR names: the draws are `{run}__fit_{arm}.npz` (theta_d_ln_K_* and
theta_ln_K_*_wt posterior arrays, 'mutations'); the sims `{run}/sim/`.
Writes results/predictability.csv and prints the fall-off tables.
"""

import argparse
import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from tfscreen.tfmodel.generative.components.theta.thermo.dimer import (
    species as sp)

import pick_backgrounds as pb

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = ["h_l", "h_o", "l_o", "h_oii", "l_oii", "d_h", "m_h_l", "m_h_o",
         "m_l_o", "m_u", "h_e", "l_e"]
assert tuple(f"ln_K_{t}" for t in SHORT) == pb.TERMS
KS = (1, 2, 3, 4, 5, 6)
NUM_COMBOS = 300
NUM_DRAWS = 200
LN_TF_MID = pb.LN_TF["mid"]
BACKGROUND_MUTS = {m for muts, _ in pb.NARRATIVE for m in muts}


def _theta(vec):
    terms = {t: vec[..., i, None] for i, t in enumerate(pb.TERMS)}
    return sp.theta(terms, jnp.asarray(pb.CONC_MM * 1e-3),
                    jnp.exp(LN_TF_MID), pb.VARIANT)


_theta_b = jax.jit(jax.vmap(_theta))


def theta(vecs):
    shape = vecs.shape[:-1]
    flat = np.asarray(_theta_b(jnp.asarray(vecs.reshape(-1, len(SHORT)))))
    return flat.reshape(*shape, len(pb.CONC_MM))


def combos(muts, k, rng):
    if k == 1:
        return np.arange(len(muts))[:, None]     # every single
    res = np.array([int(m[1:-1]) for m in muts])
    out, seen = [], set()
    while len(out) < NUM_COMBOS:
        idx = rng.choice(len(muts), k, replace=False)
        key = tuple(sorted(idx))
        if len(set(res[idx])) < k or key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return np.array(out)


def scores(pred_med, truth, lo=None, hi=None):
    err = pred_med - truth
    row = {"rmse": float(np.sqrt(np.mean(err ** 2))),
           "r2": float(1 - np.mean(err ** 2) / np.var(truth)),
           "ok": float(np.mean(np.max(np.abs(err), axis=-1) <= 0.1))}
    if lo is not None:
        row["cov95"] = float(np.mean((truth >= lo) & (truth <= hi)))
    return row


def main(draws_dir, sims_dir):
    files = sorted(glob.glob(os.path.join(draws_dir, "*__fit_*.npz")))
    fits = {}
    for f in files:
        run, arm = os.path.basename(f)[:-4].split("__fit_")
        fits[(run, arm)] = np.load(f, allow_pickle=True)
    shared = None
    for z in fits.values():
        m = set(z["mutations"].astype(str))
        shared = m if shared is None else shared & m
    lib = sorted(m for m in shared if m not in BACKGROUND_MUTS
                 and pb.AMPLICON[0] <= int(m[1:-1]) <= pb.AMPLICON[1])
    rng = np.random.default_rng(20261107)
    sets = {k: combos(lib, k, rng) for k in KS}
    print(f"{len(fits)} fits, {len(lib)} library mutations, "
          f"{NUM_COMBOS} combinations per k")

    table = pd.read_csv(pb.TABLE)
    table_d = pb.mutation_constants(table, 1.0)
    rows = []
    truth_cache = {}
    for (run, arm), z in sorted(fits.items()):
        sim = os.path.join(sims_dir, run, "sim")
        if run not in truth_cache:
            mp = pd.read_csv(os.path.join(sim, "tfs_sim_mutation_parameters.csv"),
                             index_col="mutation")
            wt = pd.read_csv(os.path.join(sim, "tfs_sim_parameters.csv"),
                             index_col="genotype").loc["wt"]
            wt_vec = np.array([wt[t] for t in pb.TERMS])
            d_true = mp.loc[lib, [f"d_{t}" for t in pb.TERMS]].to_numpy()
            d_tab = table_d.loc[lib, list(pb.TERMS)].to_numpy()
            truth = {k: theta(wt_vec + d_true[s].sum(axis=1))
                     for k, s in sets.items()}
            tab = {k: theta(wt_vec + d_tab[s].sum(axis=1))
                   for k, s in sets.items()}
            wt_curve = theta(wt_vec[None])[0]
            truth_cache[run] = (truth, tab, wt_curve)
            slope = run.split("_")[1]
            for k in KS:
                rows.append({"run": run, "arm": "table", "k": k,
                             **scores(tab[k], truth[k])})
                rows.append({"run": run, "arm": "wt", "k": k,
                             **scores(np.broadcast_to(wt_curve, truth[k].shape),
                                      truth[k])})
        truth, _, _ = truth_cache[run]

        idx = {m: i for i, m in enumerate(z["mutations"].astype(str))}
        cols = [idx[m] for m in lib]
        n = z["d_h_l"].shape[0]
        take = np.linspace(0, n - 1, min(NUM_DRAWS, n)).astype(int)
        D = np.stack([z[f"d_{t}"][take][:, cols] for t in SHORT], axis=-1)
        W = np.stack([z[f"wt_{t}"][take] for t in SHORT], axis=-1)
        for k, s in sets.items():
            vec = W[:, None, :] + D[:, s, :].sum(axis=2)       # (draws, combos, 12)
            th = theta(vec)
            q = np.quantile(th, [0.025, 0.5, 0.975], axis=0)
            rows.append({"run": run, "arm": arm, "k": k,
                         **scores(q[1], truth[k], q[0], q[2])})
        print(f"  {run} {arm}", flush=True)

    out = pd.DataFrame(rows).drop_duplicates(["run", "arm", "k"])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out.to_csv(os.path.join(HERE, "results", "predictability.csv"), index=False)
    pd.set_option("display.width", 220)
    for metric in ("rmse", "r2", "ok", "cov95"):
        print(f"\n{metric} by k:")
        print(out.pivot_table(index=["run", "arm"], columns="k", values=metric)
              .round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", default=os.path.join(HERE, "results", "draws"))
    ap.add_argument("--sims", default=os.path.join(HERE, "runs"))
    a = ap.parse_args()
    main(a.draws, a.sims)
