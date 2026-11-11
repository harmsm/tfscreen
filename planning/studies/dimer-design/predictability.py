"""
Predictive fall-off: how well does each fit predict θ for combinations of
k library mutations, k = 1, 2, 3, ...?

Per-term errors are the wrong criterion: the model has ridges the data
never see, and a term left unresolved along one costs nothing unless it
changes θ. What matters is whether the fitted constants predict genotypes
nobody measured, and how fast that degrades as mutations are combined.

A combination's truth is wt plus the sum of its mutations' true
constants plus its pairs' true epistasis. The simulation draws epistasis
only for the library's pairs (tfs_sim_pair_parameters.csv); every other
pair gets, per term, a draw from the pooled values the simulation did
draw (`truth_pair_table`, seeded by run), so in an ep run every pair is
epistatic with the simulation's own distribution. Without epistasis (pair
horseshoe scale 1e-9) those values are ~1e-9 and the truth is additive.

Each fit's prediction is the same sum over its posterior draws: wt and
per-mutation constants drawn jointly, plus pair terms (unless
--no_fit_pairs): the fit's own pair shift where it measured the pair,
else a draw from its pair horseshoe at that posterior draw's tau and c2,
so unmeasured pairs widen the interval the way the model says they
should. A fit's θ uses its own model (npz `terms` and `variant`): the
no-monomer fits (*_m0) predict with the no-monomer species model and are
scored against the full-model truth. Draw files without them (the
pilots') are the full model with no pair terms. Everything is scored in
the reference background at TF mid (held at 10 uM in every fit), at the
screen's eight IPTG concentrations.

Two test sets, both from the mutations every fit shares (backgrounds' own
mutations excluded), combinations at distinct residues:

  random      every single at k = 1, else NUM_COMBOS random sets; the
              natural mix of curve classes. Most k = 1 sets were
              measured; almost no k >= 2 set was.
  stratified  up to PER_CLASS combinations of each true curve class at
              each k: drawn from POOL random sets, and, for a class the
              pool leaves short, from a beam search on the truth toward
              that class (BEAM wide). A test set chosen by the truth, so
              rare classes (inverted curves, shifted midpoints) can be
              scored at all. Per run, since the truth differs by slope.

Per fit and k (random set): rmse of the posterior median against the
truth over every concentration; r2 = 1 - mse / var(truth θ); cov95, the
fraction of points inside the 95% interval; ok, the fraction of
combinations predicted within 0.1 at every concentration. Two baselines:
`table`, the ΔΔG table alone (c = 1 projection, no data: what the
structural prior says), and `wt`, the wt curve for everything.

Curve classes (`curve_class`, from the curve at the 8 concentrations):
repressor (θ falls by DROP or more, midpoint within SHIFT log10 units of
wt's), shifted (a repressor whose midpoint is SHIFT or more from wt's),
inverted (θ rises by DROP or more), stuck_on (flat, mean θ >= LEVEL:
bound, not induced) and off (flat, mean θ < LEVEL: no repression). Per
fit, k and class: n (true members), rmse, ok and cov95 over the true
members; recall, the fraction of true members predicted in the class;
n_pred and precision, the fraction of combinations predicted in the class
that truly are. Precision on the random set is the design-relevant number
("of what the model calls a repressor, how much is one"); on the
stratified set it reflects the set's equal class mix, not nature's.

    python predictability.py --draws DIR --sims DIR

DIR names: the draws are `{run}__fit_{arm}.npz` from extract_draws.py
(older ones: d_*, wt_* and 'mutations' only); the sims `{run}/sim/`.
Writes results/predictability.csv (random set, overall),
results/predictability_by_class.csv and
results/predictability_stratified.csv (per class), and prints tables.
"""

import argparse
import glob
import math
import os
import zlib

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
PER_CLASS = 100
POOL = 20000
BEAM = 200
LN_TF_MID = pb.LN_TF["mid"]
BACKGROUND_MUTS = {m for muts, _ in pb.NARRATIVE for m in muts}

CLASSES = ("repressor", "shifted", "inverted", "stuck_on", "off")
DROP = 0.3      # θ(0) - θ(1 mM) for a response
LEVEL = 0.5     # mean θ splitting stuck_on from off among flat curves
SHIFT = 0.5     # log10 midpoint shift from wt that makes a repressor shifted
# log10 mM of each concentration, the 0 point a decade below the lowest.
LOG_CONC = np.log10(np.where(pb.CONC_MM > 0, pb.CONC_MM,
                             pb.CONC_MM[pb.CONC_MM > 0].min() / 10))


_THETA = {}


def _theta_fn(variant):
    key = (variant.monomers, variant.unfolded_dimer, variant.unfolded_monomer,
           variant.second_engagement)
    if key not in _THETA:
        names = sp.active_terms(variant)

        def f(vec):
            terms = {t: vec[..., i, None] for i, t in enumerate(names)}
            return sp.theta(terms, jnp.asarray(pb.CONC_MM * 1e-3),
                            jnp.exp(LN_TF_MID), variant)
        _THETA[key] = (jax.jit(jax.vmap(f)), len(names))
    return _THETA[key]


def theta(vecs, variant=pb.VARIANT):
    """θ at the eight concentrations, (..., terms) -> (..., C), in the
    order of species.active_terms(variant)."""
    fn, n = _theta_fn(variant)
    shape = vecs.shape[:-1]
    flat = np.asarray(fn(jnp.asarray(vecs.reshape(-1, n))))
    return flat.reshape(*shape, len(pb.CONC_MM))


def pair_index(idx):
    """(a, b) column positions of every pair in a (n, k) combination."""
    k = idx.shape[1]
    return [(a, b) for a in range(k) for b in range(a + 1, k)]


def truth_pair_table(sim, lib, rng):
    """(L, L, 12) true epistasis for every pair of library singles: the
    simulation's value for its pairs, otherwise a draw per term from the
    pooled simulated values."""
    L, K = len(lib), len(pb.TERMS)
    E = np.zeros((L, L, K))
    path = os.path.join(sim, "tfs_sim_pair_parameters.csv")
    if not os.path.exists(path):
        return E
    p = pd.read_csv(path)
    vals = p[[f"epi_{t}" for t in pb.TERMS]].to_numpy()
    upper = np.triu_indices(L, 1)
    E[upper] = rng.choice(vals.ravel(), (len(upper[0]), K))
    pos = {m: i for i, m in enumerate(lib)}
    for label, v in zip(p["pair"], vals):
        a, b = label.split("/")
        if a in pos and b in pos:
            i, j = sorted((pos[a], pos[b]))
            E[i, j] = v
    return E + np.transpose(E, (1, 0, 2))


def truth_theta(idx, wt_vec, d_true, E):
    vec = wt_vec + d_true[idx].sum(axis=1)
    for a, b in pair_index(idx):
        vec = vec + E[idx[:, a], idx[:, b]]
    return theta(vec)


def fit_variant(z):
    if "variant" not in z.files:
        return pb.VARIANT, list(SHORT)
    m, ud, um, se = (bool(x) for x in z["variant"])
    return (sp.Variant(monomers=m, unfolded_dimer=ud, unfolded_monomer=um,
                       second_engagement=se), [str(t) for t in z["terms"]])


def fit_pair_draws(z, terms, take, lib, idx, rng):
    """(draws, n, terms) summed pair terms of each combination: the fit's
    shift for a measured pair, else a horseshoe draw at that posterior
    draw's tau and c2."""
    R, n, K = len(take), len(idx), len(terms)
    out = np.zeros((R, n, K))
    pairs = pair_index(idx)
    if not pairs:
        return out
    pos = {m: i for i, m in enumerate(lib)}
    measured = {}
    for p, label in enumerate(z["pairs"].astype(str)):
        a, b = label.split("/")
        if a in pos and b in pos:
            measured[tuple(sorted((pos[a], pos[b])))] = p
    epi = np.stack([z[f"epi_{t}"][take] for t in terms], axis=-1)   # R,P,K
    tau = z["epi_tau"][take][:, None, None]
    c2 = z["epi_c2"][take][:, None, None]
    for a, b in pairs:
        i, j = np.minimum(idx[:, a], idx[:, b]), np.maximum(idx[:, a], idx[:, b])
        p = np.array([measured.get((x, y), -1) for x, y in zip(i, j)])
        lam = np.abs(rng.standard_cauchy((R, n, K)))
        draw = rng.standard_normal((R, n, K)) * tau * np.sqrt(
            c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
        known = p >= 0
        if known.any():
            draw[:, known, :] = epi[:, p[known], :]
        out += draw
    return out


def midpoint(th):
    """log10 mM where a falling curve crosses halfway between its ends
    (linear in log concentration); NaN where it does not fall."""
    half = (th[..., :1] + th[..., -1:]) / 2
    below = th <= half
    i = np.argmax(below, axis=-1)
    i = np.clip(i, 1, th.shape[-1] - 1)
    t0 = np.take_along_axis(th, (i - 1)[..., None], -1)[..., 0]
    t1 = np.take_along_axis(th, i[..., None], -1)[..., 0]
    x0, x1 = LOG_CONC[i - 1], LOG_CONC[i]
    h = half[..., 0]
    f = np.where(t0 != t1, (t0 - h) / np.where(t0 != t1, t0 - t1, 1), 0.5)
    mid = x0 + np.clip(f, 0, 1) * (x1 - x0)
    return np.where(th[..., 0] - th[..., -1] > 0, mid, np.nan)


def curve_class(th, wt_mid):
    """Class index into CLASSES per curve, (..., C) -> (...)."""
    drop = th[..., 0] - th[..., -1]
    shifted = np.abs(midpoint(th) - wt_mid) >= SHIFT
    flat_on = th.mean(axis=-1) >= LEVEL
    return np.where(drop >= DROP, np.where(shifted, 1, 0),
                    np.where(drop <= -DROP, 2, np.where(flat_on, 3, 4)))


def class_score(th, i, wt_mid):
    """Larger is further into class i; the beam search's objective."""
    drop = th[..., 0] - th[..., -1]
    if i == 1:
        return np.minimum(drop, 0.6) + np.abs(
            np.nan_to_num(midpoint(th) - wt_mid))
    if i == 2:
        return -drop
    if i == 3:
        return th.mean(axis=-1) - np.abs(drop)
    if i == 4:
        return -th.max(axis=-1)
    return drop - np.abs(np.nan_to_num(midpoint(th) - wt_mid))


def combos(muts, k, rng, num=NUM_COMBOS):
    if k == 1:
        return np.arange(len(muts))[:, None]     # every single
    res = np.array([int(m[1:-1]) for m in muts])
    out, seen = [], set()
    while len(out) < num:
        idx = rng.choice(len(muts), k, replace=False)
        key = tuple(sorted(idx))
        if len(set(res[idx])) < k or key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return np.array(out)


def beam(i, tfun, res, k_max, wt_mid):
    """{k: combos (n, k)} found by a beam search on the truth (tfun: idx ->
    true θ) toward class i: keep the BEAM best sets at each depth by
    class_score, extend each by every mutation at a new residue."""
    L = len(res)
    cur = [(j,) for j in range(L)]
    found = {}
    for k in range(1, k_max + 1):
        idx = np.array(cur)
        th = tfun(idx)
        found[k] = idx[curve_class(th, wt_mid) == i]
        if k == k_max:
            break
        top = idx[np.argsort(-class_score(th, i, wt_mid))[:BEAM]]
        nxt = set()
        for s in map(tuple, top.tolist()):
            used = set(res[list(s)])
            for j in range(L):
                if res[j] not in used:
                    nxt.add(tuple(sorted(s + (j,))))
        cur = sorted(nxt)
    return found


def stratified_sets(lib, tfun, wt_mid, rng):
    """{k: (idx (n, k), true class (n,))}, up to PER_CLASS per class."""
    res = np.array([int(m[1:-1]) for m in lib])
    short = {}
    out = {}
    for k in KS:
        pool = combos(lib, k, rng, num=min(POOL, math.comb(len(lib), k) // 3)) \
            if k > 1 \
            else np.arange(len(lib))[:, None]
        cls = curve_class(tfun(pool), wt_mid)
        parts = {i: pool[cls == i] for i in range(len(CLASSES))}
        out[k] = parts
        for i, p in parts.items():
            if len(p) < PER_CLASS:
                short.setdefault(i, []).append(k)
    for i, ks in short.items():
        found = beam(i, tfun, res, max(ks), wt_mid)
        for k in ks:
            have = {tuple(s) for s in out[k][i]}
            extra = [s for s in found.get(k, []) if tuple(s) not in have]
            if extra:
                out[k][i] = np.concatenate([out[k][i].reshape(-1, k),
                                            np.array(extra)])
    sets = {}
    for k, parts in out.items():
        idx, cls = [], []
        for i, p in parts.items():
            if len(p) > PER_CLASS:
                p = p[rng.choice(len(p), PER_CLASS, replace=False)]
            idx += [np.asarray(p).reshape(-1, k)]
            cls += [np.full(len(p), i)]
        sets[k] = (np.concatenate(idx), np.concatenate(cls))
    return sets


def scores(pred_med, truth, lo=None, hi=None):
    err = pred_med - truth
    row = {"rmse": float(np.sqrt(np.mean(err ** 2))),
           "r2": float(1 - np.mean(err ** 2) / np.var(truth)),
           "ok": float(np.mean(np.max(np.abs(err), axis=-1) <= 0.1))}
    if lo is not None:
        row["cov95"] = float(np.mean((truth >= lo) & (truth <= hi)))
    return row


def class_rows(base, pred_med, truth, wt_mid, lo=None, hi=None):
    """Per class: n, rmse, ok, cov95 (true members), recall, n_pred,
    precision."""
    tc, pc = curve_class(truth, wt_mid), curve_class(pred_med, wt_mid)
    rows = []
    for i, name in enumerate(CLASSES):
        sel, called = tc == i, pc == i
        if not sel.any() and not called.any():
            continue
        row = dict(base, true_class=name, n=int(sel.sum()),
                   n_pred=int(called.sum()),
                   precision=float(np.mean(tc[called] == i))
                   if called.any() else np.nan)
        if sel.any():
            err = pred_med[sel] - truth[sel]
            row.update(rmse=float(np.sqrt(np.mean(err ** 2))),
                       ok=float(np.mean(np.max(np.abs(err), axis=-1) <= 0.1)),
                       recall=float(np.mean(pc[sel] == i)))
            if lo is not None:
                row["cov95"] = float(np.mean((truth[sel] >= lo[sel])
                                             & (truth[sel] <= hi[sel])))
        rows.append(row)
    return rows


def pooled(cl, metric, weight):
    """n-weighted mean over runs of the same slope."""
    d = cl.dropna(subset=[metric]).copy()
    d = d[d[weight] > 0]
    d["v"] = d[metric] * d[weight]
    g = d.groupby(["slope", "true_class", "arm", "k"])
    return (g["v"].sum() / g[weight].sum()).unstack("k")


def main(draws_dir, sims_dir, fit_pairs=True, out_dir=None):
    out_dir = out_dir or os.path.join(HERE, "results")
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
          f"{NUM_COMBOS} combinations per k", flush=True)

    table = pd.read_csv(pb.TABLE)
    table_d = pb.mutation_constants(table, 1.0)
    rows, crows, srows = [], [], []
    cache = {}
    for (run, arm), z in sorted(fits.items()):
        sim = os.path.join(sims_dir, run, "sim")
        if run not in cache:
            mp = pd.read_csv(os.path.join(sim, "tfs_sim_mutation_parameters.csv"),
                             index_col="mutation")
            wt = pd.read_csv(os.path.join(sim, "tfs_sim_parameters.csv"),
                             index_col="genotype").loc["wt"]
            wt_vec = np.array([wt[t] for t in pb.TERMS])
            d_true = mp.loc[lib, [f"d_{t}" for t in pb.TERMS]].to_numpy()
            d_tab = table_d.loc[lib, list(pb.TERMS)].to_numpy()
            wt_curve = theta(wt_vec[None])[0]
            wt_mid = float(midpoint(wt_curve))
            seed = zlib.crc32(run.encode())
            E = truth_pair_table(sim, lib, np.random.default_rng(seed + 1))

            def tfun(idx, wt_vec=wt_vec, d_true=d_true, E=E):
                return truth_theta(idx, wt_vec, d_true, E)
            strat = stratified_sets(lib, tfun, wt_mid,
                                    np.random.default_rng(seed))
            truth = {k: tfun(s) for k, s in sets.items()}
            struth = {k: tfun(s) for k, (s, _) in strat.items()}
            cache[run] = (truth, struth, strat, wt_mid)
            for name, base in (("table", lambda s: wt_vec + d_tab[s].sum(axis=1)),
                               ("wt", None)):
                for k in KS:
                    for tr, s, out, cout in ((truth[k], sets[k], rows, crows),
                                             (struth[k], strat[k][0], None, srows)):
                        pred = theta(base(s)) if base else \
                            np.broadcast_to(wt_curve, tr.shape)
                        b = {"run": run, "arm": name, "k": k}
                        if out is not None:
                            out.append(dict(b, **scores(pred, tr)))
                        cout += class_rows(b, pred, tr, wt_mid)
            n = {k: np.bincount(c, minlength=len(CLASSES))
                 for k, (_, c) in strat.items()}
            print(f"  {run}: stratified per class (k x "
                  f"{'/'.join(CLASSES)}): "
                  + "; ".join(f"{k}: " + " ".join(map(str, v))
                              for k, v in n.items()), flush=True)
        truth, struth, strat, wt_mid = cache[run]

        variant, terms = fit_variant(z)
        pairs_on = fit_pairs and "epi_tau" in z.files
        prng = np.random.default_rng(zlib.crc32(f"{run} {arm}".encode()))
        idx = {m: i for i, m in enumerate(z["mutations"].astype(str))}
        cols = [idx[m] for m in lib]
        n = z[f"d_{terms[0]}"].shape[0]
        take = np.linspace(0, n - 1, min(NUM_DRAWS, n)).astype(int)
        D = np.stack([z[f"d_{t}"][take][:, cols] for t in terms], axis=-1)
        W = np.stack([z[f"wt_{t}"][take] for t in terms], axis=-1)
        for k in KS:
            for s, tr, out, cout in ((sets[k], truth[k], rows, crows),
                                     (strat[k][0], struth[k], None, srows)):
                vec = W[:, None, :] + D[:, s, :].sum(axis=2)
                if pairs_on:
                    vec = vec + fit_pair_draws(z, terms, take, lib, s, prng)
                th = theta(vec, variant)
                q = np.quantile(th, [0.025, 0.5, 0.975], axis=0)
                b = {"run": run, "arm": arm, "k": k}
                if out is not None:
                    out.append(dict(b, **scores(q[1], tr, q[0], q[2])))
                cout += class_rows(b, q[1], tr, wt_mid, q[0], q[2])
        print(f"  {run} {arm} ({len(terms)} terms"
              + (", pair terms" if pairs_on else "") + ")", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out = pd.DataFrame(rows).drop_duplicates(["run", "arm", "k"])
    out.to_csv(os.path.join(out_dir, "predictability.csv"), index=False)
    pd.set_option("display.width", 220)
    for metric in ("rmse", "r2", "ok", "cov95"):
        print(f"\n{metric} by k (random set):")
        print(out.pivot_table(index=["run", "arm"], columns="k", values=metric)
              .round(3).to_string())

    for name, data in (("by_class", crows), ("stratified", srows)):
        cl = pd.DataFrame(data).drop_duplicates(["run", "arm", "k", "true_class"])
        cl.to_csv(os.path.join(out_dir,
                               f"predictability_{name}.csv"), index=False)
        cl["slope"] = cl["run"].str.split("_").str[1]
        label = "random set" if name == "by_class" else "stratified set"
        frac = (cl[cl["arm"] == "wt"].groupby(["slope", "k", "true_class"])["n"]
                .sum().unstack("true_class"))
        print(f"\ntrue class shares by k ({label}):")
        print(frac.div(frac.sum(axis=1), axis=0).round(2).to_string())
        for metric, w in (("recall", "n"), ("precision", "n_pred"),
                          ("rmse", "n")):
            if name == "stratified" and metric == "precision":
                continue
            print(f"\n{metric} by true class and k ({label}, pooled over "
                  "designs):")
            print(pooled(cl, metric, w).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", default=os.path.join(HERE, "results", "draws"))
    ap.add_argument("--sims", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=None,
                    help="output directory (default results/)")
    ap.add_argument("--no_fit_pairs", action="store_true",
                    help="leave the fits' pair terms out of the predictions")
    a = ap.parse_args()
    main(a.draws, a.sims, fit_pairs=not a.no_fit_pairs, out_dir=a.out)
