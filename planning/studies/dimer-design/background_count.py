"""
How many backgrounds help? Predicted θ error for unmeasured combinations
as backgrounds are added one at a time, in several orders and against
several prediction targets.

The picker (pick_backgrounds.py) found a design of two backgrounds about
as good as five: the narrative's two D149M backgrounds carried nearly all
of its gain. That may be a property of the target, not the system. Its
target was one curve, combinations predicted at the reference at TF mid,
which has only a few shape features for the backgrounds to pin. Here the
same Laplace risk (independent per-mutation posteriors, the c = 1 table
as truth, per-tube σ SIGMA) is scored against three targets:

  ref_mid   combinations at the reference, TF mid (the picker's target)
  ref_lohi  combinations at the reference, TF lo and hi
  held_out  combinations in NUM_HELD_OUT readable single-mutant
            backgrounds never offered as candidates, each at a random
            TF level (prediction in new genetic contexts)

and the backgrounds are added in four orders: greedy on ref_mid, greedy
on held_out, the narrative's five (best first), and random readable
candidates (NUM_SHUFFLES shuffles, median). Each background is a 20-tube
block at its TF level. The score is the predicted rmse, averaged over
k in pb.COMBO_KS; per-k values are written too.

    ~/miniconda3/bin/python background_count.py [--c 1.0] [--sigma 0.1]

Writes results/background_count_c1.csv and prints the curves.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd

import pick_backgrounds as pb

MAX_BACKGROUNDS = 10
NUM_HELD_OUT = 20
NUM_SHUFFLES = 5
GREEDY_POOL = 600       # best candidates by gain alone, searched greedily


def draw_combos(lib_res, rng):
    out = {}
    for k in pb.COMBO_KS:
        idx, seen = [], set()
        while len(idx) < pb.NUM_COMBOS:
            i = tuple(sorted(rng.choice(len(lib_res), k, replace=False)))
            if i in seen or len(set(lib_res[list(i)])) < k:
                continue
            seen.add(i)
            idx.append(i)
        out[k] = np.array(idx)
    return out


def jacobians(contexts, lib, combos):
    """{k: (B, n, C, 12)} over prediction contexts (base vector, level)."""
    out = {}
    for k, idx in combos.items():
        s = lib[idx].sum(axis=1)
        out[k] = np.stack([np.asarray(pb._jac_b(
            pb.jnp.asarray(base[None, :] + s), pb.LN_TF[level]))
            for base, level in contexts])
    return out


class Risk:
    def __init__(self, combos, targets):
        self.combos, self.targets = combos, targets

    def per_k(self, F):
        cov = np.linalg.inv(F)
        out = {}
        for k, idx in self.combos.items():
            C = cov[idx].sum(axis=1)
            for name, J in self.targets.items():
                v = np.einsum("bnci,nij,bncj->bnc", J[k], C, J[k])
                out[(name, k)] = float(np.sqrt(np.mean(v)))
        return out

    def mean(self, F, target):
        cov = np.linalg.inv(F)
        r = []
        for k, idx in self.combos.items():
            C = cov[idx].sum(axis=1)
            J = self.targets[target][k]
            r.append(np.sqrt(np.mean(np.einsum("bnci,nij,bncj->bnc", J, C, J))))
        return float(np.mean(r))


def main(c=1.0, sigma=0.1):
    table = pd.read_csv(pb.TABLE)
    table = table[table["mut"].str[-1] != "*"].reset_index(drop=True)
    d = pb.mutation_constants(table, c)
    wt = np.array([pb.WT[t] for t in pb.TERMS])
    res = table["mut"].str[1:-1].astype(int).to_numpy()

    lib_muts = [m for m, r in zip(table["mut"], res)
                if pb.AMPLICON[0] + 2 <= r <= pb.AMPLICON[1]]
    lib_muts = sorted(np.random.default_rng(0).choice(lib_muts, 120, replace=False))
    lib = d.loc[lib_muts].to_numpy()
    lib_res = np.array([int(m[1:-1]) for m in lib_muts])
    combos = draw_combos(lib_res, np.random.default_rng(1))

    def bg_vec(muts):
        return wt + d.loc[list(muts)].to_numpy().sum(axis=0)

    def readable(vecs, level):
        th = pb.curves(vecs, level)
        return (th[:, 0] - th[:, -1] >= pb.MIN_DROP) & (th[:, 0] >= pb.MIN_THETA0)

    outside = [(m,) for m, r in zip(table["mut"], res)
               if r < pb.AMPLICON[0] or r > pb.AMPLICON[1]]
    rng = np.random.default_rng(2)

    # Held-out contexts: readable singles at a random level, never candidates.
    held, held_muts = [], set()
    for i in rng.permutation(len(outside)):
        m = outside[i]
        level = rng.choice(list(pb.LN_TF))
        if readable(bg_vec(m)[None], level)[0]:
            held.append((bg_vec(m), level))
            held_muts.add(m[0])
        if len(held) == NUM_HELD_OUT:
            break
    targets = jacobians([(wt, "mid")], lib, combos)
    targets = {"ref_mid": targets,
               "ref_lohi": jacobians([(wt, "lo"), (wt, "hi")], lib, combos),
               "held_out": jacobians(held, lib, combos)}
    risk = Risk(combos, targets)

    F0 = np.eye(len(pb.TERMS))[None] / pb.PRIOR_SD ** 2 + \
        pb.info(wt, lib, "mid", pb.REF_TUBES, sigma)

    # Candidates, as in pick_backgrounds: readable singles, then doubles
    # of the best 40 singles (by ref_mid gain), at every level.
    cand = []

    def add(muts_list, level):
        vecs = np.array([bg_vec(m) for m in muts_list])
        ok = readable(vecs, level)
        for m, v in zip(np.array(muts_list, dtype=object)[ok], vecs[ok]):
            cand.append(dict(muts=tuple(m), level=level,
                             G=pb.info(v, lib, level, pb.BLOCK_TUBES, sigma)))

    singles = [m for m in outside if m[0] not in held_muts]
    for level in pb.LN_TF:
        add(singles, level)
    base = {t: risk.mean(F0, t) for t in targets}
    for x in cand:
        x["gain_ref_mid"] = base["ref_mid"] - risk.mean(F0 + x["G"], "ref_mid")
    top, seen = [], set()
    for x in sorted(cand, key=lambda x: -x["gain_ref_mid"]):
        r = int(x["muts"][0][1:-1])
        if r not in seen:
            seen.add(r)
            top.append(x["muts"])
        if len(top) == pb.TOP_SINGLES:
            break
    pairs = [a + b for a, b in itertools.combinations(top, 2)]
    n_single = len(cand)
    for level in pb.LN_TF:
        add(pairs, level)
    for x in cand[n_single:]:
        x["gain_ref_mid"] = base["ref_mid"] - risk.mean(F0 + x["G"], "ref_mid")
    for x in cand:
        x["gain_held_out"] = base["held_out"] - risk.mean(F0 + x["G"], "held_out")
    print(f"{len(cand)} candidates, {NUM_HELD_OUT} held-out contexts", flush=True)

    def curve(order_name, picks):
        rows, F = [], F0.copy()
        for n in range(len(picks) + 1):
            if n:
                F = F + picks[n - 1]["G"]
            pk = risk.per_k(F)
            for (t, k), v in pk.items():
                rows.append({"order": order_name, "n": n, "target": t, "k": k,
                             "rmse": v,
                             "added": "/".join(picks[n - 1]["muts"]) + " "
                             + picks[n - 1]["level"] if n else ""})
        return rows

    def greedy(target):
        pool = sorted(cand, key=lambda x: -x[f"gain_{target}"])[:GREEDY_POOL]
        F, picks, used = F0.copy(), [], set()
        for _ in range(MAX_BACKGROUNDS):
            best = min((x for x in pool if not set(x["muts"]) & used),
                       key=lambda x: risk.mean(F + x["G"], target))
            picks.append(best)
            used |= set(best["muts"])
            F = F + best["G"]
        return picks

    rows = []
    for t in ("ref_mid", "held_out"):
        rows += curve(f"greedy_{t}", greedy(t))
        print(f"greedy {t} done", flush=True)

    nar = []
    for muts, level in pb.NARRATIVE:
        v = bg_vec(muts)
        nar.append(dict(muts=muts, level=level,
                        G=pb.info(v, lib, level, pb.BLOCK_TUBES, sigma)))
    nar.sort(key=lambda x: risk.mean(F0 + x["G"], "ref_mid"))
    rows += curve("narrative", nar)

    shuf = []
    for s in range(NUM_SHUFFLES):
        order = np.random.default_rng(100 + s).permutation(len(cand))
        picks, used = [], set()
        for i in order:
            if not set(cand[i]["muts"]) & used:
                picks.append(cand[i])
                used |= set(cand[i]["muts"])
            if len(picks) == MAX_BACKGROUNDS:
                break
        shuf += [dict(r, order=f"random_{s}") for r in curve("random", picks)]
    rows += shuf

    out = pd.DataFrame(rows)
    out["c"], out["sigma"] = c, sigma
    tag = f"c{c:g}".replace(".", "")
    os.makedirs(pb.OUT, exist_ok=True)
    out.to_csv(os.path.join(pb.OUT, f"background_count_{tag}.csv"), index=False)

    # Mean over k; random orders pooled by median.
    m = out.groupby(["order", "n", "target"], as_index=False)["rmse"].mean()
    m["order"] = m["order"].where(~m["order"].str.startswith("random_"), "random")
    m = m.groupby(["order", "n", "target"], as_index=False)["rmse"].median()
    pd.set_option("display.width", 220)
    print(f"\nc = {c}, sigma = {sigma}: predicted θ rmse (mean over k = "
          f"{', '.join(map(str, pb.COMBO_KS))}) by number of backgrounds\n")
    for t in targets:
        print(f"target {t}:")
        print(m[m["target"] == t].pivot(index="order", columns="n",
                                        values="rmse").round(4).to_string())
        print()
    print("picks:")
    for o in ("greedy_ref_mid", "greedy_held_out", "narrative"):
        added = out[(out["order"] == o) & (out["target"] == "ref_mid")
                    & (out["k"] == pb.COMBO_KS[0]) & (out["n"] > 0)]["added"]
        print(f"  {o}: " + "; ".join(added))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.1)
    a = ap.parse_args()
    main(a.c, a.sigma)
