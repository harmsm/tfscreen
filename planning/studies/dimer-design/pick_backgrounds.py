"""
Pick backgrounds for the dimer-design N arm on the full dimer model.

The narrative chose its five double backgrounds by Fisher information on
a reduced seven-primitive ensemble at wt. On the full model (monomers,
unfolded monomer, second engagement) two of the five misbehave: R51S/D149M
inverts at c = 1 and goes flat at c = 0.3, and S97A/G187E barely induces at
c = 1. This script repeats the choice on the model the simulation uses.

Physics: the simulation's own (thermo.dimer species model, PddG
projection, wt and TF levels of runs/N_c1); check_truth reproduced its θ to
1e-7. A library mutation's constants are c x (P . table) in kT, its Oii
terms 0 (the table does not set them).

Criterion (default `--criterion predict`): the expected error of θ
predicted for combinations nobody measured. Per-term SEs are the wrong
target: the model has ridges the data never see, and a constant left
unresolved along one costs nothing unless it moves θ. With the Laplace
posterior Σ_m = F_m^-1 of each single (below) and the singles' errors
independent, a combination S of k singles predicted at the reference
(TF mid) has var θ_S(c) = J_S(c) (Σ_{m in S} Σ_m) J_S(c)^T, J_S = dθ/d ln K
at wt + Σ d_m. Since the prior is centered with SD PRIOR_SD, this is the
Bayes risk of the prediction. The score is the predicted rmse
sqrt(mean var) over NUM_COMBOS random combinations of distinct residues
at each k in COMBO_KS, averaged over k; the greedy minimizes it.
`--criterion dopt` is the earlier D-optimal score, reported either way.

Information: each library single m has 12 constants d_m. A context k (a
background at a TF level) measured with n_k(c) tubes at IPTG c adds
F_mk = sum_c n_k(c) J_mk(c)^T J_mk(c) / SIGMA^2, J = dθ/dd_m. With the fits'
prior (SD 1 kT on every constant) the posterior precision is
F_m = I + F_m,ref + sum_k F_mk. The design score is the mean over library
singles of log det F_m (D-optimality), and the report gives each term's
median posterior SD sqrt(diag F_m^-1) in kcal/mol. SIGMA is the θ noise
per tube, a free scale: compare designs with it, not absolute SEs.

A candidate is a table single outside the amplicon (no wt_seq edit), or a
pair of the best singles, at TF lo, mid or hi. It must leave a readable
repressor curve at its level: θ(0) - θ(1 mM) >= MIN_DROP and θ(0) >=
MIN_THETA0. Greedy: add the candidate with the largest score gain,
NUM_PICK times, one level per background.

    ~/miniconda3/bin/python pick_backgrounds.py [--c 1.0] [--sigma 0.1]

Writes results/picked_backgrounds.csv and results/design_scores.csv.
"""

import argparse
import itertools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from tfscreen.tfmodel.generative.components.theta.thermo.dimer import (
    PddG, species as sp)

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "inputs", "2026-11-03_laci-state-ddg.csv")
OUT = os.path.join(HERE, "results")
KT_PER_KCAL = 1.0 / (1.987204259e-3 * 310.15)

VARIANT = sp.Variant(monomers=True, unfolded_dimer=False,
                     unfolded_monomer=True, second_engagement=True)
TERMS = sp.active_terms(VARIANT)        # 12, the per-mutation constants

# wt and TF levels of the simulation (runs/N_c1/sim, dimerization -15).
WT = dict(ln_K_h_l=1.8403057, ln_K_h_o=15.837843, ln_K_l_o=-6.3191586,
          ln_K_h_oii=4.000732, ln_K_l_oii=3.9985056, ln_K_d_h=24.336744,
          ln_K_m_h_l=0.9195248, ln_K_m_h_o=13.81595, ln_K_m_l_o=6.908902,
          ln_K_m_u=-6.4903436, ln_K_h_e=8.229746, ln_K_l_e=12.605076)
LN_TF = {"lo": -12.74453068, "mid": -11.51292515, "hi": -9.74467945}
CONC_MM = np.array([0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0])

# Tubes per concentration: the reference's 118 at mid, a 20-tube block.
REF_TUBES = np.array([24, 12, 11, 11, 12, 12, 11, 25], dtype=float)
BLOCK_TUBES = np.array([4, 2, 2, 2, 2, 2, 2, 4], dtype=float)

AMPLICON = (27, 98)
MIN_DROP = 0.3
MIN_THETA0 = 0.5
NUM_PICK = 5
TOP_SINGLES = 40        # singles paired into double candidates
PRIOR_SD = 1.0          # kT, the fits' held per-mutation scale
COMBO_KS = (2, 3, 4, 6)
NUM_COMBOS = 200

NARRATIVE = [(("L114T", "I289D"), "hi"), (("R51P", "C281D"), "hi"),
             (("S61P", "D149M"), "mid"), (("S97A", "G187E"), "lo"),
             (("R51S", "D149M"), "mid")]


def mutation_constants(table, c):
    """Per-mutation Δ terms (kT), rows of table, columns TERMS."""
    names = PddG.structure_names_for(VARIANT)
    rows = PddG.struct_terms_for(VARIANT)
    ddg = table[list(names)].to_numpy() * KT_PER_KCAL * c
    proj = ddg @ PddG.dlnk_proj_for(VARIANT).T
    out = pd.DataFrame(0.0, index=table["mut"], columns=list(TERMS))
    for i, t in enumerate(rows):
        if t in out.columns:
            out[t] = proj[:, i]
    out["ln_K_m_u"] = table["H_monomer"].to_numpy() * KT_PER_KCAL * c
    return out


def _theta(vec, ln_tf):
    terms = {t: vec[..., i, None] for i, t in enumerate(TERMS)}
    return sp.theta(terms, jnp.asarray(CONC_MM * 1e-3), jnp.exp(ln_tf),
                    VARIANT)


_theta_b = jax.jit(jax.vmap(_theta, in_axes=(0, None)))
_jac_b = jax.jit(jax.vmap(jax.jacfwd(_theta), in_axes=(0, None)))


def curves(vecs, level):
    return np.asarray(_theta_b(jnp.asarray(vecs), LN_TF[level]))


def info(base, lib, level, tubes, sigma):
    """(L, 12, 12) Fisher information of each library mutation's
    constants in context (base, level)."""
    J = np.asarray(_jac_b(jnp.asarray(base[None, :] + lib), LN_TF[level]))
    return np.einsum("lci,c,lcj->lij", J, tubes / sigma ** 2, J)


def score(F):
    return float(np.mean(np.linalg.slogdet(F)[1]))


def combo_jacobians(wt, lib, rng):
    """{k: (idx (n, k), J (n, C, 12))}: random combinations of library
    singles at distinct residues and θ's gradient at the reference."""
    out = {}
    for k in COMBO_KS:
        idx, seen = [], set()
        while len(idx) < NUM_COMBOS:
            i = tuple(sorted(rng.choice(len(lib), k, replace=False)))
            if i in seen or len(set(LIB_RES[list(i)])) < k:
                continue
            seen.add(i)
            idx.append(i)
        idx = np.array(idx)
        J = np.asarray(_jac_b(jnp.asarray(wt[None, :] + lib[idx].sum(axis=1)),
                              LN_TF["mid"]))
        out[k] = (idx, J)
    return out


def predicted_rmse(F, combos):
    """{k: predicted θ rmse} for the combinations, given (L, 12, 12) F."""
    cov = np.linalg.inv(F)
    out = {}
    for k, (idx, J) in combos.items():
        C = cov[idx].sum(axis=1)
        out[k] = float(np.sqrt(np.mean(np.einsum("nci,nij,ncj->nc", J, C, J))))
    return out


def risk(F, combos):
    return float(np.mean(list(predicted_rmse(F, combos).values())))


def term_se(F):
    sd = np.sqrt(np.diagonal(np.linalg.inv(F), axis1=1, axis2=2))
    return pd.Series(np.median(sd, axis=0) / KT_PER_KCAL, index=TERMS)


LIB_RES = None


def main(c=1.0, sigma=0.1, criterion="predict"):
    global LIB_RES
    table = pd.read_csv(TABLE)
    table = table[table["mut"].str[-1] != "*"].reset_index(drop=True)
    d = mutation_constants(table, c)
    wt = np.array([WT[t] for t in TERMS])
    res = table["mut"].str[1:-1].astype(int).to_numpy()

    # Library: the study's singles inside the amplicon (positions 29-98).
    lib_muts = [m for m, r in zip(table["mut"], res)
                if AMPLICON[0] + 2 <= r <= AMPLICON[1]]
    rng = np.random.default_rng(0)
    lib_muts = sorted(rng.choice(lib_muts, 120, replace=False))
    lib = d.loc[lib_muts].to_numpy()
    LIB_RES = np.array([int(m[1:-1]) for m in lib_muts])
    combos = combo_jacobians(wt, lib, np.random.default_rng(1))

    F0 = np.eye(len(TERMS))[None] / PRIOR_SD ** 2 + \
        info(wt, lib, "mid", REF_TUBES, sigma)
    base_score = score(F0)

    def objective(Fd):          # larger is better
        if criterion == "predict":
            return -risk(Fd, combos)
        return score(Fd)
    base_obj = objective(F0)

    def bg_vec(muts):
        return wt + d.loc[list(muts)].to_numpy().sum(axis=0)

    def readable(vecs, level):
        th = curves(vecs, level)
        return (th[:, 0] - th[:, -1] >= MIN_DROP) & (th[:, 0] >= MIN_THETA0), th

    # Singles outside the amplicon.
    outside = [(m,) for m, r in zip(table["mut"], res)
               if r < AMPLICON[0] or r > AMPLICON[1]]
    cand = []
    for level in LN_TF:
        vecs = np.array([bg_vec(m) for m in outside])
        ok, th = readable(vecs, level)
        for m, v, t in zip(np.array(outside, dtype=object)[ok], vecs[ok], th[ok]):
            G = info(v, lib, level, BLOCK_TUBES, sigma)
            cand.append(dict(muts=tuple(m), level=level, G=G, curve=t,
                             gain=objective(F0 + G) - base_obj))
    print(f"{len(outside)} singles x 3 levels: {len(cand)} readable")

    # Doubles from the best singles (one level each), at every level.
    best = sorted(cand, key=lambda x: -x["gain"])
    seen, top = set(), []
    for x in best:
        if x["muts"][0] not in seen and \
                int(x["muts"][0][1:-1]) not in {int(t[0][1:-1]) for t in top}:
            seen.add(x["muts"][0])
            top.append(x["muts"])
        if len(top) == TOP_SINGLES:
            break
    pairs = [a + b for a, b in itertools.combinations(top, 2)]
    for level in LN_TF:
        vecs = np.array([bg_vec(p) for p in pairs])
        ok, th = readable(vecs, level)
        for p, v, t in zip(np.array(pairs, dtype=object)[ok], vecs[ok], th[ok]):
            G = info(v, lib, level, BLOCK_TUBES, sigma)
            cand.append(dict(muts=tuple(p), level=level, G=G, curve=t,
                             gain=objective(F0 + G) - base_obj))
    print(f"+ {len(pairs)} doubles x 3 levels: {len(cand)} candidates")

    # Greedy.
    F, picked, used = F0.copy(), [], set()
    for _ in range(NUM_PICK):
        best, gain = None, -np.inf
        for x in cand:
            if set(x["muts"]) & used:
                continue
            g = objective(F + x["G"])
            if g > gain:
                best, gain = x, g
        picked.append(best)
        used |= set(best["muts"])
        F = F + best["G"]

    def design_F(bgs):
        Fd = F0.copy()
        for muts, level in bgs:
            Fd = Fd + info(bg_vec(muts), lib, level, BLOCK_TUBES, sigma)
        return Fd

    rand = list(pd.read_csv(os.path.join(HERE, "inputs", "design_R.csv"))
                ["background"].unique())
    rand = [((b[2:],), "mid") for b in rand if b != "reference"]
    designs = {"W": F0, "N narrative": design_F(NARRATIVE),
               "R random 15": design_F(rand),
               "picked 5": F}
    rows = []
    for name, Fd in designs.items():
        pr = predicted_rmse(Fd, combos)
        rows.append({"design": name, "dopt": round(score(Fd) - base_score, 2),
                     **{f"rmse_k{k}": round(v, 4) for k, v in pr.items()},
                     **term_se(Fd).round(3).to_dict()})
    scores = pd.DataFrame(rows)

    pk = pd.DataFrame([{"background": "/".join(x["muts"]), "level": x["level"],
                        "gain_alone": round(x["gain"], 3),
                        "theta_0": round(x["curve"][0], 2),
                        "theta_1mM": round(x["curve"][-1], 2),
                        "curve": " ".join(f"{v:.2f}" for v in x["curve"])}
                       for x in picked])
    nar = []
    for muts, level in NARRATIVE:
        t = curves(bg_vec(muts)[None], level)[0]
        nar.append({"background": "/".join(muts), "level": level,
                    "gain_alone": round(objective(F0 + info(bg_vec(muts), lib,
                                        level, BLOCK_TUBES, sigma))
                                        - base_obj, 4),
                    "curve": " ".join(f"{v:.2f}" for v in t)})

    os.makedirs(OUT, exist_ok=True)
    tag = f"c{c:g}".replace(".", "") + ("" if criterion == "predict" else "_dopt")
    pk.to_csv(os.path.join(OUT, f"picked_backgrounds_{tag}.csv"), index=False)
    scores.to_csv(os.path.join(OUT, f"design_scores_{tag}.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"\nc = {c}, sigma = {sigma}, criterion {criterion}; gain = "
          f"objective gain over W (predict: drop in mean predicted rmse)\n")
    print("narrative backgrounds:")
    print(pd.DataFrame(nar).to_string(index=False))
    print("\npicked:")
    print(pk.to_string(index=False))
    print("\npredicted θ rmse for unmeasured k-combinations; dopt; "
          "median posterior SD per term (kcal/mol; prior 0.616):")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--criterion", choices=("predict", "dopt"), default="predict")
    a = ap.parse_args()
    main(a.c, a.sigma, a.criterion)
