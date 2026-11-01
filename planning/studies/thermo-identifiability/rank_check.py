"""
Structural identifiability of the dimeric-TF species model under a design.

A design is a set of screen conditions (background b, effector f, operator o,
TF level t). Each condition measures the background alone and M single
mutations in it, at every effector concentration of E_GRID. The parameters:

    wt:<term>          wt energy terms free in the variant
    bg<b>:<term>       background b's shift on its target terms (bg_terms
                       target), on those plus oiiH for the operator
                       background (target_oii), or on every term (all)
    eff2:<term>        effector 2's change to the effector-specific terms
    op2:<term>         operator 2's change to the operator-specific terms
    tf<t>              ln P_t of TF level t (level 0 known unless tf=latent)
    mut<m>:<term>      mutation m's shift, additive in every background
    mut<m>:eff2:<term> / mut<m>:op2:<term>   (ligand=specific only) its own
                       shift of a ligand-specific term under effector 2 /
                       operator 2
    aff<c>:a, aff<c>:s (observable=affine) X = a + s theta per affine group

There are no mutation x background coupling parameters: specific coupling
belongs in the residual. Rank is computed at random generic points from the
exact Jacobian (implicit function theorem), so a deficiency is structural.

Usage: python rank_check.py [--seeds 2] [--num_mut 6] [--out_prefix rank]
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from species_model import Species, TERMS, TI
from validate import generic_terms

E_GRID = np.concatenate([[0.0], np.logspace(-7, -2, 11)])
TF_LEVELS = np.log(1e-5) + np.log([1.0, 0.3, 3.0])
NUM_BG, NUM_EFF, NUM_OP, NUM_TF = 6, 2, 2, 3

EFF_TERMS = ["eH", "eL", "cH", "cL", "xH", "xL"]
OP_TERMS = ["oHm", "oLm", "oH", "oL", "oiiH", "oiiL"]
BG_TARGETS = {1: ["L1"], 2: ["oH"], 3: ["eL"], 4: ["eH"],
              5: ["dH", "dL", "U"]}
BG_LABELS = {0: "wt", 1: "H<->L", 2: "operator(H)", 3: "effector(L)",
             4: "effector(H)", 5: "dimer/stability"}

VARIANTS = {
    # name: (monomers, u_form, free wt terms)
    "mwc_dimer": (False, "dimer",
                  ["L1", "U", "eH", "eL", "oH", "oL", "oiiH", "oiiL"]),
    "mwc": (True, "dimer",
            ["L1", "U", "dH", "dL", "eH", "eL", "oHm", "oLm", "oH", "oL",
             "oiiH", "oiiL"]),
    "mwc_umono": (True, "monomer",
                  ["L1", "U", "dH", "dL", "eH", "eL", "oHm", "oLm", "oH",
                   "oL", "oiiH", "oiiL"]),
    "coupled": (True, "dimer",
                ["L1", "U", "dH", "dL", "eH", "eL", "cH", "cL", "xH", "xL",
                 "oHm", "oLm", "oH", "oL", "oiiH", "oiiL"]),
}

ALL = list(itertools.product(range(NUM_BG), range(NUM_EFF), range(NUM_OP),
                             range(NUM_TF)))
DESIGNS = {
    "full": ALL,
    "star": sorted({(b, 0, 0, 0) for b in range(NUM_BG)}
                   | {(b, 1, 0, 0) for b in range(NUM_BG)}
                   | {(b, 0, 1, 0) for b in range(NUM_BG)}
                   | {(b, 0, 0, t) for b in range(NUM_BG) for t in (1, 2)}),
    "wt_knobs": sorted({(0, f, o, t) for f in range(NUM_EFF)
                        for o in range(NUM_OP) for t in range(NUM_TF)}
                       | {(b, 0, 0, 0) for b in range(NUM_BG)}),
    "knobs_only": sorted({(0, f, o, t) for f in range(NUM_EFF)
                          for o in range(NUM_OP) for t in range(NUM_TF)}),
    "base": [(b, 0, 0, 0) for b in range(NUM_BG)],
    "wt_only": [(0, 0, 0, 0)],
}


class Problem:
    """Parameter registry and the linear map from parameters to row terms."""

    def __init__(self, variant, design, num_mut, observable="theta",
                 affine_by="condition", ligand="shared", bg_terms="target",
                 tf="ratio", rng=None):
        monomers, u_form, free = VARIANTS[variant]
        self.sp = Species(monomers, u_form)
        self.free = free
        self.design = design
        self.names = []
        self.true = []
        rng = rng or np.random.default_rng(0)
        self.T0 = generic_terms(rng)
        for t in ("cH", "cL", "xH", "xL"):
            if t not in free:
                self.T0[TI[t]] = 0.0
        used_b = sorted({c[0] for c in design})
        used_f = sorted({c[1] for c in design})
        used_o = sorted({c[2] for c in design})
        used_t = sorted({c[3] for c in design})

        def add(name, value):
            self.names.append(name)
            self.true.append(value)
            return len(self.names) - 1

        self.wt = {t: add(f"wt:{t}", self.T0[TI[t]]) for t in free}
        self.bg = {}
        for b in used_b:
            if b == 0:
                continue
            if bg_terms == "target":
                targets = BG_TARGETS[b]
            elif bg_terms == "target_oii":
                targets = BG_TARGETS[b] + (["oiiH"] if b == 2 else [])
            else:
                targets = free
            for t in targets:
                if t in free:
                    s = rng.choice([-1, 1])
                    self.bg[(b, t)] = add(f"bg{b}:{t}",
                                          s * rng.uniform(1.0, 2.5))
        self.eff2 = {t: add(f"eff2:{t}", rng.normal(0, 1))
                     for t in EFF_TERMS if t in free and 1 in used_f}
        self.op2 = {t: add(f"op2:{t}", rng.normal(0, 1))
                    for t in OP_TERMS if t in free and 1 in used_o}
        self.tf = {}
        for t in used_t:
            if t == 0 and tf != "latent":
                continue
            if tf == "known":
                continue
            self.tf[t] = add(f"tf{t}", TF_LEVELS[t])
        self.mut = {}
        for m in range(num_mut):
            for t in free:
                self.mut[(m, t)] = add(f"mut{m}:{t}", rng.normal(0, 0.7))
            if ligand == "specific":
                if 1 in used_f:
                    for t in EFF_TERMS:
                        if t in free:
                            self.mut[(m, "eff2", t)] = add(
                                f"mut{m}:eff2:{t}", rng.normal(0, 0.7))
                if 1 in used_o:
                    for t in OP_TERMS:
                        if t in free:
                            self.mut[(m, "op2", t)] = add(
                                f"mut{m}:op2:{t}", rng.normal(0, 0.7))
        self.ligand = ligand
        self.observable = observable
        self.aff = {}
        if observable == "affine":
            groups = sorted({self._aff_group(c, affine_by) for c in design})
            for gname in groups:
                self.aff[gname] = (add(f"aff{gname}:a", 0.0),
                                   add(f"aff{gname}:s", 1.0))
        self.affine_by = affine_by
        self.true = np.array(self.true)

    @staticmethod
    def _aff_group(c, affine_by):
        b, f, o, t = c
        if affine_by == "condition":
            return f"{b}{f}{o}{t}"
        if affine_by == "operator":
            return f"op{o}"
        raise ValueError(affine_by)

    def rows(self):
        """Yield (condition, genotype, list of (slot, param index), T fixed)."""
        for c in self.design:
            b, f, o, t = c
            for m in [None] + list(range(len({k[0] for k in self.mut}))):
                links = []
                for term in self.free:
                    s = TI[term]
                    links.append((s, self.wt[term]))
                    if (b, term) in self.bg:
                        links.append((s, self.bg[(b, term)]))
                    if f == 1 and term in self.eff2:
                        links.append((s, self.eff2[term]))
                    if o == 1 and term in self.op2:
                        links.append((s, self.op2[term]))
                    if m is not None:
                        lig = None
                        if f == 1 and (m, "eff2", term) in self.mut:
                            lig = (m, "eff2", term)
                        if o == 1 and (m, "op2", term) in self.mut:
                            lig = (m, "op2", term)
                        links.append((s, self.mut[lig or (m, term)]))
                if t in self.tf:
                    links.append((len(TERMS), self.tf[t]))
                yield c, m, links

    def jacobian(self):
        """Exact Jacobian (rows x params) at the true parameter values."""
        meta, Ts, lnPts, Ets = [], [], [], []
        for c, m, links in self.rows():
            T = self.T0.copy()
            for term in self.free:
                T[TI[term]] = 0.0
            lnPt = TF_LEVELS[c[3]]
            if c[3] in self.tf:
                lnPt = 0.0
            for s, pi in links:
                if s == len(TERMS):
                    lnPt += self.true[pi]
                else:
                    T[s] += self.true[pi]
            for e in E_GRID:
                meta.append((c, m, links))
                Ts.append(T)
                lnPts.append(lnPt)
                Ets.append(e)
        th, g = self.sp.batch_theta_grad(np.array(Ts), np.array(lnPts),
                                         np.array(Ets))
        J = np.zeros((len(meta), len(self.names)))
        for r, (c, m, links) in enumerate(meta):
            scale = 1.0
            if self.aff:
                ia, is_ = self.aff[self._aff_group(c, self.affine_by)]
                J[r, ia] = 1.0
                J[r, is_] = th[r]
                scale = self.true[is_]
            for s, pi in links:
                J[r, pi] += scale * g[r, s]
        return J, th


def analyze(J, names, tol=1e-9):
    s, V = np.linalg.svd(J, full_matrices=True)[1:]
    smax = s[0]
    rank = int(np.sum(s > tol * smax))
    null = V[rank:].T  # params x nullity
    gap = (s[rank - 1] / smax, (s[rank] / smax) if rank < len(s) else 0.0)
    proj = np.linalg.norm(null, axis=1) if null.size else np.zeros(len(names))
    return rank, null, proj, gap


def null_directions(null, names, max_terms=8):
    """Sparse readable basis of the null space (reduced row echelon)."""
    if null.size == 0:
        return []
    A = null.T.copy()
    piv_rows, r = [], 0
    for col in range(A.shape[1]):
        if r == A.shape[0]:
            break
        p = r + np.argmax(np.abs(A[r:, col]))
        if abs(A[p, col]) < 1e-8:
            continue
        A[[r, p]] = A[[p, r]]
        A[r] /= A[r, col]
        for i in range(A.shape[0]):
            if i != r:
                A[i] -= A[i, col] * A[r]
        r += 1
    out = []
    for row in A[:r]:
        idx = np.argsort(-np.abs(row))
        parts = [f"{row[i]:+.2f} {names[i]}" for i in idx[:max_terms]
                 if abs(row[i]) > 1e-6]
        n_more = int(np.sum(np.abs(row) > 1e-6)) - len(parts)
        out.append(" ".join(parts) + (f" (+{n_more} more)" if n_more > 0
                                      else ""))
    return out


def summarize(names, proj, tol=1e-6):
    """Unidentified parameters, mutation terms collapsed across mutations."""
    bad = [n for n, p in zip(names, proj) if p > tol]
    out, mut = [], {}
    for n in bad:
        if n.startswith("mut"):
            key = n.split(":", 1)[1]
            mut[key] = mut.get(key, 0) + 1
        else:
            out.append(n)
    out += [f"mut*:{k} (x{v})" for k, v in mut.items()]
    return out


def run_one(variant, design_name, seed, num_mut, **kw):
    rng = np.random.default_rng(seed)
    pr = Problem(variant, DESIGNS[design_name], num_mut, rng=rng, **kw)
    J, _ = pr.jacobian()
    rank, null, proj, gap = analyze(J, pr.names)
    return pr, rank, null, proj, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--num_mut", type=int, default=6)
    ap.add_argument("--out_prefix", default="rank")
    args = ap.parse_args()

    runs = []
    for variant in VARIANTS:
        for design in ("full", "star", "wt_knobs", "knobs_only", "base",
                       "wt_only"):
            for obs, aff_by in (("theta", "condition"),
                                ("affine", "condition"),
                                ("affine", "operator")):
                runs.append(dict(variant=variant, design=design,
                                 observable=obs, affine_by=aff_by))
    for variant in ("mwc", "coupled"):
        for design in ("full", "star"):
            runs.append(dict(variant=variant, design=design,
                             observable="theta", ligand="specific"))
            runs.append(dict(variant=variant, design=design,
                             observable="theta", bg_terms="all"))
            runs.append(dict(variant=variant, design=design,
                             observable="theta", tf="latent"))
            runs.append(dict(variant=variant, design=design,
                             observable="theta", tf="known"))

    for variant in ("mwc_dimer", "mwc"):
        for design in ("full", "star", "base"):
            runs.append(dict(variant=variant, design=design,
                             observable="theta", bg_terms="target_oii"))

    records, details = [], {}
    for spec in runs:
        spec = dict(spec)
        variant = spec.pop("variant")
        design = spec.pop("design")
        best = None
        for seed in range(args.seeds):
            res = run_one(variant, design, seed, args.num_mut, **spec)
            if best is None or res[1] > best[1]:
                best = res
        pr, rank, null, proj, gap = best
        label = dict(variant=variant, design=design,
                     observable=spec.get("observable", "theta"),
                     affine_by=(spec.get("affine_by", "condition")
                                if spec.get("observable") == "affine"
                                else "-"),
                     ligand=spec.get("ligand", "shared"),
                     bg_terms=spec.get("bg_terms", "target"),
                     tf=spec.get("tf", "ratio"))
        unid = summarize(pr.names, proj)
        rec = dict(label, n_conditions=len(DESIGNS[design]),
                   n_params=len(pr.names), rank=rank,
                   deficiency=len(pr.names) - rank,
                   gap_kept=gap[0], gap_dropped=gap[1],
                   unidentified="; ".join(unid))
        records.append(rec)
        key = "|".join(str(v) for v in label.values())
        details[key] = dict(unidentified=unid,
                            null_directions=null_directions(null, pr.names))
        print(f"{key}: params {len(pr.names)} rank {rank} "
              f"deficiency {len(pr.names) - rank} "
              f"(gap {gap[0]:.1e} / {gap[1]:.1e})")
        for u in unid:
            print(f"    unidentified: {u}")
    pd.DataFrame(records).to_csv(f"{args.out_prefix}_summary.csv",
                                 index=False)
    with open(f"{args.out_prefix}_null_directions.json", "w") as fh:
        json.dump(details, fh, indent=1)


if __name__ == "__main__":
    main()
