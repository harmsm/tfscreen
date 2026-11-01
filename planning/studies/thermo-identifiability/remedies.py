"""
Step 5: remedies for step 4's misspecifications, scored on higher-order
combinations.

The truth carries the two effects step 4 found harmful:

- ligand-specific mutation effects (Mike's expectation): a mutation's shift
  of the effector-binding terms (eH, eL) differs under effector 2, and its
  shift of the operator terms (oHm, oLm, oH, oL, oiiH, oiiL) differs under
  operator 2; everything else is shared. Deviation SD 0.5 kT for pocket
  (effector) or DNA-binding (operator) mutations, 0.1 kT otherwise.
- pair coupling between sites, sparse and distance dependent. Sites are the
  library mutations and the five background mutations, at synthetic
  positions uniform in a ball of radius --radius A. A pair is coupled with
  probability exp(-d/--length), by Normal(0, --pair_sd) kT on one of the
  terms either site acts on (its class's main terms; a background's target
  terms). A mutation x background coupling is the same object as a library
  pair: mutation m in background b is the double m+b.

Fit models (MAP of noise-free data, Laplace at the MAP, as in step 4):
    shared     effects shared across effectors and operators, no pair terms
    lig        per-mutation effector-2 and operator-2 deviations on the e
               and o terms, prior Normal(0, 0.5) (a tying prior)
    pair       lig + one pair term on each term either site of a pair acts
               on (the structure says which sites are near which function),
               prior SD pair_sd*sqrt(exp(-d/length)/n_terms) (the truth's
               second moment); pair_x0.3 / pair_x3 scale that SD

Data: the singles screen of a design (step 4's budget: 12 conditions'
reads at theta SD 0.03 per point), plus optionally a library of the close
doubles (site pairs closer than --close A, library pairs and mutation +
background-site pairs, all in the wt background) in one condition (base)
or three (base, effector 2, TF 0.3x), at theta SD --sigma_ref *
sqrt(K/--dbl_budget).

Targets, all in the wt background at the base condition: unmeasured
library doubles, and random triples and five-mutants. Scored by the error
of the noise-free MAP against the Laplace SD it reports, over every target
and by whether a target contains a coupled pair: none (clean), only
measured ones (mhit) or an unmeasured one (uhit).

Usage: python remedies.py [--seed 1] [--out_prefix remedies]
"""

import argparse
import itertools
import json

import numpy as np

from species_model import TERMS, TI
from rank_check import Problem, ALL, BG_TARGETS, E_GRID, TF_LEVELS
from design_power import CLASSES, lac_terms
import misspec
from misspec import REC6, COUPLING, SLOT_P

E_DEV = ("eH", "eL")
O_DEV = ("oHm", "oLm", "oH", "oL", "oiiH", "oiiL")
DESIGN_SET = {"wt_only": [(0, 0, 0, 0)], "rec6": REC6,
              "star": misspec.DESIGN_SET["star"]}
DBL_SET = {"none": [], "dbl1": [(0, 0, 0, 0)],
           "dbl3": [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)]}
BASE = (0, 0, 0, 0)


class Model:
    """Parameter registry, truth and priors over every condition."""

    def __init__(self, seed, num_mut, radius, length, pair_sd):
        rng = np.random.default_rng(seed)
        pr = Problem("coupled", ALL, num_mut, ligand="specific",
                     bg_terms="all", tf="ratio",
                     rng=np.random.default_rng(seed + 1000))
        pr.T0 = lac_terms(4.0)
        self.pr = pr
        self.nm = num_mut
        keys = list(CLASSES)
        p = np.array([CLASSES[k][0] for k in keys])
        self.cls = {m: keys[rng.choice(len(keys), p=p / p.sum())]
                    for m in range(num_mut)}
        names = list(pr.names)
        true = np.zeros(len(names))
        sd = np.zeros(len(names))
        kind = []
        for i, name in enumerate(names):
            head, _, term = name.partition(":")
            k = "main"
            if head == "wt":
                true[i], sd[i] = pr.T0[TI[term]], 3.0
            elif head.startswith("bg"):
                b = int(head[2:])
                if term in BG_TARGETS[b]:
                    true[i] = rng.choice([-1, 1]) * rng.uniform(1.5, 2.5)
                    sd[i] = 3.0
                else:
                    true[i], sd[i] = rng.normal(0, 0.2), 0.5
            elif head in ("eff2", "op2"):
                true[i], sd[i] = rng.normal(0, 1.0), 3.0
            elif head.startswith("tf"):
                t = int(head[2:])
                true[i], sd[i] = TF_LEVELS[t] + rng.normal(0, 0.1), 0.3
            elif head.startswith("mut"):
                m = int(head[3:])
                if ":" in term:
                    k = "dev"
                    knob, base = term.split(":")
                    big = (knob == "eff2" and self.cls[m] == "pocket") or \
                        (knob == "op2" and self.cls[m] == "dna")
                    ok = base in (E_DEV if knob == "eff2" else O_DEV)
                    true[i] = rng.normal(0, 0.5 if big else 0.1) if ok else 0
                    sd[i] = 0.5
                    term = base if ok else "cH"
                else:
                    mean, s = CLASSES[self.cls[m]][1].get(term, (0.0, 0.2))
                    true[i], sd[i] = rng.normal(mean, s), 1.5
            if term in COUPLING:
                true[i] = 0.0
                k = "coupling"
            kind.append(k)

        # sites: library mutations 0..nm-1, background b at nm + b - 1
        S = num_mut + 5
        pos = rng.normal(size=(S, 3))
        pos *= (radius * rng.uniform(size=S) ** (1 / 3)
                / np.linalg.norm(pos, axis=1))[:, None]
        self.dist = np.linalg.norm(pos[:, None] - pos[None], axis=2)
        acts = {m: list(CLASSES[self.cls[m]][1]) for m in range(num_mut)}
        for b in range(1, 6):
            acts[num_mut + b - 1] = list(BG_TARGETS[b])
        self.pair_par = {}
        self.pair_prior = {}
        for i, j in itertools.combinations(range(S), 2):
            if i >= num_mut:
                continue
            terms = sorted(set(acts[i]) | set(acts[j]))
            d = self.dist[i, j]
            prob = np.exp(-d / length)
            hit = rng.random() < prob
            which = terms[rng.integers(len(terms))]
            delta = rng.normal(0, pair_sd)
            self.pair_par[(i, j)] = {}
            self.pair_prior[(i, j)] = pair_sd * np.sqrt(prob / len(terms))
            for t in terms:
                self.pair_par[(i, j)][t] = len(names)
                names.append(f"pair{i}-{j}:{t}")
                true = np.append(true, delta if (hit and t == which) else 0)
                sd = np.append(sd, self.pair_prior[(i, j)])
                kind.append("pair")
        self.names = names
        self.true = true
        self.sd = sd
        self.kind = np.array(kind)
        self.pair_i = {}
        for (i, j), d in self.pair_par.items():
            for t, p_ in d.items():
                self.pair_i[p_] = (i, j)

    def links(self, c, sites):
        pr = self.pr
        b, f, o, t = c
        allsites = sorted(list(sites) + ([self.nm + b - 1] if b else []))
        lk = []
        for term in pr.free:
            s = TI[term]
            lk.append((s, pr.wt[term]))
            if (b, term) in pr.bg:
                lk.append((s, pr.bg[(b, term)]))
            if f == 1 and term in pr.eff2:
                lk.append((s, pr.eff2[term]))
            if o == 1 and term in pr.op2:
                lk.append((s, pr.op2[term]))
            for m in sites:
                lk.append((s, pr.mut[(m, term)]))
                if f == 1 and (m, "eff2", term) in pr.mut:
                    lk.append((s, pr.mut[(m, "eff2", term)]))
                if o == 1 and (m, "op2", term) in pr.mut:
                    lk.append((s, pr.mut[(m, "op2", term)]))
        if t in pr.tf:
            lk.append((len(TERMS), pr.tf[t]))
        for i, j in itertools.combinations(allsites, 2):
            for term, p_ in self.pair_par.get((i, j), {}).items():
                lk.append((TI[term], p_))
        return lk


class Rows(misspec.Rows):
    """Rows for arbitrary (condition, set of library sites) genotypes."""

    def __init__(self, model, entries):
        pr = model.pr
        self.sp = pr.sp
        self.cms = entries
        n = len(entries)
        self.n, self.nE = n, len(E_GRID)
        self.E = np.tile(E_GRID, n)
        base = np.repeat(pr.T0[None, :].copy(), n, 0)
        for term in pr.free:
            base[:, TI[term]] = 0.0
        self.base = base
        self.base_p = np.array([TF_LEVELS[c[3]] if c[3] not in pr.tf
                                else 0.0 for c, _ in entries])
        self.extra = np.zeros((n, SLOT_P))
        cm_i, sl, pi = [], [], []
        for k, (c, sites) in enumerate(entries):
            for s, p_ in model.links(c, sites):
                cm_i.append(k)
                sl.append(s)
                pi.append(p_)
        self.cm_i, self.sl, self.pi = map(np.array, (cm_i, sl, pi))


class FitSpec:
    """free / start / mu / sd in the form misspec.fit reads."""

    def __init__(self, model, fit_model, scale=1.0):
        kind = model.kind
        free = kind != "coupling"
        if fit_model == "shared":
            free &= ~np.isin(kind, ["dev", "pair"])
        elif fit_model == "lig":
            free &= kind != "pair"
        self.free = free
        sd = model.sd.copy()
        sd[kind == "pair"] *= scale
        self.sd = sd
        self.mu = np.zeros(len(sd))
        for i, name in enumerate(model.names):
            head, _, term = name.partition(":")
            if head.startswith("tf"):
                self.mu[i] = TF_LEVELS[int(head[2:])]
            elif head == "wt":
                self.mu[i] = model.true[i]
        self.start = np.where(free, model.true, 0.0)


def score(err, sdv, groups, label):
    maxerr = np.max(np.abs(err), axis=1)
    z = err / sdv
    out = {}
    for tag, mask in groups.items():
        if not mask.any():
            continue
        zz = z[mask]
        out.update({
            f"{label}_{tag}_n": int(mask.sum()),
            f"{label}_{tag}_err_median": float(np.median(maxerr[mask])),
            f"{label}_{tag}_err_q90": float(np.quantile(maxerr[mask], 0.9)),
            f"{label}_{tag}_sd_median": float(np.median(sdv[mask].max(1))),
            f"{label}_{tag}_cov95": float(np.mean(np.abs(zz) < 1.96)),
            f"{label}_{tag}_frac_z3": float(
                np.mean(np.max(np.abs(zz), axis=1) > 3))})
    return out


def target_groups(md, combos, measured):
    """all; clean (no coupled pair inside); measured_hit (coupled pairs
    inside, every one measured); unmeasured_hit (an unmeasured one)."""
    coupled = {p for p, d in md.pair_par.items()
               if np.any(md.true[list(d.values())] != 0)}
    mh, uh = [], []
    for s in combos:
        ps = set(itertools.combinations(s, 2)) & coupled
        uh.append(bool(ps - measured))
        mh.append(bool(ps) and not (ps - measured))
    mh, uh = np.array(mh), np.array(uh)
    return {"all": np.ones(len(combos), bool), "clean": ~mh & ~uh,
            "mhit": mh, "uhit": uh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num_mut", type=int, default=15)
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--length", type=float, default=6.0)
    ap.add_argument("--pair_sd", type=float, default=1.0)
    ap.add_argument("--close", type=float, default=10.0)
    ap.add_argument("--sigma_ref", type=float, default=0.03)
    ap.add_argument("--budget_conditions", type=int, default=12)
    ap.add_argument("--dbl_budget", type=int, default=3)
    ap.add_argument("--num_targets", type=int, default=200)
    ap.add_argument("--max_nfev", type=int, default=400)
    ap.add_argument("--designs", default="wt_only,rec6,star")
    ap.add_argument("--out_prefix", default="remedies")
    args = ap.parse_args()

    md = Model(args.seed, args.num_mut, args.radius, args.length,
               args.pair_sd)
    nm = args.num_mut
    rng = np.random.default_rng(args.seed + 77)
    close = [(i, j) for (i, j) in md.pair_par
             if md.dist[i, j] < args.close]
    close_lib = {(i, j) for (i, j) in close if j < nm}
    targets = {
        "dbl": [p for p in itertools.combinations(range(nm), 2)
                if p not in close_lib],
        "tri": [tuple(sorted(rng.choice(nm, 3, replace=False)))
                for _ in range(args.num_targets)],
        "quint": [tuple(sorted(rng.choice(nm, 5, replace=False)))
                  for _ in range(args.num_targets)]}
    trows = {k: Rows(md, [(BASE, s) for s in v]) for k, v in targets.items()}
    ttrue = {k: r.theta(md.true).reshape(len(targets[k]), -1)
             for k, r in trows.items()}
    print(json.dumps(dict(
        seed=args.seed, n_pairs=len(md.pair_par), n_close=len(close),
        n_close_lib=len(close_lib),
        n_coupled=int(np.sum(md.true[md.kind == "pair"] != 0)),
        n_coupled_close=int(sum(
            np.any(md.true[list(md.pair_par[p].values())] != 0)
            for p in close)))))

    def dbl_entries(conds):
        out = []
        for c in conds:
            for i, j in close:
                if j < nm:
                    out.append((c, (i, j)))
                else:
                    b = j - nm + 1
                    out.append(((b,) + c[1:], (i,)))
        return out

    runs = []
    for dname in args.designs.split(","):
        for fm, dk, scale in [("shared", "none", 1), ("lig", "none", 1),
                              ("pair", "none", 1), ("pair", "dbl1", 1),
                              ("pair", "dbl3", 1), ("pair", "dbl3", 0.3),
                              ("pair", "dbl3", 3), ("lig", "dbl3", 1)]:
            runs.append((dname, fm, dk, scale))
    out = []
    for dname, fm, dk, scale in runs:
        design = DESIGN_SET[dname]
        K = len(design)
        sig1 = args.sigma_ref * np.sqrt(K / args.budget_conditions)
        ent = [(c, ()) for c in design] + \
            [(c, (m,)) for c in design for m in range(nm)]
        n1 = len(ent)
        dent = dbl_entries(DBL_SET[dk])
        rows = Rows(md, ent + dent)
        sigma = np.full(rows.n * rows.nE, sig1)
        if dent:
            sig2 = args.sigma_ref * np.sqrt(len(DBL_SET[dk]) /
                                            args.dbl_budget)
            sigma[n1 * rows.nE:] = sig2
        spec = FitSpec(md, fm, scale)
        y = rows.theta(md.true)
        p_hat, res = misspec.fit(spec, rows, y, sigma, args.max_nfev)
        r = (rows.theta(p_hat) - y) / sigma
        misfit = float(np.sum(r ** 2))
        _, J = rows.theta_jac(p_hat)
        Jf = J[:, spec.free] / sigma[:, None]
        Lam = Jf.T @ Jf + np.diag(1.0 / spec.sd[spec.free] ** 2)
        Sigma = np.linalg.inv(Lam)
        rec = dict(seed=args.seed, design=dname, fit=fm, doubles=dk,
                   pair_scale=scale, K=K, n_dbl_rows=len(dent),
                   nfev=int(res.nfev), status=int(res.status),
                   n_free=int(spec.free.sum()), misfit_chi2=misfit,
                   detect_z=misfit / np.sqrt(2 * len(y)))
        meas = close_lib if dent else set()
        for k, tr in trows.items():
            groups = target_groups(md, targets[k], meas)
            th, G = tr.theta_jac(p_hat)
            Gf = G[:, spec.free]
            sdv = np.sqrt(np.einsum("ij,jk,ik->i", Gf, Sigma, Gf))
            n = len(targets[k])
            rec.update(score(th.reshape(n, -1) - ttrue[k],
                             sdv.reshape(n, -1), groups, k))
        out.append(rec)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in rec.items()}), flush=True)
    with open(f"{args.out_prefix}_s{args.seed}.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
