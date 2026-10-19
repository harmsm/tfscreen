"""
Can the data order the tubes of pheS+4CP replicate 2, t_sel 82? (README, run 7)

    python row_order.py GROWTH_PHES_CSV

Genotypes with > 2e5 pheS reads (wt left out), centered log counts per tube.
Each tube's IPTG signature is its projection, relative to its row's 0-IPTG
tube, on the mean 1 mM - 0 direction of the other complete rows. The
expected profile is the median, over those rows, of each concentration's
projection as a fraction of the row's 1 mM one. Every assignment of the
suspect row's six labels to its six tubes is then fitted as observed =
a + b * expected; the script prints the best, the labels as written and
two rack flips.
"""
import pandas as pd, numpy as np, itertools, sys
g = pd.read_csv(sys.argv[1], usecols=["sample","genotype","counts","replicate","condition_sel","t_sel","titrant_conc"])
g = g[g.condition_sel == "pheS+4CP"]
tot = g.groupby("genotype").counts.sum()
keep = tot[tot > 2e5].index.drop("wt", errors="ignore")
g = g[g.genotype.isin(keep)]
g["lf"] = np.log(g.counts + 0.5)
g["lf"] -= g.groupby("sample").lf.transform("mean")          # centered log ratio
w = g.pivot_table(index="genotype", columns="sample", values="lf")
meta = g.drop_duplicates("sample").set_index("sample")[["replicate","t_sel","titrant_conc"]]
print(len(keep), "genotypes")
def tubes(rep, t):
    m = meta[(meta.replicate == rep) & (meta.t_sel == t)].sort_values("titrant_conc")
    return list(m.index), list(m.titrant_conc)
# per-row IPTG profile, each tube relative to its row's 0 tube
rows = meta.groupby(["replicate","t_sel"]).size().index
prof = {}
for rep, t in rows:
    s, c = tubes(rep, t)
    if 0.0 not in c: continue
    base = w[s[c.index(0.0)]]
    prof[(rep, t)] = {cc: w[ss] - base for ss, cc in zip(s, c)}
# reference direction: 1 mM minus 0 averaged over the good full rows
good = [k for k in prof if k != (2, 82) and 1.0 in prof[k]]
v = pd.concat([prof[k][1.0] for k in good], axis=1).mean(axis=1)
v = v / np.sqrt((v**2).sum())
def score(x): return float((x * v).sum())
print("projection on the 1 mM direction, each tube minus its row's 0 tube:")
tab = {k: {c: round(score(x), 2) for c, x in prof[k].items()} for k in sorted(prof)}
print(pd.DataFrame(tab).T.to_string())
# expected per-concentration profile (median over good rows, as fraction of that row's 1 mM)
frac = pd.DataFrame({k: {c: score(x) / score(prof[k][1.0]) for c, x in prof[k].items()} for k in good}).T
expf = frac.median()
print("expected fraction of the 1 mM shift by concentration:"); print(expf.round(2).to_string())
s, c = tubes(2, 82)
obs = np.array([score(w[ss] - w[s[c.index(0.0)]]) for ss in s])
# fit obs = a + b * exp[label]; labels permuted (0 tube included)
e = expf.loc[c].values
out = []
for perm in itertools.permutations(range(len(c))):
    q = e[list(perm)]
    A = np.c_[np.ones_like(q), q]
    coef, *_ = np.linalg.lstsq(A, obs, rcond=None)
    out.append((((obs - A @ coef)**2).sum(), [c[i] for i in perm], coef))
out.sort(key=lambda r: r[0])
print("row 2/82 tubes as labeled:", c); print("observed:", obs.round(2))
for r in out[:6]: print(round(r[0], 4), "true conc per labeled tube:", r[1], "scale", round(r[2][1], 2))
for name, perm in [("as labeled", c), ("flip all", c[::-1]), ("flip all but 0", [0.0] + c[:0:-1])]:
    r = [x for x in out if x[1] == perm][0]
    print(f"{name}: SS {r[0]:.4f}, rank {out.index(r)+1}/{len(out)}, scale {r[2][1]:.2f}")
