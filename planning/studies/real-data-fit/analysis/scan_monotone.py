"""
Is there a shared monotone (not necessarily affine) map X = f(theta(P))
from binding theta at apparent protein P to growth-only X, across the
binding genotypes? Isotonic regression (decreasing X with decreasing theta,
i.e. increasing) per P; also with M42I left out.

    python scan_monotone.py
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import scan_protein as sp


def iso_rms(th, X):
    f = IsotonicRegression(increasing=True).fit(th, X)
    return np.sqrt(np.mean((X - f.predict(th)) ** 2))


curves = pd.read_csv(sp.CURVES)
xd = pd.read_csv(sp.X_FILE)[["genotype", "titrant_conc", "X"]]
by_g = {g: d.set_index("protein_uM_dimer").sort_index() for g, d in curves.groupby("genotype")}
rows = []
for P in sp.P_GRID[::3]:
    parts = []
    for g in sp.GENOS:
        lo, hi, logK, n, ex = sp.params_at(by_g[g], P)
        d = xd[xd.genotype == g]
        parts.append(pd.DataFrame(dict(genotype=g, X=d.X.values,
                                       theta=sp.theta(lo, hi, logK, n, d.titrant_conc))))
    t = pd.concat(parts)
    wo = t[t.genotype != "M42I"]
    _, aff = sp.affine_rms(t.theta.values, t.X.values)
    rows.append(dict(P=P, affine=aff, monotone=iso_rms(t.theta.values, t.X.values),
                     monotone_without_M42I=iso_rms(wo.theta.values, wo.X.values)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
