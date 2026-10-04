"""
Is there one apparent protein concentration P at which a single affine map,
X = a + b theta(P, IPTG), carries the binding curves onto the growth-only X
(rel_off_n05) for all binding genotypes?

    python scan_protein.py

theta(P) comes from ../processed/binding_curves.csv (Hill per genotype and
protein), interpolated linearly in ln P exactly as binding_fit.interpolate
does (logit plateaus, ln K, ln n). Mutants were measured at 2 and 20 uM
only, so P outside 2-20 uM is a linear extrapolation in ln P and is
flagged; wt has 0.02-80 uM. Writes resid/protein_scan.csv and
resid/2026-10-01_protein-scan.pdf.
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rc("pdf", fonttype=42)
plt.rc("font", size=14)
plt.rc("axes", labelsize=16)

CURVES = "../processed/binding_curves.csv"
X_FILE = "resid/X_vs_binding_off_n05.csv"
GENOS = ["wt", "M42I", "M42I/H74A", "G65H", "H74A", "K84L", "H74A/K84L"]
P_GRID = np.exp(np.linspace(np.log(0.5), np.log(80), 61))


def clip(v):
    return np.clip(v, 1e-4, 1 - 1e-4)


def params_at(d, P):
    """Hill parameters at P, linear in ln P between (or beyond) the two
    nearest measured protein concentrations."""
    ps = np.array(sorted(d.index))
    if P <= ps[0]:
        p0, p1 = ps[0], ps[1]
    elif P >= ps[-1]:
        p0, p1 = ps[-2], ps[-1]
    else:
        p1 = ps[ps >= P][0]
        p0 = ps[ps <= P][-1]
        if p0 == p1:
            r = d.loc[p0]
            return r.theta_lo, r.theta_hi, r.log_K_mM, r.hill_n, False
    t = (np.log(P) - np.log(p0)) / (np.log(p1) - np.log(p0))
    r0, r1 = d.loc[p0], d.loc[p1]
    lg = lambda k: (1 - t) * logit(clip(r0[k])) + t * logit(clip(r1[k]))
    lo, hi = expit(lg("theta_lo")), expit(lg("theta_hi"))
    logK = (1 - t) * r0.log_K_mM + t * r1.log_K_mM
    n = np.exp((1 - t) * np.log(r0.hill_n) + t * np.log(r1.hill_n))
    return lo, hi, logK, n, not (ps[0] <= P <= ps[-1])


def theta(lo, hi, logK, n, c):
    c = np.asarray(c, float)
    frac = np.where(c > 0, (np.maximum(c, 1e-300) / np.exp(logK)) ** n, 0.0)
    return hi + (lo - hi) / (1.0 + frac)


def affine_rms(th, X):
    A = np.vstack([np.ones_like(th), th]).T
    coef = np.linalg.lstsq(A, X, rcond=None)[0]
    return coef, np.sqrt(np.mean((X - A @ coef) ** 2))


def main():
    curves = pd.read_csv(CURVES)
    xd = pd.read_csv(X_FILE)[["genotype", "titrant_conc", "X"]]
    by_g = {g: d.set_index("protein_uM_dimer").sort_index()
            for g, d in curves.groupby("genotype")}

    def table(P_of):
        rows = []
        for g in GENOS:
            lo, hi, logK, n, extrap = params_at(by_g[g], P_of(g))
            d = xd[xd.genotype == g]
            rows.append(pd.DataFrame(dict(genotype=g, titrant_conc=d.titrant_conc,
                                          X=d.X, theta=theta(lo, hi, logK, n, d.titrant_conc),
                                          extrap=extrap)))
        return pd.concat(rows)

    # 1. one shared P
    out = []
    for P in P_GRID:
        t = table(lambda g: P)
        coef, rms = affine_rms(t.theta.values, t.X.values)
        per = t.assign(res=t.X - (coef[0] + coef[1] * t.theta)).groupby("genotype").res.apply(
            lambda r: np.sqrt(np.mean(r ** 2)))
        no_m42i = t[t.genotype != "M42I"]
        _, rms_wo = affine_rms(no_m42i.theta.values, no_m42i.X.values)
        out.append(dict(P=P, a=coef[0], b=coef[1], rms=rms, rms_without_M42I=rms_wo,
                        mutant_extrap=bool(t[t.genotype != "wt"].extrap.any()),
                        **{f"rms_{g}": per[g] for g in GENOS}))
    s = pd.DataFrame(out)
    s.to_csv("resid/protein_scan.csv", index=False)
    best = s.loc[s.rms.idxmin()]
    best_in = s[~s.mutant_extrap].loc[lambda d: d.rms.idxmin()]
    best_wo = s.loc[s.rms_without_M42I.idxmin()]
    pd.set_option("display.width", 250)
    show = s.iloc[::6][["P", "a", "b", "rms", "rms_without_M42I", "mutant_extrap"]
                       + [f"rms_{g}" for g in GENOS]]
    print(show.round(3).to_string(index=False))
    print(f"\nbest shared P: {best.P:.2f} uM (rms {best.rms:.3f}, a {best.a:.2f}, b {best.b:.2f}, "
          f"mutants extrapolated: {best.mutant_extrap})")
    print(f"best within 2-20 uM: {best_in.P:.2f} uM (rms {best_in.rms:.3f})")
    print(f"best without M42I: {best_wo.P:.2f} uM (rms {best_wo.rms_without_M42I:.3f})")

    # 2. one P per genotype, shared affine map (alternate to convergence)
    Pg = {g: best.P for g in GENOS}
    for _ in range(20):
        t = table(lambda g: Pg[g])
        coef, rms = affine_rms(t.theta.values, t.X.values)
        for g in GENOS:
            errs = []
            for P in P_GRID:
                lo, hi, logK, n, _ = params_at(by_g[g], P)
                d = xd[xd.genotype == g]
                th = theta(lo, hi, logK, n, d.titrant_conc)
                errs.append(np.sqrt(np.mean((d.X - coef[0] - coef[1] * th) ** 2)))
            Pg[g] = P_GRID[int(np.argmin(errs))]
    t = table(lambda g: Pg[g])
    coef, rms = affine_rms(t.theta.values, t.X.values)
    print(f"\nper-genotype P, shared map: rms {rms:.3f}, a {coef[0]:.2f}, b {coef[1]:.2f}")
    for g in GENOS:
        d = t[t.genotype == g]
        r = np.sqrt(np.mean((d.X - coef[0] - coef[1] * d.theta) ** 2))
        print(f"  {g:10s} P {Pg[g]:6.2f} uM{'  (extrapolated)' if d.extrap.iloc[0] else ''}  rms {r:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s.P, s.rms, "k-", label="all seven")
    ax.plot(s.P, s.rms_without_M42I, "--", color="#E69F00", label="without M42I")
    ax.axvspan(2, 20, color="0.92", zorder=0, label="mutants measured 2-20 µM")
    ax.set_xscale("log")
    ax.set_xlabel("apparent protein (µM dimer)")
    ax.set_ylabel("rms(X − (a + b θ))")
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig("resid/2026-10-01_protein-scan.pdf")


if __name__ == "__main__":
    main()
