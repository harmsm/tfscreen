"""
Joint fit of the anisotropy data: anisotropy -> theta and Hill curves at once.

For every well (batch b = day x biorep, genotype g, protein P, IPTG c):

    r = f_b + s_b * delta * theta_gP(c) + noise

- f_b: the batch's free-DNA level. Free-DNA wells (protein 0, DNA present)
  are observations with theta = 0.
- s_b: the batch's scale, log s_b ~ N(0, 0.25). The two wt-titration
  batches define s = 1.
- delta: fully bound minus free, shared by all batches.
- theta_gP(c) = theta_hi + (theta_lo - theta_hi) / (1 + (c / K)^n), a
  repressor curve per genotype and protein concentration.
- wt: theta_lo(P) = P / (Kd + P), a binding isotherm across 0.02-80 uM,
  which is what sets delta. wt's control wells in the mutant file share
  wt's curves at their protein concentration.

Wells are the mean of their six reads. Residuals use a robust (soft L1)
loss with a noise scale re-estimated from the fit, so a few bad wells do
not move a batch. Excluded: buffer-only wells, DNA-free wells, V95N at
20 uM (reads far above the bound level, no IPTG response: aggregation).

Then each genotype's Hill parameters are interpolated to TARGET_PROTEIN_UM
on log protein concentration, and the curve is written at the screen's
IPTG concentrations as a binding_df for tfs-configure-model.

Run from this directory:  python binding_fit.py
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, logit

OUT = "processed"
TARGET_PROTEIN_UM = 10.0
SCREEN_IPTG_MM = [0.0, 1e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 1.0]
EXCLUDE = [("V95N", 20.0)]
REFERENCE_SOURCE = "wt_20260826"
CURVE_PARAMS = 4       # theta_lo, theta_hi, K, n

# weak priors, in the fit's parameter units
PRIOR = {"a": (0.0, 3.0), "b": (0.0, 3.0),          # logit theta_lo / hi
         "logK": (np.log(0.01), 3.0),               # ln mM
         "logn": (np.log(1.5), 0.7),
         "logs": (0.0, 0.25),
         "f": (0.138, 0.005)}


def load_wells():
    w = pd.read_csv(os.path.join(OUT, "binding_wells.csv"))
    w = w[w["Buffer_only"] != 1]
    has_dna = (w["dna_nM"].isna() & (w["source"] == REFERENCE_SOURCE)) \
        | (w["dna_nM"] > 0)
    w = w[has_dna].copy()
    w["batch"] = w["Day"].astype(str) + "_b" + w["Biorep"].astype(int).astype(str)
    w["free"] = w["protein_uM_dimer"] == 0
    w = w[w["free"] | (w["genotype"] != "blank")]
    for g, p in EXCLUDE:
        w = w[~((w["genotype"] == g) & (w["protein_uM_dimer"] == p))]
    return w.reset_index(drop=True)


class Model:
    def __init__(self, wells):
        self.w = wells
        self.batches = sorted(wells["batch"].unique())
        ref = wells.loc[wells["source"] == REFERENCE_SOURCE, "batch"].unique()
        self.ref_batches = set(ref)
        curves = (wells[~wells["free"]]
                  .groupby(["genotype", "protein_uM_dimer"]).size().index)
        self.curves = [c for c in curves if c[0] != "wt"]
        self.wt_P = sorted(wells.loc[(wells["genotype"] == "wt")
                                     & ~wells["free"], "protein_uM_dimer"]
                           .unique())

        names = ["log_delta", "log_Kd"]
        for g, p in self.curves:
            names += [f"{g}|{p}|{k}" for k in ("a", "b", "logK", "logn")]
        for p in self.wt_P:
            names += [f"wt|{p}|{k}" for k in ("b", "logK", "logn")]
        for b in self.batches:
            names.append(f"f|{b}")
            if b not in self.ref_batches:
                names.append(f"logs|{b}")
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}

    def x0(self):
        x = np.zeros(len(self.names))
        x[self.idx["log_delta"]] = np.log(0.032)
        x[self.idx["log_Kd"]] = np.log(1.0)
        for n, i in self.idx.items():
            key = n.split("|")[-1]
            if key == "a":
                x[i] = 1.0
            elif key == "b":
                x[i] = -2.0
            elif key == "logK":
                x[i] = np.log(0.01)
            elif key == "logn":
                x[i] = np.log(1.5)
            elif n.startswith("f|"):
                x[i] = 0.138
        return x

    def curve_params(self, x, g, p):
        """theta_lo, theta_hi, K, n for one genotype and protein conc."""
        if g == "wt":
            kd = np.exp(x[self.idx["log_Kd"]])
            lo = p / (kd + p)
            hi = expit(x[self.idx[f"wt|{p}|b"]])
            K = np.exp(x[self.idx[f"wt|{p}|logK"]])
            n = np.exp(x[self.idx[f"wt|{p}|logn"]])
        else:
            lo = expit(x[self.idx[f"{g}|{p}|a"]])
            hi = expit(x[self.idx[f"{g}|{p}|b"]])
            K = np.exp(x[self.idx[f"{g}|{p}|logK"]])
            n = np.exp(x[self.idx[f"{g}|{p}|logn"]])
        return lo, hi, K, n

    @staticmethod
    def hill(lo, hi, K, n, c):
        c = np.asarray(c, dtype=float)
        frac = np.where(c > 0, (np.maximum(c, 1e-300) / K) ** n, 0.0)
        return hi + (lo - hi) / (1.0 + frac)

    def theta(self, x):
        w = self.w
        th = np.zeros(len(w))
        for (g, p), rows in w[~w["free"]].groupby(["genotype",
                                                   "protein_uM_dimer"]).groups.items():
            lo, hi, K, n = self.curve_params(x, g, p)
            th[rows] = self.hill(lo, hi, K, n, w.loc[rows, "iptg_mM"])
        return th

    def predict(self, x):
        w = self.w
        f = np.array([x[self.idx[f"f|{b}"]] for b in w["batch"]])
        s = np.array([1.0 if b in self.ref_batches
                      else np.exp(x[self.idx[f"logs|{b}"]]) for b in w["batch"]])
        delta = np.exp(x[self.idx["log_delta"]])
        return f + s * delta * self.theta(x)

    def residuals(self, x, sigma):
        res = [(self.w["r_mean"].to_numpy() - self.predict(x)) / sigma]
        pri = []
        for n, i in self.idx.items():
            key = "f" if n.startswith("f|") else \
                "logs" if n.startswith("logs|") else n.split("|")[-1]
            if key in PRIOR:
                mu, sd = PRIOR[key]
                pri.append((x[i] - mu) / sd)
        return np.concatenate(res + [np.array(pri)])


def fit(model):
    x = model.x0()
    sigma = 0.002
    nobs = len(model.w)
    for _ in range(3):
        sol = least_squares(model.residuals, x, args=(sigma,),
                            loss="soft_l1", f_scale=2.0, method="trf",
                            max_nfev=20000)
        x = sol.x
        resid = model.w["r_mean"].to_numpy() - model.predict(x)
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    J = sol.jac
    cov = np.linalg.pinv(J.T @ J)
    return x, cov, sigma, sol, resid[:nobs]


def curve_table(model, x, cov):
    rows = []
    se = np.sqrt(np.diag(cov))
    for g, p in list(model.curves) + [("wt", p) for p in model.wt_P]:
        lo, hi, K, n = model.curve_params(x, g, p)
        row = dict(genotype=g, protein_uM_dimer=p, theta_lo=lo, theta_hi=hi,
                   log_K_mM=np.log(K), hill_n=n)
        for k in ("a", "b", "logK", "logn"):
            name = f"{g}|{p}|{k}"
            if name in model.idx:
                row[f"{k}_se"] = se[model.idx[name]]
        rows.append(row)
    return pd.DataFrame(rows)


def interpolate(curves, target=TARGET_PROTEIN_UM):
    """Hill parameters at the target protein conc, linear in ln(protein)."""
    out = []
    for g, d in curves.groupby("genotype"):
        d = d.set_index("protein_uM_dimer").sort_index()
        lower = [p for p in d.index if p <= target]
        upper = [p for p in d.index if p >= target]
        if not lower or not upper:
            continue
        p0, p1 = max(lower), min(upper)
        t = 0.0 if p0 == p1 else (np.log(target) - np.log(p0)) / \
            (np.log(p1) - np.log(p0))
        r0, r1 = d.loc[p0], d.loc[p1]
        clip = lambda v: np.clip(v, 1e-4, 1 - 1e-4)
        lg = lambda k: (1 - t) * logit(clip(r0[k])) + t * logit(clip(r1[k]))
        lin = lambda k: (1 - t) * r0[k] + t * r1[k]
        out.append(dict(genotype=g, protein_uM_dimer=target,
                        bracket=f"{p0:g}-{p1:g}",
                        theta_lo=expit(lg("theta_lo")),
                        theta_hi=expit(lg("theta_hi")),
                        log_K_mM=lin("log_K_mM"),
                        hill_n=np.exp(lin_log(r0, r1, t))))
    return pd.DataFrame(out)


def lin_log(r0, r1, t):
    return (1 - t) * np.log(r0["hill_n"]) + t * np.log(r1["hill_n"])


def curve_sd(model, x, cov, interp, well_sd, step=1e-5):
    """
    theta_std for each (genotype, screen IPTG) point of the 10 uM curves.

    The curve's own uncertainty, by linear propagation of the fit's
    covariance through the interpolation in theta itself (J C J^T; sampling
    the logit parameters blew up where a plateau sits at theta = 1, M42I at
    20 uM), added in quadrature to the well noise, then scaled by
    sqrt(points / curve parameters): a curve's points are correlated,
    carrying about CURVE_PARAMS independent values, and the binding
    likelihood treats them as independent. The interpolation's model error
    (linear in ln protein) is not included.
    """
    def values(xv):
        ci = interpolate(curve_table(model, xv, cov)).set_index("genotype")
        return np.concatenate([
            Model.hill(ci.loc[g, "theta_lo"], ci.loc[g, "theta_hi"],
                       np.exp(ci.loc[g, "log_K_mM"]), ci.loc[g, "hill_n"],
                       SCREEN_IPTG_MM)
            for g in interp["genotype"]])

    base = values(x)
    J = np.zeros((len(base), len(x)))
    for k in range(len(x)):
        xp = x.copy()
        xp[k] += step
        J[:, k] = (values(xp) - base) / step
    var = np.einsum("ij,jk,ik->i", J, cov, J)
    inflate = np.sqrt(len(SCREEN_IPTG_MM) / CURVE_PARAMS)
    sd = inflate * np.sqrt(np.maximum(var, 0) + well_sd ** 2)
    n = len(SCREEN_IPTG_MM)
    return {g: sd[i * n:(i + 1) * n] for i, g in enumerate(interp["genotype"])}


def binding_df(interp, sd_by_genotype):
    rows = []
    for _, r in interp.iterrows():
        th = Model.hill(r["theta_lo"], r["theta_hi"], np.exp(r["log_K_mM"]),
                        r["hill_n"], SCREEN_IPTG_MM)
        sd = sd_by_genotype[r["genotype"]]
        for c, t, s in zip(SCREEN_IPTG_MM, th, sd):
            rows.append(dict(genotype=r["genotype"], titrant_name="iptg",
                             titrant_conc=c, theta_obs=float(t),
                             theta_std=float(s)))
    return pd.DataFrame(rows)


def main():
    wells = load_wells()
    model = Model(wells)
    print(f"{len(wells)} wells, {len(model.batches)} batches, "
          f"{len(model.curves)} mutant curves, wt at {model.wt_P} uM, "
          f"{len(model.names)} parameters")
    x, cov, sigma, sol, resid = fit(model)
    print(f"converged: {sol.status > 0} ({sol.message}); well noise "
          f"{sigma:.4f} in r; delta {np.exp(x[model.idx['log_delta']]):.4f}; "
          f"wt Kd {np.exp(x[model.idx['log_Kd']]):.3g} uM")

    se = np.sqrt(np.diag(cov))
    delta = np.exp(x[model.idx["log_delta"]])
    batches = []
    for b in model.batches:
        f = x[model.idx[f"f|{b}"]]
        if b in model.ref_batches:
            s, s_se = 1.0, 0.0
        else:
            i = model.idx[f"logs|{b}"]
            s, s_se = np.exp(x[i]), np.exp(x[i]) * se[i]
        batches.append(dict(batch=b, free=f, free_se=se[model.idx[f"f|{b}"]],
                            scale=s, scale_se=s_se,
                            num_wells=int((wells["batch"] == b).sum())))
    batches = pd.DataFrame(batches)
    print(batches.round(4).to_string(index=False))

    # per-well theta on the common scale, for plots and checks
    f = np.array([x[model.idx[f"f|{b}"]] for b in wells["batch"]])
    s = wells["batch"].map(batches.set_index("batch")["scale"]).to_numpy()
    wells["theta"] = (wells["r_mean"] - f) / (s * delta)
    wells["theta_fit"] = model.theta(x)
    wells["resid_r"] = resid
    wells["resid_z"] = resid / sigma

    curves = curve_table(model, x, cov)
    interp = interpolate(curves)

    bdf = binding_df(interp, curve_sd(model, x, cov, interp, sigma / delta))

    batches.to_csv(os.path.join(OUT, "binding_batches.csv"), index=False)
    wells.to_csv(os.path.join(OUT, "binding_wells_theta.csv"), index=False)
    curves.to_csv(os.path.join(OUT, "binding_curves.csv"), index=False)
    interp.to_csv(os.path.join(OUT, "binding_curves_10uM.csv"), index=False)
    bdf.to_csv(os.path.join(OUT, "binding_df_10uM.csv"), index=False)
    print(curves.round(3).to_string(index=False))
    print(interp.round(3).to_string(index=False))
    print(bdf.pivot(index="titrant_conc", columns="genotype",
                    values="theta_std").round(3).to_string())
    print("worst wells (|z| > 4):")
    print(wells.loc[wells["resid_z"].abs() > 4,
                    ["batch", "genotype", "protein_uM_dimer", "iptg_mM",
                     "r_mean", "resid_z"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
