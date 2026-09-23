"""
Step 3.0 of planning/congression-physics-plan.md: how should the fit evaluate
the congressed class's E_S exp(G_{g,S}(t))?

Ground truth comes from the simulator's own machinery (library_prediction on
examples/simulate/simulate_config.yaml, with the real design's
library_mixture and lambda = 0.357). For each bulk genotype g and each
growth condition (titrant, condition_pre/sel, t_pre, t_sel), the observable
(relative to ln_cfu0) is

    L_g = log[ w_clean exp(G_clean) + sum_o w_o E_{S~o} exp(G_{g,S}) ]

with the cell rules of simulate/cell_rules.py (max theta, activity from the
max plasmid, dilution dk) and the same growth pipeline as the simulator
(growth_rate_one_condition + growth transition). Co-residents S: N ~
Poisson(lambda) | N >= 1 plasmids drawn from origin o's plasmid
distribution (each sub-library is transformed separately).

"Exact": N = 1 enumerated exactly over the origin's genotypes; N >= 2 by
Monte Carlo (16% of congressed cells at lambda = 0.357).

Estimators of each congressed term (plugged into the same mixture):
  rate_avg   exp(E_S[G])
  cumulant2  exp(E_S[G] + Var_S[G]/2)
  n1_only    exact, but every congressed cell has exactly one co-resident
  mcK        K fixed co-resident sets per origin, shared by all focal
             genotypes (common random numbers, as a fit would use)
  pooled     exact, but co-residents from the pooled bulk library
References:
  current    today's fit: one trajectory, theta -> E[max(theta_g, M)] over
             an unweighted all-genotype background (spiked-origin genotypes
             uncorrected), own dk and activity
  none       no congression at all (G_clean)

Errors are in ln_cfu units (natural log), approx - exact.
"""

import sys
import numpy as np
import pandas as pd

from tfscreen.util.io import read_yaml
from tfscreen.simulate.library_prediction import library_prediction
from tfscreen.simulate.thermo_to_growth import growth_rate_one_condition
from tfscreen.simulate.selection_experiment import _compute_kt_arrays

LAM = 0.357
REAL_MIXTURE = {"single-1": 100, "single-2": 100, "double-1-2": 1000,
                "spiked": 1}
DK_SCALES = [1.0, 0.3, 0.1, 0.03, 0.015, 0.0]
MC_N2_DRAWS = 400          # N >= 2 draws per (focal, origin) for "exact"
K_LIST = [1, 2, 4, 8, 16, 32]
K_REPEATS = 10
DETECT = -7.0              # detectable: L_g - L_wt >= DETECT
SEED = 11

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Ground truth phenotypes and design
# ---------------------------------------------------------------------------

# Default input: the repository's example simulate config.
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3]
cfg_path = (sys.argv[1] if len(sys.argv) > 1
            else str(_REPO / "examples" / "simulate" / "simulate_config.yaml"))
cf = read_yaml(cfg_path)
cf["transformation_poisson_lambda"] = LAM
cf["library_mixture"] = dict(REAL_MIXTURE)
cf["seed"] = SEED
library_df, phenotype_df, genotype_theta_df, parameters_df = library_prediction(cf)[:4]
growth_params = cf["growth"]
theta_rescale = cf.get("theta_rescale", "passthrough")
growth_transition = cf.get("growth_transition")

genotypes = list(parameters_df["genotype"].astype(str))
gidx = {g: i for i, g in enumerate(genotypes)}
G = len(genotypes)

# theta per genotype per titrant point
th = genotype_theta_df.copy()
th["genotype"] = th["genotype"].astype(str)
titr = (th[["titrant_name", "titrant_conc"]].drop_duplicates()
        .sort_values(["titrant_name", "titrant_conc"]).reset_index(drop=True))
titr["t_idx"] = np.arange(len(titr))
th = th.merge(titr, on=["titrant_name", "titrant_conc"])
theta_gt = np.full((G, len(titr)), np.nan)
theta_gt[th["genotype"].map(gidx).to_numpy(), th["t_idx"].to_numpy()] = th["theta"].to_numpy()

dk0 = parameters_df["dk_geno"].to_numpy(dtype=float)
act = parameters_df["activity"].to_numpy(dtype=float)

# growth conditions
cond = (phenotype_df[phenotype_df["replicate"] == phenotype_df["replicate"].min()]
        [["library", "titrant_name", "titrant_conc", "condition_pre",
          "condition_sel", "t_pre", "t_sel"]]
        .drop_duplicates().reset_index(drop=True))
cond = cond.merge(titr, on=["titrant_name", "titrant_conc"])
C = len(cond)
cond_t = cond["t_idx"].to_numpy()

# per-origin plasmid distributions (design, no assembly skew)
lib = library_df.copy()
lib["genotype"] = lib["genotype"].astype(str)
lib["library_origin"] = lib["library_origin"].astype(str)
lib = lib[lib["genotype"].isin(gidx)]
origins = sorted(lib["library_origin"].unique())
bulk_origins = [o for o in origins if o != "spiked"]
probs = {}
for o in origins:
    sub = lib[lib["library_origin"] == o]
    p = np.zeros(G)
    np.add.at(p, sub["genotype"].map(gidx).to_numpy(), sub["weight"].to_numpy())
    probs[o] = p / p.sum()
mass = {o: REAL_MIXTURE[o] * probs[o] for o in origins}
total_mass = sum(mass.values())
in_spiked = mass["spiked"] > 0
pooled_bulk = sum(mass[o] for o in bulk_origins)
pooled_bulk = pooled_bulk / pooled_bulk.sum()

p_cong = 1.0 - np.exp(-LAM)
# weight of each (genotype, bulk origin) congressed class, and clean
w_orig = {o: np.divide(mass[o], total_mass, out=np.zeros(G), where=total_mass > 0) * p_cong
          for o in bulk_origins}
w_clean = 1.0 - sum(w_orig.values())
f_g = np.divide(sum(mass[o] for o in bulk_origins), total_mass,
                out=np.zeros(G), where=total_mass > 0)
present = total_mass > 0

# Poisson(lambda) | N >= 1 split into N = 1 and N >= 2
p_n = np.array([np.exp(-LAM) * LAM**n / np.prod(np.arange(1, n + 1)) for n in range(0, 12)])
p1 = p_n[1] / (1 - p_n[0])
p_n2 = p_n[2:] / p_n[2:].sum()          # distribution of N given N >= 2
n2_values = np.arange(2, 12)

# ---------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------

def kt(theta_c, act_c, dk):
    """theta_c, act_c: (..., C) over growth conditions; dk: (...). -> (..., C)"""
    shape = theta_c.shape
    out = np.empty(shape)
    dk_b = np.broadcast_to(np.asarray(dk)[..., None], shape)
    for c in range(C):
        row = cond.iloc[c]
        th_c = theta_c[..., c].ravel()
        a_c = act_c[..., c].ravel()
        d_c = dk_b[..., c].ravel()
        k_pre = growth_rate_one_condition(row["condition_pre"], th_c, a_c, d_c,
                                          growth_params, theta_rescale)
        k_sel = growth_rate_one_condition(row["condition_sel"], th_c, a_c, d_c,
                                          growth_params, theta_rescale)
        n = len(th_c)
        v = _compute_kt_arrays(k_pre, k_sel, np.full(n, float(row["t_pre"])),
                               np.full(n, float(row["t_sel"])), th_c,
                               np.full(n, row["condition_pre"], dtype=object),
                               growth_transition)
        out[..., c] = v.reshape(shape[:-1])
    return out


def cell_G(focal, co_sets, dk):
    """
    focal: (F,) genotype idx. co_sets: (F, D, Nmax) co-resident idx, -1 = none.
    Returns G (F, D, C) for cells {focal} + co_sets under max theta /
    activity-follows-theta / dilution dk.
    """
    F, D, Nmax = co_sets.shape
    slots = np.concatenate([np.broadcast_to(focal[:, None, None], (F, D, 1)), co_sets], axis=2)
    valid = slots >= 0
    sl = np.where(valid, slots, 0)
    th_s = theta_gt[sl][..., cond_t]                     # (F,D,P,C)
    th_s = np.where(valid[..., None], th_s, -np.inf)
    best = np.argmax(th_s, axis=2)                        # (F,D,C)
    th_cell = np.take_along_axis(th_s, best[:, :, None, :], axis=2)[:, :, 0, :]
    a_s = np.broadcast_to(act[sl][..., None], th_s.shape)
    a_cell = np.take_along_axis(a_s, best[:, :, None, :], axis=2)[:, :, 0, :]
    n_in = valid.sum(axis=2)
    dk_cell = np.where(valid, dk[sl], 0.0).sum(axis=2) / n_in
    return kt(th_cell, a_cell, dk_cell)


def logmeanexp(x, axis, w=None):
    m = np.max(x, axis=axis, keepdims=True)
    if w is None:
        return (m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)
    return (m + np.log(np.sum(w * np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def draw_sets(p, F, D, n_values, n_probs, rng):
    n = rng.choice(n_values, size=(F, D), p=n_probs)
    nmax = int(n.max())
    co = rng.choice(G, size=(F, D, nmax), p=p)
    co = np.where(np.arange(nmax)[None, None, :] < n[..., None], co, -1)
    return co


def congressed_terms(focal, p, dk, rng, chunk=60):
    parts = [_congressed_terms(focal[i:i + chunk], p, dk, rng)
             for i in range(0, len(focal), chunk)]
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}


def _congressed_terms(focal, p, dk, rng):
    """
    For focal genotypes and one co-resident distribution p, return
    dict of per-(focal, condition) log E exp(G) for exact / n1_only, and the
    moments E[G], Var[G] (exact mixture of N=1 enumeration and N>=2 MC).
    """
    F = len(focal)
    support = np.where(p > 0)[0]
    # N = 1: enumerate
    co1 = np.broadcast_to(support[None, :, None], (F, len(support), 1)).copy()
    G1 = cell_G(focal, co1, dk)                                # (F,S,C)
    w1 = p[support][None, :, None]
    log_e1 = logmeanexp(G1, 1, w1)
    m1 = np.sum(w1 * G1, axis=1)
    s1 = np.sum(w1 * G1**2, axis=1)
    # N >= 2: Monte Carlo
    co2 = draw_sets(p, F, MC_N2_DRAWS, n2_values, p_n2, rng)
    G2 = cell_G(focal, co2, dk)
    log_e2 = logmeanexp(G2, 1)
    m2 = G2.mean(axis=1)
    s2 = (G2**2).mean(axis=1)
    exact = np.logaddexp(np.log(p1) + log_e1, np.log(1 - p1) + log_e2)
    mean = p1 * m1 + (1 - p1) * m2
    var = p1 * s1 + (1 - p1) * s2 - mean**2
    return {"exact": exact, "n1_only": log_e1, "mean": mean, "var": var}


def mixture(clean_G, terms_by_origin, focal):
    """log[w_clean e^Gclean + sum_o w_o e^{term_o}] for focal genotypes."""
    acc = np.log(w_clean[focal])[:, None] + clean_G
    for o, term in terms_by_origin.items():
        wo = w_orig[o][focal]
        with np.errstate(divide="ignore"):
            acc = np.logaddexp(acc, np.log(wo)[:, None] + term)
    return acc


# today's fit rule: E[max(theta_g, M)], M = max of Poisson(lambda) draws
# from the unweighted all-genotype theta distribution (per titrant point)
def current_theta():
    grid = np.linspace(0, 1, 2001)
    out = theta_gt.copy()
    for t in range(theta_gt.shape[1]):
        pop = np.sort(theta_gt[present, t])
        F = np.searchsorted(pop, grid, side="right") / len(pop)
        P_M_le = np.exp(LAM * (F - 1.0))                    # includes N = 0
        tail = 1.0 - P_M_le                                  # P(M > t)
        cum = np.concatenate([[0], np.cumsum((tail[1:] + tail[:-1]) / 2 * np.diff(grid))])
        G_tail = cum[-1] - np.interp(theta_gt[:, t], grid, cum)
        out[:, t] = theta_gt[:, t] + G_tail
    return out

theta_current = current_theta()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

focal = np.where(present & (f_g > 0))[0]
wt = gidx["wt"]
rows = []
by_time = []
for s in DK_SCALES:
    dk = s * dk0
    clean_G = kt(theta_gt[focal][:, cond_t], np.broadcast_to(act[focal][:, None], (len(focal), C)),
                 dk[focal])
    per_origin = {o: congressed_terms(focal, probs[o], dk, rng) for o in bulk_origins
                  if np.any(w_orig[o][focal] > 0)}
    pooled = congressed_terms(focal, pooled_bulk, dk, rng)

    exact = mixture(clean_G, {o: t["exact"] for o, t in per_origin.items()}, focal)
    est = {
        "rate_avg": mixture(clean_G, {o: t["mean"] for o, t in per_origin.items()}, focal),
        "cumulant2": mixture(clean_G, {o: t["mean"] + t["var"] / 2 for o, t in per_origin.items()}, focal),
        "n1_only": mixture(clean_G, {o: t["n1_only"] for o, t in per_origin.items()}, focal),
        "pooled": mixture(clean_G, {o: pooled["exact"] for o in per_origin}, focal),
        "none": clean_G,
    }
    cur_theta = np.where(in_spiked[focal][:, None], theta_gt[focal], theta_current[focal])
    est["current"] = kt(cur_theta[:, cond_t], np.broadcast_to(act[focal][:, None], (len(focal), C)),
                        dk[focal])
    for K in K_LIST:
        errs = []
        for r in range(K_REPEATS):
            terms = {}
            for o in per_origin:
                co = draw_sets(probs[o], 1, K, np.arange(1, 12),
                               p_n[1:] / p_n[1:].sum(), rng)
                co = np.broadcast_to(co, (len(focal),) + co.shape[1:]).copy()
                terms[o] = logmeanexp(cell_G(focal, co, dk), 1)
            errs.append(mixture(clean_G, terms, focal) - exact)
        est[f"mc{K}"] = errs   # list of repeats

    # detectable: relative to wt's exact observable in the same condition
    wt_pos = np.where(focal == wt)[0]
    ref = exact[wt_pos[0]] if len(wt_pos) else np.median(exact, axis=0)
    detect = (exact - ref[None, :]) >= DETECT

    for name, val in est.items():
        reps = val if isinstance(val, list) else [val - exact]
        for subset, mask in (("all", np.ones_like(detect)), ("detectable", detect)):
            e = np.concatenate([r[mask] for r in reps])
            rows.append({"dk_scale": s, "estimator": name, "subset": subset,
                         "rms": np.sqrt(np.mean(e**2)),
                         "mean": np.mean(e),
                         "p95_abs": np.quantile(np.abs(e), 0.95),
                         "max_abs": np.max(np.abs(e)),
                         "n": e.size})
        for c in range(C):
            e = np.concatenate([r[detect[:, c], c] for r in reps])
            if e.size:
                by_time.append({"dk_scale": s, "estimator": name,
                                "library": cond.loc[c, "library"],
                                "t_sel": cond.loc[c, "t_sel"],
                                "rms": np.sqrt(np.mean(e**2))})

    # size of the congression effect itself (exact vs none), detectable
    print(f"dk_scale={s}: done", flush=True)

res = pd.DataFrame(rows)
res.to_csv("study_summary.csv", index=False)
bt = (pd.DataFrame(by_time).groupby(["dk_scale", "estimator", "library", "t_sel"])
      ["rms"].mean().reset_index())
bt.to_csv("study_by_time.csv", index=False)

pd.set_option("display.width", 200)
print(f"\n{len(focal)} bulk genotypes x {C} conditions; lambda={LAM}; "
      f"dk_geno SD (scale 1) = {np.std(dk0[present]):.4f}/min")
print(res[res.subset == "detectable"].pivot_table(index="estimator", columns="dk_scale",
                                                   values="rms").round(3).to_string())
print("\np95 |error| (detectable)")
print(res[res.subset == "detectable"].pivot_table(index="estimator", columns="dk_scale",
                                                   values="p95_abs").round(3).to_string())
print("\nmean error (detectable)")
print(res[res.subset == "detectable"].pivot_table(index="estimator", columns="dk_scale",
                                                   values="mean").round(3).to_string())
