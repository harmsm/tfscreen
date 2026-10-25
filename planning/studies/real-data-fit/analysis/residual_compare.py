"""
Where does the tube-offset fit gain its likelihood over the no-offset fit?

    python residual_compare.py <arm_a> <arm_b> [out_prefix]

Rebuilds each MAP arm's ln_cfu prediction for every observation in the full
library (linear growth, instant transition, single transformation, fixed
activity):

    ln_cfu = ln_cfu0[rep, cond_pre, g] + (k_pre + dk + m_pre θ) t_pre
             + (k_sel + dk + m_sel θ) t_sel + offset[tube]

from the extracted parameters and tfs_pred_theta.csv. The tube offset is
read off tfs_pred_growth.csv (its subset block): the median of q0.5 minus
the rebuilt prediction, per tube. The rebuild is checked against q0.5 there.
Then it scores every observation under each arm's count likelihood
(observe/growth_counts.count_distribution, the arm's own phi and inv_r) and
reports where ll_a - ll_b comes from: by tube, by genotype, by read depth.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.special import gammaln

pd.set_option("display.width", 250)

GROWTH = "inputs/growth.csv.gz"
TUBE = ["replicate", "condition_pre", "condition_sel", "titrant_conc", "t_sel"]
TUBE_INDEX = "resid/tube_index.csv"
TUBE_SHAPE = (3, 3, 2, 2, 1, 8)


def nb_logpmf(k, log_mu, phi, inv_r):
    log_c = -np.logaddexp(np.log(phi) - log_mu, np.log(inv_r))
    log_c = np.logaddexp(log_c, -30.0)
    c = np.exp(log_c)
    log_p = -np.logaddexp(0, log_mu - log_c)
    log_q = -np.logaddexp(0, log_c - log_mu)
    return (gammaln(k + c) - gammaln(c) - gammaln(k + 1)
            + c * log_p + k * log_q)


def q50(path, keys):
    return pd.read_csv(path, usecols=keys + ["q0.5"]).set_index(keys)["q0.5"]


def predict(arm, obs):
    k = q50(f"{arm}/tfs_params_growth_k.csv", ["condition_rep"])
    m = q50(f"{arm}/tfs_params_growth_m.csv", ["condition_rep"])
    dk = q50(f"{arm}/tfs_params_dk_geno.csv", ["genotype"])
    l0 = q50(f"{arm}/tfs_params_ln_cfu0.csv",
             ["replicate", "condition_pre", "genotype"])
    th = q50(f"{arm}/tfs_pred_theta.csv", ["genotype", "titrant_conc"])

    d = obs
    lo = l0.reindex(pd.MultiIndex.from_arrays(
        [d.replicate, d.condition_pre, d.genotype])).to_numpy()
    t = th.reindex(pd.MultiIndex.from_arrays(
        [d.genotype, d.titrant_conc])).to_numpy()
    g = dk.reindex(d.genotype).to_numpy()
    kp, mp = k.reindex(d.condition_pre).to_numpy(), m.reindex(d.condition_pre).to_numpy()
    ks, ms = k.reindex(d.condition_sel).to_numpy(), m.reindex(d.condition_sel).to_numpy()
    pred = (lo + (kp + g + mp * t) * d.t_pre.to_numpy()
            + (ks + g + ms * t) * d.t_sel.to_numpy())
    assert np.isfinite(pred).all(), f"{arm}: missing parameters"
    return pred


def tube_offsets(arm, obs, pred):
    """
    Per-tube offsets from the fitted params, mapped onto tubes through the
    tensor indices in tfs_pred_growth.csv. tfs-predict-growth zeroes the
    offsets (a prediction is for a typical tube), so its q0.5 must equal the
    rebuilt prediction without them; that is checked here.
    """
    idx = ["replicate_idx", "time_idx", "condition_pre_idx",
           "condition_sel_idx", "titrant_name_idx", "titrant_conc_idx"]
    pg = pd.read_csv(f"{arm}/tfs_pred_growth.csv",
                     usecols=TUBE + ["genotype", "ln_cfu", "q0.5"])
    pg = pg[pg.ln_cfu.notna()]
    o = obs[TUBE + ["genotype"]].copy()
    o["pred"] = pred
    j = pg.merge(o, on=TUBE + ["genotype"])
    diff = (j["q0.5"] - j["pred"]).abs()
    print(f"  {arm}: rebuilt vs q0.5 (no offsets) on {len(j)} rows: "
          f"median |diff| {diff.median():.2e}, q99 {diff.quantile(0.99):.2e}, "
          f"max {diff.max():.2e}")
    # a MAP rebuild is exact; an SVI rebuild uses parameter medians, which
    # only approximate the median prediction
    if os.path.exists(f"{arm}/tfs_fit_model_params.npz"):
        assert diff.max() < 1e-3
    else:
        return pd.Series(0.0, index=pd.read_csv(TUBE_INDEX).set_index(TUBE).index)
    z = np.load(f"{arm}/tfs_fit_model_params.npz")
    # training-grid indices (from smoke_map's config; same 118 tubes)
    tubes = pd.read_csv(TUBE_INDEX).set_index(TUBE)
    if "sample_offset_offset_auto_loc" not in z.files:
        return pd.Series(0.0, index=tubes.index)
    off = np.asarray(z["sample_offset_offset_auto_loc"])
    shape = TUBE_SHAPE
    assert np.prod(shape) == off.size, (shape, off.size)
    off = off.reshape(shape)
    v = off[tuple(tubes[c].to_numpy() for c in idx)]
    print(f"  {arm}: {len(v)} tube offsets, sd {v.std():.3f}, "
          f"range {v.min():.2f} to {v.max():.2f}")
    return pd.Series(v, index=tubes.index)


def dispersion(arm):
    """phi and inv_r: the MAP values, or an SVI guide's LogNormal medians."""
    f = f"{arm}/tfs_fit_model_params.npz"
    if os.path.exists(f):
        z = np.load(f)
        return float(z["growth_phi_auto_loc"]), float(z["growth_inv_r_auto_loc"])
    import dill
    from numpyro.optim import Adam
    with open(f"{arm}/tfs_fit_model_checkpoint.pkl", "rb") as fh:
        ck = dill.load(fh)
    p = Adam(1e-3).get_params(ck["svi_state"].optim_state)
    # SVI arms carry no tube offsets here (svi_nooffset); refuse otherwise
    assert "sample_offset_offset_loc" not in p, "SVI offsets not handled"
    return float(np.exp(p["growth_phi_loc"])), float(np.exp(p["growth_inv_r_loc"]))


def score(arm, obs):
    phi, inv_r = dispersion(arm)
    pred = predict(arm, obs)
    off = tube_offsets(arm, obs, pred)
    o = off.reindex(pd.MultiIndex.from_frame(obs[TUBE])).to_numpy()
    assert np.isfinite(o).all(), f"{arm}: tube without offset"
    pred = pred + o
    log_mu = obs.ln_sample_reads.to_numpy() + pred - obs.sample_ln_cfu.to_numpy()
    ll = nb_logpmf(obs.counts.to_numpy(float), log_mu, phi, inv_r)
    print(f"  {arm}: phi {phi:.3g}, inv_r {inv_r:.3g}, sum ll {ll.sum():.6g}")
    return pred, ll, off


def main():
    a, b = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "resid"
    obs = pd.read_csv(GROWTH, usecols=TUBE + ["genotype", "t_pre", "counts",
                                              "sample_reads", "sample_ln_cfu"])
    obs = obs[obs.genotype != "__unknown__"].reset_index(drop=True)
    obs["ln_sample_reads"] = np.log(obs.sample_reads)
    print(f"{len(obs)} observations, {obs.genotype.nunique()} genotypes, "
          f"{obs.groupby(TUBE).ngroups} tubes")

    pa, lla, offa = score(a, obs)
    pb, llb, offb = score(b, obs)
    obs["pred_a"], obs["pred_b"] = pa, pb
    obs["dll"] = lla - llb
    obs["ll_b"] = llb
    tot = obs.dll.sum()
    print(f"\nll_a - ll_b, whole library: {tot:.4g} nats")

    tube = (obs.groupby(TUBE)
            .agg(dll=("dll", "sum"), reads=("counts", "sum"),
                 dpred=("pred_a", "median"))
            .reset_index())
    tube["dpred"] = (obs.pred_a - obs.pred_b).groupby(
        [obs[c] for c in TUBE]).median().to_numpy()
    tube["offset_a"] = offa.reindex(pd.MultiIndex.from_frame(tube[TUBE])).to_numpy()
    tube = tube.sort_values("dll", ascending=False)
    tube.to_csv(f"{out}_tube.csv", index=False)
    print("\nby condition x IPTG (sum dll, 1e3 nats):")
    print((tube.pivot_table(index=["condition_pre", "condition_sel"],
                            columns="titrant_conc", values="dll",
                            aggfunc="sum") / 1e3).round(1).to_string())
    print("\nmean offset_a by condition x IPTG:")
    print(tube.pivot_table(index=["condition_pre", "condition_sel"],
                           columns="titrant_conc", values="offset_a",
                           aggfunc="mean").round(2).to_string())
    print("\ntubes, dll vs offset: corr(dll, |offset_a|) = "
          f"{np.corrcoef(tube.dll, tube.offset_a.abs())[0, 1]:.2f}")
    s = tube.dll.sort_values(ascending=False)
    for f in (0.1, 0.25):
        print(f"top {f:.0%} of tubes carry {s.head(int(f * len(s))).sum() / tot:.2f} of the gain")
    print(tube.head(12).round(3).to_string(index=False))

    geno = (obs.groupby("genotype")
            .agg(dll=("dll", "sum"), reads=("counts", "sum"), n=("dll", "size"))
            .reset_index().sort_values("dll", ascending=False))
    geno.to_csv(f"{out}_geno.csv", index=False)
    print("\ntop 15 genotypes:")
    print(geno.head(15).round(1).to_string(index=False))
    print("bottom 8 genotypes:")
    print(geno.tail(8).round(1).to_string(index=False))
    s = geno.dll.sort_values(ascending=False)
    for n in (10, 100, 1000, 10000):
        print(f"top {n} genotypes: {s.head(n).sum() / tot:.2f} of the gain")
    print(f"genotypes with dll > 0: {(geno.dll > 0).mean():.2f}")
    geno["read_bin"] = pd.cut(geno.reads, [-1, 0, 10, 100, 1e3, 1e4, 1e5, 1e9])
    print("\nby genotype total reads:")
    print(geno.groupby("read_bin", observed=True).dll
          .agg(["sum", "mean", "size"]).round(1).to_string())

    obs["count_bin"] = pd.cut(obs.counts, [-1, 0, 1, 5, 20, 100, 1e3, 1e9])
    print("\nby observation count:")
    print(obs.groupby("count_bin", observed=True).dll
          .agg(["sum", "mean", "size"]).round(2).to_string())
    obs.drop(columns="count_bin").to_parquet(f"{out}_obs.parquet")


if __name__ == "__main__":
    main()
