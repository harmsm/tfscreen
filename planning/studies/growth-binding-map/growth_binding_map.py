"""
Step 0a of planning/analysis-roadmap.md: a model-free look at how growth
relates to in vitro binding, for the genotypes with binding data.

For genotype g and wt in the same tube, ln(reads_g / reads_wt) does not
depend on the tube's total population, so its slope over t_sel is

    s_g(c, x) = dk_g + m_c * (X_g(x) - X_wt(x))

for condition c and IPTG concentration x, with no OD600 and no tube
noise. If growth is linear in the in vitro occupancy theta, then for each
selective condition the points (theta_g(x) - theta_wt(x), s_g(c, x))
from every genotype lie on one line per genotype with a common slope m_c,
offset by dk_g. Genotype-specific slopes or curvature mean the linear map
does not hold.

Second pass (optional SAMPLE_DF): absolute slopes. The wt-relative
design removes the tube total but also most of the signal, because the
binding genotypes' curves differ from wt's by only a few tenths of theta.
With the tube totals from the global smoothed fit (SAMPLE_DF's fit_lncfu),
ln_cfu_g = ln(reads_g / D) + fit_lncfu, and each genotype's absolute slope

    s_g(c, x) = k_c + dk_g + m_c * X_g(x)

can be tested against theta_g(x) itself, which for wt spans nearly 0-1.

Usage:
    python growth_binding_map.py DATA_DIR OUT_DIR [SAMPLE_DF]

DATA_DIR holds counts.parquet, samples.parquet and binding.csv from
planning/studies/step0-data/extract_snapshot.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt


def wt_relative_slopes(counts, samples, genotypes):
    """Weighted slope of ln(reads_g / reads_wt) on t_sel per genotype x
    replicate x condition_sel x titrant_conc (Poisson weights)."""
    c = counts.pivot_table(index="sample", columns="genotype",
                           values="counts", aggfunc="sum", observed=True)
    rows = []
    s = samples.set_index("sample")
    for g in genotypes:
        if g == "wt":
            continue
        d = pd.DataFrame({"cg": c[g], "cw": c["wt"]}).join(s)
        d = d[(d["cg"] > 0) & (d["cw"] > 0)]
        d["y"] = np.log(d["cg"] / d["cw"])
        d["w"] = 1.0 / (1.0 / d["cg"] + 1.0 / d["cw"])
        for key, grp in d.groupby(["replicate", "condition_sel",
                                    "titrant_conc"]):
            if grp["t_sel"].nunique() < 2:
                continue
            f = smf.wls("y ~ t_sel", data=grp, weights=grp["w"]).fit()
            rows.append({"genotype": g, "replicate": key[0],
                         "condition_sel": key[1], "titrant_conc": key[2],
                         "slope": f.params["t_sel"],
                         "slope_se": f.bse["t_sel"] if len(grp) > 2 else np.nan,
                         "n_tubes": len(grp),
                         "min_reads": int(grp["cg"].min())})
    return pd.DataFrame(rows)


def absolute_slopes(counts, samples, sample_df, genotypes):
    """Slope of ln(reads_g / D) + fit_lncfu on t_sel per genotype x
    replicate x condition_sel x titrant_conc."""
    tot = pd.read_csv(sample_df, index_col=0)
    tot["condition_sel"] = tot["condition_sel"].str.strip()
    tot = tot.rename(columns={"rep": "replicate"})
    keys = ["replicate", "condition_sel", "titrant_conc", "t_sel"]
    smp = samples.merge(tot[keys + ["fit_lncfu"]], on=keys, how="inner")
    print(f"\nsecond pass: {len(smp)} of {len(samples)} tubes matched to a "
          f"smoothed total")
    c = counts.pivot_table(index="sample", columns="genotype",
                           values="counts", aggfunc="sum", observed=True)
    smp = smp.set_index("sample")
    rows = []
    for g in genotypes:
        d = smp.join(c[g].rename("cg"), how="inner")
        d = d[d["cg"] > 0]
        d["y"] = np.log(d["cg"] / d["freq_denominator"]) + d["fit_lncfu"]
        for key, grp in d.groupby(["replicate", "condition_sel",
                                    "titrant_conc"]):
            if grp["t_sel"].nunique() < 2:
                continue
            f = smf.ols("y ~ t_sel", data=grp).fit()
            rows.append({"genotype": g, "replicate": key[0],
                         "condition_sel": key[1], "titrant_conc": key[2],
                         "slope": f.params["t_sel"], "n_tubes": len(grp)})
    return pd.DataFrame(rows)


def test_map(sl, theta_col, label):
    """Per condition: common m vs genotype-specific m vs curvature, with the
    slope noise estimated from replicate differences."""
    w = sl.pivot_table(index=["genotype", "condition_sel", "titrant_conc"],
                       columns="replicate", values="slope")
    diff = (w.iloc[:, 1] - w.iloc[:, 0]).dropna()
    sd = np.sqrt(diff.var() / 2)
    print(f"{label}: slope SD from replicate differences {sd:.5f} per min")
    out = []
    for cond, d in sl.groupby("condition_sel"):
        common = smf.ols(f"slope ~ C(genotype) + {theta_col}", d).fit()
        per_g = smf.ols(f"slope ~ C(genotype) + C(genotype):{theta_col}", d).fit()
        curved = smf.ols(f"slope ~ C(genotype) + {theta_col} + "
                         f"I({theta_col}**2)", d).fit()
        # F-tests with the replicate-based noise as the known variance.
        def chi_test(big, small):
            from scipy import stats
            dchi = (small.ssr - big.ssr) / sd ** 2
            ddf = small.df_resid - big.df_resid
            return stats.chi2.sf(dchi, ddf)
        m, m_se = common.params[theta_col], sd * np.sqrt(
            common.normalized_cov_params.loc[theta_col, theta_col])
        red = common.ssr / sd ** 2 / common.df_resid
        out.append({"condition_sel": cond, "m": m, "m_se": m_se,
                    "red_chi2": red,
                    "p_genotype_slopes": chi_test(per_g, common),
                    "p_curvature": chi_test(curved, common), "n": len(d)})
        print(f"  {cond:10s} m = {m:+.5f} +/- {m_se:.5f}; reduced chi2 "
              f"{red:.1f}; genotype-specific m p={out[-1]['p_genotype_slopes']:.3g}; "
              f"curvature p={out[-1]['p_curvature']:.3g}")
    return pd.DataFrame(out)


def hill_fits(binding):
    """Four-parameter Hill fit (in log concentration) per genotype, so
    theta can be evaluated off the measured grid."""
    from scipy.optimize import least_squares
    x0 = binding["titrant_conc"][binding["titrant_conc"] > 0].min() / 10
    out = {}
    for g, d in binding.groupby("genotype"):
        lx = np.log(np.maximum(d["titrant_conc"].to_numpy(), x0))
        y = d["theta_obs"].to_numpy()

        def f(p, lx=lx):
            lo, hi, lk, n = p
            return hi + (lo - hi) / (1 + np.exp(-np.exp(n) * (lx - lk)))
        r = least_squares(lambda p: f(p) - y,
                          [y.min(), y.max(), np.log(0.1), 0.0],
                          bounds=([-0.2, 0, -20, -3], [1, 1.2, 10, 3]))
        out[g] = (f, r.x)
    return out, x0


def scan_concentration_scale(ab, binding, scales):
    """For each s, theta_g evaluated at s * x (in vivo effective
    concentration), then the common linear map's residual sum of squares
    per selective condition."""
    fits, x0 = hill_fits(binding)
    rows = []
    for sc in scales:
        d = ab.copy()
        d["theta_s"] = [fits[g][0](fits[g][1], np.log(max(x * sc, x0)))
                        for g, x in zip(d["genotype"], d["titrant_conc"])]
        for cond, dd in d.groupby("condition_sel"):
            f = smf.ols("slope ~ C(genotype) + theta_s", dd).fit()
            rows.append({"s": sc, "condition_sel": cond, "ssr": f.ssr,
                         "m": f.params["theta_s"], "df": f.df_resid})
    return pd.DataFrame(rows)


def main(data_dir, out_dir, sample_df=None):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binding = pd.read_csv(data_dir / "binding.csv")
    genotypes = sorted(binding["genotype"].unique())
    counts = pq.read_table(data_dir / "counts.parquet",
                           filters=[("genotype", "in", genotypes)]).to_pandas()
    samples = pd.read_parquet(data_dir / "samples.parquet")
    print(f"binding genotypes: {genotypes}")
    print(f"binding concentrations: {sorted(binding['titrant_conc'].unique())}")
    print(f"growth concentrations:  {sorted(samples['titrant_conc'].unique())}")

    sl = wt_relative_slopes(counts, samples, genotypes)

    th = binding.pivot_table(index="titrant_conc", columns="genotype",
                             values="theta_obs")
    th_sd = binding.pivot_table(index="titrant_conc", columns="genotype",
                                values="theta_std")
    sl["dtheta"] = [th.loc[x, g] - th.loc[x, "wt"]
                    for g, x in zip(sl["genotype"], sl["titrant_conc"])]
    sl["dtheta_sd"] = [np.hypot(th_sd.loc[x, g], th_sd.loc[x, "wt"])
                       for g, x in zip(sl["genotype"], sl["titrant_conc"])]
    sl.to_csv(out_dir / "wt_relative_slopes.csv", index=False)

    print(f"\nslope SE (per min): median {sl['slope_se'].median():.5f}; "
          f"min reads in a tube: median {sl['min_reads'].median():.0f}")

    # Replicate agreement.
    w = sl.pivot_table(index=["genotype", "condition_sel", "titrant_conc"],
                       columns="replicate", values="slope")
    diff = w.iloc[:, 1] - w.iloc[:, 0]
    print(f"replicate difference in slope: SD {diff.std():.5f} "
          f"(expected from SEs ~ "
          f"{np.sqrt(2) * sl['slope_se'].median():.5f})")

    # Counting SEs miss most of the noise (study 0b: per-tube variance is
    # 5-10x Poisson). Add an excess slope variance tau^2, estimated from
    # the replicate differences, to every slope's SE.
    tau2 = max(diff.var() / 2 - (sl["slope_se"] ** 2).mean(), 0.0)
    sl["slope_se_counting"] = sl["slope_se"]
    sl["slope_se"] = np.sqrt(sl["slope_se"] ** 2 + tau2)
    print(f"excess slope SD tau = {np.sqrt(tau2):.5f} per min, added in "
          f"quadrature to every slope SE")

    # Per selective condition: common m vs genotype-specific m vs curvature.
    print("\nper condition (both replicates pooled; genotype = dk offset):")
    results = []
    for cond, d in sl.groupby("condition_sel"):
        d = d.copy()
        d["w"] = 1.0 / d["slope_se"] ** 2
        common = smf.wls("slope ~ C(genotype) + dtheta", d, weights=d["w"]).fit()
        per_g = smf.wls("slope ~ C(genotype) + C(genotype):dtheta", d,
                        weights=d["w"]).fit()
        curved = smf.wls("slope ~ C(genotype) + dtheta + I(dtheta**2)", d,
                         weights=d["w"]).fit()
        f_g = per_g.compare_f_test(common)
        f_c = curved.compare_f_test(common)
        red_chi2 = common.ssr / common.df_resid
        results.append({"condition_sel": cond, "m_common": common.params["dtheta"],
                        "m_se": common.bse["dtheta"], "red_chi2_common": red_chi2,
                        "p_genotype_slopes": f_g[1], "p_curvature": f_c[1],
                        "n": len(d)})
        print(f"  {cond:10s} m = {common.params['dtheta']:+.5f} "
              f"+/- {common.bse['dtheta']:.5f}; reduced chi2 {red_chi2:.1f}; "
              f"genotype-specific m p={f_g[1]:.3g}; curvature p={f_c[1]:.3g}")
        for g in sorted(d["genotype"].unique()):
            dg = d[d["genotype"] == g]
            if dg["dtheta"].std() > 0.05:
                fg = smf.wls("slope ~ dtheta", dg, weights=dg["w"]).fit()
                print(f"      {g:10s} m_g = {fg.params['dtheta']:+.5f} "
                      f"+/- {fg.bse['dtheta']:.5f} "
                      f"(dtheta range {dg['dtheta'].min():+.2f}.."
                      f"{dg['dtheta'].max():+.2f})")
    pd.DataFrame(results).to_csv(out_dir / "map_tests.csv", index=False)

    # Figure: slope vs dtheta per condition, coloured by genotype.
    conds = sorted(sl["condition_sel"].unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 3.6))
    for ax, cond in zip(np.atleast_1d(axes), conds):
        d = sl[sl["condition_sel"] == cond]
        for g, dg in d.groupby("genotype"):
            ax.errorbar(dg["dtheta"], dg["slope"], xerr=dg["dtheta_sd"],
                        yerr=dg["slope_se"], fmt="o", ms=3, label=g)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(cond)
        ax.set_xlabel("theta_g - theta_wt (binding)")
    np.atleast_1d(axes)[0].set_ylabel("d ln(reads_g/reads_wt) / dt (per min)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "growth_vs_binding.pdf")

    if sample_df is None:
        return
    ab = absolute_slopes(counts, samples, sample_df, genotypes)
    ab["theta"] = [th.loc[x, g] for g, x in zip(ab["genotype"], ab["titrant_conc"])]
    ab.to_csv(out_dir / "absolute_slopes.csv", index=False)
    test_map(ab, "theta", "absolute slopes vs binding theta").to_csv(
        out_dir / "map_tests_absolute.csv", index=False)

    scales = [0.3, 1, 3, 10, 30, 100, 300]
    sc = scan_concentration_scale(ab, binding, scales)
    sc.to_csv(out_dir / "concentration_scale_scan.csv", index=False)
    w = ab.pivot_table(index=["genotype", "condition_sel", "titrant_conc"],
                       columns="replicate", values="slope")
    sd2 = (w.iloc[:, 1] - w.iloc[:, 0]).dropna().var() / 2
    print("\nconcentration scale s (in vivo [IPTG] = s * in vitro): "
          "reduced chi2 of the common linear map")
    tab = sc.assign(red=sc["ssr"] / sd2 / sc["df"]).pivot(
        index="s", columns="condition_sel", values="red")
    print(tab.round(2).to_string())
    print("m at each s:")
    print(sc.pivot(index="s", columns="condition_sel", values="m")
          .round(4).to_string())

    conds = sorted(ab["condition_sel"].unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 3.6))
    for ax, cond in zip(np.atleast_1d(axes), conds):
        d = ab[ab["condition_sel"] == cond]
        for g, dg in d.groupby("genotype"):
            ax.plot(dg["theta"], dg["slope"], "o", ms=3, label=g)
        ax.set_title(cond)
        ax.set_xlabel("theta_g (binding)")
    np.atleast_1d(axes)[0].set_ylabel("d ln_cfu_g / dt (per min)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "absolute_growth_vs_binding.pdf")


if __name__ == "__main__":
    main(*sys.argv[1:4])
