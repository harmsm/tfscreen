"""
Step 0c of planning/analysis-roadmap.md: what the per-tube OD600 readings
say about each tube's total population.

Questions:
  1. How much of the difference between bioreplicates is a level (shared
     split density) and how much a slope?
  2. How far do tubes scatter about a smooth curve, against what the
     reader's own noise (2% of OD600, through the calibration) predicts?
     The excess is tube-to-tube growth noise (or protocol noise).
  3. How close are tubes to the detection threshold, by condition?
  4. The population growth rate per condition (does strong selection
     stall or shrink the total?).

Usage:
    python od600_anatomy.py SAMPLE_DF CALIBRATION_YAML OUT_DIR

SAMPLE_DF has one row per sequenced tube with od600, rep, library,
condition_sel, titrant_conc, t_sel (combined_fit_sample_df.csv).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from matplotlib import pyplot as plt


def od_to_ln_cfu(od, cal):
    """ln(cfu/mL) and the SD of ln(cfu/mL) from OD600 reading noise alone
    (the calibration curve's own error is excluded: it is shared by every
    tube, roadmap C11)."""
    cfu = cal["A_CFU"] + cal["B_CFU"] * od + cal["C_CFU"] * od ** 2
    dcfu_dod = cal["B_CFU"] + 2 * cal["C_CFU"] * od
    read_sd = np.abs(dcfu_dod) * cal["OD600_PCT_STD"] * od / cfu
    return np.log(cfu), read_sd


def main(sample_df, cal_yaml, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = yaml.safe_load(open(cal_yaml))

    df = pd.read_csv(sample_df, index_col=0)
    df["condition_sel"] = df["condition_sel"].str.strip()
    df["cond"] = (df["condition_sel"] + "@" + df["titrant_conc"].astype(str))
    df["rep"] = df["rep"].astype(str)
    df["lnN"], df["read_sd"] = od_to_ln_cfu(df["od600"].to_numpy(), cal)
    df["t"] = df["t_sel"].astype(float)

    print(f"{len(df)} tubes, {df['cond'].nunique()} conditions, "
          f"reps {sorted(df['rep'].unique())}")
    print(f"OD600 reading noise in ln units: median "
          f"{df['read_sd'].median():.4f} (range {df['read_sd'].min():.4f}-"
          f"{df['read_sd'].max():.4f})")

    # --- 1 and 2: nested models -------------------------------------------
    forms = {
        "shared curve":            "lnN ~ C(cond) + C(cond):t",
        "+ replicate level":       "lnN ~ C(rep) + C(cond) + C(cond):t",
        "+ replicate x cond level": "lnN ~ C(rep)*C(cond) + C(cond):t",
        "+ replicate slopes":      "lnN ~ C(rep)*C(cond) + C(rep):C(cond):t",
    }
    fits = {k: smf.ols(f, data=df).fit() for k, f in forms.items()}
    print("\nmodel                     dof  resid SD   ")
    for k, f in fits.items():
        print(f"{k:25s} {int(f.df_resid):4d}  {np.sqrt(f.scale):.4f}")
    names = list(fits)
    for a, b in zip(names[:-1], names[1:]):
        ft = fits[b].compare_f_test(fits[a])
        print(f"F-test {b!r} vs {a!r}: F={ft[0]:.2f} p={ft[1]:.3g} "
              f"df={int(ft[2])}")

    rep_level = fits["+ replicate level"].params.filter(like="C(rep)")
    print(f"\nreplicate level offset (rep 2 - rep 1): "
          f"{rep_level.iloc[0]:+.4f} ln units "
          f"(SE {fits['+ replicate level'].bse.filter(like='C(rep)').iloc[0]:.4f})")

    resid_sd = np.sqrt(fits["+ replicate level"].scale)
    read_sd = df["read_sd"].median()
    excess = np.sqrt(max(resid_sd ** 2 - read_sd ** 2, 0))
    print(f"scatter about shared curve + replicate level: {resid_sd:.4f}; "
          f"reading noise {read_sd:.4f}; excess (tube noise) {excess:.4f}")

    # does the scatter grow with time (tube growth noise ~ sigma * t)?
    r = fits["+ replicate level"].resid
    by_t = df.assign(r2=r ** 2).groupby(pd.cut(df["t"], 3), observed=True)["r2"]
    print("residual SD by t_sel tercile:")
    for k, v in by_t:
        print(f"  {str(k):16s} n={len(v):3d} SD={np.sqrt(v.mean()):.4f}")

    # --- 3: distance from the detection threshold --------------------------
    thr = cal["OD600_MEAS_THRESHOLD"]
    near = (df.groupby(["cond"])["od600"].min()
              .rename("min_od600").to_frame())
    near["fold_above_threshold"] = near["min_od600"] / thr
    print(f"\ndetection threshold {thr:.4f}; tubes below: "
          f"{(df['od600'] < thr).sum()}")
    print("lowest OD600 per condition (fold above threshold):")
    print(near.sort_values("min_od600").head(10).to_string())

    # --- 4: population growth rate per condition ---------------------------
    slopes = []
    for cond, g in df.groupby("cond"):
        f = smf.ols("lnN ~ C(rep) + t", data=g).fit()
        slopes.append({"cond": cond, "slope_per_min": f.params["t"],
                       "slope_se": f.bse["t"], "n": len(g)})
    slopes = pd.DataFrame(slopes).sort_values("cond")
    slopes.to_csv(out_dir / "population_slopes.csv", index=False)
    print("\npopulation growth rate (per min) by condition, shared across "
          "replicates with a replicate level:")
    print(slopes.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # --- figure --------------------------------------------------------------
    conds = sorted(df["cond"].unique())
    ncol = 8
    nrow = int(np.ceil(len(conds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.2 * nrow),
                             sharex=True, sharey=True)
    for ax, cond in zip(axes.flat, conds):
        g = df[df["cond"] == cond]
        for rep, gg in g.groupby("rep"):
            ax.errorbar(gg["t"], gg["lnN"], yerr=gg["read_sd"], fmt="o",
                        ms=3, label=f"rep {rep}")
        ax.set_title(cond, fontsize=7)
    axes.flat[0].legend(fontsize=6)
    fig.supxlabel("t_sel (min)")
    fig.supylabel("ln(cfu) from OD600")
    fig.tight_layout()
    fig.savefig(out_dir / "od600_by_condition.pdf")
    df.to_csv(out_dir / "tubes.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
