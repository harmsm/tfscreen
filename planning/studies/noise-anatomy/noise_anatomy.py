"""
Step 0b of planning/analysis-roadmap.md: the anatomy of read-count noise.

Within one replicate x condition x concentration there are three tubes
(timepoints). A per-genotype straight line in t_sel through three points
leaves one residual degree of freedom per line; its projection
z = r . v (v the unit vector orthogonal to [1, t]) carries everything the
straight-growth model does not explain in that group. That gives:

  1. Low-count prevalence: fraction of rows at 0, < 5 and < 20 reads, by
     genotype class and selection condition.
  2. The per-tube composition offset u_s (PCR, calling efficiency): a shift
     shared by every genotype's ln(frequency) in a tube. Its projection is
     the same for every genotype in a group, so the mean of z over many
     well-measured genotypes estimates it, and
         var(u) ~ mean_groups(zbar^2) - mean_groups(var(z)) / K.
     Compared with the tube's __unknown__ share.
  3. The count noise model: for ln(reads_g / reads_wt) (tube total and u
     cancel), E[z^2] against the Poisson expectation
     sum_i v_i^2 (1/c_g,i + 1/c_wt,i). A fit E[z^2] = a + b * poisson gives
     b (1 = Poisson; > 1 = effective depth smaller than the reads) and a
     (a floor independent of counts: PCR jackpotting and real
     genotype-by-tube differences).

Usage:
    python noise_anatomy.py DATA_DIR OUT_DIR
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SPIKED = {"wt", "M42I", "H74A", "K84L", "M42I/H74A"}
GROUP = ["replicate", "condition_sel", "titrant_conc"]


def genotype_class(g):
    if g in SPIKED:
        return "spiked"
    return "double" if "/" in g else "single"


def residual_projector(t):
    """Unit vector orthogonal to [1, t] (3 points -> exactly one)."""
    X = np.column_stack([np.ones_like(t), t])
    q, _ = np.linalg.qr(X, mode="complete")
    return q[:, 2]


def main(data_dir, out_dir):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(data_dir / "samples.parquet")
    counts = pd.read_parquet(data_dir / "counts.parquet")
    genos = counts["genotype"].cat.categories
    samples["unknown_share"] = 1 - samples["reads_listed"] / samples["freq_denominator"]
    print(f"{len(samples)} samples, {len(genos):,} genotypes, "
          f"{len(counts):,} rows")
    print(f"reads per tube (listed genotypes): median "
          f"{samples['reads_listed'].median():,.0f}; __unknown__ share by "
          f"condition:")
    print(samples.groupby("condition_sel")["unknown_share"]
          .describe()[["mean", "min", "max"]].to_string())

    # --- 1. low-count prevalence -------------------------------------------
    counts = counts.merge(samples[["sample", "condition_sel"]], on="sample")
    cls = pd.Series({g: genotype_class(g) for g in genos})
    counts["class"] = counts["genotype"].map(cls).astype("category")
    low = counts.groupby(["class", "condition_sel"], observed=True)["counts"].agg(
        rows="size",
        zero=lambda c: (c == 0).mean(),
        lt5=lambda c: (c < 5).mean(),
        lt20=lambda c: (c < 20).mean(),
        median="median")
    print("\n1. low-count prevalence (fraction of rows):")
    print(low.to_string(float_format=lambda x: f"{x:.3f}"))
    low.to_csv(out_dir / "low_counts.csv")

    # Matrix: tube x genotype.
    mat = counts.pivot_table(index="sample", columns="genotype",
                             values="counts", aggfunc="sum", observed=True,
                             fill_value=0)
    s = samples.set_index("sample").loc[mat.index]
    lnf = np.log(mat.to_numpy(dtype=float) + 0.5) - \
        np.log(s["freq_denominator"].to_numpy())[:, None]

    # --- 2. composition offset u_s ------------------------------------------
    rows = []
    well = mat.columns[(mat.to_numpy() >= 1000).all(axis=0)]
    wi = mat.columns.get_indexer(well)
    print(f"\n2. genotypes with >= 1000 reads in every tube: {len(well)}")
    for key, g in s.reset_index().groupby(GROUP):
        if len(g) != 3:
            continue
        idx = mat.index.get_indexer(g["sample"])
        v = residual_projector(g["t_sel"].to_numpy(dtype=float))
        z = v @ lnf[idx][:, wi]              # one z per well-measured genotype
        uz = v @ g["unknown_share"].to_numpy()
        rows.append({**dict(zip(GROUP, key)), "zbar": z.mean(),
                     "var_z": z.var(ddof=1), "K": len(z), "unknown_proj": uz})
    u = pd.DataFrame(rows)
    var_u = (u["zbar"] ** 2).mean() - (u["var_z"] / u["K"]).mean()
    print(f"   groups with 3 tubes: {len(u)}")
    print(f"   composition offset SD (shared by all genotypes in a tube): "
          f"{np.sqrt(max(var_u, 0)):.4f} ln units")
    print(f"   per-genotype residual SD (well-measured, projection): "
          f"{np.sqrt(u['var_z'].mean()):.4f}")
    r = np.corrcoef(u["zbar"], u["unknown_proj"])[0, 1]
    print(f"   correlation of the shared offset with the __unknown__ share "
          f"(projected): {r:+.2f}")
    u.to_csv(out_dir / "composition_offset.csv", index=False)

    # --- 3. count noise: ln(reads_g / reads_wt) -----------------------------
    wt = mat["wt"].to_numpy(dtype=float)
    M = mat.to_numpy(dtype=float)
    rec = []
    for key, g in s.reset_index().groupby(GROUP):
        if len(g) != 3:
            continue
        idx = mat.index.get_indexer(g["sample"])
        v = residual_projector(g["t_sel"].to_numpy(dtype=float))
        c = M[idx]
        ok = (c > 0).all(axis=0)
        cg = c[:, ok]
        y = np.log(cg) - np.log(wt[idx])[:, None]
        z = v @ y
        pois = (v[:, None] ** 2 * (1 / cg + 1 / wt[idx][:, None])).sum(axis=0)
        rec.append(pd.DataFrame({"z2": z ** 2, "poisson": pois,
                                 "min_reads": cg.min(axis=0),
                                 "condition_sel": key[1]}))
    rec = pd.concat(rec, ignore_index=True)
    rec = rec[rec["poisson"] > 0]
    bins = np.quantile(rec["poisson"], np.linspace(0, 1, 21))
    rec["bin"] = pd.cut(rec["poisson"], bins, include_lowest=True)
    b = rec.groupby("bin", observed=True).agg(poisson=("poisson", "mean"),
                                              z2=("z2", "mean"),
                                              min_reads=("min_reads", "median"),
                                              n=("z2", "size"))
    b["ratio"] = b["z2"] / b["poisson"]
    print(f"\n3. ln(reads_g/reads_wt), {len(rec):,} genotype-group lines "
          f"(all 3 tubes > 0 reads)")
    print(b.to_string(float_format=lambda x: f"{x:.4g}"))
    A = np.column_stack([np.ones(len(b)), b["poisson"]])
    coef, *_ = np.linalg.lstsq(A, b["z2"], rcond=None)
    print(f"   E[z^2] = {coef[0]:.4g} + {coef[1]:.3f} * poisson  "
          f"(floor SD {np.sqrt(max(coef[0], 0)):.3f} ln units; "
          f"Poisson multiplier {coef[1]:.2f})")
    for cond, d in rec.groupby("condition_sel"):
        bb = d.groupby(pd.cut(d["poisson"], bins, include_lowest=True),
                       observed=True).agg(p=("poisson", "mean"), z2=("z2", "mean"))
        A = np.column_stack([np.ones(len(bb)), bb["p"]])
        cc, *_ = np.linalg.lstsq(A, bb["z2"], rcond=None)
        print(f"   {cond:10s} floor SD {np.sqrt(max(cc[0], 0)):.3f}, "
              f"Poisson multiplier {cc[1]:.2f}, n={len(d):,}")
    b.to_csv(out_dir / "count_noise_bins.csv")

    # --- 3b. same, on ln(frequency) with the shared offset removed ----------
    # The ratio to wt carries wt's own residual in every line. Here each
    # genotype's projection has the group's shared offset (zbar over the
    # well-measured genotypes) subtracted instead, and only lines with at
    # least MIN_READS reads in every tube are used (below that, ln counts
    # are biased and the Poisson expansion fails).
    MIN_READS = 30
    zbar = u.set_index(GROUP)["zbar"]
    rec = []
    for key, g in s.reset_index().groupby(GROUP):
        if len(g) != 3 or key not in zbar.index:
            continue
        idx = mat.index.get_indexer(g["sample"])
        v = residual_projector(g["t_sel"].to_numpy(dtype=float))
        c = M[idx]
        ok = (c >= MIN_READS).all(axis=0)
        cg = c[:, ok]
        lnD = np.log(s.loc[g["sample"], "freq_denominator"].to_numpy())
        z = v @ (np.log(cg) - lnD[:, None]) - zbar.loc[key]
        pois = (v[:, None] ** 2 / cg).sum(axis=0)
        rec.append(pd.DataFrame({"z2": z ** 2, "poisson": pois,
                                 "min_reads": cg.min(axis=0),
                                 "condition_sel": key[1],
                                 "genotype": mat.columns[ok]}))
    rec = pd.concat(rec, ignore_index=True)
    rec["class"] = rec["genotype"].map(cls)
    bins = np.quantile(rec["poisson"], np.linspace(0, 1, 16))
    rec["bin"] = pd.cut(rec["poisson"], bins, include_lowest=True)
    b = rec.groupby("bin", observed=True).agg(poisson=("poisson", "mean"),
                                              z2=("z2", "mean"),
                                              min_reads=("min_reads", "median"),
                                              n=("z2", "size"))
    b["ratio"] = b["z2"] / b["poisson"]
    print(f"\n3b. ln(frequency) minus the shared offset, lines with >= "
          f"{MIN_READS} reads in every tube ({len(rec):,} lines)")
    print(b.to_string(float_format=lambda x: f"{x:.4g}"))
    # Fit only where the >= MIN_READS selection does not truncate the draws.
    FIT_READS = 100
    bf = b[b["min_reads"] >= FIT_READS]
    rec = rec[rec["min_reads"] >= FIT_READS]
    print(f"   fit on bins with median min reads >= {FIT_READS} "
          f"({len(bf)} bins)")
    A = np.column_stack([np.ones(len(bf)), bf["poisson"]])
    coef, *_ = np.linalg.lstsq(A, bf["z2"], rcond=None)
    print(f"   E[z^2] = {coef[0]:.4g} + {coef[1]:.2f} * poisson: variance "
          f"multiplier (phi) {coef[1]:.2f}, floor SD "
          f"{np.sqrt(max(coef[0], 0)):.3f} ln units")
    for (cond), d in rec.groupby("condition_sel"):
        bb = d.groupby(pd.cut(d["poisson"], bins, include_lowest=True),
                       observed=True).agg(p=("poisson", "mean"), z2=("z2", "mean"))
        bb = bb.dropna()
        A = np.column_stack([np.ones(len(bb)), bb["p"]])
        cc, *_ = np.linalg.lstsq(A, bb["z2"], rcond=None)
        print(f"   {cond:10s} phi {cc[1]:.2f}, floor SD "
              f"{np.sqrt(max(cc[0], 0)):.3f}, n={len(d):,}")
    b.to_csv(out_dir / "count_noise_bins_lnfreq.csv")

    # --- 4. why is a low count low? -----------------------------------------
    # A double that is rare everywhere carries little information in any
    # tube; one that is abundant without the drug and crashes under
    # selection carries its signal in the crash. Classify each low-count
    # row in a selective tube by the genotype's median reads in the no-drug
    # tubes of the same library (same replicate), which stand in for its
    # abundance before selection.
    low_rows = counts.merge(samples[["sample", "library", "replicate"]],
                            on="sample")
    nodrug = low_rows[low_rows["condition_sel"].isin(["kanR-kan", "pheS-4CP"])]
    base = (nodrug.groupby(["library", "replicate", "genotype"], observed=True)
            ["counts"].median().rename("nodrug_median"))
    sel = low_rows[low_rows["condition_sel"].isin(["kanR+kan", "pheS+4CP"])]
    sel = sel.join(base, on=["library", "replicate", "genotype"])
    edges = [-0.1, 5, 20, 100, np.inf]
    labels = ["rare (<5)", "low (5-20)", "moderate (20-100)", "abundant (>=100)"]
    sel["abundance_without_drug"] = pd.cut(sel["nodrug_median"], edges,
                                           labels=labels)
    print("\n4. doubles in selective tubes: reads in the tube vs the "
          "genotype's median reads without the drug")
    d = sel[sel["class"] == "double"]
    tab = pd.crosstab(d["abundance_without_drug"],
                      pd.cut(d["counts"], [-1, 0, 4, 19, np.inf],
                             labels=["0", "1-4", "5-19", ">=20"]),
                      normalize="all")
    print("   fraction of all double rows in selective tubes:")
    print(tab.round(3).to_string())
    by_cond = (d.assign(low=d["counts"] < 5)
                .groupby(["condition_sel", "abundance_without_drug"],
                         observed=True)["low"].agg(["mean", "size"]))
    print("   fraction below 5 reads, by condition and abundance without drug:")
    print(by_cond.round(3).to_string())
    g = (d.assign(low=d["counts"] < 5)
          .groupby(["library", "genotype"], observed=True)
          .agg(nodrug=("nodrug_median", "median"), frac_low=("low", "mean")))
    g = g[g["nodrug"] >= 20]
    print(f"   doubles with >= 20 median reads without the drug: {len(g):,}; "
          f"of these, share with some selective tube below 5 reads: "
          f"{(g['frac_low'] > 0).mean():.3f}; with at least half below 5: "
          f"{(g['frac_low'] >= 0.5).mean():.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
