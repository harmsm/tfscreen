"""
Do genotypes whose X disagrees between markers also grow differently in
the two libraries' control conditions? (planning/deep-coverage.md step 3)

    python control_dk_check.py tfs_growth.csv both/tfs_pred_theta.csv \
        kanR_only/tfs_pred_theta.csv --out control_dk.csv

A library-specific dk_geno (the genotype growing differently on the kanR
and pheS backbones) shows up directly in each library's control condition
(kanR-kan, pheS-4CP), where m is near 0: there a genotype's growth
relative to wt is its dk. For each deep double (at least --min_reads total
reads) and each control series (replicate x condition x IPTG level), the
slope over time of ln((counts_g + 0.5) / (counts_wt + 0.5)) estimates dk_g
(weighted least squares, Poisson weights 1 / (1/c_g + 1/c_wt)); series
combine by inverse variance into dk_kanR and dk_pheS, and delta_dk =
dk_kanR - dk_pheS.

The X disagreement is each genotype's mean offset between the kanR-only
and both-marker fits after the best affine map (the level part, 91-92% of
the disagreement). If the kanR library's dk differs from the shared value
by delta, the kanR-only fit's X moves by about -delta / m_kanR relative to
the shared fit; with the shared dk near the two libraries' mean, the
offset is about delta_dk / (2 |m_kanR|) (--m_kan, default -0.0147).

Prints the correlation between the two libraries' dk estimates (with a
shared dk, how well the estimator recovers it), the correlation of offset
and delta_dk over all deep doubles and
over the worst --tail share by disagreement, the regression slope against
the expected 1 / (2 |m_kanR|), and the noise in delta_dk. Writes one row
per genotype. A simulation that shares dk_geno across libraries gives the
no-signal benchmark.
"""

import argparse

import numpy as np
import pandas as pd

CHUNK = 5_000_000
CONTROLS = {"kanR": "kanR-kan", "pheS": "pheS-4CP"}


def depth_and_controls(path, min_reads):
    tot = None
    for c in pd.read_csv(path, usecols=["genotype", "counts"], chunksize=CHUNK):
        s = c.groupby("genotype")["counts"].sum()
        tot = s if tot is None else tot.add(s, fill_value=0)
    g = tot.index.to_series().astype(str)
    deep = set(tot.index[(g.str.count("/") == 1) & (tot >= min_reads)])
    keep = deep | {"wt"}
    cols = ["genotype", "library", "replicate", "condition_sel",
            "titrant_conc", "t_sel", "sample", "counts"]
    parts = []
    for c in pd.read_csv(path, usecols=cols, chunksize=CHUNK):
        c = c[c["condition_sel"].isin(CONTROLS.values()) & c["genotype"].isin(keep)]
        parts.append(c)
    return deep, pd.concat(parts)


def control_dk(rows):
    """Per genotype and library: dk relative to wt and its SE."""
    wt = rows[rows["genotype"] == "wt"][["sample", "counts"]].rename(
        columns={"counts": "c_wt"})
    d = rows[rows["genotype"] != "wt"].merge(wt, on="sample")
    d["y"] = np.log((d["counts"] + 0.5) / (d["c_wt"] + 0.5))
    d["w"] = 1.0 / (1.0 / (d["counts"] + 0.5) + 1.0 / (d["c_wt"] + 0.5))
    key = ["genotype", "library", "replicate", "condition_sel", "titrant_conc"]
    d["wt_"] = d["w"] * d["t_sel"]
    d["wy"] = d["w"] * d["y"]
    s = d.groupby(key).agg(sw=("w", "sum"), swt=("wt_", "sum"),
                           swy=("wy", "sum"), n=("y", "size"))
    d = d.join(s, on=key)
    d["tc"] = d["t_sel"] - d["swt"] / d["sw"]
    d["yc"] = d["y"] - d["swy"] / d["sw"]
    d["wtt"] = d["w"] * d["tc"] ** 2
    d["wty"] = d["w"] * d["tc"] * d["yc"]
    s = d.groupby(key).agg(stt=("wtt", "sum"), sty=("wty", "sum"),
                           n=("y", "size")).reset_index()
    s = s[(s["n"] >= 2) & (s["stt"] > 0)]
    s["slope"] = s["sty"] / s["stt"]
    s["var"] = 1.0 / s["stt"]
    s["iv"] = 1.0 / s["var"]
    s["iv_slope"] = s["iv"] * s["slope"]
    out = s.groupby(["genotype", "library"]).agg(
        iv=("iv", "sum"), ivs=("iv_slope", "sum"), n_series=("slope", "size"))
    out["dk"] = out["ivs"] / out["iv"]
    out["dk_se"] = 1.0 / np.sqrt(out["iv"])
    return out[["dk", "dk_se", "n_series"]].unstack("library")


def x_offsets(both, kan, genotypes):
    cols = ["genotype", "titrant_conc", "q0.5"]
    m = pd.read_csv(both, usecols=cols).merge(
        pd.read_csv(kan, usecols=cols), on=["genotype", "titrant_conc"],
        suffixes=("_b", "_k"))
    m = m[m["genotype"].isin(genotypes)]
    p = np.polyfit(m["q0.5_b"], m["q0.5_k"], 1)
    m["res"] = m["q0.5_k"] - np.polyval(p, m["q0.5_b"])
    g = m.groupby("genotype")["res"]
    return pd.DataFrame({"offset": g.mean(),
                         "rms": g.apply(lambda r: np.sqrt(np.mean(r ** 2)))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("growth_csv")
    ap.add_argument("both_pred")
    ap.add_argument("kan_pred")
    ap.add_argument("--min_reads", type=float, default=1e4)
    ap.add_argument("--tail", type=float, default=0.08)
    ap.add_argument("--m_kan", type=float, default=-0.0147)
    ap.add_argument("--out", default="control_dk.csv")
    args = ap.parse_args()

    deep, rows = depth_and_controls(args.growth_csv, args.min_reads)
    dk = control_dk(rows)
    dk.columns = [f"{a}_{b}" for a, b in dk.columns]
    off = x_offsets(args.both_pred, args.kan_pred, deep)
    df = off.join(dk, how="inner").dropna(subset=["dk_kanR", "dk_pheS"])
    df["delta_dk"] = df["dk_kanR"] - df["dk_pheS"]
    df["delta_dk_se"] = np.hypot(df["dk_se_kanR"], df["dk_se_pheS"])
    df.to_csv(args.out)

    expect = 1.0 / (2 * abs(args.m_kan))
    tail = df[df["rms"] >= df["rms"].quantile(1 - args.tail)]
    print(f"{len(df)} deep doubles with both controls; tail {len(tail)}")
    print(f"delta_dk: SD across genotypes {df['delta_dk'].std():.4f}, "
          f"median SE {df['delta_dk_se'].median():.4f} per minute")
    print(f"r(dk_kanR, dk_pheS) {np.corrcoef(df['dk_kanR'], df['dk_pheS'])[0, 1]:.3f} "
          f"(how well the two controls agree on each genotype's dk)")
    for name, sub in (("all", df), ("tail", tail)):
        r = np.corrcoef(sub["offset"], sub["delta_dk"])[0, 1]
        slope = np.polyfit(sub["delta_dk"], sub["offset"], 1)[0]
        print(f"{name}: r(offset, delta_dk) {r:.3f}, slope {slope:.1f} "
              f"(expected about {expect:.0f})")


if __name__ == "__main__":
    main()
