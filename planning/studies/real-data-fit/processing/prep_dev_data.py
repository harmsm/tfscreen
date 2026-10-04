"""
Turn the raw dev-data into pipeline inputs, written to processed/.

Lab-specific choices live here (see README.md, "Answers from Mike"):

- initial-kan and initial-phe count files are swapped (data-level evidence;
  origin unknown), so each is assigned to the other library.
- Resequenced files of one tube are the same PCR product and are summed.
- Tubes with fewer than MIN_READS called reads after pooling are dropped.
- OD600 is the measurement; sample_df's lncfu columns are ignored.
- Tube totals: calibrated CFU/mL times TUBE_VOLUME_ML.

Run from this directory:  python prep_dev_data.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import yaml

from tfscreen.process_raw.od600 import (
    cfu_per_mL_error_components,
    od600_to_cfu_per_mL,
    read_calibration,
)

OUT = "processed"
MIN_READS = 1_000_000
TUBE_VOLUME_ML = 5.0

# Split: shared culture at OD600 ~0.35, 200 uL into 15 mL (Mike, approximate)
PRESPLIT_OD600 = 0.35
PRESPLIT_DILUTION = 0.2 / 15.2

LIBRARY_DESIGN = {
    "transform_sizes": {"single-1": 100_000, "single-2": 100_000,
                        "double-1-2": 300_000, "spiked": 1_000},
    "library_mixture": {"single-1": 100, "single-2": 100,
                        "double-1-2": 1_000, "spiked": 1},
}


def canonical(name):
    """Sample name with '.' decimals as 'o' and any barcode suffix dropped."""
    name = re.sub(r"(\d)\.(\d)", r"\1o\2", name)
    name = re.sub(r"-[ACGT]{8,}-[ACGT]{8,}$", "", name)
    return name.replace("initial_", "initial-")


# ---------------------------------------------------------------------------
# library config
# ---------------------------------------------------------------------------

# The spikes actually in the experiment (Mike, 2026-09-30), as codon edits
# of wt_seq ('.' keeps the wt base). counts/run_config.yaml, the processing
# config, lists only the three the library design cannot make (H74A/K84L,
# M42I/H74A/K84L, D88A), and its M42I codon is att, not the spikes' ata.
# The reads were still called right: tfs-process-fastq ran with
# --allowed_diff_from_expected 2, and every spike is 0-2 mismatches from a
# unique expected sequence of the same genotype. But the fit needs all nine
# as spiked (in_spiked_origin: ln_cfu0 prior class, bulk_fraction).
_P = "." * 45
SPIKE_EDITS = {
    "wt": "",
    "M42I": _P + "ata",
    "H74A": "." * 141 + "gcg",
    "K84L": "." * 171 + "ctg",
    "M42I/H74A": _P + "ata" + "." * 93 + "gcg",
    "M42I/K84L": _P + "ata" + "." * 123 + "ctg",
    "H74A/K84L": "." * 141 + "gcg" + "." * 27 + "ctg",
    "M42I/H74A/K84L": _P + "ata" + "." * 93 + "gcg" + "." * 27 + "ctg",
    "D88A": "." * 184 + "cc",
}


def spiked_seqs(wt_seq):
    out = []
    for edit in SPIKE_EDITS.values():
        edit = edit.ljust(len(wt_seq), ".")
        assert len(edit) == len(wt_seq)
        out.append("".join(w if e == "." else e for w, e in zip(wt_seq, edit)))
    return out


def library_config():
    with open("counts/run_config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.update(LIBRARY_DESIGN)
    cfg["spiked_seqs"] = spiked_seqs(cfg["wt_seq"])
    path = os.path.join(OUT, "library_config.yaml")
    with open(path, "w") as f:
        f.write("# counts/run_config.yaml plus the realized library design\n"
                "# (serial dilution matched the spec; Mike, 2026-09-28) and\n"
                "# the nine spikes actually used (Mike, 2026-09-30; see\n"
                "# SPIKE_EDITS in prep_dev_data.py)\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# OD600 calibration
# ---------------------------------------------------------------------------

def od600_calibration():
    d = "od600-to-cfu/"
    r = pd.read_excel(d + "2025-09-17_od600-serial-dilution-of-keio-cells-"
                      "with-10x-technical-replicate.xlsx",
                      sheet_name="1-3dilution", header=None)
    reps = pd.DataFrame([(3.0 ** -i, v) for i, row in r.iterrows()
                         for v in row.values], columns=["dilution", "od600"])
    p = pd.read_excel(d + "2025-09-19_od600-vs-cfu-per-mL-by-plate-"
                      "counting.xlsx")
    plates = pd.DataFrame(dict(od600=p["OD600 plate reader"],
                               colonies=p["CFU"], dilution=p["Dilution"],
                               plated_volume_mL=p["Volume_mL"],
                               num_dilutions=p["num_dilutions"],
                               plating_steps=p["plating_steps"],
                               day=p["replicate"]))
    reps.to_csv(os.path.join(OUT, "od600_replicates.csv"), index=False)
    plates.to_csv(os.path.join(OUT, "od600_plate_counts.csv"), index=False)
    prefix = os.path.join(OUT, "od600_calibration")
    from tfscreen.process_raw.scripts.calibrate_od600_cli import calibrate_od600
    calibrate_od600(os.path.join(OUT, "od600_replicates.csv"),
                    os.path.join(OUT, "od600_plate_counts.csv"),
                    out_prefix=prefix)
    return read_calibration(prefix + ".yaml")


# ---------------------------------------------------------------------------
# counts: pool resequencing, fix the initial swap, drop failed tubes
# ---------------------------------------------------------------------------

def pooled_counts():
    files = {}
    for f in sorted(glob.glob("counts/counts/counts_*.csv")):
        name = canonical(re.match(r".*counts_(.*?)_S\d+_", f).group(1))
        files.setdefault(name, []).append(f)

    pooled = {}
    for name, fs in files.items():
        df = pd.concat([pd.read_csv(f) for f in fs])
        pooled[name] = (df.groupby("genotype", as_index=False)["counts"].sum(),
                        [os.path.basename(f) for f in fs])

    # the two initial files are swapped (README: initial-kan has the pheS
    # library's composition)
    pooled["initial-kan"], pooled["initial-phe"] = (pooled["initial-phe"],
                                                    pooled["initial-kan"])
    return pooled


# ---------------------------------------------------------------------------
# sample table
# ---------------------------------------------------------------------------

def sample_table(cal, pooled):
    s = pd.read_csv("sample_df.csv")
    s["sample"] = s["sample"].map(canonical)
    s = s[~s["sample"].str.startswith("initial")]

    # phe-1-0o03-2-2 has counts and OD600 but no row: rebuild it from its
    # replicate's other rows and the OD600 table
    od = pd.read_excel("all_reps_screen_od600.xlsx")
    od["condition_sel"] = od["condition_sel"].str.replace("4cp", "4CP")
    if "phe-1-0o03-2-2" not in set(s["sample"]):
        row = s[s["sample"] == "phe-1-0o03-1-2"].iloc[0].copy()
        hit = od[(od["rep"] == 2) & (od["condition_sel"] == "pheS+4CP")
                 & np.isclose(od["titrant_conc"], 0.03)]
        t = sorted(hit["time"])[1]
        row["sample"] = "phe-1-0o03-2-2"
        row["t_sel"] = t
        row["od600"] = float(hit.loc[hit["time"] == t, "od600"].iloc[0])
        s = pd.concat([s, row.to_frame().T], ignore_index=True)

    # the sheet's od600 must match the raw OD600 table
    chk = s.merge(od, left_on=["rep", "condition_sel", "titrant_conc", "t_sel"],
                  right_on=["rep", "condition_sel", "titrant_conc", "time"],
                  suffixes=("", "_raw"))
    assert len(chk) == len(s), "every sequenced tube has a raw OD600"
    assert np.allclose(chk["od600"].astype(float), chk["od600_raw"])

    od600 = s["od600"].astype(float).to_numpy()
    cfu_mL, cfu_mL_sd, detectable = od600_to_cfu_per_mL(od600, cal)
    curve_sd, reading_sd = cfu_per_mL_error_components(od600, cal)
    assert detectable.all()

    reads = {n: int(p[0]["counts"].sum()) for n, p in pooled.items()}
    s["called_reads"] = s["sample"].map(reads)
    s["count_files"] = s["sample"].map(lambda n: ";".join(pooled[n][1])
                                       if n in pooled else "")

    out = pd.DataFrame({
        "sample": s["sample"],
        "library": s["library"],
        "replicate": s["rep"].astype(int),
        "condition_pre": s["condition_pre"],
        "t_pre": s["t_pre"].astype(int),
        "condition_sel": s["condition_sel"],
        "t_sel": s["t_sel"].astype(int),
        "titrant_name": s["titrant_name"],
        "titrant_conc": s["titrant_conc"].astype(float),
        "od600": od600,
        "sample_cfu": cfu_mL * TUBE_VOLUME_ML,
        "sample_cfu_std": cfu_mL_sd * TUBE_VOLUME_ML,
        "sample_cfu_curve_std": curve_sd * TUBE_VOLUME_ML,
        "sample_cfu_reading_std": reading_sd * TUBE_VOLUME_ML,
        "called_reads": s["called_reads"],
        "count_files": s["count_files"],
    })
    dropped = out[out["called_reads"].fillna(0) < MIN_READS]
    kept = out[out["called_reads"].fillna(0) >= MIN_READS]
    return kept.sort_values("sample").reset_index(drop=True), dropped


def write_counts(pooled, samples):
    d = os.path.join(OUT, "counts")
    os.makedirs(d, exist_ok=True)
    for name in samples["sample"]:
        pooled[name][0].to_csv(os.path.join(d, f"counts__{name}__.csv"),
                               index=False)


def initial_composition(pooled):
    rows = []
    for name, lib in (("initial-kan", "kanR"), ("initial-phe", "pheS")):
        df = pooled[name][0].copy()
        df["library"] = lib
        called = df.loc[df["genotype"] != "__unknown__", "counts"].sum()
        df["frequency"] = df["counts"] / called
        df.loc[df["genotype"] == "__unknown__", "frequency"] = np.nan
        df["source_files"] = ";".join(pooled[name][1])
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# wt monoculture controls
# ---------------------------------------------------------------------------

def wt_controls(cal):
    c = pd.read_excel("2025-10-02_control-experiments.xlsx")
    c.columns = ["condition_pre", "t_pre", "condition_pre_dup",
                 "condition_sel", "t_sel", "titrant_name", "titrant_conc",
                 "od600", "replicate", "library", "genotype"]
    c = c.drop(columns="condition_pre_dup")
    cfu_mL, cfu_mL_sd, detectable = od600_to_cfu_per_mL(c["od600"].to_numpy(),
                                                        cal)
    c["cfu_per_mL"] = cfu_mL
    c["ln_cfu_per_mL"] = np.log(cfu_mL)
    c["ln_cfu_per_mL_std"] = cfu_mL_sd / cfu_mL
    c["detectable"] = detectable

    rows = []
    for key, g in c.groupby(["condition_sel", "titrant_conc", "replicate"]):
        g = g[g["detectable"]]
        if g["t_sel"].nunique() < 2:
            continue
        (slope, icpt), cov = np.polyfit(g["t_sel"], g["ln_cfu_per_mL"], 1,
                                        cov=g["t_sel"].nunique() > 2)
        se = np.sqrt(cov[0, 0]) if np.ndim(cov) == 2 else np.nan
        rows.append(dict(zip(["condition_sel", "titrant_conc", "replicate"],
                             key), rate=slope, rate_se=se,
                         num_timepoints=g["t_sel"].nunique()))
    rates = pd.DataFrame(rows)
    summary = (rates.groupby(["condition_sel", "titrant_conc"])["rate"]
               .agg(rate_mean="mean", rate_sd="std", num_replicates="size")
               .reset_index())
    return c, rates, summary


# ---------------------------------------------------------------------------
# binding: tidy anisotropy (no conversion to theta yet)
# ---------------------------------------------------------------------------

def binding_tidy():
    w = pd.read_csv("binding/clean_anisotropy_df_20260826.csv")
    w["Day"] = "20260826"
    w["source"] = "wt_20260826"
    m = pd.read_csv("binding/clean_anisotropy_df_20260923-muts.csv")
    m["source"] = "muts_20260923"
    m["genotype"] = m["genotype"].fillna("blank")
    cols = ["source", "Day", "Biorep", "well", "genotype", "protein_conc",
            "DNA", "Buffer_only", "IPTG_conc", "read", "r"]
    both = pd.concat([w[cols], m[cols]], ignore_index=True)
    both = both.rename(columns={"protein_conc": "protein_uM_dimer",
                                "DNA": "dna_nM", "IPTG_conc": "iptg_mM"})
    # wt file records DNA as a present/absent flag (1), not nM
    both.loc[both["source"] == "wt_20260826", "dna_nM"] = np.nan
    wells = (both.groupby(["source", "Day", "Biorep", "well", "genotype",
                           "protein_uM_dimer", "dna_nM", "Buffer_only",
                           "iptg_mM"], dropna=False)["r"]
             .agg(r_mean="mean", r_sd="std", num_reads="size")
             .reset_index())
    return wells


def main():
    os.makedirs(OUT, exist_ok=True)
    print("library config ->", library_config())

    cal = od600_calibration()
    presplit = od600_to_cfu_per_mL(np.array([PRESPLIT_OD600]), cal)[0][0]
    print(f"presplit {presplit:.3g} CFU/mL; tubes start at "
          f"{presplit * PRESPLIT_DILUTION:.3g} CFU/mL "
          f"(ln per tube {np.log(presplit * PRESPLIT_DILUTION * TUBE_VOLUME_ML):.2f})")

    pooled = pooled_counts()
    samples, dropped = sample_table(cal, pooled)
    samples.to_csv(os.path.join(OUT, "sample_df.csv"), index=False)
    dropped.to_csv(os.path.join(OUT, "sample_df_dropped.csv"), index=False)
    write_counts(pooled, samples)
    print(f"{len(samples)} tubes kept, {len(dropped)} dropped: "
          f"{dropped['sample'].tolist()}")

    initial_composition(pooled).to_csv(
        os.path.join(OUT, "initial_composition.csv"), index=False)

    tubes, rates, summary = wt_controls(cal)
    tubes.to_csv(os.path.join(OUT, "wt_control_tubes.csv"), index=False)
    rates.to_csv(os.path.join(OUT, "wt_control_rates.csv"), index=False)
    summary.to_csv(os.path.join(OUT, "wt_control_rate_summary.csv"),
                   index=False)

    binding_tidy().to_csv(os.path.join(OUT, "binding_wells.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
