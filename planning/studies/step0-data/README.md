# Step 0 data: compact extract of the 2026-07-23 snapshot

Support for the roadmap's step 0 studies (`planning/analysis-roadmap.md`),
not a study itself.

## What it makes

`growth.csv` in `~/Desktop/2026-07-23_data-snapshot.tar.gz` is 8.2 GB: one
row per sequenced tube x genotype, with every tube-level column repeated on
every row. `extract_snapshot.py` streams it out of the tarball without
unpacking (the disk had 17 GB free) and writes:

| File | Content |
|---|---|
| `counts.parquet` | `sample, genotype, counts`: 20,904,551 rows, 113 tubes, 215,402 genotypes |
| `samples.parquet` | one row per tube: design columns, per-tube `od600`, the processed `cfu_per_mL`/`sample_cfu` (from that tube's own OD600, per mL), total reads over listed genotypes, and `freq_denominator` = `adjusted_counts / frequency` (includes `__unknown__` reads and pseudocounts) |
| `binding.csv`, `base_growth.csv`, `presplit.csv` | copied as-is (the snapshot's `presplit.csv` has the off-by-one residue numbering; the corrected one is `dev/presplit.csv` in the main clone) |

Condition names are stripped of trailing spaces (`'pheS-4CP '`).

## How to run

```bash
python extract_snapshot.py ~/Desktop/2026-07-23_data-snapshot.tar.gz \
    ~/work/programming/git-clones/tfscreen/dev/real-2026-07-23
```

About 15 minutes. The output goes to the main clone's untracked `dev/` so
it survives worktrees; nothing from it is committed.

## Other inputs the studies use

- `~/Desktop/2026-07-23_onward/combined_fit_sample_df.csv`: one row per
  sequenced tube (118, of which 113 appear in `growth.csv`) with per-tube
  `od600`, `cfu_est` and `fit_lncfu`/`fit_lncfu_sem`, the global smoothed
  total (a polynomial in time per condition from repeated OD600 runs,
  through the calibration). Sample names differ in format from
  `growth.csv` (`0.0001` vs `0o0001`), so join on
  `replicate, condition_sel, titrant_conc, t_sel`.
- `~/Desktop/tfscreen-notebooks/od600-to-cfu/od600_to_cfu.yaml`: the lab's
  OD600-to-CFU calibration.

Units differ between files: `growth.csv`'s `sample_cfu` is cfu/mL from the
tube's own OD600; `combined_fit_sample_df.csv`'s `cfu_est` is 10x cfu/mL.
The studies use only within-file differences and slopes, which do not
depend on the scale.
