# Study 0c: what per-tube OD600 says about each tube's total

**Status: first pass (2026-09-26)** on the two sequenced replicates'
per-tube OD600; waiting on the third bioreplicate, the repeated OD600 runs
and the presplit OD600 and dilution.

## Question

How do bioreplicates differ (level vs slope), how far do tubes scatter
about a smooth curve compared with the reader's own noise, which tubes are
near or below the detection threshold, and how fast does the total grow in
each condition? Feeds roadmap steps 2, 4 and 6 and open item D9 (the
population curve form) (`planning/analysis-roadmap.md`).

## How to run

```bash
python od600_anatomy.py ~/Desktop/2026-07-23_onward/combined_fit_sample_df.csv \
    ~/Desktop/tfscreen-notebooks/od600-to-cfu/od600_to_cfu.yaml out/
```

## Method

Per-tube OD600 through the lab calibration to ln(cfu); reading noise from
the calibration's 2% OD600 SD alone (the calibration curve's own error is
shared by every tube and left out). Nested least-squares models over the
118 sequenced tubes (2 replicates x 20 conditions x 3 timepoints, a few
missing): one curve per condition (a line in `t_sel`), plus a replicate
level, plus replicate x condition levels, plus replicate-specific slopes.

## Commit

Scripts as in this directory on top of `b949095`.

## Results (2026-09-26)

- **Replicates differ by a level:** replicate 2 reads +0.24 ln units above
  replicate 1 (SE 0.03; F = 52, p = 3e-10). Separate levels per condition
  add nothing (p = 0.14), consistent with one split density per replicate.
- **Replicate-specific slopes are significant** (F = 4.2, p = 7e-5, residual
  SD 0.167 -> 0.115). This contradicts slopes being shared across
  bioreplicates for these two replicates; with three tubes per line it
  could also be structured noise.
- **Tube scatter dwarfs reading noise:** 0.176 ln units about a shared
  curve plus replicate level, against 0.031 from OD600 reading noise. The
  excess, 0.17, is tube-to-tube (growth or handling). It does not grow
  with time (0.14, 0.11, 0.18 by tercile of `t_sel`), so it is not
  `sigma * t` tube growth noise alone.
- **No tube is below the threshold** (0.096). The lowest are kan+ at 0 mM
  (1.5x threshold), kan- at 1 mM and kan+ at 0.001 mM (1.6-1.7x).
- **Population growth per condition** (per min, shared across replicates
  with a replicate level): kan+ rises from -0.001 +/- 0.006 at 0 mM to
  about 0.011 at >= 0.01 mM; 4CP+ falls from 0.020 at 0 mM to 0.014-0.015
  at >= 0.01 mM; no drug 0.016-0.030. Standard errors are 0.003-0.007 per
  condition: three tubes per replicate over 45-120 min.

## What it means

- **Per-tube OD600 pins condition growth rates only to +/- 0.003-0.007 per
  min,** comparable to `m` (~0.01-0.025). The absolute scale (`k_c`, `m_c`;
  roadmap C4) cannot rest on these tubes alone; the earlier smoothed total
  drew on repeated OD600 runs, and those data (and the third bioreplicate)
  are the next input for this study.
- **The population model needs a replicate level** and probably
  per-replicate slope deviations (D9), and the OD600 likelihood a per-tube
  scatter term of about 0.17 ln units that is not proportional to time.
- **The sequenced tubes were chosen to be visible;** strong selection
  shows up as low but readable OD600 (kan+ 0 mM barely grows), not as
  censored tubes, in this dataset.
- Missing for the rest of 0c: OD600 for the third (unsequenced)
  bioreplicate and the repeated OD600 runs behind `fit_lncfu`, and the
  presplit OD600 and dilution factor (the known start).
