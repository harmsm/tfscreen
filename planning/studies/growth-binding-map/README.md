# Study 0a: growth vs in vitro binding, model-free

**Status: provisional (2026-09-26).** Run on the 2026-07-23 snapshot:
counts from a FASTQ processor that assigned every unknown read to wt (about
two thirds of kan reads), a stringent Q cutoff, and binding data for five
genotypes at the original (high) protein concentration. The wt-relative
pass is unusable in kan and wt's own points in the absolute pass are
suspect; the mutants' absolute slopes are unaffected. The user has since
collected binding at two lower repressor concentrations (curves much closer
to the in vivo response, wt no longer capped at theta = 1) for 10 genotypes
with diverse behavior, which confirms the concentration offset and should
give this study real leverage. The 4CP null result conflicts with an
earlier per-genotype MLE analysis that showed reasonable pheS response
curves; re-check it on the new data before drawing conclusions.

## Question

For the genotypes with in vitro binding data, is growth rate a single
linear function of the binding occupancy theta (the joint fit's
assumptions 1 and 2), a monotone but nonlinear one, or neither? Decides the
base case for Track G (`planning/growth-only-relative-fit.md`, step 0;
`planning/analysis-roadmap.md`, steps 5, 8, 9).

## How to run

```bash
python growth_binding_map.py ~/work/programming/git-clones/tfscreen/dev/real-2026-07-23 out/ \
    ~/Desktop/2026-07-23_onward/combined_fit_sample_df.csv
```

Inputs from `planning/studies/step0-data/`. The third argument enables the
second pass (absolute slopes and the concentration-scale scan).

## Method

Five genotypes have binding data: wt, M42I, H74A, K84L, M42I/H74A (all
spiked; 22,000-56,000 reads per tube). Binding and growth use the same
eight IPTG concentrations (0 to 1 mM).

1. **wt-relative, total-free.** Slope over `t_sel` of
   `ln(reads_g / reads_wt)` per replicate x condition x concentration:
   `dk_g + m_c (X_g - X_wt)`, with no tube total and no tube noise.
2. **Absolute.** Slope of `ln(reads_g / D) + fit_lncfu` (the global
   smoothed total): `k_c + dk_g + m_c X_g`. Uses theta itself, which for wt
   spans 1.00 to 0.33, but trusts the total of each IPTG column: an error in
   a column's total shifts every genotype at that concentration together.
3. **Concentration scale.** Hill fits to the binding data, then theta_g
   evaluated at `s x` (in vivo effective [IPTG] = s times external) for
   s = 0.3-300.

Slope noise is taken from replicate differences (counting SEs understate it
about 2.7-fold; study 0b).

## Commit

Scripts as in this directory on top of `b949095`.

## Results (2026-09-26)

**1. wt-relative: no leverage.** Slope noise 0.0032-0.0037 per min
(replicate differences; counting alone predicts 0.0016). The four mutants'
binding curves differ from wt's by at most a few tenths across IPTG, so the
common-m fits are consistent with the linear map and measure nothing (kan
m = +0.005 +/- 0.006, 4CP m = +0.002 +/- 0.009 per min per unit theta).

**2. Absolute, s = 1: the common linear map fails.**

| condition | m (per min) | reduced chi2 | genotype-specific m | curvature |
|---|---|---|---|---|
| kan+ | -0.0124 +/- 0.0019 | 6.7 | p = 2e-5 | p = 4e-4 |
| 4CP+ | +0.0006 +/- 0.0019 | 2.0 | p = 6e-4 | p = 0.08 |

The curves (mean over replicates, per min):

- kan+: every genotype's growth rises from about 0 to 0.015-0.027 between
  0.001 and 0.01 mM IPTG, then flattens or falls slightly. In vitro, wt,
  M42I and M42I/H74A are still fully bound (theta 1.00) at 0.01 mM, and wt
  falls only to 0.33 at 1 mM. H74A (theta 0.57 at 0 mM) grows like wt at
  0 mM.
- 4CP+: growth falls with IPTG to a minimum near 0.01-0.03 mM and partly
  recovers, for every genotype; no ordering by binding.
- Total-free check of the same point: at 0.01 mM in kan, all four mutants
  outgrow wt (replicate 1: +0.006 to +0.015 per min) although M42I and
  M42I/H74A bind like wt there in vitro.

**3. An in vivo concentration scale helps in kan.** Reduced chi2 of the
common linear map by s:

| s | 0.3 | 1 | 3 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|---|---|
| kan+ | 7.1 | 6.7 | 6.0 | 4.8 | 3.4 | 3.0 | 3.7 |
| 4CP+ | 2.0 | 2.0 | 2.0 | 1.9 | 1.8 | 1.8 | 1.8 |

At s = 30-100, kan m = -0.024 to -0.026 per min. 4CP stays weak
(m = +0.005 to +0.006).

The no-drug conditions (only 0 and 1 mM) differ between the two
concentrations for every genotype (kanR library faster at 1 mM, pheS
library slower), in line with the OD600 population slopes (study 0c):
marker expression changes growth even without the drug, so "no drug" is not
a pure `dk_geno` control.

## What it means

- **The base case (one linear map at s = 1) is not supported.** In kan the
  in vivo response sits at roughly 30-100x lower external IPTG than the in
  vitro assay, with genotype-specific structure left over; in 4CP growth
  does not track in vitro binding for these five genotypes. The joint fit's
  assumption 1 (in vitro binding = in vivo occupancy at the same [IPTG])
  is likely violated, which is what Track G was built to find out. The map
  in step 8 must include the concentration scale s.
- Plausible sources of s: active IPTG uptake (LacY) concentrating IPTG in
  cells; the in vitro assay's conditions (22 uM protein; not all curves
  taken with the final protocol, per the snapshot README).
- **Caveats.** Five genotypes, all similar to wt in vitro; the absolute
  pass depends on per-column totals (roadmap C4), and the non-monotone
  shapes shared by every genotype (the kan peak at 0.01 mM, the 4CP dip)
  are what a column-total error would look like.
- **The direct test is a bench measurement:** monoculture growth rates of
  these five genotypes across the eight IPTG concentrations in kan and 4CP
  (`planning/absolute-abundance-measurements.md`, experiment 1, extended
  from the gauge concentrations to the full titration). It measures the
  growth-binding map with no library and no totals problem.
