---
title: Per-sample (tube-level) ln_cfu offset in the growth likelihood
status: idea
filed: 2026-09-24
area: tfmodel
revisit_when: >-
  After censored floor observations are in (step 3.5 of the congression
  plan), or sooner if theta coverage stays well below nominal once the
  floor bias is fixed.
related:
  - planning/congression-physics-plan.md
  - planning/count-likelihood.md
  - planning/studies/congression-calibration/README.md
  - src/tfscreen/tfmodel/generative/observe/growth.py
  - src/tfscreen/tfmodel/generative/components/growth_noise/normal_kt.py
  - src/tfscreen/tfmodel/generative/components/sample_offset/normal.py
  - src/tfscreen/process_raw/counts_to_lncfu.py
---

**Context:** Diagnosing the congression calibration grid (2026-09-24). On
well-measured simulated rows (more than 100 reads) the error in `ln_cfu` was
0.38 ln units, almost all of it shared by every genotype in a sample:
per-sample SD 0.374, within-sample 0.054. It is the simulator's per-tube
growth noise (`tube_noise_sigma` x t). Real data add a second shared term:
every genotype's `ln_cfu` in a tube carries the same plating (`sample_ln_cfu`)
error, and the per-tube denominator review found a PCR component too.

The model treats these as independent per-row noise: `growth_noise`
(`normal_kt`) adds `sigma_k` in quadrature to each row's `ln_cfu_std`, and
`sample_ln_cfu_std` is folded into each row's `ln_cfu_var` in
`counts_to_lncfu`. A shared offset seen as N independent errors gives N-fold
too much information about everything that differs between tubes, and too
little about what is shared. Bulk theta 95% coverage was about 0.65 in every
arm of the grid, `single` included; this is one candidate cause.

**Idea:** Add one latent level offset per sample, `o_s`, shared by all
genotypes in the tube: `ln_cfu_obs[g, s] ~ ln_cfu_pred[g, s] + o_s + e[g, s]`.
Its prior width comes from the known shared sources (plating SD, tube growth
noise over the elapsed time, a PCR term); the per-row noise keeps only the
counting part. `sigma_k` then shrinks to what is genuinely per-row.

**Why not now:** The detection-floor bias (censoring) comes first; it is a
concrete bias the mixture exploits. This one is a calibration improvement
whose size in real data is not yet measured.

**What it would take:**

- A new component (or a replacement for `sample_offset`, whose current form
  offsets the growth *rate*, not the level; it is also not exposed by
  `tfs-configure-model`).
- `counts_to_lncfu` to write the counting variance and the plating variance
  separately instead of summing them into `ln_cfu_var`.
- `o_s` is a per-sample latent: mini-batch safe as a global site, like the
  condition parameters.
- Interaction with `ln_cfu0` and the k/dk_geno slide: `o_s` at the first
  timepoint trades against `ln_cfu0`; the prior has to keep it identified.
- The sim to test it: the calibration grid, comparing theta coverage with and
  without the offset.
- It is the `o_s` of `planning/count-likelihood.md`; building it first makes
  that project smaller.
