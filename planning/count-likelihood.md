---
title: Fit read counts directly instead of ln_cfu
status: idea
filed: 2026-09-24
area: tfmodel
revisit_when: >-
  After censoring and the per-sample offset are in and the congression
  calibration grid has been re-run. The user expects to get here (2026-09-24):
  the data side is mostly propagating information we drop today.
related:
  - planning/congression-physics-plan.md
  - planning/per-sample-level-offset.md
  - planning/studies/congression-calibration/README.md
  - src/tfscreen/process_raw/counts_to_lncfu.py
  - src/tfscreen/tfmodel/generative/observe/growth.py
  - src/tfscreen/tfmodel/generative/observe/presplit.py
---

**Context:** The congression calibration grid (2026-09-24) found that the
mixture used its congressed classes to fit the read-count detection floor.
`ln_cfu` is computed from counts with a pseudocount of 1, so rows with 0
reads sit +1.26 ln units above the truth and 1-2 reads +0.62. Censoring those
rows is the next step. It keeps "below the floor" but loses the difference
between 0 and 4 reads, and it leaves the rest of the approximation in place.

**Idea:** Make read counts the observation. Per sample s and genotype g:

```
reads[g, s] ~ NegBin(mean = exp(ln_cfu_pred[g, s] + o_s), dispersion)
```

with `o_s` a per-sample intercept whose prior is centered on plating,
`o_s ~ ln N_s - sample_ln_cfu_s` with the plating uncertainty as its width.
With `o_s` free this matches a multinomial over the tube (the Poisson trick),
but each genotype's term stands alone, so genotype mini-batching still works.
The reads carry how genotypes compare within a tube; plating carries the
absolute scale. Predictions and all downstream outputs stay in `ln_cfu`.

What it gives that `ln_cfu` with censoring cannot:

- **Exact low-count information:** 0 vs 4 reads pins down when a genotype
  crashed, which carries theta information at strong selection.
- **The right noise model everywhere:** no first-order binomial approximation
  (poor below ~20 reads), no pseudocount, and a learned overdispersion for
  PCR jackpotting, which is probably why high-count rows look overconfident.
- **Misassignment as additive physics:** index hopping, cross-contamination
  and barcode-calling efficiency add reads in count space,
  `E[reads] = N (p_g + h q_g + ...)`, which is what produces a floor. Index
  hopping into sample s comes from genotype g in the other samples of the
  lane, so it only couples a genotype to itself across samples (batch safe).
  Sequencing errors that turn one genotype into another would couple
  genotypes; leave them to the `__unknown__` bucket.

**Why not now:** The user wants the masked grid results and censoring first
(2026-09-24). The core noise model needs design: today tube growth noise is
per-row variance (`sigma_k`). In a count model, the per-tube shared part goes
into `o_s` (see `planning/per-sample-level-offset.md`) and the per-row part
into the NegBin dispersion; a per-(tube, genotype) latent would be far too
many parameters.

**What it would take:**

- **Data:** keep raw counts, per-sample depth `N_s` (including the
  `__unknown__` reads) and plating through `tfs-process-counts` and
  `tfs-process-presplit`. Most of it is already in the processed CSVs.
- **Model:** new growth and presplit observers; per-sample `o_s` latents
  (built by the per-sample offset idea); a dispersion model; optionally index
  hopping and calling efficiency.
- **Interactions:** `o_s` vs `ln_cfu0` and the k/dk_geno slide; the congression
  mixture is unchanged (it acts on `ln_cfu_pred`).
- **Checks:** the calibration grid with index hopping and PCR overdispersion
  turned on in the simulator, against the censored `ln_cfu` model.
