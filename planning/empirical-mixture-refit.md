---
title: Correct the empirical pipeline's per-genotype fits for congression
status: idea
filed: 2026-09-24
area: simulate
revisit_when: >-
  When the empirical-phenotype pipeline (tfs-build-empirical,
  tfs-fit-genotypes) is reviewed as a whole, and after steps 4-5 of the
  congression plan settle the theta and dk cell rules.
related:
  - planning/congression-physics-plan.md
  - src/tfscreen/tfmodel/genotype_fit/fit.py
  - src/tfscreen/simulate/scripts/build_empirical_cli.py
  - src/tfscreen/tfmodel/scripts/fit_genotypes_cli.py
  - src/tfscreen/simulate/cell_rules.py
  - src/tfscreen/tfmodel/generative/components/transformation/mixture.py
---

**Context:** Step 3.4 of the congression plan. `tfs-build-empirical` builds
simulation ground truth from real data: per-genotype maximum-likelihood fits
(Stage 1), a population distribution (Stage 2), resampling (Stage 3). Real
bulk genotypes are measured with congression built in, so resampling their
fits and simulating congression again counts it twice. The optional Stage 1.5
(`--congression_lambda`) removed that bias from the theta curves with the
theta-only `E[max(theta_g, co-residents)]` operator. Step 3.3c dropped that
operator from the main fit (the 3.0 study showed averaging theta cannot
represent a mixture of growth rates), and Stage 1.5 also used Poisson rather
than zero-truncated co-resident weights, treated purity as spiked/bulk, and
ignored dk dilution. It was retired on 2026-09-24 (user); the pipeline now
builds its distribution from uncorrected fits.

**Idea:** Replace Stage 1.5 with an iterated Stage 1 refit under the same
observable-level mixture as `tfs-fit-model`:

- Pass 1 is plain Stage 1.
- In each later pass, refit every genotype with
  `ln_cfu = ln_cfu0 + logsumexp_c(log w_c + G_c)`. The genotype's own
  parameters are free; its congressed classes combine them with fixed
  co-resident sets whose theta, activity and dk come from the previous pass,
  through the simulator's cell rules (`simulate/cell_rules.py`) and the
  zero-truncated weights with `bulk_fraction`.
- Iterate until the fits stop moving (expect 2-3 passes).

This corrects dk as well as theta, follows any change to the cell rules, and
gives each genotype a covariance under the corrected model.

**Why not now:** The user wants the whole empirical pipeline reviewed before
it is used again, and the theta and dk rules are about to change (steps 4-5).
In the measured regime the bias is modest (about 0.05 ln units RMS; see
`planning/studies/congression-estimator/`).

**What it would take:**

- A mixture likelihood in `genotype_fit/fit.py`, sharing the class
  construction with `transformation/mixture.py` or `cell_rules.py` rather
  than a third copy.
- Co-resident sets as in `ModelOrchestrator._draw_coresident_sets`.
- Cost: about (1 + K) times the work per fit, times the number of passes. The
  fits within a pass stay independent and parallel.
- Tests: reduction to plain Stage 1 as lambda -> 0; recovery of simulated
  phenotypes from a congressed simulation.
- Open question: whether the refit should share lambda with, or estimate it
  alongside, the main fit.
