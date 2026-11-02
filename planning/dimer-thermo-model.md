---
title: A general dimeric-TF thermodynamic model in place of the O2 models
status: active
filed: 2026-10-31
area: tfmodel
revisit_when: >-
  Active. Step 1 (the port) is done; step 2, feasibility in the full
  pipeline on the laptop, is under way; the data axes (step 3) come
  before any cluster grid.
related:
  - planning/studies/thermo-identifiability/README.md
  - src/tfscreen/tfmodel/generative/components/theta/thermo/dimer/
  - docs/source/model.rst
---

## Context

The identifiability study (`planning/studies/thermo-identifiability/`) asked
which species and energy terms of a dimeric-TF model a screen can resolve.
It found everything identifiable except the split between the first and
second operator engagement, recommended keeping the second engagement as a
softplus term, and found that predictions of combinations need a pair term,
per-effector deviations on the effector terms, and posterior draws from a
sampler rather than a Laplace.

## Decisions (Mike, 2026-10-31)

1. The data axes stay as they are: one operator, one known TF level, the wt
   background. Operator, TF-level and background axes need theta to depend
   on condition, a data-model change of its own.
2. The species set is chosen by static priors flags, not by registry names.
3. The terms are referenced to the dimer and reuse the O2_C12 names, so the
   default model nests O2_C12 U1.
4. PK puts a hierarchical normal on every per-mutation shift. PddG and PnnC
   put the existing structural prior on the six C12 constants, with ΔΔG_HO
   setting the first engagement. The other terms get the PK prior.
5. Effector effects are shared across titrants plus a per-titrant deviation,
   prior SD 0.5 (the study's step 5 remedy).
6. The pair term is a horseshoe on every term, scaled by contact distance
   where the structure file gives one.
7. The solver works in log space with implicit-function derivatives, as
   `mwc.py` does.
8. Not in the port: sampler intervals for combinations, the extra data axes,
   and retiring the O2 models.
9. Order after the port (2026-11-01): a cheap laptop feasibility check, then
   the data axes, then the compute-heavy utility grids on the axis-capable
   model.

## Steps

1. Done (2026-10-31). Port the model: `thermo/dimer/` (`species.py`,
   `model.py`, `PK.py`, `PddG.py`, `PnnC.py`). Validated against an
   independent nested-brentq solve in every variant (1e-11), exactly against
   `mwc.theta_c12`, and by finite differences of the gradient and Hessian.
   Wired into the orchestrator, extraction, the simulator's truth tables and
   the registry-wide tests. A pipeline smoke test runs configure, MAP,
   Laplace, extraction and prediction.
2. Feasibility in the full pipeline, on the laptop
   (`planning/studies/dimer-utility/`, stage A). A1: with the second
   engagement off the model is O2_C12 U1, so a dimer.PK and an O2_C12 U1 PK
   fit of the same simulation should agree. A2: simulate the default
   variant from the AF3 state table and fit PK and PddG on a small library;
   record parameter count, MAP convergence, Laplace time and memory, and
   recovery. A machinery problem found here is fixed before step 3.
   Found (2026-11-01, study README): joint MAP does not estimate the
   hierarchical scales, so they are now held and updated between fits
   (direct marginal-likelihood update works for PK σ_d, not yet for PddG
   σ_s); with them held the constants cover. Open: θ intervals undercover
   (width 0.04 against RMSE 0.09) in every fit, scales held or not.
3. Operator, TF-level and background axes in the data model, and with them
   the per-operator deviations. Before any cluster grid (Mike, 2026-11-01):
   at one operator, one TF level and wt only, a failed utility test cannot
   tell a wrong model from an under-measured one (saturation hides the
   effector-site shifts; study step 3), and the backgrounds analysis feeds
   the NSF preliminary data.
4. Utility in full simulations, on the cluster, on the axis-capable model:
   the ΔΔG prior (PK, PddG learned σ_s, PddG σ_s held at the dial, over the
   table error) and combinations (planted distance-scaled pair coupling,
   close-doubles library, pair term on and off, scored on five-mutants),
   with `wt_only` as one design arm. PnnC once its LigandMPNN features are
   built. Revalidates the study's steps 3-5.
5. Sampler-based intervals for combinations (NUTS on the sites a prediction
   touches), or a guide checked against a sampler.
6. Retire the O2 models once the general model reproduces their fits.

## Known limits of the port

- With the unfolded dimer off, the structural block still carries a
  Δln K_u row. It reaches no θ, so the per-structure ΔΔG common mode then
  comes from the prior.
- Guesses cover only the wt sites every variant has, because they are made
  before the priors CSV, so before the flags, are read.
- TF total is fixed, not a latent.
