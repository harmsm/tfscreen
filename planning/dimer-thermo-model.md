---
title: A general dimeric-TF thermodynamic model in place of the O2 models
status: active
filed: 2026-10-31
area: tfmodel
revisit_when: >-
  Active. Step 1 (the port) is done; step 2, testing its use in full
  simulations, is the next commit.
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

## Steps

1. Done (2026-10-31). Port the model: `thermo/dimer/` (`species.py`,
   `model.py`, `PK.py`, `PddG.py`, `PnnC.py`). Validated against an
   independent nested-brentq solve in every variant (1e-11), exactly against
   `mwc.theta_c12`, and by finite differences of the gradient and Hessian.
   Wired into the orchestrator, extraction, the simulator's truth tables and
   the registry-wide tests. A pipeline smoke test runs configure, MAP,
   Laplace, extraction and prediction.
2. Next commit. Test whether the model is useful in full simulations with
   realistic structural and energetic inputs, with and without the ΔΔG and
   NN priors. Revalidate the study's steps 3-5 in the full simulation.
3. Sampler-based intervals for combinations (NUTS on the sites a prediction
   touches), or a guide checked against a sampler.
4. Operator, TF-level and background axes in the data model, and with them
   the per-operator deviations.
5. Retire the O2 models once the general model reproduces their fits.

## Known limits of the port

- With the unfolded dimer off, the structural block still carries a
  Δln K_u row. It reaches no θ, so the per-structure ΔΔG common mode then
  comes from the prior.
- Guesses cover only the wt sites every variant has, because they are made
  before the priors CSV, so before the flags, are read.
- TF total is fixed, not a latent.
