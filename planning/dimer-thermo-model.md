---
title: A general dimeric-TF thermodynamic model in place of the O2 models
status: active
filed: 2026-10-31
area: tfmodel
revisit_when: >-
  Active. Steps 1 (the port) and 2 (laptop feasibility) are done; step 3,
  the data axes, is designed (2026-11-02) and comes before any cluster
  grid.
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
   on condition, a data-model change of its own. (For the port only; step
   3 adds them, see **Step 3 design**.)
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
   the NSF preliminary data. Design below (**Step 3 design**); sub-steps
   3a-3g.

## Step 3 design (Mike, 2026-11-02)

Each axis is declared where the lab already records the physical thing it
belongs to. The effector axis exists already as `titrant_name`.

**Decisions.**

1. Backgrounds are genotypes, not an axis. A genotype is named in absolute
   terms, relative to the reference `wt_seq`: A100V made in the K84L/M42I
   background is `K84L/M42I/A100V`. The θ model needs nothing new: a
   background's shift is the sum of its mutations' shifts plus their pair
   terms, its own parent is a measured genotype, and background x library
   coupling is the existing pair horseshoe.
2. Operator and TF level are text labels in two optional tube-table
   columns, `operator` and `tf_level`. A table without them is one operator
   and one level, as now.
3. TF levels are given in absolute concentration (µM protomer), as the
   location of a prior on each level's total TF, not as a fold over a
   reference. They are learned unless the prior SD is 0.
4. Operators carry literature or measured offsets as priors. One operator
   is the reference, where the wt constants are defined.

**The contexts file.** `tfs-configure-model --contexts contexts.yaml`,
snapshotted into the priors CSV like `--growth_priors`, so the CSV stays
the place to override it:

```yaml
units: kcal/mol            # operator shifts; as thermo_options ddG_units
temperature_K: 310.15
operators:
  O1: {reference: true}
  O2: {shift: 1.5, shift_sd: 0.3}     # ΔΔG of operator binding vs O1
tf_levels:                 # total TF per cell, µM protomer
  low:  {conc_uM: 3,  ln_sd: 0.5}
  high: {conc_uM: 30, ln_sd: 0.5}
```

An operator's `shift` moves the effective operator constant of H and L by
the same ΔΔG, which is what one literature number per operator describes
(positive binds more weakly); it is subtracted from the first engagement
(`ln_K_h_o`, `ln_K_l_o`), so the effective constant K_o (1 + K_oii) moves
by exactly the shift. 3b added it; the 3e simulator test caught the sign
on 2026-11-02. Every
operator term (first and second engagement, monomer half-sites) also gets
a per-operator deviation, N(0, 0.5 kT), as the per-titrant effector
deviations do.
Mutation effects are shared across operators and TF levels. Labels in the
tube table that the file does not name are an error; a file is required
once there is more than one label of either kind.

**Binding data.** `binding_df` gains optional `operator` and `tf_total_uM`
columns: the in vitro protein concentration is pipetted, so it is known,
not learned. `tfs-process-binding` writes `tf_total_uM` and keeps every
protein concentration instead of filtering to one (`--protein_conc`
becomes an optional filter). This fixes a present mismatch: binding
measured at 2 µM protein is now fit at the in vivo TF total.

**Implementation.** Inside the fit, (`titrant_name`, `operator`,
`tf_level`) fold into one θ-condition index on the existing `titrant_name`
tensor dimension. Every growth component already indexes tubes by that
dimension, so tube identity stays unique and nothing outside the θ layer
changes shape. Static per-index arrays (`effector_idx`, `operator_idx`,
`tf_level_idx`) tell the θ components what each index is. Every output
splits the index back into `titrant_name`, `operator` and `tf_level`
columns. Per-genotype curve models (`hill_geno` and the like) fit each
context as its own curve, which is what they mean; the O2 thermodynamic
models refuse more than one operator or TF level.

**Sub-steps.**

- 3a. Data: tube-table and `growth_df` columns, `binding_df` columns, the
  contexts file and its priors rows, the folded θ-condition index in the
  orchestrator. Without the columns, configs and fits are unchanged
  (regression tests on an existing config). Done 2026-11-02 except the
  priors rows, which need the dimer's per-operator and per-level fields
  and move to 3b (`tfmodel/contexts.py`, `tests/.../test_contexts.py`).
  Found on the way: binding data measuring a subset of the growth
  titrants were indexed against the wrong titrant; fixed by aligning the
  binding axis with the growth titrants.
- 3b. thermo.dimer: per-operator shift and deviations, per-level total TF,
  binding at its known `tf_total_uM`. Done 2026-11-02: effector terms are
  indexed by effector, not θ condition (`model.condition_terms`); shifts,
  deviations and TF levels are non-centered (loc + scale · z), so a scale
  of 0 holds a value without changing the sites; the contexts file's
  values reach the priors CSV as `operator`/`tf_level`-tagged rows, joined
  by label on read.
- 3c. Outputs: `tfs-predict-theta`, extraction, `tfs-summarize-fit` and
  `tfs-compare-runs` carry the `operator` and `tf_level` columns; new
  `operator_shift` and `tf_total` tables. Done 2026-11-02. The dimer's
  θ prediction resolves each row's effector, operator and TF level
  (`model.resolve_context_rows`), so a manual grid may also give
  `tf_total_uM` to predict at any protein concentration; curve models'
  tables gain the context columns in the generic extraction layer
  (`extraction._with_context_columns`, `_attach_context_columns`,
  `_split_condition_labels`) and refuse manual grids with contexts;
  `copy_orchestrator` keeps the context columns; summarize-fit joins on
  shared context columns and skips binding at a known protein
  concentration against growth-context predictions. Its truth tables
  get context columns with the simulator (3e).
- 3d. Backgrounds: declared in the experiment's library YAML (below), with
  a `background` tube-table column; `tfs-process-fastq --background` names
  genotypes in absolute terms; the composition table, genotype check and
  congression pools work per background, since each background is its own
  transformation. Done 2026-11-02: `genetics/backgrounds.py`;
  `LibraryManager` names against the reference (`name_reference_seq`,
  `name_extra_mutations`); the composition table gains a `background`
  column; configure checks each growth row against its background; the
  orchestrator draws one pool per library (`_congression_pools`;
  `GrowthData.pool_bulk_fraction`, `pool_coresident_idx`,
  `condition_pre_pool`) and the mixture picks each condition_pre slot's
  pool. Pool 0 draws exactly as a single pool did. Not yet: the library
  subsample keys with backgrounds, and backgrounds in the simulator (3e).
- 3e. Simulator: the design table's new columns, the contexts and
  backgrounds in the truth, truth tables for operator shifts and TF levels.
  Done 2026-11-02. The simulator's θ axis is the fit's θ conditions
  (`sim_data_class.sim_theta_conditions`; `theta_gc` columns are θ
  condition-major), so several titrants also get their own curves; a
  `contexts` key reads the contexts file; a `thermo.dimer` draw writes
  `{out_prefix}_operator_shift.csv`, `_operator_dev.csv` and
  `_ln_tf_total.csv`, which `tfs-summarize-fit` joins to the fit's
  tables; `binding_data` takes `operator` and `tf_total_uM`. A
  `backgrounds` block builds one library per background
  (`backgrounds.background_library_df`); each library is transformed from
  its background's origins and mixture, and its tubes' counts list only
  its background's genotypes. Found on the way: the operator shift had the
  wrong sign (below). Not yet: presplit with backgrounds.
- 3f. `tfs-process-binding` as above. Done 2026-11-02. The lab's protein
  column is a dimer concentration, so `tf_total_uM` is twice it
  (`--protein_species dimer`, the default). Theta SDs pool per protein
  concentration, and a concentration with no replicated point takes the
  pool over all of them. `--no_write_tf_total` writes the old table for
  curve models, which cannot link binding to a protein concentration.
- 3g. Laptop validation: A2 again with two operators, two TF levels and
  two backgrounds, checked against the identifiability study's step 1.

**How a background is declared (Mike, 2026-11-02).** One library YAML per
experiment stays the rule, with an optional `backgrounds` block. Each
background gives its amplicon DNA (`wt_seq`) so the expected reads match
its real codons, and `outside_mutations` for the mutations the reads never
show. Congression uses one pool per library, since a library is one
transformation of one background's plasmids into one marker strain
(Mike, 2026-11-02: per background and per marker, with one global λ). Each
pool draws its own co-resident sets from its background's table, and each
library's condition_pre slots use their library's pool.
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
