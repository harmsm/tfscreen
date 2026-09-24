---
title: Match simulator and fit on the physics of congression
status: active
filed: 2026-09-13
area: tfmodel
revisit_when: >-
  Active now. Update the step list as steps finish.
related:
  - src/tfscreen/simulate/selection_experiment.py
  - src/tfscreen/tfmodel/generative/components/transformation/mixture.py
  - src/tfscreen/tfmodel/generative/model.py
  - src/tfscreen/tfmodel/model_orchestrator.py
  - src/tfscreen/tfmodel/analysis/prediction.py
  - planning/empirical-mixture-refit.md
  - src/tfscreen/simulate/transformation_lam_output.py
  - src/tfscreen/genetics/library_design.py
  - planning/estimate-dk-alpha-by-varying-lambda.md
  - planning/plasmid-segregation-low-copy.md
  - planning/combination-specific-expression-effects.md
  - planning/estimate-spike-fraction-from-growth-curvature.md
---

## Why

A code analysis revealed the simulator and the fit describe congression
(cells carrying more than one plasmid) differently:

1. **The simulator combines whole growth rates.** `_sim_growth` in
   `simulate/selection_experiment.py` takes the min (kanR) or max (pheS) of each
   co-resident plasmid's full `k*t`, including its `dk_geno` and the condition
   baseline. The min/max is meant to encode "the tightest binder sets theta,"
   but `dk_geno` varies more across genotypes (SD about 0.018 per minute in the
   sim prior) than the whole theta signal (|m| = 0.0099 kanR, 0.0062 pheS). So
   the selection is driven by `dk_geno`: in cells with a co-resident, the chosen
   plasmid is the highest-theta one only about 52% of the time (chance level).
   Monte Carlo over the sim prior (theta uniform as a stand-in) gives a growth
   error with SD of about 0.01 per minute, 1 to 1.5 theta units. The fit
   corrects theta only, so the error lands on lambda, `m`, the growth
   baselines and theta.
2. **"Spiked" genotypes are not congression-free.** Their codons also occur in
   the bulk sub-libraries. In the old sim config only 0.3% of wt cells and 14 to
   20% of M42I/H74A/K84L cells came from the monoclonal spike, yet
   `--spiked` masks all of them as clean. This is also true of the real data,
   because spiked and library copies can't be told apart by sequence.

Checked and ruled out: the definition of lambda (sim cell sizes are
zero-truncated Poisson(lambda), so a focal plasmid's co-residents are exactly
Poisson(lambda), as the fit assumes; true only while every plasmid got its
whole cell's abundance. Since step 1 a cell's abundance is shared among its
plasmids, and the co-resident weights are zero-truncated; corrected in 3.4),
and the fit's unweighted background
distribution (the lambda that reproduces the sim's true theta inflation is 0.30
to 0.35 against a true 0.357).

## Where we start (code on `main`, 2026-09-23)

**Fit.** Congression is a correction to theta alone, applied in
`generative/model.py` through the `transformation` component:

- A congressed genotype's growth theta is replaced by the expected dominant
  maximum `E[max(theta_g, theta_background)]` over Poisson(lambda)
  co-residents (`_congression.update_thetas`). The background distribution is
  logit-normal (`logit_norm`) or the empirical theta of the whole population
  (`empirical`); `single` turns the correction off.
- Which genotypes are corrected is a yes/no `congression_mask`, built in
  `model_orchestrator.py` from the spiked genotypes (from the library
  composition table's `in_spiked_origin`, or the legacy `spiked_genotypes`
  list). `ln_cfu0_spiked_mask` is exactly `~congression_mask`, so one flag sets
  both purity and the `ln_cfu0` prior class.
- dk_geno is never touched by congression.
- The corrected theta then passes through `theta_growth_noise` and
  `theta_rescale` before growth is assembled.
- Under mini-batching the empirical background is computed over the full
  population (`population_theta`), and prediction code that subsets genotypes
  must supply `external_theta_population` (`analysis/prediction.py`).
- The same `update_thetas` operator is reused by Stage 1.5 of
  `tfs-fit-genotypes` / `tfs-build-empirical`
  (`tfmodel/genotype_fit/congression.py`).

**Library composition.** `tfs-configure-model --library_config` writes
`{out_prefix}_library.csv` (`genetics.library_composition_table`), with a
per-genotype `bulk_fraction` (the `f_g` below) and `pool_fraction`. The model
reads the table but does not yet use either fraction; it only derives the
spiked set from `in_spiked_origin`.

**Simulator.** `_sim_growth` (`simulate/selection_experiment.py`) combined each
co-resident plasmid's whole `k*t` with a per-marker-library
`multi_plasmid_combine_fcn` (`gmean`/`mean`/`min`/`max`/`sum`; default `mean`;
the example configs used `min` for kanR and `max` for pheS). Replaced in
step 2 (2026-09-23) by cell-level theta and dk rules.

**Known simulator/design mismatch.** `_sim_sequencing` and
`_calc_genotype_cfu0` credit every plasmid in a cell with that cell's whole
abundance, so a co-transformed cell counts once per plasmid.
`expected_library_composition` assumes the opposite: a cell's plasmid copies
are shared among the variants it carries, so `bulk_fraction` does not depend
on lambda. Fixed in step 1 (2026-09-23): the simulator now splits a cell's
abundance among its plasmids.

## Biology and assumptions

Provenance: user, 2026-09-13, unless noted.

- **The repressor is a dimeric lac repressor, not tetrameric.**
- **Protein is in large excess over DNA:** about 10 uM dimer against about
  10 nM operator. No depletion, so each variant's statistical weight on the
  operator is linear in its share of the protein.
- **Heterodimers: unknown.** Working assumption is mostly homodimers, which is
  intuition, not data. Native mass spectrometry (collaborator) should resolve
  it.
- **Total repressor expression per cell does not depend on plasmid number.**
  Each variant is diluted by its share. Combination-specific effects on
  expression are out of scope (`planning/combination-specific-expression-effects.md`).
- **Plasmids are high copy and assumed to persist without segregating** over
  the experiment (roughly 3 to 7 doublings). The next design moves to a
  low-copy plasmid, where segregation matters
  (`planning/plasmid-segregation-low-copy.md`).
- **The dataset has a single lambda** (0.3572) and is headed for publication.
  Parameters that only a lambda series could identify are set from biology
  and checked by sensitivity, not fit.
- **Spiked and library copies are indistinguishable** in this dataset. Future
  runs will give spikes distinct codons.
- **dk_geno spread is tight in this library** (bench, 2026-09-23): monoculture
  growth rates of eight genotypes, chosen to be biased toward expected poor
  growers, were 0.024 +/- 0.001. Not guaranteed for other libraries.
- **Overnight outgrowth does not shift the clean/congressed class weights**
  (user, 2026-09-23). The mixture weights `w_clean`/`w_congressed` are the
  split at transformation (`f_g`, lambda), but the mixture needs the split at
  `t = -t_pre`, after an overnight of non-selective growth. Congressed cells
  of genotype g would be reweighted by roughly
  `exp((dk_cell - dk_g) * t_overnight)`; theta plays little role without
  selection. Presplit sees only the total, so it cannot correct this. We
  assume the shift is negligible and leave it unmodeled. Revisit if step 3's
  calibration shows bias concentrated in bulk genotypes with large |dk_geno|.

## Decisions

### Combine each physical quantity at its own level

Build the cell's theta and dk from its plasmids, then assemble growth once:

```
k_cell = k_condition + dk_cell + m * A * theta_cell
```

Never combine whole growth rates across plasmids.

### Theta: partition function over the protein mixture

Write `z_g = theta_g / (1 - theta_g)` (odds at single-variant concentration)
and `x_g` for variant g's share of subunits in the cell. Two limits:

| Assembly | Cell odds | Logit form |
|---|---|---|
| Homodimers only | `sum_g x_g * z_g` | `log sum_g x_g exp(l_g)` |
| Random heterodimers, additive half-site energies | `(sum_g x_g * sqrt(z_g))^2` | `2 log sum_g x_g exp(l_g / 2)` |

Both are soft maxima between the share-weighted mean logit and the max; the
heterodimer form sits closer to the mean. A strong binder in a 1:1 cell loses
log 2 (homodimer) or log 4 (heterodimer) logit units compared with a max rule.
The heterodimer rule is exact per titrant concentration only if subunits bind
ligand independently; allosteric coupling between subunits makes it
approximate. Default: homodimers, until native mass spec says otherwise.
The choice is made in step 4 from the bench experiments below.

### dk_geno: soft-min family

```
dk_cell = -(1/alpha) * log sum_g x_g exp(-alpha * dk_g)
```

`alpha -> 0` is dilution (share-weighted mean burden); `alpha -> inf` is
dominance (the worst variant sets the cost). Alpha is set from biology with a
sensitivity check, not fit (`planning/estimate-dk-alpha-by-varying-lambda.md`).
Under dilution the average effect is a uniform shrinkage of each genotype's
dk toward the population mean, by `E[1/M]` over the abundance-weighted cell
size (`M` zero-truncated Poisson, see 3.4) = 0.913 at lambda = 0.357 (0.841
under the old size-biased weighting), which the hierarchical dk_geno prior
mostly absorbs.

The fit's current behavior (each plasmid keeps its own dk) has no cell-level
counterpart: a cell carrying two plasmids has one growth rate. It is wrong,
not just approximate, so we accept a breaking change here (user, 2026-09-23).
Both simulator and fit start from **dilution** (alpha -> 0); step 5 adds the
rest of the family and the sensitivity check.

Like theta's background, dk's co-resident background must come from the full
genotype population, not the current mini-batch, and prediction paths that
subset genotypes need an external dk population reference.

### Mix at the observable level

A genotype's cells are a mixture of classes growing at different rates, so
mix `exp(ln_cfu)`, not rates or theta. With `f_g` the fraction of genotype g's
cells from the bulk library (computed from the library design), a focal plasmid
has no co-residents with probability `exp(-lambda)` (superseded in 3.4: with
a cell's abundance shared among its plasmids, the clean fraction of a bulk
genotype's abundance is the zero-truncated `P(M = 1) = lambda exp(-lambda) /
(1 - exp(-lambda))`, so `w_congressed = f_g (1 - P(M = 1))`):

```
w_congressed = f_g * (1 - exp(-lambda))
w_clean      = 1 - w_congressed
ln_cfu_g(t)  = ln_cfu0_g + log[ w_clean * exp(k_clean t)
                              + w_congressed * E_S exp(k_{g,S} t) ]
```

`S` is the random set of co-resident genotypes. This form covers every
genotype: a pure spike has `f = 0`, a pure bulk genotype has `f = 1` (and is
still 70% clean cells). Whether `E_S exp(k t)` can be approximated by
`exp(E_S[k] t)` must be checked numerically on the real design and timepoints.

`k t` above is shorthand. The model's growth is pre-growth plus selection,
routed through the `growth_transition` component, and both `t_pre` and `t_sel`
growth can carry congression artifacts (user, 2026-09-23). So the mixture is
over each class's whole `ln_cfu` trajectory: each class gets its own `k_pre`
and `k_sel`, the `growth_transition` model is applied per class, and the
classes are mixed last.

Other observations:

- **base_growth** measures a constant rate (`k_ref + dk_geno`) and is not
  mixed (user, 2026-09-23).
- **presplit** observes `ln_cfu0` directly at `t = -t_pre`. The pool there is
  a mixture of clean and congressed cells, but no growth has accrued yet in
  the model's time frame, so the mixture term is `log(w_clean + w_cong) = 0`
  and the presplit likelihood stays `ln_cfu0_g ~ presplit`. Presplit is a
  direct estimate of `ln_cfu0_g` for every genotype. Congression still enters
  it in two ways:
  - *Counting.* For presplit and growth to estimate the same `ln_cfu0_g`,
    both must count a co-transformed cell with its abundance split among its
    variants (simulator: step 1; fit: same definition of `ln_cfu0`).
  - *Overnight growth.* Differential growth of clean and congressed cells
    during the overnight outgrowth before `t = -t_pre` is absorbed into the
    free per-genotype `ln_cfu0_g`. It also shifts the class weights away from
    the design values; that shift is not modeled (see assumptions).
- **binding** stays clean (below).

Open: `theta_growth_noise` and `theta_rescale` currently act on the corrected
theta. In the mixture they must act on each class's theta before growth is
assembled. Decide in step 3 whether the noise term is drawn once per genotype
and shared across classes (the natural reading) or per class.

### Split `--spiked` into two concepts

`--spiked` currently sets both congression purity and the high `ln_cfu0` prior
class. Keep one `ln_cfu0` latent per genotype for total abundance, with the
existing spiked/bulk prior on it; carry purity in `f_g`. Genotypes can't simply
be removed from `--spiked`, because their abundance really is far higher than
the bulk's.

### Fit design (step 3)

Agreed 2026-09-23 (user). Today's fit replaces a bulk genotype's theta by
`E[max]` over co-residents and grows one trajectory with the genotype's own
dk, i.e. it averages theta over the genotype's cells where the reads average
`exp(ln_cfu)`. The replacement:

```
ln_cfu_g(t) = ln_cfu0_g + logsumexp_c [ log w_{g,c} + G_{g,c}(t) ] + delta_sample
```

- **Classes c.** *Clean*: the genotype's own theta, activity and dk.
  *Congressed*: the genotype plus co-residents; cell theta and activity from
  the theta rule (max), dk from the dk rule (dilution), matching the
  simulator (`simulate/cell_rules.py`).
- **Weights.** `w_cong = f_g * (1 - exp(-lambda))`, `w_clean = 1 - w_cong`
  (zero-truncated form since 3.4: `w_cong = f_g (1 - P(M = 1))`);
  time-independent. `f_g` is `bulk_fraction` from the library table.
- **G** is each class's whole trajectory through the existing pipeline
  (`theta_rescale` -> `calculate_growth` -> `growth_transition`) over pre and
  selection growth. Classes go on a new leading axis; the growth and
  transition components are elementwise and should broadcast (verify).
- **Unchanged:** `ln_cfu0`, `delta_sample`, growth noise, presplit,
  base_growth, binding.
- **`congression_mask` is split.** Purity -> `f_g`; the `ln_cfu0` prior
  class -> `in_spiked_origin` (spikes really are far more abundant). Legacy
  `spiked_genotypes` -> `f_g = 0` for spiked, 1 otherwise.
- **Co-resident background** needs the full-population theta (exists) and
  dk (new), mini-batch safe, with external references for prediction.
  Pooled vs per-origin background is decided by 3.0 (each sub-library is
  transformed separately, so a single's co-residents come from the single
  library and a double's from the double library).
- **`theta_growth_noise`** acts on the genotype's own theta before classes
  are built; co-residents use noiseless population values.
- **lambda** stays a latent with the measured prior; it now enters the
  weights as well as the co-resident draws.
- **Estimator (from 3.0; user, 2026-09-23): fixed Monte Carlo quadrature.**
  Before fitting, draw once, per genotype, K co-resident sets from the pooled
  bulk composition (`pool_fraction * bulk_fraction`), stratified by count:
  K_1 sets with N = 1, K_2 with N = 2, K_3 with N = 3 (e.g. 12 / 3 / 1; N >= 4
  is about 0.3% of congressed cells at lambda = 0.357 and is dropped). The sets
  are integer indices stored with the data and never redrawn. Each step builds
  1 + K cells per genotype from the *current* parameters (focal and
  co-resident theta/activity/dk looked up by index), weights set k by
  `w_cong * P(N = n_k | N >= 1; lambda) / K_{n_k}` so lambda stays
  differentiable, and evaluates the likelihood once. Per-genotype sets
  (rather than one shared set) keep the quadrature errors independent across
  genotypes, so they do not push shared parameters (k, m, lambda) in a common
  direction. Redrawing every step is rejected: the log of a fresh K-sample
  mean is biased low and adds gradient noise. K is configurable (default 16).
- **Components.** Keep `single` and `empirical`; retire `logit_norm` (a
  parametric theta background with no dk counterpart).

### Binding data stays uncongressed

`congression_mask` already acts only on the growth theta path
(`generative/model.py`), so binding measurements on monoclonal stocks stay
clean even when a genotype's growth data is mixed. Keep it that way.

## Real library design

Tune simulations to the design used for the dataset:

```yaml
transform_sizes:
  single-1: 100_000
  single-2: 100_000
  double-1-2: 300_000
  spiked: 1_000

library_mixture:
  single-1: 100
  single-2: 100
  double-1-2: 1_000
  spiked: 1
```

## Experiments (bench)

Underway (user, 2026-09-23); first result 2026-09-24 (below). They decide the
theta rule in step 4.

- **Double transformation, tight + dead binder:** relative to the tight binder
  alone, a max rule predicts no shift, homodimer mixing about 2x weaker
  apparent K, heterodimer mixing about 4x, dominant negative more. The sign
  (shift or not) is expected to be easy to resolve; magnitude is hard. Control:
  the same variant on two distinguishable plasmids should reproduce the single
  transformant under every rule.
- **Native mass spectrometry** of co-expressed variants (collaborator) to
  measure heterodimer formation.

Record results here when they arrive.

**Result, 2026-09-24 (user; student's experiment).** Growth in the pheS
selection (4CP) at three IPTG concentrations; high IPTG induces the toxin and
slows growth. Transformations at high DNA concentration of wt, D88A (super
repressor, tight binder), K84L (barely binds), and the pairs wt:D88A and
wt:K84L. At the intermediate IPTG: D88A fastest, wt intermediate, K84L no
growth. wt:D88A grew faster than wt, slightly slower than D88A; wt:K84L grew
faster than K84L, slightly slower than wt. So each pair grows between its
parents, close to the tighter binder.

- User's reading: not consistent with weaker-binder dominance; does not
  separate stronger-binder dominance (max) from a partition-function rule.
  Encouraging that the sign of the correction is right.
- Limits (user): no direct lambda, so the mix of double transformants and
  single transformants of each parent in the paired cultures is unknown
  (high DNA concentration should give many doubles); pheS arm only.
- Caveat (Claude): single transformants of both parents are present in a
  paired culture, and the faster parent's singles take over under
  exponential growth. So "between the parents, close to the faster" is
  partly expected under any rule, including weak-binder dominance, if singles
  make up much of the culture. What separates the rules is how close to the
  faster parent the pair is given the fraction of doubles. It could be
  firmed up by estimating that fraction (genotyping single colonies from the
  paired transformation), by the shape of the growth curve (a mixture of
  subpopulations speeds up over time as the fast one takes over; a clonal
  double grows at a constant rate), or by growing verified clonal doubles,
  though those segregate over generations
  (`planning/plasmid-segregation-low-copy.md`).
- Native MS: not available.

## Steps

Keep this list current. Work the steps in order and keep this conversation's
scope to them; anything else goes into its own `planning/` file or task.
Each step from 1 on ends with a simulation and a fit that use the same rules,
checked with `tfs-summarize-calibration`.

-2. **Bring in the mini-batch scrambling fix.**
   - [x] Port the MAP/autoguide genotype-aliasing fix and its tests
     (`inference/batch_safety.py`, `test_batch_safety.py`) from
     `guide-selection` as a separate commit, code only. Every fit used to
     check the steps below depends on it.
-1. **Update this plan** with the review of 2026-09-23.
   - [x] Starting-point description, simulator/design mismatch, dk and
     mixture decisions, bench pointer, step list. 2026-09-23.
0. **Groundwork.**
   - [x] Per-genotype `f_g` from the library definition:
     `genetics.library_composition_table` → `bulk_fraction` in
     `{out_prefix}_library.csv`, via `tfs-configure-model --library_config`
     (commit 77befcb).
   - [x] Bring in `tfs-summarize-calibration`
     (`analysis/calibration_grid.py`, `analysis/scripts/summarize_calibration_cli.py`)
     from `guide-selection` (commit e925b90), code only, no grids or results.
     2026-09-23.
1. **Simulator: share a cell's plasmid copies.**
   - [x] `_sim_sequencing` and `_calc_genotype_cfu0` split a cell's abundance
     among the variants it carries, so simulated composition matches
     `bulk_fraction` and no longer depends on lambda. 2026-09-23
     (`_plasmid_shares`; a test checks simulated abundance against the
     design `pool_fraction` at lambda = 0, 0.357 and 1.5).
2. **Simulator restructure.**
   - [x] Build each cell's theta and dk from its plasmids with pluggable rules,
     then assemble growth once
     (`k_cell = k_condition + dk_cell + m * A * theta_cell`). Replaces
     `multi_plasmid_combine_fcn` on whole `k*t`. 2026-09-23:
     `simulate/cell_rules.py`, `selection_experiment._cell_kt`; config keys
     `congression_theta_rule`, `congression_dk_rule`.
   - [x] First rules: max theta (the fit's current theta rule) and dilution dk.
     TF activity follows the theta rule: it comes from the plasmid that sets
     theta (user, 2026-09-23). Revisit in step 4, where a partition-function
     theta rule would put activity into the plasmid weights.
3. **Fit: observable-level mixture.** Design agreed 2026-09-23 (see "Fit
   design (step 3)"). 3.0 runs alongside 3.1/3.2; only 3.3's estimator
   waits on it. 3.1, 3.2 and 3.4 are behavior-preserving commits; 3.3 is the
   breaking one.
   - [x] **3.0 Numerical study** (no fitting), 2026-09-23. Ground truth from
     `library_prediction` on `examples/simulate/simulate_config.yaml` (843
     genotypes, thermo theta prior, memory growth transition, 48 growth
     conditions, t_sel 95 to 200 min) with the real design's
     `library_mixture` and lambda = 0.357, and the step 2 cell rules. "Exact"
     `E_S exp(G)`: N = 1 enumerated over the co-resident library, N >= 2 by
     Monte Carlo. dk_geno deviations scaled by s (sim prior at s = 1: SD
     0.068/min, 5th to 95th percentile -0.13 to 0.014/min). Error = approx -
     exact `ln_cfu`, RMS over genotype x condition cells within 7 ln units of
     wt (detectable):

     | estimator | s = 0 | s = 0.015 | s = 0.03 | s = 0.1 | s = 0.3 | s = 1 |
     |---|---|---|---|---|---|---|
     | no congression | 0.041 | 0.053 | 0.097 | 0.287 | 0.758 | 1.507 |
     | today's fit rule | 0.019 | 0.041 | 0.093 | 0.293 | 0.770 | 1.524 |
     | `exp(E_S[G])` | 0.006 | 0.006 | 0.008 | 0.036 | 0.206 | 1.086 |
     | `E[G] + Var[G]/2` | 0.000 | 0.000 | 0.001 | 0.025 | 0.509 | 12.6 |
     | N = 1 only (exact) | 0.003 | 0.004 | 0.010 | 0.030 | 0.080 | 0.127 |
     | MC, K = 4 | 0.016 | 0.016 | 0.032 | 0.072 | 0.181 | 0.379 |
     | MC, K = 16 | 0.008 | 0.011 | 0.010 | 0.028 | 0.072 | 0.187 |
     | MC, K = 32 | 0.005 | 0.006 | 0.010 | 0.019 | 0.064 | 0.142 |
     | pooled background | 0.002 | 0.002 | 0.002 | 0.004 | 0.008 | 0.024 |

     s = 0.015 is the measured regime for the current library (dk SD about
     0.001/min; see "Biology and assumptions").

     Findings:
     - The congression effect is large once dk varies (0.29 to 1.5 ln units
       RMS), and **today's fit rule is no better than no correction** at any
       s > 0: averaging theta cannot represent a mixture of growth rates.
     - Averaging the rate first is adequate only for a narrow dk spread
       (s <= 0.1) and worsens with time; the second-order cumulant fails
       (heavy dk tails).
     - Monte Carlo with K fixed co-resident sets (common random numbers,
       shared by all genotypes) has near-zero mean error at every s, and its
       RMS falls roughly as 1/sqrt(K). This is the only estimator that holds
       across the dk range.
     - **Pooled background is fine** (<= 0.024): no per-origin backgrounds
       needed. Caveat: in the sim prior, singles and doubles have similar
       theta/dk distributions; real doubles may be more often broken.
     - N >= 2 matters little but not nothing (N = 1 only: 0.13 at s = 1);
       Monte Carlo covers it at no extra cost.
     - **In the measured regime (s ~ 0.015) the whole effect is modest**:
       0.053 RMS, 0.105 p95 (comparable to per-point `ln_cfu` noise, but
       systematic), and today's rule removes only about a quarter of it. Every
       estimator except small-K MC is accurate there. We still build MC
       (decision 2026-09-23): tight dk is not guaranteed for other libraries,
       and K is a knob.
     Study: `planning/studies/congression-estimator/`.
   - [x] **3.1 Plumbing.** Done 2026-09-23: `GrowthData.bulk_fraction`,
     `ModelOrchestrator._build_bulk_fraction`; `congression_mask` still
     drives today's correction until 3.3. `f_g` (from `bulk_fraction`) and the `ln_cfu0`
     prior class (from `in_spiked_origin`) as separate `GrowthData` fields,
     built by the orchestrator from the library table or the legacy
     `spiked_genotypes` list (spiked -> `f_g = 0`, others 1). No behavior
     change: fits identical.
   - [x] **3.2 Full-population dk_geno and activity access** (API only).
     Done 2026-09-23. Activity was added to the step: the cell's activity
     follows the theta rule, so co-residents' activity is needed too. Every
     `dk_geno` and `activity` component takes `return_population=False`;
     when True it also returns the library-ordered per-genotype values from
     the same draw (`generative/components/_population.py::per_genotype`), or
     `None` when the latents arrived as a batch-sized substitution.
     `GrowthData.external_dk_population` / `external_activity_population`
     are the fallback. Nothing requests the population yet, so fits are
     identical. Building the external arrays in `analysis/prediction.py`
     moves to 3.4.
     Found for 3.3: `theta/categorical_geno` slices its latent to the batch
     at sample time, so `model.py`'s full-population `calc_theta` (arange
     `batch_idx`/`geno_theta_idx`) returns batch-ordered theta at full batch
     and indexes past the batch under mini-batching. Fixed in 3.3a.
   - 3.3 lands as three commits: 3.3a library-ordered population theta
     (bug fix), 3.3b co-resident sets drawn into `GrowthData` (no behavior
     change), 3.3c the mixture itself (breaking; includes the
     `analysis/prediction.py` change, since prediction runs the model and
     must supply full-library populations and the fit's co-resident sets).
   - [x] **3.3a Library-ordered population theta.** Done 2026-09-23:
     `categorical_geno` keeps theta library-sized and indexes through
     `batch_idx[geno_theta_idx]`; `test_population_theta.py` checks every
     theta component.
   - [x] **3.3b Co-resident sets.** Done 2026-09-23:
     `ModelOrchestrator._draw_coresident_sets` / `_coresident_pool`;
     `GrowthData.coresident_idx` (num_genotype, K, N_max; -1 = empty) and
     `coresident_n`; settings `congression_sets` (default [12, 3, 1]) and
     `congression_seed` (default 0), recorded in the config. Pool: bulk share
     (`pool_fraction * bulk_fraction`) of genotypes with growth data,
     `__unknown__` excluded; legacy path uniform over non-spiked (user,
     2026-09-23). An empty pool leaves every slot -1; 3.3c must refuse a
     congression model with `bulk_fraction > 0` and an empty pool.
   - [x] **3.3c Mixture.** Done 2026-09-23. Code plan and decisions agreed 2026-09-23 (user):
     - *Transformation interface.* `define_model`/`guide` sample lambda only
       (no `anchors`). New `cell_classes(focal, population, params, data)`
       returns `(theta, activity, dk, log_w)` with a leading class axis.
       `NEEDS_POPULATION` replaces `NEEDS_FULL_POPULATION_THETA`. `single`:
       one class, `log_w = 0`, identical to today. The mixture component is
       named **`mixture`**; `transformation: empirical` is refused with a
       message explaining the change, so old configs do not silently switch
       physics. `logit_norm` is removed. `_congression.update_thetas` stays as
       a plain function for Stage 1.5 until 3.4.
     - *`model.py`.* theta -> noise on the genotype's own theta -> population
       theta/dk/activity (external if supplied, else the full-population
       `calc_theta` and `return_population=True`) -> gather
       `coresident_idx[batch_idx]` -> classes (max theta and its activity per
       titrant point; dilution dk; -1 slots masked) on a class axis ahead of
       `(rep, time, cp, cs, tn, tc, batch)` -> per class `theta_rescale`,
       `calculate_growth`, `growth_transition` (each called once; they
       broadcast) -> `ln_cfu_pred = ln_cfu0 + logsumexp(log_w + G, 0) +
       delta_sample`. `theta_growth_pred` stays the genotype's own
       uncorrected theta. The guide skips class construction.
     - *Weights* (superseded in 3.4 by the zero-truncated form). `w_cong = f_g (1 - exp(-lambda))`; `log w_clean =
       log1p(-w_cong)`; `log w_k = log f_g + log(1 - exp(-lambda)) + log P(N
       = n_k | N in strata; lambda) - log K_{n_k}` (strata renormalized over
       the counts that have sets), so the lambda gradient stays finite for
       `f_g = 0`.
     - *Refusals.* A `mixture` model with some `bulk_fraction > 0` and an
       empty co-resident pool is refused at build time. The
       `categorical_geno`/`empirical` incompatibility check is deleted (fixed
       in 3.3a).
     - *`congression_mask` is removed* from `GrowthData`, `batch.py`, the
       orchestrator and tests.
     - *Prediction.* `predict()` passes full-library
       `external_{theta,dk,activity}_population` (dk/activity: posterior
       medians of their deterministic sites; theta: the theta component at
       posterior-median parameters, evaluated on the prediction's
       concentration grid) and the subset's rows of
       the original `coresident_idx`. A raw MAP checkpoint (no stored
       deterministic sites) with a `mixture` model raises, pointing to
       `tfs-sample-posterior`.
     - *Cost.* Growth tensors grow by (1 + K); default K stays 16
       (`[12, 3, 1]`). Main runs mini-batch. A memory-flat `lax.scan` over
       classes would need growth_transition components split into sample and
       compute; deferred unless memory bites.
     - *Tests.* `single` regression; mixture vs an independent numpy
       implementation; lambda -> 0 and `f_g = 0` reduce to `single`; finite
       gradients; registry-wide batch shape/order checks; subset prediction
       equals full prediction rows; refusals; smoke tests.
   - [x] **3.4 Other consumers.** Done 2026-09-24, three commits:
     - *Weights (a 3.3c fix found while checking the lambda echo).* Since
       step 1 the simulator shares a cell's abundance among its plasmids, so
       a genotype's abundance by co-resident count n is the zero-truncated
       `P(M = n + 1)`, not Poisson(n; lambda) (which assumed each plasmid got
       the whole cell). At lambda = 0.357 a bulk genotype is 83% clean, not
       70%; the old weights would have recovered lambda ~ 0.18 against a
       simulated 0.357. `mixture._log_class_weights` now uses
       `w_cong = f_g (1 - P(M = 1))`, `P(M = 1) = lambda e^-lambda /
       (1 - e^-lambda)`, and strata by `P(M = n + 1)`; a test checks them
       against simulated cell shares. Zero-truncation is physical: the zero
       class (no plasmid, marker dead, ...) never grows under selection
       (user, 2026-09-24). The 3.0 study used the old weights for its
       "exact" reference as well as the estimators, so its comparisons
       stand.
     - *Lambda echo.* With the weights fixed, the fit's `lam` is the
       simulator's `transformation_poisson_lambda`; docstrings and
       `docs/source/model-inputs.rst` define it.
     - *Stage 1.5 retired* (user, 2026-09-24): `--congression_lambda` removed
       from `tfs-fit-genotypes`/`tfs-build-empirical`; `genotype_fit/
       congression.py` and `transformation/_congression.py` deleted. The
       whole empirical pipeline is to be reviewed before it is used again;
       the replacement is filed as `planning/empirical-mixture-refit.md`.
     - *`tfs-summarize-calibration`* strata: `origin` (spiked/bulk) became
       `purity` (`spike`/`mixed`/`bulk`, from the sub-libraries encoding each
       genotype, i.e. `bulk_fraction` 0, between, or 1).
   - [ ] **3.5 Calibration.** Step 2 simulator vs the new fit, with
     `tfs-summarize-calibration`; watch bulk genotypes without binding data.
     Grid set up 2026-09-24 (user: cluster, both extra arms, `mixed` left
     unstratified): `planning/studies/congression-calibration/`, 54 runs =
     simulated lambda {0, 0.357, 1.0} x dk spread {wide, tight} x 3 seeds x
     fit {single, mixture with matched lambda prior, mixture with the
     measured 0.357 prior}, paired against `single`. Both sides use the max
     theta rule, so this checks the machinery and also serves as the
     baseline for step 4. Whether to run it before the bench result on the
     dominance rule is still open (2026-09-24).
     Ran 2026-09-24 (all 54 converged); not yet usable. The k/dk_geno slide
     is unanchored in this pipeline (no wt pin, no `base_growth`, prefit
     `k_scale` at its ceiling); congression sits below the design's noise
     floor (`single`'s errors do not change with lambda); and lambda is
     pulled up whatever the truth. Diagnosis: the mixture uses its
     congressed classes to fit the read-count detection floor (pseudocount
     1 puts 0-read rows +1.3 ln above the truth), which real data share.
     Next: `grid_masked.yaml` (rows with < 5 reads dropped) to test the
     diagnosis; then a censored likelihood for floor observations (user's
     preferred fix), then the full grid re-run with a `base_growth` anchor.
     Details: `planning/studies/congression-calibration/README.md`.
4. **Theta rule.** Homodimer vs heterodimer soft max, chosen from the bench
   results, in both simulator and fit.
5. **dk rule.** Soft-min family with an alpha sensitivity check, in both
   simulator and fit.
