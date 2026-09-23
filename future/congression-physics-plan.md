---
title: Match simulator and fit on the physics of congression
status: active
filed: 2026-09-13
area: tfmodel
revisit_when: >-
  Active now. Update the step list as steps finish.
related:
  - src/tfscreen/simulate/selection_experiment.py
  - src/tfscreen/tfmodel/generative/components/transformation/_congression.py
  - src/tfscreen/tfmodel/generative/model.py
  - src/tfscreen/tfmodel/model_orchestrator.py
  - src/tfscreen/tfmodel/analysis/prediction.py
  - src/tfscreen/tfmodel/genotype_fit/congression.py
  - src/tfscreen/simulate/transformation_lam_output.py
  - src/tfscreen/genetics/library_design.py
  - future/estimate-dk-alpha-by-varying-lambda.md
  - future/plasmid-segregation-low-copy.md
  - future/combination-specific-expression-effects.md
  - future/estimate-spike-fraction-from-growth-curvature.md
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
Poisson(lambda), as the fit assumes), and the fit's unweighted background
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

**Simulator.** `_sim_growth` (`simulate/selection_experiment.py`) combines each
co-resident plasmid's whole `k*t` with a per-marker-library
`multi_plasmid_combine_fcn` (`gmean`/`mean`/`min`/`max`/`sum`; default `mean`;
the example configs use `min` for kanR and `max` for pheS).

**Known simulator/design mismatch.** `_sim_sequencing` and
`_calc_genotype_cfu0` credit every plasmid in a cell with that cell's whole
abundance, so a co-transformed cell counts once per plasmid.
`expected_library_composition` assumes the opposite: a cell's plasmid copies
are shared among the variants it carries, so `bulk_fraction` does not depend
on lambda. Until the simulator shares copies (step 1), simulated composition
and the design `bulk_fraction` disagree.

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
  expression are out of scope (`future/combination-specific-expression-effects.md`).
- **Plasmids are high copy and assumed to persist without segregating** over
  the experiment (roughly 3 to 7 doublings). The next design moves to a
  low-copy plasmid, where segregation matters
  (`future/plasmid-segregation-low-copy.md`).
- **The dataset has a single lambda** (0.3572) and is headed for publication.
  Parameters that only a lambda series could identify are set from biology
  and checked by sensitivity, not fit.
- **Spiked and library copies are indistinguishable** in this dataset. Future
  runs will give spikes distinct codons.
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
sensitivity check, not fit (`future/estimate-dk-alpha-by-varying-lambda.md`).
Under dilution the average effect is a uniform shrinkage of each genotype's
dk toward the population mean, by `E[1/(1+N)] = (1 - exp(-lambda))/lambda` =
0.841 at lambda = 0.357, which the hierarchical dk_geno prior mostly absorbs.

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
has no co-residents with probability `exp(-lambda)`:

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

Underway (user, 2026-09-23); results expected 2026-09-24. They decide the
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

## Steps

Keep this list current. Work the steps in order and keep this conversation's
scope to them; anything else goes into its own `future/` file or task.
Each step from 1 on ends with a simulation and a fit that use the same rules,
checked with `tfs-summarize-calibration`.

-2. **Bring in the mini-batch scrambling fix.**
   - [ ] Port the MAP/autoguide genotype-aliasing fix and its tests
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
   - [ ] Bring in `tfs-summarize-calibration`
     (`analysis/calibration_grid.py`, `analysis/scripts/summarize_calibration_cli.py`)
     from `guide-selection`, code only, no grids or results.
1. **Simulator: share a cell's plasmid copies.**
   - [ ] `_sim_sequencing` and `_calc_genotype_cfu0` split a cell's abundance
     among the variants it carries, so simulated composition matches
     `bulk_fraction` and no longer depends on lambda.
2. **Simulator restructure.**
   - [ ] Build each cell's theta and dk from its plasmids with pluggable rules,
     then assemble growth once
     (`k_cell = k_condition + dk_cell + m * A * theta_cell`). Replaces
     `multi_plasmid_combine_fcn` on whole `k*t`.
   - [ ] First rules: max theta (the fit's current theta rule) and dilution dk.
3. **Fit: observable-level mixture.** Breaking change.
   - [ ] Clean/congressed log-sum-exp mixture in the growth observation, over
     each class's whole pre + selection trajectory through
     `growth_transition`.
   - [ ] Consume `bulk_fraction` as `f_g`; retire `congression_mask`. Keep the
     spiked/bulk `ln_cfu0` prior class as its own concept.
   - [ ] Dilution dk with a full-population dk background (mini-batch safe;
     external reference for prediction).
   - [ ] Place `theta_growth_noise` / `theta_rescale` per class.
   - [ ] Check whether `exp(E[k] t)` is an adequate approximation.
   - [ ] Update the other users of the congression operator:
     `analysis/prediction.py`, `genotype_fit/congression.py` (Stage 1.5 of
     `tfs-fit-genotypes` / `tfs-build-empirical`),
     `simulate/transformation_lam_output.py`.
   - [ ] Calibrate against step 2's simulations (current max theta as a
     placeholder).
4. **Theta rule.** Homodimer vs heterodimer soft max, chosen from the bench
   results, in both simulator and fit.
5. **dk rule.** Soft-min family with an alpha sensitivity check, in both
   simulator and fit.
