# Congression calibration study

**Question.** Fit to data from the step 2 simulator, is the observable-level
congression mixture (`transformation: mixture`) calibrated, and does it beat
`single` where congression matters? In particular:

- Does the fit recover lambda?
- Are theta and the per-genotype parameters calibrated for bulk genotypes
  *without* binding data, whose theta is informed only by their growth?
- What does fitting the mixture cost when there is no congression
  (lambda = 0), and how much does a misspecified lambda prior hurt?

**Decision it feeds.** Step 3.5 of
[`planning/congression-physics-plan.md`](../../congression-physics-plan.md):
whether the mixture is ready to use on real data. The simulator and the fit
use the same cell rules (max theta, dilution dk), so this checks the fit's
machinery given matched physics, not whether the rules are biologically
right. The mixture-vs-single effect sizes are specific to the max rule;
re-run the grid when step 4 changes the rule.

## Design

Base config: [`simulate_config.yaml`](simulate_config.yaml), from
`examples/simulate-and-analyze/` (483 genotypes, hill_mut theta, instant
growth transitions, kanR and pheS selections) with the real design's
`library_mixture` and 20 in-library binding anchors (`library_binding`).
Purity: 474 bulk genotypes, 3 pure spikes (D88A, H74A/K84L,
M42I/H74A/K84L), and 6 mixed (wt and the spiked singles, `bulk_fraction`
0.99 to 1.0; M42I/H74A and M42I/K84L, 0.74).

[`grid.yaml`](grid.yaml): 3 x 2 x 3 x 3 = 54 runs.

| axis | levels |
|---|---|
| simulated lambda | 0 (null), 0.357 (measured), 1.0 (stress) |
| dk_geno spread (`dk_geno_hyper_loc`) | -3.5 (wide, SD 0.018/min), -7.0 (tight, SD 0.0006/min, the measured regime) |
| seed (simulation and fit) | 1, 2, 3 |
| fit (`fit`, `lam_prior`) | `single`/`off`; `mixture`/`matched` (prior at the simulated lambda, 0.01 when it is 0, SD 0.05); `mixture`/`measured` (prior 0.357 +/- 0.05 whatever the truth) |

At simulated lambda 0.357 the matched and measured fits are the same fit;
their results should agree up to GPU nondeterminism (a reproducibility
check).

Each run ([`run.srun`](run.srun)): `tfs-simulate` -> `tfs-configure-model`
(the simulate-and-analyze components; `--transformation_model` and
`--transformation_lambda` from the fit arm) -> `tfs-prefit-calibration` ->
`tfs-fit-model` (SVI) -> `tfs-sample-posterior` -> `tfs-extract-params` ->
`tfs-predict-theta` -> `tfs-predict-growth` -> `tfs-summarize-fit`.

## How to run

On the cluster, from a scratch directory, with this repository checked out.
`tfs-setup-sim-grid` copies `hill_params.csv` into the grid's `inputs/`
directory and each run refers to it as `../inputs/hill_params.csv`, so the grid
directory can be moved as a unit after setup:

```bash
tfs-setup-sim-grid /path/to/tfscreen/planning/studies/congression-calibration/grid.yaml \
    --out_prefix congression_calibration
for d in congression_calibration/run_*/; do (cd "$d" && sbatch run.srun); done
```

When the runs finish:

```bash
tfs-summarize-calibration congression_calibration --out_prefix calib/congression \
    --baseline fit=single lam_prior=off \
    --facet_by transformation_poisson_lambda dk_geno_hyper_loc
```

Arms average over `seed` (the default `--replicate_keys`).

## What to look at

- `lam`: posterior against the simulated value, by lambda and prior.
- theta (`theta_test`) and per-genotype parameters by `purity` x
  `has_binding`: coverage, calibration bias, width, RMSE; paired deltas
  against `single`.
- The lambda = 0 arm: the mixture's cost when nothing is congressed.
- The `measured` prior at lambda 0 and 1.0: robustness to a wrong prior.

## Commit

Grid set up and run on `6e4735e` (2026-09-24).

## Results (grid.yaml, 2026-09-24)

All 54 runs finished and SVI converged. `tfs-summarize-calibration` outputs
are in `calib/` (not committed). The grid does not yet answer its question,
for three reasons:

1. **The k/dk_geno slide is unanchored.** In every run the `growth_k` error
   mirrors the `dk_geno` error, and the spread of the `dk_geno` errors is
   small (0.001 to 0.01/min): each genotype's dk is recovered up to one
   global constant. Nothing pins it here: `hierarchical_geno` does not pin
   wt, there is no `base_growth` data, and the prefit's Hessian-derived
   `k_scale` hit its 0.1 ceiling. `single` fits drift -0.02 to 0 in k,
   mixture fits +0.02 to +0.07 (true baselines 0.01 to 0.03). `dk_geno` and
   `growth_k` calibration are therefore meaningless in this grid.
2. **Congression is below the noise floor of this design.** `single`'s
   errors do not change with the simulated lambda (growth RMSE 0.469 at
   lambda 0 vs 0.471 at 1.0, tight dk; bulk theta without binding 0.180 vs
   0.181), although 42% of a bulk genotype's cells are congressed at
   lambda 1.0.
3. **Lambda is pulled up whatever the truth.** With the 0.357 +/- 0.05
   prior, posterior medians were 0.46 to 0.66 at a true lambda of 0,
   0.46 to 0.60 at 0.357, and 0.44 to 0.63 at 1.0. With the matched prior at
   1.0: 1.00 to 1.28. Seed 3 is highest in every arm.

Also: the mixture's theta RMSE is slightly worse than `single`'s in every
arm (0.20 to 0.22 vs 0.18, bulk genotypes), including lambda 0 with the
matched prior, where the fitted lambda is ~0.004. Bulk theta 95% coverage is
~0.65 in every arm, `single` included (a baseline overconfidence, separate
from congression). Binding genotypes are recovered well everywhere (theta
RMSE 0.015).

## First diagnosis: the detection floor (2026-09-24; superseded, see below)

Scripts: [`diagnosis/`](diagnosis/) (run from that directory against the
pulled grid). Runs 0001 (`single`) and 0002 (mixture, matched prior, true
lambda 0, fitted lambda 0.004) were fit to the same data.

- The mixture's forward model is right: at `single`'s posterior-median
  latents with lambda = 1e-8 it reproduces `single`'s `growth_pred` to 1e-6.
- The mixture fit leans on its congressed classes anyway: switching them off
  at its own latents costs 36,700 in growth log-likelihood. A class weighted
  0.2% can only matter that much by outgrowing the clean class by ~6 ln
  units. The fit gets there through dk dilution: slow genotypes (true
  dk_geno -0.02 to -0.05) are fit at ~-0.12, and their congressed cells,
  with dk averaged against a co-resident, carry the late timepoints.
- What those classes fit is the read-count floor. `ln_cfu` uses a
  pseudocount of 1 (`process_raw/counts_to_lncfu.py`, the same path as real
  data), so a dying genotype's late timepoints stall above the truth:

  | reads | mean error vs truth (ln) | reported SD |
  |---|---|---|
  | 0 | +1.26 | 1.00 |
  | 1-2 | +0.62 | 0.71 |
  | 3-5 | +0.22 | 0.45 |
  | 6-20 | +0.07 | 0.29 |

  A single exponential cannot bend upward; a dying clean class plus a slower
  congressed class can. Genotypes relying on congressed classes have zero
  reads in 29% of their rows, against 6% for the rest.
- This accounts for the grid results: more congressed weight fits the floor
  better (lambda pulled up everywhere), the dk shifts move the fits along
  the unanchored slide, and the mixture spends flexibility on the artifact.
  `single` is biased by the same points, less visibly.
- Real data share the pseudocount path, so a mixture fit to real data would
  do the same.

The 0.4 ln-unit error on well-measured rows is the simulator's per-tube
growth noise (`tube_noise_sigma` x t; per-sample SD 0.374, within-sample
0.054), which `growth_noise` is meant to absorb. Not a data problem.

## Masked re-run (grid_masked.yaml, 2026-09-24): floor not the driver

`grid_masked.yaml` dropped growth rows with fewer than 5 reads (`min_counts`
in `run.srun`) and refit grid runs 0001-0003 and 0037-0039. The cluster
reproduced the original simulations exactly. Comparison
([`diagnosis/compare_masked.py`](diagnosis/compare_masked.py)), true lambda 0:

| | full data | masked |
|---|---|---|
| lambda, measured prior (0.357 +/- 0.05) | 0.55 | 0.53 |
| k offset, mixture (matched prior) | +0.062 | +0.065 |
| bulk theta RMSE, mixture vs `single` | 0.267 vs 0.219 | 0.270 vs 0.220 |

Lambda 1.0 behaved the same. The mixture still leans on its congressed
classes (switching them off costs 21,100 in growth log-likelihood, against
36,700 unmasked). The floor bias is real and censoring is still worth doing,
but it does not drive the mixture's behavior.

## Second diagnosis: the binding weight (2026-09-24; superseded, see below)

Scripts: [`diagnosis/diag_basin.py`](diagnosis/diag_basin.py) and
[`diagnosis/diag_sites.py`](diagnosis/diag_sites.py) (runs 0001/0002). The
joint log density of the two lambda-0 fits differs almost entirely in the
binding likelihood:

| term | mixture fit - `single` fit |
|---|---|
| binding likelihood, as weighted in the fit | +1,044,800 |
| growth likelihood | -37,600 |
| everything else | about -1,100 |

`ModelOrchestrator` scales the binding likelihood by `binding_weight`,
which defaults to growth rows / binding rows (about 54,300 / 216 = 251
here), and the simulated binding data have `theta_std` 0.001. Together they
dominate the objective; `single` cannot fit them (residuals ~15 sigma,
theta ~0.015). The mixture gives the fit a way out: pushing a genotype's own
dk_geno far down removes its clean cells from its trajectory, its congressed
classes (dk averaged with a co-resident) carry its growth, and its theta is
free to follow binding. The 20 in-library binding genotypes gain 1,430; the
4 spiked ones gain 2,730, mostly H74A (2,020), through hill_mut mutation
effects shared with bulk genotypes carrying H74 changes. This accounts for
lambda being pulled up, the slide, and the worse bulk theta and growth.
Not a mixture bug: the weight also bends `single` fits, less visibly.

Background (user, 2026-09-24): the weight was added when, with ~200,000
growth genotypes and ~10 binding genotypes, fits ignoring the binding data
won. That points to a conflict between the modalities over shared
parameters (m, k, shared mutation effects) rather than a volume problem;
the binding assay has since moved closer to cellular conditions. The
binding SDs used on real data are the SE of a Hill curve fitted to the
anisotropy points, which counts 9 correlated values as independent and
leaves no room for differences between assay and cell.

## Binding weight 1 (grid_bw1.yaml, 2026-09-24): weight not the driver

The same 6 simulations refit with `--binding_weight 1`
([`diagnosis/compare_masked.py`](diagnosis/compare_masked.py), which now
takes the follow-up grids as arguments). True lambda 0:

| | original | masked | weight 1 |
|---|---|---|---|
| lambda, measured prior (0.357 +/- 0.05) | 0.55 | 0.53 | 0.47 |
| k offset, mixture (matched prior) | +0.062 | +0.065 | +0.080 |
| bulk theta RMSE, mixture | 0.267 | 0.270 | 0.247 |
| bulk theta RMSE, `single` | 0.219 | 0.220 | 0.227 |

Theta 95% coverage improved in every arm at weight 1 (0.71 to 0.75, from
0.61 to 0.70), so the weight matters for calibration, but the mixture's
behavior survived. At weight 1 the mixture fit is worse than `single` on
both binding and growth, so no trade-off explains it.

## Third diagnosis: the unanchored slide (2026-09-24; superseded, see below)

- **Optimization, not the objective.** Under the mixture model's own joint
  density, `single`'s solution at lambda 0.004 scores ~18,000 log units
  better than where the mixture's SVI stopped
  ([`diagnosis/diag_basin.py`](diagnosis/diag_basin.py), bw1 grid).
- **The drift happens in SVI, after the MAP warm-up.** After the warm-up all
  fits have sensible `growth_k` (truth 0.011 to 0.029); the matched-prior
  mixture then climbs to 0.089 to 0.112, with the between-condition
  differences collapsing, and every `dk_geno` falls by ~0.09 (`ln_cfu0`
  barely moves, so it is not a per-genotype trap).
- **No gradient bug.** The mean SVI gradient (64 draws) at `single`'s
  solution is the same for both models (kanR+kan `growth_k`: -1.90e6 vs
  -1.86e6), as is the exact joint-density gradient
  ([`diagnosis/diag_gradient.py`](diagnosis/diag_gradient.py)). Both pull k
  up from `single`'s point and back down from the mixture's: `single`
  stopped short, the mixture overshot.
- **The only anchor is weak.** `hierarchical_geno` pins wt's `dk_geno` to 0,
  and it stays 0 in both fits. The mixture fit abandons wt instead: it
  predicts wt's `ln_cfu` ~13.6 ln units too high (`single`: +1.3). The
  growth likelihood is Student-t (fitted nu ~9 `single`, ~6.4 mixture),
  whose cost grows only logarithmically: that miss costs ~2,000 log units
  here, against ~44,000 under a Normal.
- **The intended anchor never engaged.** The prefit's per-condition k prior
  is meant to pin the slide, but its Hessian-based `k_scale` hit the 0.1
  ceiling in every run.

## Anchored grid (grid_anchor.yaml, 2026-09-25)

Binding weight 1 plus a tight prefit k prior (`--k_scale_ceiling 0.005`,
prefit `k_scale` 0.005). True lambda 0:

| | original | weight 1 | weight 1 + tight k prior |
|---|---|---|---|
| k offset, `single` | -0.015 | -0.007 | -0.0004 |
| k offset, mixture (matched prior) | +0.062 | +0.080 | +0.083 |
| lambda, measured prior (0.357 +/- 0.05) | 0.55 | 0.47 | 0.45 |
| bulk theta RMSE, mixture vs `single` | 0.267 vs 0.219 | 0.247 vs 0.227 | 0.261 vs 0.230 |

The prior fixed `single`; the mixture drifted ~17 prior SDs past it. Two
further endpoint checks were inconclusive or negative: the ELBO at the two
solutions ([`diagnosis/diag_elbo.py`](diagnosis/diag_elbo.py)) and the
population arrays the mixture looks co-residents up in, which match each
genotype's own values exactly
([`diagnosis/diag_population.py`](diagnosis/diag_population.py)).

## Controls (grid_controls.yaml, 2026-09-25)

Lambda-0 simulation, anchored settings. All three were still improving at
the stop.

| run | k offset | fitted lambda | final loss |
|---|---|---|---|
| mixture, MAP | +0.0003 | 0.0047 | 6.5e4 |
| `single`, MAP | -0.0013 | | 6.2e4 |
| mixture, lambda pinned ~1e-6, SVI | +0.016 | 1e-6 | 1.7e5 |
| mixture, matched prior, SVI (anchored grid) | +0.083 | 0.004 | 1.8e5 |

Without SVI's guide draws the mixture lands where `single` does; MAP also
reaches much lower loss. The pinned-lambda SVI fit drifts less but still
drifts.

## Review: the fits never converged (2026-09-25)

An independent review
([`../congression-calibration-convergence/`](../congression-calibration-convergence/README.md))
found that none of the 3.5 fits converged, `single` included, and it
supersedes the endpoint diagnoses above. Verified here:

- The stop rule (`RunInference._update_loss_deque`) divides the loss change
  by the improvement since the start of SVI, whose loss is ~500 times the
  final loss. Runs stop while the loss is still falling: run 0002 of the
  anchored grid went 5.5e5, 3.6e5, 2.5e5, 1.9e5 over epochs 6000-9000 and
  stopped at 9234. "SVI run converged" in the logs did not mean converged;
  the diagnoses above took it at face value.
- The pre-MAP warm-up's step size decays to 1e-6 within its own 1000
  epochs, so SVI starts far from the mode with prior-width guide scales.
- Epoch checkpoints show `single` holding k at the truth while the
  mixture's k jumps early (to +0.12 near epoch 4000, with nu ~1.3) and
  returns slowly; it was cut off mid-return. The controls agree: under MAP
  the mixture matches `single`.
- The simulated co-resident pool is 39% wt (the library enumerator adds a
  wt entry per non-degenerate codon); being checked separately.

What stands from the earlier sections: the read-count floor bias (0 reads
+1.26 ln) and the binding weight's dominance of the objective, as facts about
the data and the objective. Their effect on converged fits is open.

## Resumed to convergence (resume.srun, 2026-09-25)

Anchored runs 0001 (`single`, lambda 0), 0002 (mixture, matched prior,
lambda 0) and 0005 (mixture, matched prior, lambda 1.0), resumed from their
checkpoints for the full default 100,000 epochs. Traced with
`../congression-calibration-convergence/trace_checkpoints.py`: all three
stop moving after ~70,000 epochs, at losses of 5.2e4 to 5.4e4 (1.7e5 to
1.9e5 at the old stops). The resumed stop rule never fired, as expected.

| growth_k | kanR+kan | kanR-kan | pheS+4CP | pheS-4CP |
|---|---|---|---|---|
| truth | 0.0107 | 0.0154 | 0.0214 | 0.0286 |
| `single`, lambda 0 | 0.0102 | 0.0139 | 0.0212 | 0.0286 |
| mixture, lambda 0 | 0.0109 | 0.0145 | 0.0220 | 0.0294 |
| mixture, lambda 1 | 0.0120 | 0.0170 | 0.0226 | 0.0295 |

| run | fit | lambda (95%) | k offset | dk offset | bulk theta RMSE | bulk theta 95% coverage |
|---|---|---|---|---|---|---|
| 0001 `single`, lambda 0 | stopped | | -0.0004 | +0.0020 | 0.230 | 0.75 |
| | converged | | -0.0005 | +0.0014 | 0.129 | 0.66 |
| 0002 mixture, lambda 0 | stopped | 0.004 (0.003-0.005) | +0.083 | -0.095 | 0.261 | 0.70 |
| | converged | 0.043 (0.039-0.048) | +0.0002 | +0.0003 | 0.131 | 0.66 |
| 0005 mixture, lambda 1 | stopped | 0.98 (0.92-1.04) | +0.021 | -0.031 | 0.227 | 0.73 |
| | converged | 1.30 (1.29-1.32) | +0.0013 | -0.0029 | 0.173 | 0.55 |

- **The transient explanation is confirmed.** Converged, the lambda-0
  mixture matches `single` (k, dk_geno, theta RMSE 0.131 vs 0.129). The k
  drift, the abandoned wt and the worse theta were artifacts of stopping
  early.
- **Converging halves the theta error** (bulk, no binding: 0.23 to 0.13).
- **Coverage got worse** as the guide scales shrank (0.66, and 0.55 for the
  lambda-1 mixture): converged mean-field SVI is overconfident here.
- **Lambda is still pulled up, now with tight intervals that exclude the
  truth:** 0.043 at a true 0, and 1.30 at a true 1.0 under a 1.0 +/- 0.05
  prior (six prior SDs). Open.
- Growth-noise nu at convergence: 61 (`single`), 28 (mixture lambda 0),
  6.5 (mixture lambda 1). The lambda-1 mixture needs much heavier tails.

`single` on the lambda-1 data (run 0004), resumed the same way, also
converged (parameters flat after ~70,000 epochs). Against the lambda-1
mixture (run 0005), same data:

| lambda 1, converged | `single` (0004) | mixture (0005) |
|---|---|---|
| final loss | 5.59e4 | 5.40e4 |
| dk_geno RMSE | 0.0105 | 0.0049 |
| growth_m RMSE | 0.0012 | 0.0009 |
| bulk theta RMSE | 0.155 | 0.173 |
| log_hill_K RMSE | 1.03 | 1.17 |
| bulk theta 95% coverage | 0.58 | 0.55 |
| lambda | | 1.30 (truth 1.0) |

The mixture fits the data better and halves the dk_geno error (the 3.0
study's main congression channel, dk dilution) but does not recover theta:
`single`'s theta RMSE rises from 0.129 at lambda 0 to 0.155 at lambda 1,
and the mixture's is 0.173. The overestimated lambda may be
over-correcting theta; the lambda profile tests that.

These grids used the library enumeration from before bff2c51 (wt 39% of the
co-resident pool; see the convergence study, result 5). Simulation and fit
shared it, so these comparisons stand; new grids from `simulate_config.yaml`
simulate a different library.

## Baseline and profile grids on 5bf5b78: the gradient clip (2026-09-25)

All 13 runs of `grid_baseline.yaml` and `grid_profile.yaml` ran to 100,000
steps without converging, and the lambda-1 seed-1 fits (baseline 0005/0006,
profile 0.8-1.4) jumped to 4-5 times their best loss and stayed there
through three step-size cuts. Neither is a convergence-rule problem at
root; both come from `ClippedAdam`'s elementwise gradient clip at 1.0
(`adam_clip_norm`, older than 5bf5b78).

- **The per-step loss is quantized**: 41k + n x 243k (n = 0-5) on run 0005.
  Each quantum is one binding point of M42C/K84N (from `hill_params.csv`, so
  in every simulation): true `hill_n` 24.6, K between the 0.001 and 0.003
  grid points, measured with SD 0.001. A guide draw that shifts the step
  across a grid point flips theta 0.99 <-> 0.007 there (`diagnosis/diag_clip.py`,
  part 1).
- **Every run carries these penalties at a steady share of steps**, old code
  and new: ~5% (the resumed anchored runs), 5-15% (seed 2), 27-32% (lambda-0
  seed 1), 46-89% (lambda-1 seed 1). The "blow-ups" are the 250-step block
  medians flipping to the penalized level once the share passes 50%; guide
  parameters moved by at most ~0.4 across the jump.
- **The share is set by the clip.** Gradients here are 1e3-1e6 per element,
  so every element is clipped every step and Adam follows each draw's
  gradient sign; it settles where the signs balance, not where the mean
  gradient vanishes. At checkpoint 40000 of run 0005, mutation 1's
  `theta_d_logit_delta_offset_locs` has mean gradient -7.9e3 +/- 0.9e3 (away
  from the penalty) and mean clipped gradient -0.04; for
  `theta_d_logit_low_offset_locs` the clip reverses the sign (-1.2e3 raw,
  +0.14 clipped). The balance point does not depend on the step size (48%
  at 1e-3 and at 1e-6), so the cuts locked it in.
- **Resumed from run 0005's checkpoints** (`diag_clip.py` part 3, local CPU):

| from | clip | step size | penalized share per 2000 steps | mean loss |
|---|---|---|---|---|
| step 8000 (before the jump) | 1.0 | 1e-3 | 0.17, 0.42, 0.57, 0.77, 0.97 | 96k -> 485k |
| step 8000 | none | 1e-3 | 0.01, 0.00 | 47k -> 38k |
| step 40000 (after) | none | 1e-3 | 0.21, 0.07, 0.04, 0.02, 0.00 | 102k -> 40k |
| step 40000 | 1.0 | 1e-4 | 0.48 throughout | ~200k |

- **Fresh fits of run 0005's data** (`tfs-fit-model`, CPU): unclipped, the
  penalties vanish by step 8000 and SVI **converges at step 46,000** (3 cuts,
  loss 38.5k, below the clipped run's best of 41k). With the clip and the old
  start (`--pre_map_num_epoch 0`) it settles at 8-11% penalized (mean loss
  68k), not 48%: 5bf5b78's pre-MAP start only picked a worse clipped
  equilibrium on this simulation.
- **The non-stop (failure 1) is the same clip, plus a yardstick problem.**
  At 1e-6 the flagged locations crept in one direction at ~10% of Adam's top
  speed for 60k steps (clipped dynamics), measured against guide SDs that had
  collapsed to ~4e-4 (hill_mut `theta_sigma_d_*`, `theta_epi_tau`). The
  collapse itself is the mean-field ELBO (it happens unclipped too; the raw
  and clipped gradients agree on shrinking), so those reported widths are
  not honest posterior SDs. Unclipped, the worst movement at 1e-6 was
  0.02-0.03 SDs and the rule stopped the run.

Fixed in the optimizer (CHANGELOG, "Gradient clipping is off by default"):
plain Adam by default (`--adam_clip_norm` opts back in); at the final step
size a window whose mean sits more than 3 robust SDs above its median (rare
penalties the median hides) is not a plateau; posterior SDs used to measure
parameter movement are floored at 1% of the prior SD. Every result on these
two grids, and any clipped fit with this binding set, is biased and should
be rerun. Separately, binding noise 0.001 on step-like curves is much tighter
than real assays; to be loosened in the simulation config.

## Next

The convergence work has landed (5bf5b78: noise-referenced stop rule,
step-size cuts on stalls, SVI started from a converged pre-MAP with guide
scales capped at 0.1; `--convergence_tolerance` is gone and `run.srun` /
`resume.srun` no longer pass it). New grids use it and simulate the
corrected library (bff2c51).

1. Why lambda is pulled up on converged fits: profile lambda (fixed lambda
   on a grid, everything else converged) on the lambda-0 and lambda-1 data,
   then test candidate causes one at a time (the read-count floor, which
   curves dying trajectories the way a mixture does; realized vs design
   library composition in the simulator's transformation, including
   `lib_assembly_skew_sigma`; the fixed co-resident sets).
2. Coverage: why converged fits are overconfident (guide family, the
   binding weight).

Set up (2026-09-25), anchored settings (binding weight 1, prefit
`--k_scale_ceiling 0.005`), corrected library (483 genotypes as before;
wt's co-resident share 0.0026, from 0.39):

- `grid_baseline.yaml`: lambda 0 and 1.0 x {`single`, mixture with the
  matched prior} x seeds 1 and 2 (8 runs). Does the new stop rule end runs
  cleanly, and do the converged results above hold on the corrected
  library?
- `grid_profile.yaml`: the mixture with lambda held at 0.6, 0.8, 1.0, 1.2,
  1.4 (`lam_prior: fixed`, prior SD 0.1% of the value) on the lambda-1,
  seed-1 simulation (5 runs). Loss against lambda says whether the data
  prefer lambda > 1; theta error against lambda says whether theta is
  recovered at the true lambda.

## Baseline and profile grids (2026-09-25)

Both on the noise-referenced convergence code (5bf5b78) and the corrected
library (bff2c51). **The stop rule never fired: all 13 runs ran to the
100,000-step cap** ("SVI run has not yet converged"). After three 10x
step-size cuts (to 1e-6) the parameter criterion still reads hill_mut's
hierarchical scales as moving by 0.3-0.5 posterior SDs; their guide SDs
have collapsed to 0.001-0.005, so noise-level movement counts.

**Blow-ups on one simulation.** On the lambda-1, seed-1 simulation, both
`single` (baseline 0005) and the mixture (0006), and four of the five
profile runs (lambda fixed at 0.8-1.4), reach a good loss (4-6e4) and then
jump ~5x at full step size between steps 8,000 and 14,000 (profile run
0003: one step at epoch 8,750 took it from 7.2e4 to 2.8e5 with guide
parameters moving at most ~0.4). The step-size cuts then lock the damage in.
Flagged parameters: hill_mut's `theta_sigma_d_*` scales. Not
congression-specific (`single` does it too) and not seen before 5bf5b78 on
the analogous runs. The profile grid is therefore uninterpretable: only the
lambda-0.6 run avoided the blow-up. Both failures went to the convergence
work as a separate task.

**The stable pairs** (losses ended within ~2% of their best, so usable with
that caveat):

| | lambda (95%) | dk_geno RMSE | bulk theta RMSE | bulk theta 95% coverage |
|---|---|---|---|---|
| lambda 0, seed 1: `single` / mixture | - / 0.155 (0.151-0.160) | 0.0023 / 0.0017 | 0.133 / 0.121 | 0.82 / 0.83 |
| lambda 0, seed 2: `single` / mixture | - / 0.149 (0.145-0.154) | 0.0021 / 0.0017 | 0.116 / 0.101 | 0.79 / 0.82 |
| lambda 1, seed 2: `single` / mixture | - / 0.98 (0.97-0.99) | 0.0077 / 0.0023 | 0.106 / 0.121 | 0.76 / 0.81 |

- At lambda 1 the mixture recovers lambda (0.98 at a true 1.0; 1.30 on the
  old library) and cuts the dk_geno error ~3x; bulk theta is still slightly
  worse than `single`'s.
- At lambda 0 the mixture still finds lambda ~0.15 (matched prior centered
  at 0.01), yet its theta and dk_geno are slightly better than `single`'s.
  What the congressed classes absorb there is open (the read-count floor is
  one candidate).
- Coverage is better than on the old code (0.76-0.83, from 0.55-0.66).

Next: wait for the convergence fixes, then re-run the profile grid and more
seeds of the baseline.

## Realism fixes before the re-run (2026-09-26)

The blow-ups traced in part to unrealistic binding data: SD 0.001 on
theta, and near-step curves (the simulator drew Hill coefficients up to 270;
50 of 483 genotypes above 4, including 5 of the 20 in-library binding
anchors). From here on the study config uses binding noise 0.025 (the
experimental theta error, user) and `theta_sim_priors.max_hill_n: 4.0`
(new; larger draws are set to 4, so ~10% of genotypes sit exactly at 4).
The optimizer was also updated. Grids after this date are not comparable
with earlier ones.

The model's default binding weight (growth rows / binding rows) is also to
be lowered, but that is a change to the model for real fits; it is left
for its own step. These grids run at `--binding_weight 1` regardless.

## Re-run with realistic data and the fixed optimizer (2026-09-26)

`grid_baseline.yaml` and `grid_profile.yaml` again, with binding noise
0.025, `max_hill_n` 4 and the updated optimizer. **All 13 runs converged**
under the stop rule (40,000-48,000 steps), each ending at its best loss; no
blow-ups. `tfs-summarize-calibration` outputs: `calib/baseline_*`.

Baseline (bulk theta and parameters; 95% intervals for lambda):

| lambda, seed | fit | lambda | dk_geno RMSE | log_hill_K RMSE | bulk theta RMSE | bulk theta coverage | loss |
|---|---|---|---|---|---|---|---|
| 0, 1 | `single` | | 0.0023 | 0.72 | 0.108 | 0.79 | 35,874 |
| 0, 1 | mixture | 0.131 (0.128-0.134) | 0.0015 | 0.67 | 0.089 | 0.82 | 35,153 |
| 0, 2 | `single` | | 0.0024 | 0.72 | 0.097 | 0.80 | 32,480 |
| 0, 2 | mixture | 0.113 (0.110-0.116) | 0.0013 | 0.71 | 0.079 | 0.86 | 32,116 |
| 1, 1 | `single` | | 0.0092 | 0.94 | 0.125 | 0.58 | 37,837 |
| 1, 1 | mixture | 0.942 (0.936-0.948) | 0.0032 | 0.75 | 0.104 | 0.81 | 36,285 |
| 1, 2 | `single` | | 0.0080 | 0.65 | 0.084 | 0.77 | 29,437 |
| 1, 2 | mixture | 0.803 (0.796-0.810) | 0.0024 | 0.61 | 0.074 | 0.87 | 28,182 |

Paired against `single` (`calib/baseline_paired_summary.csv`, theta, mean
of 2 seeds), the gain is in bulk genotypes without binding data:

| lambda | stratum | delta theta RMSE | delta 95% coverage |
|---|---|---|---|
| 0 | bulk, no binding | -0.018 | +0.05 |
| 0 | with binding / spike / mixed | ~0 | small |
| 1 | bulk, no binding | -0.015 | +0.16 |
| 1 | with binding / spike / mixed | ~0 | small |

Profile (mixture, lambda fixed, lambda-1 seed-1 data):

| lambda held at | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 |
|---|---|---|---|---|---|
| loss | 36,319 | 35,812 | 36,294 | 35,870 | 36,013 |
| bulk theta RMSE | 0.111 | 0.107 | 0.105 | 0.102 | 0.102 |
| dk_geno RMSE | 0.0044 | 0.0034 | 0.0032 | 0.0041 | 0.0052 |

- **The mixture now beats `single` where congression is real** (lambda 1):
  bulk theta RMSE 0.104 vs 0.125 and 0.074 vs 0.084, dk_geno error cut
  ~3x, coverage 0.81-0.87 vs 0.58-0.77, and a better fit. The earlier
  "mixture worse on theta" results came from unconverged or unstable fits.
- **Lambda is recovered roughly, with overconfident intervals:** 0.94 and
  0.80 at a true 1.0; intervals ~+/-0.007 exclude the truth. At a true 0
  the mixture still finds 0.11-0.13, yet fits better than `single` on
  theta, dk_geno and coverage there too, so its congressed classes absorb
  something real in the data (the read-count floor is the leading
  candidate).
- **The profile is flat within the loss noise** (differences of a few
  hundred, not monotone): these data barely pin lambda between 0.6 and 1.4.
  Theta error changes little across that range (0.111 to 0.102), and
  dk_geno error is lowest near the true lambda. So a misestimated lambda
  costs little in theta.
- Coverage remains below nominal for both fits (0.78-0.87), the mixture's
  less so.

## Step 4 changes that affect this study (2026-09-26)

The fit's default congression theta rule is now `homodimer` and the default
activity `fixed`. `simulate_config.yaml` keeps `congression_theta_rule: max`
so the recorded grids stay reproducible, and `run.srun` now passes the fit
the rule the simulation used (`fit_theta_rule` overrides it, for cross-rule
grids). The confirming round (`grid_measured.yaml`, `grid_seed3.yaml`) was
set up before this change and runs `max` on both sides.

