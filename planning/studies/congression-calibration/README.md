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

These grids used the library enumeration from before bff2c51 (wt 39% of the
co-resident pool; see the convergence study, result 5). Simulation and fit
shared it, so these comparisons stand; new grids from `simulate_config.yaml`
simulate a different library.

## Next

1. `single` on the lambda-1 data, converged (resume anchored run 0004 with
   `resume.srun`; no new code needed), to see whether the mixture beats it
   where congression is real.

Items 2 and 3 need new fits, so they wait for the convergence work
(separate session):

2. Why lambda is pulled up on converged fits: profile lambda (fixed lambda
   on a grid, everything else converged) on the lambda-0 and lambda-1 data,
   then test candidate causes one at a time (the read-count floor, which
   curves dying trajectories the way a mixture does; realized vs design
   library composition in the simulator's transformation, including
   `lib_assembly_skew_sigma`; the fixed co-resident sets).
3. Coverage: why converged fits are overconfident (guide family, the
   binding weight).
