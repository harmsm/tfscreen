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
`tfs-setup-sim-grid` writes each run's `binding_data.spiked_binding.choose_by`
as a path from the run directory to this study's `hill_params.csv`, so nothing
is copied, but keep the grid and the checkout where they are after setup:

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

## Diagnosis: the mixture fits the detection floor (2026-09-24)

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

## Next: masked re-run (grid_masked.yaml)

Tests the diagnosis by dropping growth rows with fewer than 5 reads before
fitting (`min_counts`, handled in `run.srun`; 13,253 of 54,300 rows in the
lambda 0 simulation). Same simulations as grid runs 0001-0003 and 0037-0039
(wide dk, seed 1), 6 runs. If the diagnosis is right, the lambda 0 mixture
matches `single`, and lambda stops being pulled up. The principled fix is a
censored likelihood for floor observations (user, 2026-09-24: preferred
next step over masking; fitting read counts directly is a longer-term
option). The anchored re-run of the full grid (a `base_growth` block) waits
on this.
