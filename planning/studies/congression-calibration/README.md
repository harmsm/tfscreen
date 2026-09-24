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

On the cluster, from a scratch directory, with this repository checked out
(the rendered `run.srun` finds `hill_params.csv` by a path relative to the
study directory, so do not move the grid after setting it up):

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

Pending (set up 2026-09-24).

## Results

Pending.
