# Joint stage 1 on a simulation

**Question.** With binding back at weight 1 and the joint model's pieces
(absolute theta from `hill_geno`, a per-marker warp of occupancy, the
binding-to-growth titrant scale s, one binding offset per binding
genotype), does stage 1 (wt, spikes and singles, with the congression
mixture) recover m, the maps, s and theta on a simulation where we know
them? Does binding at weight 1 leave growth undistorted, and does the full
model return the null when there is nothing to find?

**Decision it feeds.** `planning/joint-fit-return.md` step 5: whether to
run step 6 (the real stage 1) with this model, and which pieces it needs.

## Simulations (`make_sim.py`)

Both start from the full-size realistic configuration
(`planning/studies/full-size-sim/make_sim_config.py`; on the cluster
`full-sized-sims-3/sim_realistic_s2`), seed 1, so the library, the
phenotypes and the spikes' curves are the same. Every spike gets binding
at the lab's 2 uM IPTG grid (16 concentrations, 0 to 5 mM) with noise 0.04,
the scatter within one real curve. Its curve comes from `spike_hill.csv`:
wt at the simulator's reference, the other eight drawn by the script.

| arm | kanR+kan | pheS+4CP | s | binding offsets |
|---|---|---|---|---|
| `joint` | switch-like, warp a = 8, x0 = 0.5 | graded, a = 1.5, x0 = 0.4 | 0.5 | SD 0.1 per spike |
| `null` | linear in theta | linear | 1 | none |

New simulator keys (2026-10-16): a linear growth condition's `warp_a` and
`warp_x0`, the top-level `growth_titrant_scale` and
`binding_data.genotype_offset_sd`. Their truths are in
`tfs_sim_growth_parameters.csv` (`warp_log_a`, `warp_x0`),
`tfs_sim_growth_titrant_scale.csv` (log s) and
`tfs_sim_binding_offsets.csv`.

## Fits (`stage1.py`)

Stage 1 of the two-stage fit, joint: wt, spikes and singles; `hill_geno`;
the mixture with the homodimer rule (lambda prior 0.357 +- 0.15); binding
at weight 1; counts with level tube offsets (SD 0.17); log(n) population
SD 0.5; one k and m per condition with loose priors. MAP (staged), Laplace
with 500 draws, extraction, theta predictions.

- `full`: plus the warp on both selection conditions, s (log s ~
  Normal(0, 1)) and the binding offsets (`genotype_offset`).
- `plain`: none of those.

| run | simulation | model | asks |
|---|---|---|---|
| `fit_joint_s1_full` | joint | full | recovery of m, warps, s, offsets, theta |
| `fit_joint_s1_plain` | joint | plain | what the pieces buy (the misspecified model) |
| `fit_null_s1_full` | null | full | the null comes back (a small, s near 1, tau near 0) without costing theta |

**Pass.** On `fit_joint_s1_full`: m within its 95% interval and within
about 10% of the truth in both selection conditions; warp log a and x0
covered; log s covered; the offsets covered and tau's interval holding
0.1; theta coverage near 0.95 for wt, spikes and singles. On
`fit_null_s1_full`: kanR's and pheS's warps compatible with near-linear,
log s covering 0, tau's interval reaching toward 0, and theta no worse than
`fit_joint_s1_full`'s. `fit_joint_s1_plain` should miss somewhere; where
it does says which piece matters.

## How to run

On the cluster, after pulling the commit that adds this study:

```bash
mkdir -p /gpfs/projects/harmslab/harms/studies/joint-fit-sim && cd /gpfs/projects/harmslab/harms/studies/joint-fit-sim
```

```bash
for a in joint null; do sbatch --export=ALL,ARM=$a,SEED=1 --output=sim_$a.out /gpfs/home/harms/tfscreen/planning/studies/joint-fit-sim/simulate.srun; done
```

Once both finish (about 90 minutes each; check `ls sim_*_s1/tfs_sim_binding.csv`):

```bash
for r in joint_s1:full joint_s1:plain null_s1:full; do sbatch --export=ALL,SIM=sim_${r%%:*},MODEL=${r##*:} --output=fit_${r%%:*}_${r##*:}.out /gpfs/home/harms/tfscreen/planning/studies/joint-fit-sim/stage1.srun; done
```

Then, locally after downloading the fits' outputs and each simulation's
truth files (`simulate_config.yaml`, `tfs_sim_parameters.csv`,
`tfs_sim_growth_parameters.csv`, `tfs_sim_growth_titrant_scale.csv`,
`tfs_sim_binding_offsets.csv`):

```bash
python analyze.py runs/fit_joint_s1_full:runs/sim_joint_s1 runs/fit_joint_s1_plain:runs/sim_joint_s1 runs/fit_null_s1_full:runs/sim_null_s1 --out out
```

`analyze.py` writes `out/params.csv` (each fitted parameter against its
truth, with its 95% interval) and `out/theta.csv` (theta RMSE, bias and
95% coverage by group: wt, spikes, singles). Theta is compared in binding
units, the fitted curve against the truth curve at the growth
concentrations.

## Results

### Run 1 (2026-10-17, commit of 2026-10-16)

Simulations from `sim_realistic_s2`'s config, seed 1; 955 stage-1
genotypes (wt, 8 spikes with binding, 946 singles); M42I/H74A survived
this time. Final MAP losses: joint/full 780,800 (converged), joint/plain
798,807 (converged), null/full 784,304 (still descending at step 1e-5).

Shared parameters, median [95%] against the truth:

| | truth | joint/full | joint/plain | null/full (truth) |
|---|---|---|---|---|
| m kanR+kan | -0.0141 | -0.0144 [-0.0145, -0.0143] | -0.0187 | -0.0167 (-0.0141) |
| m pheS+4CP | +0.0144 | +0.0164 [0.0160, 0.0167] | +0.0155 | +0.0169 (+0.0144) |
| kanR warp log a / x0 | 2.08 / 0.50 | 2.20 / 0.57 | - | 2.78 / 0.70 (linear) |
| pheS warp log a / x0 | 0.41 / 0.40 | -0.14 [-0.68, 0.44] / 1.25 [0.28, 2.16] | - | 2.71 / 0.70 (linear) |
| s | 0.50 | 0.42 [0.37, 0.48] | (1) | 0.25 [0.17, 0.35] (1) |
| offset scale tau | 0.10 | 0.25 [0.11, 0.51] | - | 0.34 (0) |
| k, every condition | | 0.002-0.005 high | 0.003-0.005 high | 0.000-0.003 high |

Theta RMSE of the median (95% coverage), binding units, and the RMSE as
growth sees it (each curve at its own s times nominal):

| | joint/full | joint/plain | null/full |
|---|---|---|---|
| singles (946) | 0.072 (0.18); in vivo 0.074 | 0.130 (0.04); in vivo 0.091 | 0.293 (0.04); in vivo 0.302 |
| spikes (8) | 0.059 (0.31) | 0.047 (0.31) | 0.226 (0.09) |
| wt | 0.055 (0.25) | 0.137 (0.00) | 0.151 (0.13) |

What it says:

- **On the joint simulation the pieces work and are needed.** The full
  model gets kanR's m within 2% and its warp close (a 9.0 against 8,
  x0 0.57 against 0.50), s within 16%, and halves the singles' theta
  error of the plain model, which steepens kanR's m by 33% to bend a
  linear map round a switch. pheS's graded warp is weakly identified
  (wide, covers) and its m is 14% steep. Most offsets are near their
  truth; tau is high (0.25 against 0.1).
- **The null fit finds a wrong mode.** With linear truth, both markers get
  the same steep warp (a about 16, x0 0.70), s drops to 0.25 and the
  spikes take large offsets (K84L -0.32, M42I/K84L -0.38); theta is off by
  0.3 whether read in binding or growth units. A warp shared by both
  markers trades exactly with every single's curve (growth sees f(theta),
  and singles have no binding), so only the binding genotypes pin it, and
  the offsets let those same genotypes disagree with their binding. On
  the joint arm the two markers' warps differ, so the singles pin their
  difference and binding only sets the common scale.
- **Intervals are far too narrow everywhere** (theta 18-31% coverage
  where the median error is small; m's interval excludes the truth),
  the MAP plus Laplace holding the shared parameters' error out of each
  genotype, as before.
- k sits high in every condition, the k/dk_geno slide, as in every stage
  1 so far.

### Run 2 (2026-10-17): trap or degeneracy?

`fit_null_s1_plain` (the null simulation's own model) converged at a MAP
loss of 783,286, about 1,000 nats below `fit_null_s1_full` (784,304, still
descending at step 1e-5 when it hit the epoch cap). The full model nests
the plain one at a cost of tens of nats in prior terms, so run 1's null
fit was trapped, not unidentified: the data do separate the linear truth
from the steep shared warp. It reached the steep warps early, while the
step size was large.

The plain fit's theta is good (singles RMSE 0.056, bias -0.002; spikes
0.036) but its intervals still undercover (singles 0.10), and m is about
20% steep in both markers with the true model (kanR -0.0173 against
-0.0141, pheS +0.0173 against +0.0144) while k sits 0.005 high: an open
question for stage 1 itself, separate from the warp.

Next (run 3): the full model started from the plain fit's MAP
(`stage1.py --init_from`, `stage1.srun` `INIT=`), on both simulations. Pass:
the null fit returns to the plain fit's loss or below with near-linear
warps, s near 1 and small offsets; the joint fit reaches run 1's 780,800
or below.

```bash
for s in null joint; do sbatch --export=ALL,SIM=sim_${s}_s1,MODEL=full,INIT=fit_${s}_s1_plain --output=fit_${s}_s1_full_from_plain.out /gpfs/home/harms/tfscreen/planning/studies/joint-fit-sim/stage1.srun; done
```

