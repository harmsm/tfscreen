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

### Run 3 (2026-10-17): the full model started from the plain fit

Both converged below the fits they had to beat: null 783,021 (plain
783,286; run 1's full 784,304), joint 780,782 (run 1's full 780,800).
Starting the full model from the plain one is the recipe.

| | truth | null, full from plain | truth | joint, full from plain |
|---|---|---|---|---|
| m kanR+kan | -0.0141 | -0.0176 | -0.0141 | -0.0144 |
| m pheS+4CP | +0.0144 | +0.0180 | +0.0144 | +0.0166 |
| kanR warp a / x0 | linear | 2.6 / 0.59 | 8 / 0.50 | 9.0 [8.7, 9.5] / 0.53 [0.50, 0.56] |
| pheS warp a / x0 | linear | 1.7 / 0.57 | 1.5 / 0.40 | 0.9 [0.2, 4.3] / 0.59 [-0.24, 1.32] |
| log s | 0 | -0.05 [-0.22, 0.11] | -0.69 | -0.80 [-0.94, -0.66] |
| offsets covered | 0 | 9 of 9 | | 8 of 9 |
| tau | 0 | 0.17 [0.05, 0.52] | 0.10 | 0.22 [0.09, 0.50] |
| singles theta RMSE (95% cov.) | | 0.077 (0.14) | | 0.070 (0.17) |
| spikes theta RMSE (95% cov.) | | 0.056 (0.64) | | 0.039 (0.48) |

- **The null comes back.** The warps sit at about the prior's median
  steepness (a about 2.7, close to linear on [0, 1]), s covers 1, every
  offset covers 0. The extra freedom costs theta a little against the
  plain fit (singles 0.077 against 0.056).
- **The joint truth comes back.** kanR's switch, s and the offsets are
  recovered; pheS's graded warp is weakly identified (wide, covers).
- **Still open, both independent of the warp:**
  1. m is steep: 22-25% in both markers on the null simulation, with the
     plain (true) model as well; pheS 15% on the joint one (kanR right).
     The simulator and the fit use the same congression rules (homodimer,
     dilution). k is high by 0.002-0.005, the k/dk_geno slide.
  2. Theta intervals undercover badly (singles 14-17%, wt 13-25%) while
     the median error is small: the MAP plus Laplace leaves the shared
     parameters' error (m above all) out of each genotype's interval.

### Run 4 (2026-10-17): the loss at the true k and m

The null simulation, plain (true) model, k and m pinned at their truth:
from `fit_null_s1_plain`'s MAP, converged at 784,031; from scratch,
784,068 (still descending at step 1e-5). The free fit is 783,286, so the
truth costs about 745 nats: stage 1 prefers the steep m, by far more
than four parameters' noise. At the truth, theta is much better
(singles RMSE 0.020 against 0.056, coverage 0.70 against 0.10; spikes
0.012, wt 0.011): the steep m is most of stage 1's theta error.

Attribution (`attribute.py`, pinned minus free at each MAP; total -720):

| | nats |
|---|---|
| growth, pheS+4CP | -396 |
| growth, pheS-4CP | -106 |
| growth, kanR+kan / kanR-kan | -64 / -31 |
| log prior (theta population scales and offsets, mostly) | -161 |
| binding | +37 |

- pheS+4CP's loss is in the deep singles (> 1e5 reads), at 0.1 and 1 mM
  IPTG (-261 together), growing with selection time, and in rows whose
  true theta is below 0.1 (-366 of -396): the cells that grow slowest
  under 4CP, whose counts their co-transformed partners come to
  dominate. Fitted lambda is 0.335 free and 0.301 pinned, against the
  simulated 0.357.
- The prior's share: at the true m the singles' curves must spread wider,
  which the theta population scales charge for; shrinkage of the theta
  range is part of what a steeper m buys.
- Binding agrees with the truth better (+37).

So the steep m most likely compensates for how the fit's congression
mixture (fixed co-resident sets, `congression_sets` [12, 3, 1]) describes
slow growers, plus some shrinkage. The simulator and the fit share the
congression rules (homodimer, dilution); what differs is that the fit
approximates each cell's co-residents by a few fixed draws.

### Run 5 (2026-10-17): without congression

The control: the null simulation without congression
(`make_sim.py --arm nocong`, `transformation_poisson_lambda` 0, same seed
and library), fitted with the plain model without the mixture
(`stage1.py --no_mixture`, `NOMIX=1`), free and with k and m pinned at the
truth. If m comes back unbiased and the truth costs a few nats, the
mixture's approximation is the cause.

```bash
cd /gpfs/projects/harmslab/harms/studies/joint-fit-sim && sim=$(sbatch --parsable --export=ALL,ARM=nocong,SEED=1 --output=sim_nocong.out /gpfs/home/harms/tfscreen/planning/studies/joint-fit-sim/simulate.srun) && echo "sim $sim"
```

```bash
sbatch --dependency=afterok:$sim --export=ALL,SIM=sim_nocong_s1,MODEL=plain,NOMIX=1 --output=fit_nocong_s1_plain_single.out /gpfs/home/harms/tfscreen/planning/studies/joint-fit-sim/stage1.srun && sbatch --dependency=afterok:$sim --export=ALL,SIM=sim_nocong_s1,MODEL=plain,NOMIX=1,PIN=1 --output=fit_nocong_s1_plain_single_pinkm.out /gpfs/home/harms/tfscreen/planning/studies/joint-fit-sim/stage1.srun
```

Results. Free 776,488, pinned at the truth 776,871 (both converged): the
truth still costs 383 nats, half of run 4's 745. m is 14% steep in both
markers (kanR -0.0161, pheS +0.0164), against 22-25% with congression. At
the truth theta is nearly exact (singles RMSE 0.010, 95% coverage 0.91;
spikes 0.006, wt 0.002); free, singles 0.046 (coverage 0.11).

Attribution (pinned minus free, total -357): growth -235, spread over all
four conditions and the deep singles with no hot spot; log prior -144, of
which the theta population scales and offsets take about -130 (logit low,
logit delta and log K, about -21 to -23 each for scale and offsets);
binding +22.

So the bias has two parts of about equal size:

1. **The MAP over-shrinks the theta population** (no congression needed).
   The population scales of the singles' Hill parameters run small, which
   compresses their theta range, and a steeper m restores the growth swing;
   at the true m the curves need their real spread, which the population
   prior charges for. The doubles' pull on m (the m-bias study) works the
   same way. A hierarchical MAP has no reason to get those scales right
   (the funnel); `hill_relative` holds them for the same reason
   (`theta_*_hyper_scale_fixed`), `hill_geno` only holds log(n)'s.
2. **The congression mixture's approximation** adds the rest (run 4's
   pheS slow growers).

