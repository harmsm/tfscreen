# Validating tfs-fit-two-stage against the m-bias study

**Question.** Does `tfs-fit-two-stage` reproduce what the m-bias study
(`planning/studies/m-bias-factorial/`) found with study scripts, on both
full-size simulation seeds and on the real data? And where should the
consistency step's flag sit?

**Decision it feeds.** `planning/two-stage-fit.md` step 4: whether the
command is the recipe the students and the NSF preliminary data use.

## Runs

| run | growth table | library config | lambda prior |
|---|---|---|---|
| `run_seed1` | `sim_realistic_s2` (seed 1) | its simulate config (wt mostly spiked, as simulated) | 0.357 +- 0.15 |
| `run_seed2` | `sim_realistic_seed2` | its simulate config | 0.357 +- 0.15 |
| `run_real` | the recipe's growth table | the real config as written (wt mostly bulk) | 0.38 +- 0.15 (the lab's 0.357, 95% CI 0.156-0.674) |

Every run uses the study's model (growth-only relative fit, level offsets
at SD 0.17, log(n) population SD 0.5, loose k/m priors in stage 1), 30
jackknife subsamples dropping 20% of tubes, 500 posterior draws, and the
consistency step on kanR alone. The YAMLs (`two_stage_*.yaml`) hold
cluster paths.

## What each should reproduce

| | study | from |
|---|---|---|
| seed 1 stage 1 m / truth | kanR 1.056, pheS 1.037 | `m_subsets/no_doubles_mix` |
| seed 2 stage 1 m / truth | kanR 1.047, pheS 1.023 | `sim_realistic_seed2/stage1/no_doubles_mix` |
| seed 1 jackknife k SE | kanR+kan 0.0048, pheS+4CP 0.0028 | `jk_seed1` |
| seed 1 X coverage with the shift, 1e4-1e5 / > 1e5 reads | 0.94 / 0.91 | `jk_seed1/x_kband.csv` |
| seed 1 X without the shift (stage 2) | 0.90 / 0.76 | `stage2_single_seed1` |
| seed 2 X without the shift (stage 2) | 0.88 / 0.67 | `stage2_single_seed2` |
| real stage 1 | kanR+kan k 0.0167, m -0.0147 | `m_subsets_real/no_doubles_mix_cfg` |
| real X | r about 1 with `held_mix_real` | `held_mix_real` |

Differences to expect: the command picks the stage-1 genotypes from the
library config (`in_spiked_origin` or one mutation) where the study used
every non-double; the real data's three spiked doubles with growth data
now join stage 1. The jackknife's k step is 1.7% smaller (covariance over
N, not N - 1). The k-shift refits predict X at their MAP, where the study
used their Laplace medians. Seed 2's coverage with the shift is new.

For the consistency flag: the m-bias study found 2.5% of deep simulated
doubles' rows and 8.2% of real ones beyond 2 SD between the kanR-only and
both-marker fits (`kanR_only_seed1`, `kanR_only_real`). The command's
statistic is per genotype (level offset over the median combined SD), so
its seed 1 and seed 2 rates are the no-signal baseline for the real
rate.

## How to run

On the cluster, after pulling the commit that adds this study (the job
calls the command as a module, so no reinstall is needed):

```bash
mkdir -p /gpfs/projects/harmslab/harms/studies/two-stage-validation && cd /gpfs/projects/harmslab/harms/studies/two-stage-validation
```

```bash
for c in seed1 seed2 real; do bash /gpfs/home/harms/tfscreen/planning/studies/two-stage-validation/submit.sh /gpfs/home/harms/tfscreen/planning/studies/two-stage-validation/two_stage_${c}.yaml run_${c}; done
```

Each `submit.sh` prints its five job ids (stage1; the jackknife array and
stage2 after it; kshift + predict after both; consistency after stage2).
Logs are in `run_*/logs/`. Each step skips finished work, so resubmitting
a chain resumes it. Expect about a day of wall time per run with 10
jackknife tasks at once, longer if GPUs are short.

Then, per run (locally after downloading, or on the cluster),
`validate.py` with the matching study outputs:

```bash
python validate.py run_seed1 --depth <seed-1 depth.csv> --truth <stage2_single_seed1>/summary/tfs_summarize_theta_corr_test.csv --sim_dir <sim_realistic_s2> --reference <stage2_single_seed1>/tfs_pred_theta.csv --study_stage1 <m_subsets/no_doubles_mix>
```

The follow-ups, from the same directory once a run is done:

```bash
sbatch --export=ALL,RUN=real /gpfs/home/harms/tfscreen/planning/studies/two-stage-validation/control_refit.srun
```

```bash
sbatch --export=ALL,RUN=real,LIB=pheS /gpfs/home/harms/tfscreen/planning/studies/two-stage-validation/one_library.srun
```

`control_refit.py` refits stage 2 as the k-shift step does but with k
unchanged (`kshift/zero/`) and extracts the tube offsets of all three
refits. `one_library.py` fits stage 2 on one library's tubes alone
(`consistency_{library}/`).

## Commit

The three runs: 3ce04cc5. The control: 4f52e68f. The pheS-only fit:
9477a578.

## Results

### Run 1 (2026-10-13 to 2026-10-15)

All three runs finished without errors.

**The command reproduces the study.** Stage 1's k and m match the study's
to about 1%, except the real pheS+4CP k (0.0132 against 0.0120), which we
put down to the three spiked doubles now in stage 1. Seed 1's jackknife k
SEs match (kanR+kan 0.0048, pheS+4CP 0.0028). Seed 1's stage 2 covers the
deep doubles exactly as the study's did:

| | 1e4-1e5 reads | > 1e5 reads |
|---|---|---|
| seed 1, study, no shift | 0.90 | 0.76 |
| seed 1, run, no shift | 0.90 | 0.76 |
| seed 1, run, with shift | 0.915 | 0.858 |
| seed 1, study, with shift (`x_kband.csv`) | 0.94 | 0.91 |
| seed 2, run, no shift | 0.907 | 0.814 |
| seed 2, run, with shift | 0.921 | 0.867 |

Stage 2 against the study's matching fit, deep genotypes: r 0.9997 (seed
2) and 0.9999 (seed 1, real), slope 0.99-1.02, widths within 2%, and a
uniform offset of -0.024, +0.020 and -0.017. The whole-library slopes
(0.91 and 1.17 in `validate.py`'s line) come from genotypes below 100
reads, whose X is mostly prior.

**The marker-consistency flag.** It flagged 1.9% of the deep (>= 1e4
reads) genotypes on both simulations and 10.6% on the real data (kanR
alone). The real tail is signal, about 5.5x the no-signal rate.

**The k shift does not work as built.** The shift came out 25% below the
study's k-band SD on seed 1 (and about 3x below it at high IPTG), so the
coverage with the shift fell short of the study's. The two refits did not
bracket stage 2: on seed 1 both landed 0.06-0.11 above stage 2 in X for
the deep doubles, and on the real data both landed 0.25-0.32 above it,
1.4-1.6e4 nats lower in loss.

The control (k left at stage 1's values) moved X by +0.32 as well,
uniformly over depth, IPTG, singles and doubles, and ended 2.3e3 nats
above stage 2. So the refit procedure moves X, not k:

| fit (real) | steps | final exact loss | first window's loss |
|---|---|---|---|
| stage 2 (staged MAP) | 54,000 | 61,106,742 | 61,162,781 |
| control | 188,000 | 61,109,081 | 61,773,654 |
| +k | 410,000 | 61,092,275 | 61,796,625 |
| -k | 540,000 | 61,090,397 | 61,771,936 |

The refits start at stage 2's MAP but at Adam's default step size 1e-3,
where stage 2 ended at 1e-6, and the first 2,000 steps knocked them about
6.7e5 nats off it. Each re-descended and stopped where its convergence
rule fired. Where it stopped set the X level.

**What slides: the dk_geno population against wt.** Stage 2's arrowhead
Laplace held one negative direction (eigenvalue -9.5e3), the dk_geno
population location (loading 0.89) and scale (-0.45). In the refits every
genotype's dk_geno moved by nearly the same amount, about 5 jackknife k
SEs:

| fit (real) | dk_geno median | change from stage 2 (q10 to q90) | X_delta population location |
|---|---|---|---|
| stage 2 | 0.0034 | | -0.53 |
| control | 0.0138 | +0.0106 (+0.0081 to +0.0108) | -0.27 |
| +k | 0.0097 | +0.0063 (+0.0055 to +0.0066) | -0.30 |
| -k | 0.0149 | +0.0116 (+0.0108 to +0.0119) | -0.28 |

This is the k/dk_geno slide reopened through the population: pinning k
fixes wt's growth, but every other genotype can move together against wt
through their shared dk_geno location, and only wt's own data and the
tube offsets resist. In the refits the offsets moved most in the pheS-4CP
tubes (+0.4 to +0.5, both IPTG). The tube-offset mode of
`planning/offset-mode-growth-transition.md` may be the same direction seen
in a fit with k free.

**pheS alone compresses X** (confirmed with wt's curve held, below). Deep
genotypes, X from one library's tubes
against the both-library stage 2:

| | r | slope | intercept | rms |
|---|---|---|---|---|
| kanR alone vs both | 0.941 | 1.04 | -0.03 | 0.12 |
| pheS alone vs both | 0.689 | 0.71 | +0.15 | 0.24 |
| pheS alone vs kanR alone | 0.559 | 0.52 | +0.24 | 0.31 |
| control vs kanR alone | 0.941 | 0.86 | +0.40 | 0.35 |

The both-library fit follows kanR (median SD 0.09 against pheS's 0.14).
pheS alone sits 0.14 below it at 0 IPTG and 0.12 above it at 1 mM: the
mutants' pheS response spans less than wt's. A mutant in wt's repressed
or derepressed state should grow as wt does there whatever the shape of
the theta-to-growth map, so the compression says wt and the library
differ in pheS. Since X is gauged on wt, an anomaly in wt's pheS growth
gives exactly such an affine map. Neither library alone supports the
control's level (+0.40 and +0.46).

The deep genotypes flagged by the two one-library checks barely overlap:
10.6% (kanR alone), 8.7% (pheS alone), 1.8% both, and their level
offsets do not correlate (r -0.14).

**The compression is real (2026-10-15, `planning/studies/x-vs-binding/`).**
The pheS-only fit is degenerate at wt (its wt X falls from 1 to 0 below
1 uM IPTG, where wt's library trajectories fall at 10-30 uM), but
refitting with wt's curve held at stage 2's (`consistency_pheS_wtheld/`)
changed no mutant (r 0.996 with the first pheS fit) and kept the
compression (slope 0.52, intercept +0.24 against kanR). Binding theta at
2 uM agrees with kanR's X (mean -0.03) and not pheS's: pheS puts the
weak binders 0.25-0.5 below their theta and below X = 0. Binding also
supports stage 2's X level over the refits' (+0.3).

**Proposed next steps** (not yet decided).

- Drop the k-shift refits and take k's (and m's) uncertainty from the
  Laplace instead: stage 2 with k and m free under a prior equal to the
  jackknife covariance. The
  real jackknife puts m's SE at 0.001 (7% of kanR's m; 0.5% on the
  simulations), which the shift never carried.
- That needs a real optimum. Candidate: hold the dk_geno population
  location at stage 1's estimate, as the k and m pins do. Its scale
  probably cannot be held: stage 1 (singles) gave 0.53, stage 2 1.53.
- Find why wt does better in 4CP, derepressed, than mutants at the same
  or lower occupancy (`planning/studies/x-vs-binding/`).

### Run 2 setup: the k-shift repair (2026-10-19)

Decided (user, 2026-10-19; `planning/release-0.5.0.md` decision 1):
repair the shift, and if one round fails, ship without it.
`tfs-fit-two-stage`'s kshift step now:

- starts each refit from stage 2's MAP at `two_stage.kshift_step_size`
  (default 1e-5, final 1e-6), not Adam's 1e-3, which knocked run 1's
  refits about 6.7e5 nats off the MAP in the first window;
- holds the dk_geno population (`hyper_loc`, `hyper_scale`,
  `hyper_shift`) at stage 2's MAP in every refit, the direction stage 2's
  Laplace held and run 1's refits slid along. Holding it at stage 1's
  values, as first proposed, would change stage 2 itself (stage 1's
  singles give a scale of 0.53 against stage 2's 1.53) and need stage 2
  revalidated;
- runs a third refit with k unchanged (`kshift/zero/`, on unless
  `kshift_control: false`) and writes `kshift/drift.csv`: each refit's
  median change in X from stage 2's MAP (`stage2/tfs_pred_theta_map.csv`)
  over non-wt rows, with q10 and q90. It warns when the control moves
  more than a quarter of the median |shift|.

A risk that remains: the data preferred run 1's slid state by about
1.5e4 nats, and with the population held, every genotype can still move
its own dk_geno offset together, at a prior cost of a few hundred nats.
The step size slows that slide but does not forbid it; the control
refit is there to show whether it happens.

Stage 1, the jackknife and stage 2 are unchanged, so only kshift and
predict rerun. On the cluster, after pulling, in
`/gpfs/projects/harmslab/harms/studies/two-stage-validation`, set run 1's
outputs aside:

```bash
for c in seed1 seed2 real; do mv run_${c}/kshift run_${c}/kshift_run1; mv run_${c}/predict run_${c}/predict_run1; done
```

```bash
ls -d run_*/kshift_run1 run_*/predict_run1
```

Then one job per run. Every path is absolute: an earlier form built `OUT`
from `$PWD`, and jobs submitted from another directory made empty run
directories there and failed (2026-10-19). The partition and time on the
command line override the script's `#SBATCH` lines:

```bash
D=/gpfs/projects/harmslab/harms/studies/two-stage-validation; S=/gpfs/home/harms/tfscreen/planning/studies/two-stage-validation; for c in seed1 seed2 real; do sbatch --partition=gpu --time=00-06:00:00 --job-name=ts_kshift_${c} --output=${D}/run_${c}/logs/kshift2_%j.out --error=${D}/run_${c}/logs/kshift2_%j.err --export=ALL,CFG=${S}/two_stage_${c}.yaml,OUT=${D}/run_${c},STEP=kshift_predict ${S}/run_step.srun; done
```

Three refits per run, each probably 1-3
hours at the small step size (run 1's took 188,000-540,000 steps at
1e-3); these ask for `gpu`, 6 hours (`gpulong` is congested).

Once a run's kshift job is done, after pulling the commit that adds the
summary step, its results table (`run_*/tfs_x.csv`, a few minutes on
CPU; the GPU partition is not needed but `run_step.srun` asks for it):

```bash
D=/gpfs/projects/harmslab/harms/studies/two-stage-validation; S=/gpfs/home/harms/tfscreen/planning/studies/two-stage-validation; for c in seed1 seed2 real; do sbatch --time=00-01:00:00 --job-name=ts_summary_${c} --output=${D}/run_${c}/logs/summary_%j.out --error=${D}/run_${c}/logs/summary_%j.err --export=ALL,CFG=${S}/two_stage_${c}.yaml,OUT=${D}/run_${c},STEP=summary ${S}/run_step.srun; done
```

Pass:

- `kshift/drift.csv`: the control (`zero`) moves X by less than a quarter
  of the median |shift|, and plus and minus move by about the same
  amount in opposite directions;
- seed 1: the shift within about 15% of the study's k-band SD
  (`jk_seed1/x_kband.csv`; run 1 was 25% low), and deep X coverage with
  the shift near the study's 0.94 / 0.91 (`validate.py`, as run 1);
- seed 2: coverage with the shift above run 1's 0.921 / 0.867;
- real: the control near 0 (run 1: +0.32).

### Run 2 results (2026-10-20)

All three ran on commit ccd5544c (after a first submission from the
wrong directory and one on code before the repair; both set aside as
`kshift_oldcode/`, `predict_oldcode/` on seed 1). Every refit converged
at 1e-6. `kshift/drift.csv`, median change in X from stage 2's MAP over
non-wt rows:

| run | plus | minus | control (k unchanged) | median shift | a quarter of it |
|---|---|---|---|---|---|
| seed 1 | +0.002 | -0.024 | -0.004 | 0.013 | 0.003 |
| seed 2 | -0.007 | +0.000 | -0.001 | 0.004 | 0.001 |
| real | +0.022 | -0.026 | -0.008 | 0.024 | 0.006 |

The drift is mostly gone (the real control moved -0.008, against +0.32
in run 1), but the repair fails its pass criteria:

- On the simulations the shift is a third (seed 1) and a tenth (seed 2)
  of the study's k-band SD (one SD of k moved X by 0.032-0.038), and the
  plus and minus refits are lopsided.
- The control moved more than a quarter of the shift on seed 1 and the
  real data, and as much on seed 2.

Reading: holding the dk_geno population blocks the slide but also part
of X's real response to k, in which the population moves with k. The
real data's symmetric refits cannot be checked against a truth.

Seed 1, `validate.py` (the doubles' X against the truth by read depth;
the shift column for the record):

| reads | genotypes | slope | RMSE | 95% coverage, no shift | with run 2's shift | median width |
|---|---|---|---|---|---|---|
| <= 1e2 | 37,707 | 0.04 | 2.60 | 0.977 | 0.977 | 5.87 |
| 1e2-1e3 | 89,022 | 0.39 | 0.54 | 0.896 | 0.896 | 1.18 |
| 1e3-1e4 | 64,362 | 0.88 | 0.17 | 0.883 | 0.885 | 0.48 |
| 1e4-1e5 | 15,283 | 1.02 | 0.07 | 0.899 | 0.906 | 0.20 |
| > 1e5 | 1,225 | 1.02 | 0.05 | 0.759 | 0.830 | 0.07 |

Run as `python validate.py run_seed1 --depth $F/depth.csv --truth
$F/summary/tfs_summarize_theta_corr_test.csv` with
`F=/gpfs/projects/harmslab/harms/studies/full-sized-sims-3/sim_realistic_s2`
(seed 2's simulation has no summary to take the truth from).

**Decision (user, 2026-10-20):** ship without the shift (decision 1's
fallback in `planning/release-0.5.0.md`); undercoverage of 0.76-0.90 is
acceptable for now. `tfs-fit-two-stage` now has `two_stage.kshift`
(default false): the jackknife and kshift steps are skipped and predict
runs without `--shared_shift`. The usable-genotype threshold is 1,000
reads (decision 5), where the slope reaches 0.88.
