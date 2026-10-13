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

## Commit

The commit that adds this study.

## Results

Pending.
