# Joint stage 1 on the real screen

**Question.** With the recipe from `planning/studies/joint-fit-sim/`
(absolute theta from `hill_geno`, binding at weight 1, the plain model
first and the full model started from it, 75 co-resident sets), what do
the real data say: is kanR's growth a switch-like function of occupancy
and pheS's graded, what is the in vivo titrant scale s, which binding
genotypes disagree with their growth (M42I), and does the full model
explain growth and binding better than the plain one?

**Decision it feeds.** `planning/joint-fit-return.md` step 6, and through
it step 7 (stage 2 with the full library) and which model pieces stay.

## Inputs

- Growth: the recipe's growth table (`dev-data/real_fit/recipe/tfs_growth.csv`
  on the cluster), wt, spikes and singles (`two_stage.stage1_genotypes`),
  with the probably swapped pheS+4CP replicate 2, t_sel 82 row left out
  (`planning/studies/relative-m/`, tube QC; the lab judged the swap
  probable, and the 1 uM tube is ambiguous, so the row is dropped rather
  than relabeled).
- Library: `dev-data/processed/library_config.yaml`.
- Binding: the 2026-10-15 table through `tfs-process-binding` at 2 uM:
  110 points over seven genotypes (wt, M42I, H74A, K84L, M42I/H74A,
  H74A/K84L, G65H), theta_std from the pooled replicate SD (0.13, wt's
  day-to-day scatter; the mutants have one replicate each). D88A, V95N and
  the other spikes have no 2 uM data. The lab's data stay out of the
  repository: the table is made locally in `planning/dev-data` (gitignored)
  and copied to the cluster.

## Model

`stage1_real.py`: `hill_geno`; the mixture with the homodimer rule, lambda
prior 0.38 +- 0.15, co-resident sets 50 20 5 (`--congression_sets`,
joint-fit-sim run 9); binding at weight 1; counts with level tube offsets
(SD 0.17); log(n) population SD 0.5; one k and m per condition with loose
priors. `plain`: linear growth, s = 1, no offsets. `full`: warps on both
selection conditions, s, and one offset per binding genotype; started from
the plain fit. The Laplace uses a Hessian chunk of 8 (75 sets ran the
default 64 out of GPU memory on the simulation).

## What to expect from the simulations

The real data have fewer binding anchors than the simulations: wt and
five spikes at 2 uM (the simulations had wt and eight) plus one in-library
genotype (G65H). At 0 to 5 extra binding singles the congression
simulation put stage 1's m 17-21% steep and the growth-only singles'
theta RMSE at 0.046-0.056, intervals 3-6 times too narrow
(`planning/studies/joint-fit-sim/`, runs 8 and 10). Treat the real
fit's theta intervals as provisional; m, the warps, s and the offsets are
the results here.

## How to run

Locally, make the binding table (already done on 2026-10-18):

```bash
cd planning/dev-data && tfs-process-binding binding/2026-10-15_binding.csv --out_prefix binding/processed/2026-10-15_binding_2uM
```

Copy it to the cluster:

```bash
ssh harms@login2.talapas.uoregon.edu 'mkdir -p /gpfs/projects/harmslab/harms/studies/dev-data/binding'
```

```bash
scp planning/dev-data/binding/processed/2026-10-15_binding_2uM.csv harms@login2.talapas.uoregon.edu:/gpfs/projects/harmslab/harms/studies/dev-data/binding/
```

Verify:

```bash
ssh harms@login2.talapas.uoregon.edu 'wc -l /gpfs/projects/harmslab/harms/studies/dev-data/binding/2026-10-15_binding_2uM.csv'
```

(111 lines.) Then on the cluster, the plain fit and the full fit started
from it:

```bash
mkdir -p /gpfs/projects/harmslab/harms/studies/joint-fit-real && cd /gpfs/projects/harmslab/harms/studies/joint-fit-real && R=/gpfs/home/harms/tfscreen/planning/studies/joint-fit-real/stage1_real.srun && j=$(sbatch --parsable --export=ALL,MODEL=plain,OUT=plain --output=plain.out $R) && sbatch --dependency=afterok:$j --export=ALL,MODEL=full,OUT=full,INIT=plain --output=full.out $R
```

Then locally, after downloading each fit's `tfs_params_*.csv`,
`tfs_pred_theta*.csv` and convergence tables into `runs/`:

```bash
python report.py runs/plain runs/full --binding ../../dev-data/binding/processed/2026-10-15_binding_2uM.csv --out out
```

## Results

### Run 1 (2026-10-18): 2 uM binding

Both fits stopped at the epoch cap still descending slowly (about 20 nats
over the last five windows at step 1e-5); the Laplace ran (chunk 8).
Final MAP losses: plain 813,484, full 804,973.

| | plain | full |
|---|---|---|
| m kanR+kan (within-tube -0.0046) | -0.0280 | -0.063 [-0.089, -0.035] |
| m pheS+4CP (within-tube +0.0125) | +0.0171 | +0.0156 |
| kanR warp a / x0 | - | 8.9 / 0.81 |
| pheS warp a / x0 | - | 6.6 / 0.52 |
| s | (1) | 0.53 [0.23, 1.26] |
| lambda (prior 0.38 +- 0.15) | 2.10 | 0.82 |
| offset scale tau | - | 0.34 |

Binding offsets (full): G65H -0.18, H74A -0.23, H74A/K84L -0.27, K84L
-0.28, M42I -0.07, M42I/H74A -0.20, wt -0.17: every one negative, and in
the plain fit every genotype's binding sits below its fitted curve (mean
residual -0.03 to -0.26).

wt shows why. Its 2 uM binding falls from 0.75 at 0 IPTG to about 0 by
0.01 mM (midpoint near 0.003 mM); the curve growth wants has its midpoint
near 0.03 mM and a floor of 0.15-0.24. An IPTG scale of about 0.1 would
reconcile the two for wt; the fit keeps s near 0.5 and leaves wt's binding
misfit (RMS residual 2.9-4.0 measurement SDs), since growth's rows
outweigh eighteen binding points. The offsets then absorb a common level
difference (a hierarchical offset with mean zero is the wrong shape for
it), and lambda runs to 0.8-2.1 against the measured 0.36 to absorb the
rest. So m, the warps and lambda here are not results.

The 2 uM binding and growth disagree by more than s plus per-genotype
offsets can express. This matches the open question of
planning/joint-fit-return.md (wt's IPTG midpoint matched 20 uM binding,
the mutants' levels 2 uM): the in vivo occupancy may correspond to more
than 2 uM of repressor.
