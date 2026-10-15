# m from relative information only

**Question.** With the per-tube level offsets free, so that the count
likelihood keeps only relative abundances within each tube, does stage 1's
m land on the within-tube values on the real data and on the truth on the
simulations? (`planning/relative-m.md`, step 1.)

**Decision it feeds.** Whether the two-stage fit should take m (and X)
from relative information only, dropping the tube totals and the k shift
(`planning/relative-m.md`, steps 2-6).

## Design

`stage1_relative.py` reruns the two-stage fit's stage 1 (wt, spikes and
singles with the congression mixture) with:
- `sigma_fixed=10` for the level offsets (nearly free; they absorb each
  tube's total and any growth shift shared by a tube);
- k pinned per condition at the run's current stage 1 (it drops out of the
  likelihood, so its value does not matter);
- m free under Normal(0, 0.05), wider than the absolute stage 1's 0.01 so
  the prior cannot hold pheS's m near 0.

Runs: real, seed 1, seed 2.

## How to run

From the cluster's two-stage validation directory (each needs its
`run_*/stage1/km.csv`):

```bash
for r in real seed1 seed2; do sbatch --export=ALL,RUN=$r /gpfs/home/harms/tfscreen/planning/studies/relative-m/stage1_relative.srun; done
```

Writes `rel_{run}/stage1/`.

## Pass

- Seeds: m within about 5% of the truth (the absolute stage 1's m / truth
  was 1.04-1.06).
- Real: m near the within-tube values (kanR+kan about -0.005, pheS+4CP
  about +0.012 to +0.018; ratio about -2.3).
- The within-tube posterior-predictive check
  (`../within-tube-m/ppc_within_tube.py`) reproduces the observed
  contrast.
- No held direction in the Laplace beyond what stage 1 has now.

## Commit

The commit that adds this study.

## Results

Pending.
