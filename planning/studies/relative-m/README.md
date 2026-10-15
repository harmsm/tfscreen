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

### Run 1 (2026-10-15)

`analyze.py` and `swing_ppc.py` (the fit's per-genotype within-tube swings
against the observed, after `tfs-predict-growth` at the MAP, locally).

**The seeds pass; the real data do not move.**

| | relative m | absolute m | within tube | pheS/kanR (relative, absolute) |
|---|---|---|---|---|
| real kanR+kan | -0.0177 | -0.0148 | -0.0049 | |
| real pheS+4CP | +0.0086 | +0.0070 | +0.0178 | -0.49, -0.47 |
| seed 1, m / truth | 1.06, 1.05 | 1.06, 1.04 | | -1.00, -1.00 |
| seed 2, m / truth | 1.05, 1.03 | 1.05, 1.02 | | -0.99, -0.99 |

On the simulations the relative fit recovers the truth as well as the
absolute one. On the real data freeing the offsets left the ratio at
-0.49, so the tube totals are not what sets the fit's m, and the OD600
explanation of `planning/studies/within-tube-m/` does not hold. The real
run had not converged at its 100,000-step cap (still descending, not by
enough to close a 5x gap); the seeds converged.

**The fit discards pheS's IPTG response.** Across 903 deep singles, the
fit's predicted swing against the observed: kanR+kan slope 0.63 (r 0.64,
median 0.0037 predicted, 0.0045 observed); pheS+4CP slope 0.29 (r 0.59,
median 0.0020 predicted, 0.0165 observed). The data's pheS/kanR ratio is
-2.2, the fit's -0.42; wt 0.004 predicted against 0.018 observed.

**Reading.** The constraint is the X shared by both markers: one Hill
curve per genotype. In the library, wt's within-tube pheS response starts
at about 0.1-1 uM IPTG and kanR's at 10-100 uM
(`../within-tube-m/`, run 1), so one curve cannot fit both; the fit
follows kanR and shrinks pheS's m. That would make pheS growth a strongly
nonlinear function of occupancy (a little derepression is already toxic
in 4CP). wt's monoculture pheS curve does not drop early, so if it holds
it is specific to the library.

**Next.** Model-free: an IPTG midpoint for each deep single's within-tube
curve in each marker. pheS midpoints systematically below kanR's would
mean the markers read occupancy differently (a per-marker map from X to
growth). Steps 2-6 of `planning/relative-m.md` wait on it.
