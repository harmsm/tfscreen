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

### Run 2: per-marker midpoints (2026-10-15)

`midpoints.py` fits each deep genotype's within-tube curve (against
M42I/H74A/K84L) in each marker with y = A (1 - s(c)), s a Hill in IPTG
(n bounded to 0.3-4), and compares the 136 singles whose swing is beyond
3 SE in both (`out/midpoints.pdf`).

**Midpoints differ modestly.** pheS's K is a median 1.5x lower than
kanR's (ln K difference -0.42, IQR -0.66 to +0.34, 71% below 0; median K
18 uM against 29 uM).

**Steepness differs a lot.** kanR's curves are switch-like (median n at
the bound of 4; the drop between about 30 and 100 uM). pheS's are broad
(median n 0.93; a gradual decline over about three decades). wt: kanR K
44 uM, n 4; pheS K 18 uM, n 0.90.

**Reading.** One Hill curve per genotype cannot be steep in one marker
and broad in the other, so the shared-X fit follows kanR and flattens
pheS (run 1's 8x shortfall in pheS swings). The markers seem to turn
occupancy into growth through differently shaped maps: kanR like a
threshold in expression, 4CP toxicity graded. The model assumes both are
linear in X.

**An anomaly at 0 IPTG in pheS.** For the singles and for wt the pheS
curve at 0 IPTG is well below its value at 0.1 uM (median single: the
0.1 uM value is about 1.5x the 0 IPTG value). Binding does not change
between 0 and 0.1 uM, so the 0-IPTG pheS+4CP tubes may be off
systematically. They are the gauge point, so this needs settling before
modeling pheS.

**Next.** (1) The 0-IPTG pheS tubes: their t_sel, totals and reads
against the 0.1 uM tubes, and whether the anomaly holds against other
flat references. (2) If it is a tube problem, a per-marker monotone map
from X to growth (for example a Hill or logistic warp per selection
condition, gauged on wt) in the relative fit, tested on the simulations
(which need a nonlinear marker map added) and against binding.

### Run 3: the 0-IPTG pheS anomaly is one row of tubes (2026-10-15)

Tube totals cancel in a within-tube ratio, so the anomaly had to be a
difference in relative growth. Against both flat pure spikes the 421
singles with >= 1e6 reads grew 0.010 per minute faster at 0.1 uM than at 0
in pheS+4CP (the two spikes agreed with each other; against wt the singles
moved -0.002), and not in kanR+kan. Per replicate, one row drove it:
pheS+4CP replicate 2, t_sel 82.

`tube_qc.py` scores every tube's apparent IPTG from its genotype
composition (0 like the 0-IPTG tubes, 1 like the 1 mM tubes). Every
replicate x time row rises smoothly from about 0 to 1.1-1.3 in both
markers except that one:

| pheS+4CP rep 2, t_sel 82 | 0 | 0.1 uM | 1 uM | 10 uM | 30 uM | 1 mM |
|---|---|---|---|---|---|---|
| apparent IPTG | 0.01 | 0.91 | 0.48 | 0.19 | 0.22 | 0.06 |

The tube labeled 0.1 uM looks like a 1 mM tube and the one labeled 1 mM
like a low-IPTG tube; 1 uM is also off. The rows with missing
concentrations (replicate 1 t 200, replicate 2 t 185 and 199,
replicate 3) are shifted by the centering, not swapped. A physical swap
or a sample-sheet label swap would both do this; the lab's records can
say which (the user: tube mixups were cross-checked and unlikely).

Without that row the 0 vs 0.1 uM difference is +0.0010 (was +0.0097).
The rest stands: kanR switch-like (median n at the bound of 4, K 35 uM)
and pheS broad (median n 1.08, K 19 uM) over 69 singles responsive in
both; wt's within-tube swing 0.0125 in pheS (near the monoculture's
0.0145, was 0.018) and -0.0046 in kanR.

**Next.** Check the records for pheS+4CP replicate 2, t_sel 82
(0.1 uM, 1 uM, 1 mM) and exclude or relabel those tubes. Then a
per-marker monotone map from X to growth (`planning/relative-m.md`).

