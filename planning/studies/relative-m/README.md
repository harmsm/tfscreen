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

### Run 4 setup: a warp for one marker (2026-10-15)

`linear` growth now takes `m_warp` per condition: that condition's growth
follows f(X) = (s(X) - s(0)) / (s(1) - s(0)), s = sigmoid(a (X - x0)),
with log a ~ Normal(1, 1.5) and x0 ~ Normal(0.5, 0.5) per warped
condition; f(0) = 0 and f(1) = 1 keep the wt gauge, and f is linear as
a -> 0.

Which marker to warp: a monotone map of X changes values, not where in
IPTG they change, so the broad marker (pheS) must carry X itself and the
switch-like one (kanR) a steep map of it. A threshold x0 below 1 also
predicts kanR responding at higher IPTG than pheS, as measured (35 against
19 uM). The reverse is run as a check, and seed 1 (linear maps) as a null.

`stage1_warp.py` is the relative stage 1 (offsets free, k pinned, m free)
with the warp, leaving out the pheS+4CP replicate 2, t_sel 82 row on the
real data until the lab's records settle it (run 3). Runs:

| run | warped | tubes left out |
|---|---|---|
| warp_real_kanR | kanR+kan | pheS+4CP rep 2, t_sel 82 |
| warp_real_pheS | pheS+4CP | the same |
| warp_real_none | none | the same (the reference without a warp) |
| warp_seed1_kanR | kanR+kan | none |

Pass: on the real data, the kanR warp lowers the loss against both the
no-warp and the pheS-warp runs, its fit follows the within-tube swings in
both markers (`swing_ppc.py`), and m's pheS/kanR ratio comes near the
within-tube one; on seed 1 the warp stays close to linear (small a or x0
near the middle) and m stays on the truth.

### Run 4 results (2026-10-16)

The three real runs first failed (sbatch `--export` split the
`--drop_tubes` triple on its commas; now `:`-separated, and the stage 1
table is moved into place only when complete). `compare_warp.py`:

| real (rep 2, t 82 left out) | kanR warped | pheS warped | no warp |
|---|---|---|---|
| final exact loss | 806,900 | 816,700 | 814,900 |
| m kanR+kan / pheS+4CP | -0.0120 / +0.0088 | -0.0102 / +0.0041 | -0.0174 / +0.0095 |
| pheS/kanR (within tube -2.7) | -0.74 | -0.41 | -0.55 |
| wt's pheS swing predicted (observed 0.0125) | 0.0060 | 0.0031 | 0.0047 |
| singles' pheS swing, predicted / observed (median) | 0.36 | 0.28 | 0.22 |
| warp a, x0 | 2.96, 1.35 | 3264, 0.39 | |

The kanR warp lowers the loss by about 8,000 nats for two parameters; the
pheS warp collapsed to a step at X = 0.39 and fits worse than none. The
fitted kanR map, f(0.2, 0.4, 0.6, 0.8) = 0.06, 0.16, 0.33, 0.60, holds
kanamycin growth down only near full repression: a little derepression
already gives much of the resistance. On seed 1 the warp is nearly linear
(a 0.84, x0 1.07; f(0.2..0.8) = 0.18, 0.38, 0.58, 0.79) and m stays near the
truth (1.07-1.09 of it, against 1.05-1.06 without), so the null passes.
None of the real runs converged by the 100,000-step cap.

**Reading.** The kanR warp is real and worth keeping, but it closes little
of the pheS gap: pheS's swings are still predicted about 3x too small and
the ratio reaches -0.74. Something else in the likelihood holds pheS's m
down.

**Next.** Likelihood attribution: refit stage 1 with the kanR warp and m
pinned at the within-tube values (kanR -0.0046, pheS +0.0125), both runs
to convergence, and compare the per-observation log likelihood with the
free fit by marker, genotype class and tube.

### Run 5 setup: likelihood attribution (2026-10-16)

`attribute.py` traces a stage 1 at its MAP and keeps the count log
likelihood of every growth observation (the `growth_obs` site, mapped back
to its row through the growth table's index columns) and the log prior of
every latent site, then sums the difference between two fits by marker,
genotype class, IPTG and tube. Checked on run 4 (kanR warp against no
warp): the warp's 8,100 nats come from the pheS+4CP singles (+9,700), with
the kanR conditions losing about 1,900; wt's 40 pheS+4CP rows move by 15
nats, so the gauge genotype carries almost no weight in the likelihood.

Two runs, both with the kanR warp, without the replicate 2, t_sel 82 row
and with the step cap raised to 500,000:

| run | m |
|---|---|
| warp_real_attrfree | free |
| warp_real_attrpinned | pinned at the within-tube values: kanR+kan -0.0046, pheS+4CP +0.0125 |

`attribute.py` on the pair says what loses likelihood when the selection
markers' m equal wt's own within-tube swings.

### Run 5 results (2026-10-16)

Both runs converged: free at 50,000 steps (loss 806,876; m kanR+kan
-0.0119, pheS+4CP +0.0088, as run 4), pinned at 204,000 (810,725).

Pinning m at the within-tube values costs 3,689 nats of count log
likelihood and 155 of log prior (mostly the log K population SD and
offsets), so the data, not the priors, prefer the free m. By group
(pinned minus free):

| | kanR+kan | kanR-kan | pheS+4CP | pheS-4CP |
|---|---|---|---|---|
| wt | -833 | -180 | -610 | -105 |
| singles >= 1e6 reads | -678 | -8 | -1,297 | -235 |
| singles 1e5-1e6 | -18 | -11 | +254 | -137 |
| spikes | +42 | +7 | +23 | +7 |
| total | -1,459 | -180 | -1,593 | -457 |

wt alone loses 1,729 nats over its rows, in every condition including
both no-selection controls, and in most series the loss grows with time
(pheS+4CP replicate 1: -99, -120, -175 over t_sel 94, 111, 125). The
pinned fit's kanR warp went step-like (a 11, x0 -0.05) and its X
population locations moved (X_low 0.87 -> 0.53).

Against the geometric mean of all deep singles instead of the flat spikes,
the arithmetic is consistent (wt - spike = (wt - library) - (spike -
library)), so the spikes are not the odd ones; the library on average
swings about 0.83 of wt in kanR and 0.55 in pheS, which one shared X
cannot do.

**Reading.** The model can match wt's swing against the flat spikes or
wt's levels against the rest of the library, not both. A wt whose growth
relative to the library is not constant over the selection (a lag, a
transient, a dk that changes with time) would do this; wt losing
likelihood in the no-selection tubes, where m hardly matters, points the
same way.

**Next.** wt's observed and predicted ln(wt / library) against t_sel per
condition and IPTG under both fits: a time trend or a level.

### Run 6: wt against the library over time (2026-10-16)

`wt_trajectory.py` plots ln(wt / library) per tube (the library: the
geometric mean of the 881 deep singles in every tube) against t_sel for
each condition and IPTG, observed and predicted by run 5's free and
pinned fits (`out/*_wt-trajectory.pdf`, `out/wt_trajectory_residuals.csv`).

**Not a wt transient.** The free fit follows wt against the library in
every condition: mean residuals within +-0.06, trends within +-0.0025 per
minute. The pinned fit misses by 0.3-0.8 in level and about -0.01 per
minute in trend everywhere, the no-selection controls included: pinning m
at the spike-based values makes the fit move the whole library's growth
against wt, which the controls reject.

**Not bulk origin.** wt's within-tube swing against weak binders that are
pure spikes and against ones that come 69% from the bulk is the same
(kanR+kan: -0.0051, -0.0046 pure; -0.0052 K84L; pheS+4CP +0.0093, +0.0125
pure; +0.0087 K84L, +0.0074 H74A), so congression is not the difference.

**The scale of X.** A within-tube swing is m (wt's X range - the
reference's X range); turning it into m needs a reference whose X does not
change. Each m so far rests on a different anchor:

| anchor | kanR+kan | pheS+4CP |
|---|---|---|
| tube totals (absolute stage 1) | -0.0148 | +0.0070 |
| priors and Hill shape (relative fit, kanR warp) | -0.0119 | +0.0088 |
| weak binders flat (binding and expression) | -0.0046 to -0.0052 | +0.0087 to +0.0125 |

pheS roughly agrees once the swapped row is out; kanR differs about 2.5x.
The free fit reconciles kanR by giving the weak binders a partial kanR
response (X range about 0.4-0.6 of wt's), against their flat binding. But
kanR-alone X at the absolute m matched binding theta at 2 uM with no
correction (`planning/studies/x-vs-binding/`), which a 2.5x smaller m
would undo. So binding supports the larger kanR m and the weak binders'
flatness the smaller one.

**Decision needed** (user): which external fact anchors kanR's scale:
binding (calibrate X to the spikes' 2 uM curves), flat weak binders (hold
the pure weak-binder spikes' X constant), or neither for now (report
kanR's scale as uncertain over that range). Knowing whether weak binders'
kanR growth can respond to IPTG at flat occupancy would settle it.

