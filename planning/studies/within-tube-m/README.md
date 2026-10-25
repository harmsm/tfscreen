# Each marker's growth slope m, within tube and absolute

**Question.** The two-stage fit pins each condition's slope m at stage
1's. In the library, does wt's growth against IPTG span that m when
measured against genotypes in the same tube, where tube totals cancel?
And does the absolute growth (through the tube totals) agree?

**Decision it feeds.** Whether stage 1's m, and the linear growth model's
single k per condition, can be the pins for stage 2
(`planning/two-stage-fit.md`), and what the pheS/kanR disagreement in
`planning/studies/x-vs-binding/` is.

## Method

With growth = k + dk + m X and wt gauged to X = 1 at 0 IPTG and 0 at
1 mM, the slope over the selection of ln(counts_wt / counts_ref) is
(dk_wt - dk_ref) + m (X_wt - X_ref). For a reference whose X does not
change with IPTG, the change of that slope from 0 to 1 mM is m: dk, the
reference's X, tube totals and tube offsets all cancel. The references
are the pure spikes H74A/K84L and M42I/H74A/K84L, flat in binding and
expression (user, 2026-10-15; the second with a slightly different lacI
construct).

The slopes come from one weighted fit per condition and reference,
ln ratio = a[IPTG, replicate] + b[IPTG] t_sel, weights from Poisson
counting, covariance scaled by the reduced chi-square (64-740:
overdispersed). The absolute versions fit wt's and the reference's
ln_cfu the same way, unweighted. Early and late m use the first and last
interval of each replicate's three time points.

## How to run

```bash
python within_tube_m.py ../../dev-data/processed --stage1_km ../two-stage-validation/run_real/stage1/km.csv --out out
```

Outputs in `out/` (gitignored): `m_estimates.csv`, `wt_curves.csv`
(wt's within-tube growth against IPTG, minus its value at 1 mM) and
`YYYY-MM-DD_within-tube-m.pdf`.

## Commit

The commit that adds this study.

## Results

### Run 1 (2026-10-15)

m per minute, +- SE:

| | within tube, vs H74A/K84L | within tube, vs M42I/H74A/K84L | wt absolute | wt monoculture | stage 1 (pinned) |
|---|---|---|---|---|---|
| kanR+kan | -0.0051 +- 0.0030 | -0.0046 +- 0.0020 | -0.0152 +- 0.0070 | -0.0141 +- 0.0014 | -0.0148 |
| pheS+4CP | +0.0175 +- 0.0041 | +0.0181 +- 0.0042 | +0.0124 +- 0.0061 | +0.0144 +- 0.0022 | +0.0070 |

**The markers disagree with stage 1 in opposite directions.** Within
tube, kanR's m is a third of the monoculture's and of stage 1's (2.7
and 3.9 SE below the monoculture). pheS's m matches
the monoculture and is 2.5x stage 1's (about 2.7 SE above it). The two
references agree with each other in both markers.

**The flat references' own absolute growth moves with IPTG in the
selection conditions.** Their absolute slope at 1 mM minus at 0 IPTG:

| | H74A/K84L | M42I/H74A/K84L |
|---|---|---|
| kanR+kan | +0.0094 +- 0.0064 | +0.0105 +- 0.0072 |
| pheS+4CP | +0.0082 +- 0.0070 | +0.0096 +- 0.0064 |
| kanR-kan | +0.0037 +- 0.0067 | +0.0003 +- 0.0055 |
| pheS-4CP | -0.0035 +- 0.0058 | -0.0002 +- 0.0050 |

Under the model it is 0 (a flat X, one k per condition). Each is about
1.5 SE, but all four selection entries point the same way: every
genotype, references included, grows about 0.01 per minute faster at
1 mM than at 0 IPTG under kanamycin and under 4CP, and not without
selection. wt's absolute swing is then its within-tube m plus that
shared term, which is why the absolute swing sits near the monoculture
in kanR while the within-tube m does not.

**Reading.** The linear model has one k per condition, so it has no
place for a growth shift that depends on IPTG and is shared by every
genotype in the selection conditions. Fitting the absolute and the
within-tube information with one m forces a compromise that differs by
marker, and since X is gauged on wt, the misfit falls between wt and the
library. The shared term's sign is the same under kanamycin (derepressed
cells grow) and 4CP (derepressed cells die), so a public-good effect of
the resistance enzyme alone does not explain it; an IPTG effect on the
tube totals in selection (OD600 to CFU under stress), or on growth under
selection, would.

**wt's curve.** Within tube, wt's pheS response keeps falling from
0.1 uM to 1 mM and spans about 0.03 per minute from 0.1 uM, where the
monoculture's is flat to 10 uM and drops between 10 and 30 uM. In kanR
the within-tube curve is compressed but has the monoculture's shape.
Early and late intervals agree within their noise.

**Next.** Test a per-(condition, IPTG) k in selection conditions (a
growth shift shared by every genotype in those tubes) against the
current model on the dev data, and see whether stage 1's m then matches
the within-tube m.

### Run 2 setup: stage 1 with a titrant-dependent k (2026-10-15)

`linear` growth now takes `k_titrant_free_scale` (a k shift per
selection condition and nonzero IPTG, prior SD 0.02) and
`k_titrant_loglinear_scale` (a slope in log IPTG per selection
condition, from a tenth of the lowest nonzero concentration, prior SD
0.005), both 0 at no IPTG. The old monoculture model, delta_k = s log(IPTG)
+ b (user, 2026-10-15), is the log-linear form. `stage1_titrant.py`
reruns the two-stage fit's stage 1 with either, four runs: real and
seed 1 (a null check: the simulation has no such term, so the shift
should come out near 0 and m unchanged), each free and log-linear. From
the cluster's two-stage validation directory:

```bash
for r in real seed1; do for m in free loglinear; do sbatch --export=ALL,RUN=$r,MODE=$m /gpfs/home/harms/tfscreen/planning/studies/within-tube-m/stage1_titrant.srun; done; done
```

Compare each `ktitrant_*/stage1/km.csv` with `run_*/stage1/km.csv`
(no term) and with the within-tube m above (kanR+kan about -0.005,
pheS+4CP about +0.018); the shift itself is
`condition_growth_k_titrant[_slope]_auto_loc` in the stage's
`tfs_fit_model_params.npz`.

### Run 2 results (2026-10-15)

The free form was removed from the code after this run.

The four runs failed first (the shift used numpy on the traced
titrant_conc; fixed) and then ran, about 30 minutes each.

**The free form is not identified.** On seed 1, which has no such term,
it put the shift at -0.021 (kanR+kan) and +0.023 (pheS+4CP) per minute at
1 mM and made both m's 2.5x too steep (-0.0365, +0.0369 against
-0.0146, +0.0146 without it). Its shift took wt's own IPTG curve: a k
free at every concentration can stand in for m X_wt(c), so k(c) trades
with m along the gauge. On the real data its numbers mean nothing.

**The log-linear form passes the null and moves kanR, not pheS.**

| | seed 1 m, none -> log-linear | seed 1 shift at 1 mM | real m, none -> log-linear | real shift at 1 mM | real within-tube m |
|---|---|---|---|---|---|
| kanR+kan | -0.0146 -> -0.0146 | +0.0002 | -0.0148 -> -0.0089 | +0.0030 | -0.0049 |
| pheS+4CP | +0.0146 -> +0.0146 | +0.0013 | +0.0070 -> +0.0041 | +0.0021 | +0.0178 |

On the real data it finds a small shift of the expected sign, about a
quarter of the 0.01 the flat spikes' absolute growth suggested, and it
moves kanR's m most of the way to the within-tube value. pheS's m moves
away from it.

**Reading.** A shared IPTG shift is probably real but small and accounts
for part of the kanR discrepancy. It does not account for pheS: within
tube, wt against the flat spikes swings like the monoculture (0.015-0.018
per minute); every fit puts pheS's m at 0.004-0.011.

**Next.** A posterior-predictive check of the within-tube statistic:
stage 1's predicted ln_cfu for wt and the two flat spikes in pheS+4CP,
the same ratio slopes, against the observed. Whether the fit reproduces
the 0.018 swing says whether something in its likelihood outweighs it or
wt's X in stage 1 does not span 1 to 0 as assumed.

### Run 3: posterior-predictive check and the marker ratio (2026-10-15)

`ppc_within_tube.py` (stage 1 folders downloaded to `runs/ppc/`, gitignored;
`tfs-predict-growth` at the MAP, locally) computes the within-tube swing
from the fit's predicted ln_cfu on the same tubes. `marker_ratio.py`
computes each deep genotype's swing in each marker against
M42I/H74A/K84L, model-free.

**Stage 1 does not reproduce the within-tube contrast.** pheS+4CP:
predicted 0.004, observed 0.018 (0.018-0.03 at low IPTG, 3-4 SE).
kanR+kan: predicted -0.009, observed -0.005 (1-2 SE). The log-linear
titrant k changes neither. On seed 1 the fit reproduces the statistic
(both near 0, since the simulation's spikes are not flat), so the check
works. The real fits' count overdispersion floor (inv_r 0.0065, about 8%
per tube at high counts) is 6x the simulation's.

**The pheS-only fit's curves follow from the pinned m.** Read with
m = 0.018, H74A/K84L sits near X 0.4 at both IPTG ends, as its binding and
kanR X say.

**The singles' marker ratio matches wt, not stage 1.** With one X shared
by both markers, each genotype's pheS swing over its kanR swing is
m_pheS / m_kanR:

| | pheS / kanR |
|---|---|
| 903 singles with >= 1e5 reads, weighted line through 0 | -2.2 |
| median over the 372 with a kanR swing > 2 SE | -2.5 |
| M42I/H74A | -2.4 |
| wt | -3.9 (0.018 +- 0.004 over -0.005 +- 0.002) |
| stage 1's pinned m | -0.47 |
| wt monoculture | -1.03 |

wt sits with the library. Every within-tube measure agrees, and stage 1's
m's are about 5x off in ratio: kanR's m about 3x too steep, pheS's about
2x too shallow.

**Reading.** The within-tube ratio is the information free of tube totals.
The fit's absolute scale comes from the OD600-derived totals, which tie
each condition's k and m to the population's growth; the per-tube level
offsets cannot absorb a rate difference across IPTG, so m does. The
monoculture is OD-based too. An OD-to-CFU conversion that drifts with
IPTG or cell state under selection (filamentation under kanamycin, say)
would do this, as the staged-MAP study already listed. Under the wt gauge,
m is identified by wt's contrast with the library without the totals, so
a fit that takes m from relative information alone is possible in
principle; the IPTG-structured offset mode is a warning that freeing the
totals has not gone cleanly before.
