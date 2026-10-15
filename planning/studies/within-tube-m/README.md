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
