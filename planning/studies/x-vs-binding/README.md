# Growth X against in vitro binding theta

**Question.** Does the two-stage fit's growth X follow in vitro binding
theta for the genotypes we measured, and does it follow it the same way
in each selection marker? Which X level does binding support when fits
disagree (the k-shift refits moved every mutant by +0.3 against wt,
`planning/studies/two-stage-validation/`)?

**Decision it feeds.** How to replace the two-stage fit's k shift
(`planning/two-stage-fit.md` step 4) and whether the pheS disagreement is
growth physics or a fit problem (`planning/offset-mode-growth-transition.md`).

## Inputs

- Binding: `planning/dev-data/binding/2026-10-15_binding.csv` (gitignored,
  C10). Nine genotypes, theta per well normalized to day-of internal
  controls, mostly at 2 and 20 uM repressor dimer; wt at 0.02-120 uM. A
  work in progress (more replicates coming).
- Growth X: the real run of `planning/studies/two-stage-validation/`
  (`run_real/`): stage 2 (both libraries), `consistency/` (kanR alone)
  and `consistency_pheS/` (pheS alone), all with k and m pinned at
  stage 1.
- wt monoculture rates and the processed growth tables:
  `planning/dev-data/processed/`.

Spikes (user, 2026-10-15): wt, M42I, H74A, K84L, M42I/H74A, M42I/K84L,
H74A/K84L, M42I/H74A/K84L and D88A, mixed evenly and spiked into the
growth culture at single tile 1 : single tile 2 : doubles : spikes =
100 : 100 : 1000 : 1, as `processed/library_config.yaml` has it. Of the
binding genotypes, G65H and V95N are bulk only. All nine have at least
1e5 reads.

## How to run

```bash
python x_vs_binding.py ../../dev-data/binding/2026-10-15_binding.csv ../two-stage-validation/run_real ../../dev-data/processed --out out
```

Binding theta goes onto the X gauge through wt at the same protein
concentration, `(theta - theta_wt(1 mM)) / (theta_wt(0) - theta_wt(1 mM))`.
Outputs in `out/` (gitignored): `x_vs_binding.csv`, `summary.csv`,
`wt_rates.csv` and `YYYY-MM-DD_x-vs-binding.pdf` (per-genotype curves, X
against gauged theta, wt's rates).

## Commit

The commit that adds this study.

## Results

### Run 1 (2026-10-15)

**At 2 uM protein, kanR's X follows binding with no correction.** Mutant
rows (6 genotypes at 2 uM, 8 at 20 uM):

| protein | fit | X - gauged theta, mean | rms | affine rms | r |
|---|---|---|---|---|---|
| 2 uM | kanR alone | -0.03 | 0.12 | 0.09 | 0.89 |
| 2 uM | both libraries | -0.10 | 0.15 | 0.09 | 0.89 |
| 2 uM | pheS alone | -0.20 | 0.32 | 0.24 | 0.67 |
| 20 uM | kanR alone | -0.16 | 0.32 | 0.27 | 0.63 |
| 20 uM | both libraries | -0.23 | 0.36 | 0.27 | 0.64 |
| 20 uM | pheS alone | -0.33 | 0.44 | 0.30 | 0.70 |

The 2026-10-01 comparison (`planning/dev-data/real_fit`, theta
interpolated to 10 uM, the old full-library X) found no affine map (rms
0.30). H74A, H74A/K84L, K84L, M42I/H74A and G65H now sit on their 2 uM
curves. D88A and V95N, measured only at 20 uM, match that.

**wt's curve has the 20 uM midpoint.** Its growth X falls at about
30 uM IPTG, as its 20 uM binding does, where its 2 uM binding falls at
about 3 uM. The mutants' levels match 2 uM. One protein concentration
does not reproduce both.

**M42I** grows as if less repressed than its 2 uM binding at 0-1 uM IPTG
(X 0.3-0.4 below), the same direction as before.

**Binding supports stage 2's level, not the refits'.** The k-shift
control moved every mutant's X by about +0.32 over stage 2, which would
take the 2 uM mean difference from -0.10 to about +0.2, against kanR
alone's -0.03.

**The pheS-only fit is degenerate at wt.** Its wt X falls from 1 to 0
between 0 and 1 uM IPTG. wt's own library trajectories put the drop
between 10 and 30 uM, as in the monoculture. So the pheS-only X is not a
usable reading of the pheS response, and the compression reported in
the validation README needs a healthy pheS-only fit before it means
anything.

**wt's library rates are too noisy to settle the pheS question.** The
slope of wt's ln_cfu over the selection, per replicate, scatters by
0.006-0.015 per minute (tube totals from single OD600 readings). In
pheS+4CP wt's library rate averages 0.024 at 0-3 uM IPTG and 0.015 at
30 uM-1 mM, against the monoculture's 0.016 and 0.004: a smaller drop in
the library, but at about one SE per IPTG level.

**Next.** A pheS-only fit that does not collapse wt's curve (hold wt's
log K or n near kanR's, or the log(n) population SD), then the marker
comparison again. Repeat with the replicated binding data.
