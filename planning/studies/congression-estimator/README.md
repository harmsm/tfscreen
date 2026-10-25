# Congression estimator study

**Question.** In the observable-level congression mixture, how should the fit
evaluate a genotype's congressed-cell term `E_S exp(G_{g,S}(t))`, the average
over random co-resident sets S of the congressed cells' exponentiated
log-growth? And do co-residents need per-origin backgrounds?

**Decision it fed.** Step 3.0 of
[`planning/congression-physics-plan.md`](../../congression-physics-plan.md):
the fit uses fixed Monte Carlo quadrature (K per-genotype co-resident sets,
stratified by plasmid count, drawn once), with a pooled co-resident
background. See the plan's "Fit design (step 3)".

## How to run

From a scratch directory (outputs are written to the working directory):

```bash
python /path/to/tfscreen/planning/studies/congression-estimator/study.py
```

An optional argument replaces the simulate config. Runtime is about 2.5 min
and peak memory about 1.3 GB.

Outputs: `study_summary.csv` (one row per dk scale x estimator x subset) and
`study_by_time.csv` (RMS per growth condition and selection time). Not
committed.

## Inputs

- `examples/simulate/simulate_config.yaml` (default) for the library, theta
  prior, growth conditions and growth transition: 843 genotypes, thermo theta
  prior, memory growth transition, 48 growth conditions, t_sel 95 to 200 min.
- Overridden in the script: `library_mixture` = the real design
  (single-1 100, single-2 100, double-1-2 1000, spiked 1),
  `transformation_poisson_lambda` = 0.357, seed 11.
- Cell rules as in `simulate/cell_rules.py`: max theta, activity from the max
  plasmid, dilution dk.

## Method

For each bulk genotype g and growth condition, the observable relative to
`ln_cfu0` is

```
L_g = log[ w_clean exp(G_clean) + sum_o w_o E_{S~o} exp(G_{g,S}) ]
```

with co-residents S drawn as N ~ Poisson(lambda) | N >= 1 plasmids from
sub-library o's plasmid distribution (each sub-library is transformed
separately). "Exact" enumerates N = 1 over every genotype in the sub-library
and uses Monte Carlo for N >= 2 (16% of congressed cells). dk_geno deviations
are scaled by s; at s = 1 (the sim prior) dk_geno has SD 0.068/min, with 5th
to 95th percentiles -0.13 and 0.014/min. s = 0.015 corresponds to the bench
measurement for the current library (SD about 0.001/min).

Estimators of each congressed term, plugged into the same mixture:

| name | estimator |
|---|---|
| `rate_avg` | `exp(E_S[G])` |
| `cumulant2` | `exp(E_S[G] + Var_S[G] / 2)` |
| `n1_only` | exact, but every congressed cell has exactly one co-resident |
| `mcK` | K fixed co-resident sets per sub-library, shared by all genotypes |
| `pooled` | exact, but co-residents from the pooled bulk library |
| `current` | today's fit: one trajectory, theta -> `E[max(theta_g, M)]` over an unweighted all-genotype background, own dk (reference) |
| `none` | no congression (reference) |

Error = approximate minus exact `L_g`, in natural-log `ln_cfu` units, over
genotype x condition cells within 7 ln units of wt ("detectable"). `mcK`
pools 10 independent draw sets.

## Commit

Produced on `abec95f`; reproduced exactly on `fc0f2b9` (no simulator changes
in between).

## Results

RMS error (detectable):

| estimator | s = 0 | s = 0.015 | s = 0.03 | s = 0.1 | s = 0.3 | s = 1 |
|---|---|---|---|---|---|---|
| none | 0.041 | 0.053 | 0.097 | 0.287 | 0.758 | 1.507 |
| current | 0.019 | 0.041 | 0.093 | 0.293 | 0.770 | 1.524 |
| rate_avg | 0.006 | 0.006 | 0.008 | 0.036 | 0.206 | 1.086 |
| cumulant2 | 0.000 | 0.000 | 0.001 | 0.025 | 0.509 | 12.599 |
| n1_only | 0.003 | 0.004 | 0.010 | 0.030 | 0.080 | 0.127 |
| mc1 | 0.032 | 0.043 | 0.049 | 0.104 | 0.391 | 1.113 |
| mc2 | 0.018 | 0.028 | 0.038 | 0.075 | 0.207 | 0.515 |
| mc4 | 0.016 | 0.016 | 0.032 | 0.072 | 0.181 | 0.379 |
| mc8 | 0.013 | 0.017 | 0.020 | 0.037 | 0.136 | 0.360 |
| mc16 | 0.008 | 0.011 | 0.010 | 0.028 | 0.072 | 0.187 |
| mc32 | 0.005 | 0.006 | 0.010 | 0.019 | 0.064 | 0.142 |
| pooled | 0.002 | 0.002 | 0.002 | 0.004 | 0.008 | 0.024 |

95th percentile |error| (detectable):

| estimator | s = 0 | s = 0.015 | s = 0.03 | s = 0.1 | s = 0.3 | s = 1 |
|---|---|---|---|---|---|---|
| none | 0.099 | 0.105 | 0.121 | 0.376 | 1.671 | 3.882 |
| current | 0.048 | 0.064 | 0.108 | 0.414 | 1.707 | 3.900 |
| rate_avg | 0.014 | 0.016 | 0.018 | 0.057 | 0.460 | 2.418 |
| mc16 | 0.021 | 0.028 | 0.020 | 0.044 | 0.147 | 0.391 |
| mc32 | 0.013 | 0.013 | 0.020 | 0.034 | 0.113 | 0.296 |
| pooled | 0.002 | 0.002 | 0.003 | 0.006 | 0.016 | 0.053 |

Mean error (detectable) of `mcK` is within ±0.01 at s <= 0.1 and within ±0.09
at s = 1, for every K >= 2; `rate_avg` is biased low (-0.15 at s = 0.3,
-0.76 at s = 1).

Error grows with selection time. At s = 0.3, `rate_avg` RMS rises from 0.20
to 0.31 over kanR t_sel 145 to 200 min, and `mc16` from 0.07 to 0.09.

## Conclusions

- Once dk varies, the congression effect is large (`none`), and today's fit
  rule removes almost none of it: averaging theta cannot represent a mixture
  of growth rates.
- `rate_avg` is adequate only for a narrow dk spread and worsens with time;
  `cumulant2` fails on heavy dk tails.
- `mcK` has near-zero mean error at every spread, with RMS falling roughly as
  1/sqrt(K). It is the only estimator that holds across the dk range.
- A pooled co-resident background is adequate. Caveat: in the sim prior,
  singles and doubles have similar theta/dk distributions; real doubles may be
  more often broken.
- At the measured spread for the current library (s ~ 0.015), the whole effect
  is modest (0.053 RMS, 0.105 p95) and every estimator except small-K MC is
  accurate.

## Limits

- One simulated library (843 genotypes, 2 sites) and one theta prior.
- This study shares draw sets across genotypes; the adopted fit design uses
  per-genotype sets, which gives the same per-genotype error with errors
  independent across genotypes.
