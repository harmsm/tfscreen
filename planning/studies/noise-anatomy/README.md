# Study 0b: anatomy of read-count noise

**Status: provisional (2026-09-26).** Run on the 2026-07-23 snapshot, whose
counts came from a FASTQ processor that assigned every unknown read to wt
and used a very stringent Q cutoff; a new processing batch is running.
Affected here: section 3 (ratios to wt) and the `__unknown__` shares, which
are wrong (unknown reads sat in wt). Expected to survive, to be re-checked:
the low-count prevalence (may improve with a looser Q cutoff), the
composition offset, and the 5-18x excess count variance (uniform read
thinning does not create overdispersion). Next analysis: the variance
multiplier by condition and IPTG column against the tube's fold-growth
since the split; it rises with growth if founder sampling dominates and is
flat if a template bottleneck does.

## Question

How much of the real data sits in the low-count regime, how large is the
per-tube composition offset `u_s`, and what does per-tube count noise look
like compared with Poisson? Feeds roadmap steps 4 (simulator), 6 (`u_s`
prior) and 7 (the count likelihood's dispersion model)
(`planning/analysis-roadmap.md`).

## How to run

```bash
python noise_anatomy.py ~/work/programming/git-clones/tfscreen/dev/real-2026-07-23 out/
```

Inputs from `planning/studies/step0-data/`. About 15 s.

## Method

Each replicate x condition x IPTG concentration has three tubes
(timepoints). A per-genotype straight line in `t_sel` through three points
leaves one residual degree of freedom; its projection `z` on the unit
vector orthogonal to `[1, t]` carries whatever straight-line growth does not
explain. The shared composition offset is the mean of `z` over the 494
genotypes with at least 1000 reads in every tube; its variance is corrected
for the finite number of genotypes. Count noise compares `E[z^2]` with the
Poisson expectation `sum_i v_i^2 / c_i`, on ln frequency with the shared
offset removed, using lines with at least 100 reads in every tube (below
that, requiring nonzero counts truncates the draws and ln counts are
biased).

## Commit

Scripts as in this directory on top of `b949095`.

## Results (2026-09-26)

**1. The library is overwhelmingly low-count.** 113 tubes, median 17.9M
reads each (listed genotypes); `__unknown__` 0.4-3% of reads.

| class | median reads per tube | 0 reads | < 5 | < 20 |
|---|---|---|---|---|
| doubles (kan+ / 4CP+) | 4 / 3 | 23% / 33% | 53% / 55% | 79% / 76% |
| singles | ~5,300 | 0.5-1% | 1-3% | 1.5-4.5% |
| spiked (5 genotypes) | 29,000-56,000 | 0 | 0 | 0 |

Doubles are about 97% of rows. At 3-5 reads, `ln(counts + pseudocount)` is
far from Gaussian and the pseudocount dominates.

**2. The per-tube composition offset is small:** SD 0.018 ln units, shared
by every genotype in a tube (consistent with the earlier denominator review,
0.026). Correlation with the tube's `__unknown__` share: -0.34. Per-genotype
residual SD for well-measured genotypes: 0.047.

**3. Per-tube count noise is 5-10 times Poisson at high counts.** On ln
frequency, lines with >= 100 reads:

`E[z^2] = 0.0055 + 8.4 x Poisson` (pooled): a variance multiplier of about
8 on top of counting, plus a floor of about 0.07 ln units. By condition the
split between the two trades off (multiplier 3-5 with floors 0.07-0.15)
because each line has one residual degree of freedom; the robust statement
is that at 100-3000 reads the observed variance is 5-18 times the Poisson
variance. The ratio to wt gives the same picture (13 times Poisson at ~500
reads).

**4. Low counts are mostly low abundance, not crashes.** Each double's
row in a selective tube, classified by the genotype's median reads in the
no-drug tubes of the same library and replicate (its abundance before
selection):

| reads without drug | share of selective double rows | of which < 5 reads (kan+ / 4CP+) |
|---|---|---|
| < 5 | 59% | 74% / 88% |
| 5-20 | 22% | 19% / 38% |
| 20-100 | 13% | 2% / 9% |
| >= 100 | 6% | 0% / 0.2% |

About 88% of the low-count (< 5) double rows belong to genotypes that are
already under 5 reads without the drug. Of the 69,245 doubles with >= 20
reads without the drug, 39% dip below 5 reads in at least one selective
tube, but only 0.1% do so in half of their selective tubes. The double
sub-library's design gives about 1.5 transformants per double
(`transform_sizes` 300,000 for ~200,000 doubles), so a double's abundance
is set early and varies widely between doubles.

## What it means

- **The count likelihood (roadmap step 7) is the main event, not a
  refinement.** The doubles, which carry the mutant cycles, are almost all
  at 0-20 reads.
- **Today's `ln_cfu_var` (binomial counting) understates the noise 5-10
  fold** for well-measured genotypes, so fits are overconfident by roughly
  2-3x in SD. This is a leading candidate for the low theta coverage seen
  in simulation-based calibration being worse on real data.
- **The dispersion model must allow variance proportional to the mean**
  (NB1 / quasi-Poisson, `var = phi * mu`), not only a constant-CV floor
  (NB2, `var = mu + mu^2 / r`). Variance scaling with 1/reads at 5-10x
  Poisson is the signature of a bottleneck upstream of the reads: fewer
  template molecules than reads (PCR input), and/or few founder cells per
  genotype per tube.
- **Founder sampling is a physical per-tube noise the model and simulator
  both lack.** A typical double has ~4 reads at harvest, ~15 cells in the
  tube, and so only a few cells when the tube was seeded from the split.
  Each tube draws its own founders, so a rare genotype's starting abundance
  differs from tube to tube, independently across timepoints; `ln_cfu0`
  shared by all tubes of a replicate x `condition_pre` is wrong at low
  abundance. The simulator seeds every tube deterministically
  (`_sim_growth`: `total_cfu0 * freq`) and samples reads once
  (multinomial), so simulated noise is pure Poisson.
- **`u_s` needs only a tight prior** (SD ~0.02).
- **Most of the doubles' weak data is weak everywhere.** Low counts come
  from low abundance before selection, not from genotypes crashing under
  it, so a count likelihood mostly makes their uncertainty honest (and
  lets the hierarchical priors carry them) rather than recovering a lost
  crash signal. The crash information exists for the better-represented
  doubles (39% of them dip below 5 reads somewhere) and is where the count
  likelihood beats a pseudocount. The design lever is more cells per double:
  deeper transformation of the double sub-library and more founders per
  tube at the split (which also shrinks founder noise).
- Separating template bottleneck from founder sampling needs a technical
  replicate: re-amplify and re-sequence the same extracted DNA for a few
  tubes (see the roadmap).
