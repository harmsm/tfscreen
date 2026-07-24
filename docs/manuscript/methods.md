# Methods (draft)

> **Draft status.** This file covers the inference protocol only: the model, the
> fitting workflow, and the multi-seed aggregation used for all downstream
> analyses. Sections marked `[TODO]` are placeholders for material we still need
> to write (k-fold cross validation, comparisons against independent
> measurements, library construction, sequencing).

## Software overview

We analyzed the screen with `tfscreen`, a Python package we wrote for
simulating and analyzing high-throughput screens of transcription factor (TF)
libraries [TODO: version, DOI, repository URL]. `tfscreen` treats the screen as
a single generative process that runs from TF–operator occupancy through
bacterial growth to sequencing read counts, and inverts that process by
hierarchical Bayesian inference. The package covers the whole path — read
counts to occupancy — but the analyses reported here used three of its stages:
conversion of read counts into per-genotype colony-forming-unit (CFU)
trajectories, joint inference of the generative model, and posterior
prediction of the quantities we interpret biologically (fractional occupancy
θ and second-order epistasis in θ).

We implemented the generative model in NumPyro (v0.19) on JAX (v0.8), and fit
it by stochastic variational inference (SVI) with the Adam optimizer as
implemented in `optax` (v0.2).

## The generative model

### What the model says

The central quantity is θ, the fractional occupancy of the TF on its operator.
θ is not observed directly in the screen; it is observed through growth. Each
genotype in the library carries a TF variant that regulates a selection marker,
so a genotype's growth rate under selection reports on how well its TF occupies
the operator at a given effector (titrant) concentration. We modeled the log
population size of every genotype in every tube as

```
ln_cfu = ln_cfu0 + (k_pre + dk_geno + m_pre · A · θ) · t_pre
                 + (k_sel + dk_geno + m_sel · A · θ) · t_sel
```

where

- `ln_cfu0` is the genotype's log abundance at the start of the experiment,
- `k_pre` / `k_sel` are per-condition baseline growth rates (pre-selection and
  selection phases),
- `m_pre` / `m_sel` are the per-condition slopes coupling occupancy to growth,
- `dk_geno` is the pleiotropic growth effect of the genotype, independent of TF
  activity,
- `A` is per-genotype TF activity (how strongly occupancy translates into
  regulation), and
- `t_pre` / `t_sel` are the durations of the two phases.

The model is modular: each term above is supplied by an interchangeable
component, and the components we selected are listed below. This structure let
us test alternative functional forms (e.g. different growth linkages or
occupancy parameterizations) against the same data without rewriting the model.

### Components we used

| Model axis | Choice | What it does |
|---|---|---|
| `theta` | `hill_mut` | θ follows a Hill curve in titrant concentration. Each of the four Hill parameters — `logit(θ_low)`, `logit(θ_high − θ_low)`, `log K`, `log n` — is written as a wild-type value plus additive per-mutation deltas in the transformed space, plus pairwise epistasis terms (`--epistasis`). Per-mutation deltas are hierarchical (`Normal(0, σ_d)`, σ_d inferred); pairwise terms carry a regularized horseshoe prior, so epistasis is sparse by default and only supported where the data demand it. |
| `condition_growth` | `linear` | Growth rate is linear in occupancy: `g = k_condition + dk_geno + m · A · θ`, with per-condition `k` and `m`. |
| `growth_transition` | `instant` | Genotypes switch from the pre-selection to the selection growth rate instantaneously at the split. |
| `ln_cfu0` | `hierarchical_factored` | `ln_cfu0[r, c, g] = geno_baseline[r, g] + tube_offset[r, c]`, separating genotype abundance in the library from tube-to-tube dilution differences. |
| `dk_geno` | `hierarchical_geno` | Per-genotype pleiotropic growth effect drawn from a pooled, left-skewed prior (a shifted negative log-normal), encoding the expectation that most mutations are neutral, a few beneficial, and a long tail deleterious. Wild type is pinned to `dk_geno = 0`. |
| `activity` | `fixed` | `A ≡ 1`. The TF in this system is a repressor that either blocks or does not; residual leaky repression is absorbed into `θ_low`. |
| `transformation` | `empirical` | Corrects for congression — cells that took up more than one plasmid during transformation. |
| `theta_rescale` | `passthrough` | θ enters the growth model directly (identity). |
| `theta_growth_noise` | `logit_normal` | Additive `Normal(0, σ_logit)` noise on `logit(θ)`, so occupancy noise is largest near θ = 0.5 and vanishes at saturation or depletion. σ_logit is a single global scalar inferred from the data. |
| `theta_binding_noise` | `zero` | The binding measurements carry their own reported errors; no extra noise term. |
| `growth_noise` | `normal_kt` | A single global `σ_k`, added in quadrature to the per-observation `ln_cfu` error, capturing biological growth variation not explained by θ or `dk_geno`. |

### Congression correction

Transformation of the library placed more than one plasmid in some cells. Since
the tightest-binding TF variant in a cell sets the effective occupancy at the
operator, a cell's growth reports the *maximum* occupancy over its plasmid
complement rather than the focal genotype's own occupancy — an attenuation that
pulls every measured curve toward the population's upper tail. We modeled this
explicitly: given the co-transformation rate λ, the corrected occupancy is
`E[max(θ_focal, background)]`, where the background distribution is the
empirical distribution of θ over the full genotype population at that
concentration. Because the background is itself an inferred quantity, we
evaluated it over all genotypes on every forward pass rather than over the
training minibatch.

We measured λ independently [TODO: how] and supplied it as a moment-matched
log-normal prior (`--transformation_lambda 0.3572 0.1296`). Genotypes that were
spiked into the library as clean monoclonal controls (`wt`, `M42I`, `H74A`,
`K84L`, `M42I/H74A`, `M42I/K84L`, `H74A/K84L`, `D88A`) are congression-free by
construction, and we exempted them from the correction.

### Observation channels

We fit four data sets jointly, each entering the likelihood through its own
observation layer:

1. **Growth** (`growth.csv`) — per-genotype `ln_cfu` trajectories across
   replicates, timepoints, conditions, and titrant concentrations. Observed
   under a Student-*t* likelihood with scale
   `sqrt(ln_cfu_std² + σ_k²)`, with a boolean mask excluding
   low-quality or missing cells.
2. **Binding** (`binding.csv`) — directly measured θ values for genotypes with
   independent binding curves. These pin the occupancy scale that growth alone
   cannot fix.
3. **Pre-split** (`presplit.csv`) — sequencing observations taken at
   `t = −t_pre`, before the culture was split into conditions. These constrain
   `ln_cfu0` directly.
4. **Base growth** (`base_growth.csv`) — direct reference-condition growth-rate
   measurements for a subset of genotypes, entering as
   `rate_obs ~ Normal(k_ref + dk_geno, rate_std)`. Together with the wild-type
   `dk_geno = 0` pin, these anchor the otherwise-degenerate additive slack
   between the per-condition baselines `k` and the per-genotype `dk_geno`.

The identifiability problem the last two channels address is worth stating
explicitly: the growth likelihood is invariant to `k += C, dk_geno −= C`, so
without anchors the whole system slides by a global constant, inflating every
condition baseline. We closed this in two complementary ways — the base-growth
channel above, and per-condition priors on the baselines set by the calibration
pre-fit described next.

## Fitting workflow

We ran the following pipeline once per random seed. Each step is a `tfscreen`
command-line entry point; the full script is reproduced in
[TODO: supplementary file / repository path].

### 1. Configure the model (`tfs-configure-model`)

We assembled the model specification from the four input tables and the
component choices above. This step wrote a YAML configuration plus two CSVs —
one holding the prior distribution for every parameter, one holding initial
values — which every later step read. It also built the mutation-by-genotype and
mutation-pair-by-genotype incidence matrices that `hill_mut` uses to decompose
genotype phenotypes into per-mutation and pairwise terms, declared the spiked
(congression-free) genotypes, and set the genotype minibatch size (65,536).

### 2. Calibrate the growth linking function (`tfs-prefit-calibration`)

Fitting the per-condition growth parameters simultaneously with the full
hierarchy is poorly conditioned, so we calibrated them first. We ran a MAP fit
of a deliberately collapsed model in which θ was pinned to the measured binding
values and every non-growth component was reduced to its simplest form
(`dk_geno` pinned from the base-growth measurements, `activity` and `ln_cfu0`
hyperparameters pinned to their prior locations, no noise components), against
the subset of cells observed in both the growth and binding data. We then wrote
the resulting per-condition MAP estimates of `k` and `m` back into the
production priors as per-condition prior locations, with scales floored from
the Hessian at the MAP point, and used them as warm-start values.

Because a soft prior in SVI is only a KL penalty — one the growth likelihood
can outvote across millions of observations — we additionally hard-clamped the
occupancy slope `m` to its calibrated value (`--pin_m`). The calibration
estimate of `m` is unbiased (`dk_geno` is uncorrelated with θ), whereas the
baseline `k` carries genuine per-experiment tube-to-tube variance and sits in
the additive slide described above; we therefore pinned `k`'s prior location but
left it free to move.

### 3. Fit the full model (`tfs-fit-model`)

We fit the full joint model by SVI with a structured (per-component) variational
guide, optimizing the ELBO with Adam over genotype minibatches. We ran to a
relative-ELBO convergence tolerance of 5 × 10⁻⁷ [TODO: report typical epoch
counts and wall time].

### 4. Draw the posterior (`tfs-sample-posterior`)

We drew 500 samples from the fitted variational posterior. Because these are
joint draws over all latent parameters, every downstream quantity inherits the
full cross-genotype posterior covariance rather than a per-genotype marginal
summary.

### 5. Predict occupancy, growth, and epistasis

From the posterior we computed:

- **Occupancy** (`tfs-predict-theta`) — θ for every genotype at every titrant
  concentration, summarized as posterior quantiles.
- **Second-order epistasis** (`tfs-predict-epistasis`) — for every double
  mutant, epistasis on the mutant cycle formed by the double, its two
  single-mutant parents, and wild type. We computed epistasis *within* each
  posterior draw and then took quantiles across draws, so the reported
  uncertainty reflects the posterior covariance among the four corners of each
  cycle rather than treating them as independent. We worked on the logit scale,
  which is the natural additive scale for an occupancy in [0, 1]: since
  `logit(θ) = −ΔG/RT`, we multiplied by `−RT = −0.6159` kcal mol⁻¹ (310.15 K)
  to report epistasis as an interaction free energy in kcal mol⁻¹.
  Each cycle also carries an `in_regime` flag, set only when all four corners
  have their central 95% θ interval inside [0.01, 0.99]. Outside that band
  `logit(θ)` saturates and the linear-in-θ growth likelihood constrains it
  weakly, so those estimates lean on the functional form of the θ model; we
  treated them as model-conditional throughout.
- **Growth and parameters** (`tfs-predict-growth`, `tfs-extract-params`) —
  posterior predictions of the observed `ln_cfu` data and per-parameter
  posterior summaries, used for the fit diagnostics in
  [TODO: figure reference].

## Aggregating across seeds

SVI finds a local optimum, so a single fit conflates the posterior with the
particular basin the optimizer reached. We therefore repeated the entire
workflow — configure through prediction — for 10 independent random seeds, and
combined the results with `tfs-compare-runs`.

For each feature (θ, epistasis) and each `(genotype, titrant, concentration)`
cell, we reconstructed each run's marginal posterior from its stored quantile
ladder and formed the **equal-weight mixture** across the 10 runs. By the law of
total variance the mixture's width folds in both each run's own posterior width
and the run-to-run spread of the point estimates, and — correctly — it does not
shrink with the number of runs, since different-seed fits are not independent
replicates of the experiment. All downstream analyses use these aggregate
posteriors. Any bias shared by all 10 runs is invisible to this procedure; it
captures optimization variability, not model misspecification.

The same tool also scored every genotype on two independent axes, which we used
to filter the results. It reports both as raw statistics rather than grades, so
the thresholds below are ours and are stated here rather than buried in a
command line:

- **Reproducibility** (`rms_sd`) — the run-to-run spread (RMS standard deviation
  over the concentration grid) of the median estimate, in the feature's own
  units. We required `rms_sd <` [TODO: cutline per feature].
- **Self-consistency** (`overdispersion`) — a χ²-per-degree-of-freedom statistic
  asking whether the run-to-run disagreement is explained by each run's own
  reported uncertainty, reported with its p-value and a Benjamini-Hochberg
  q-value across genotypes. It is independent of the reproducibility axis: it
  separates genotypes that are honestly uncertain from those that are
  confidently inconsistent across runs. We used [TODO: threshold, or state that
  it was reported but not used to filter].

[TODO: report how many genotypes passed each filter, and the reproducibility ×
self-consistency cross-tabulation.]

## [TODO] k-fold cross validation

We held out [TODO] and refit ... `tfs-compare-runs` in reference mode scores
each dropout fit by its deviation from the full-data fit rather than against the
cross-run mean.

## [TODO] Comparison against independent measurements

[TODO: which measurements, how they were matched to model predictions, what
agreement statistic.]

## [TODO] Data and code availability
