# Which parts of a dimeric-TF thermodynamic model can a screen resolve

**Question.** We want to replace the hand-written `O2_*` theta models with one
general dimeric-TF model whose variants (species present, energy terms
shared or free) are chosen in YAML, with PK, PnnC and PddG as priors on its
terms. Before building it: which species and energy terms can the planned
experiments resolve at all, from θ and from an affine transform of θ (the
relative fit's X)? And what does folding the second operator half-site
engagement (Oii) into one effective constant cost in spurious epistasis?

**Decision it feeds.** The design of the generalized thermodynamic model
(no plan file yet; this study and the review below are its starting point)
and the design of the next screen: six genetic backgrounds, two effectors,
two operators and two or three TF expression levels, under a fixed budget.

**Date and commit.** 2026-10-31, on `cc8d13a7` (main) with this directory
uncommitted.

## Background: review of the reaction-tree prototype

The prototype is
`~/work/Documents/writing/grants/2026-11-20_nsf-grant/code/thermo_model`
(outside the repository): a spanning tree of reactions over 27 species
(monomers, dimers, 0-2 effectors, monomer- and dimer-bound operator with a
first (Oi) and second (Oii) half-site engagement), pruned and merged by
named models, flags and address patterns. A review on 2026-10-31 found:

- The bookkeeping is right: monomial species, `root_composition`, the
  statistical link K1 = 2k, K2 = k/2, log-space mass balances.
- At equilibrium a model is its species set and their free energies. A tree
  is only a choice of variables, `merge` is the same as dropping the
  interior species (removed species contribute nothing to the partition
  function), and the root has no physical meaning. Agreed: the spec should
  be a species set plus an energy model; the tree, if kept, is internal.
- With θ alone, at one genotype and condition, observables depend only on
  the summed weight of each composition class (protomers, E, O): 27 species
  give 10 classes, so the full model's 26 constants collapse to 9. H vs L
  and Oi vs Oii are separated only by perturbations that act differently on
  members of a class. This study asks which perturbations do.
- The `concerted` named model has no free apo dimer, so it has no solution
  when TF exceeds effector (every 0-IPTG tube).
- `apply_k_eff` adds its shift to every constant with n_O != 0; the right
  shift is `shift * n_O` (a merged edge has n_O = -1). With operator a
  probe a uniform K_eff is only a logit offset on θ.
- The prototype's solver fails often in the lac regime (47/60 draws at
  E_t = 0, 23/60 at 1e-6 M; never silently wrong). E_t = 0 needs an
  analytic branch.
- The current O2 models are not special cases of the prototype: its U is a
  monomer (2U <-> H2), tfscreen's U1 is a dimer state, and MWC independence
  (operator binding the same at every effector rung) has no flag.

Decisions taken in the discussion that followed (Mike, 2026-10-31):

- Assume absolute θ is recoverable (other experimental knobs pin the
  scale); report the affine case alongside.
- The operator is an integrated single-copy reporter: fixed and far below
  TF, so it is a probe and leaves the mass balances.
- TF level is set by promoter choice rather than by a dimerization
  background.
- No explicit mutation x background coupling terms: specific coupling goes
  to the residual (a structural sparse prior is a possible later step).

## The model (`species_model.py`)

Species and their log formation constants from H monomers and effector are
linear in 16 energy terms: per-protomer H/L (`L1`), unfolding (`U`, as a
monomer or as a dimer state), dimerization (`dH`, `dL`), per-site effector
binding (`eH`, `eL`), second-site coupling (`cH`, `cL`), effector-operator
coupling (`xH`, `xL`), monomer half-site binding (`oHm`, `oLm`), dimer first
engagement (`oH`, `oL`) and second engagement (`oiiH`, `oiiL`). A dimer
contributes K_oi (1 + K_oii) to the operator binding polynomial; θ is the
operator-bound fraction. A mutation in a homodimer acts on both protomers,
so a shift of `L1` by d moves L2/H2 by 2d. The solve is a closed-form
quadratic for free protomer inside a bracketed 1D root for free effector,
and the Jacobian comes from the implicit function theorem.

`validate.py` checks it (all PASS):

| check | worst error |
|---|---|
| mass balances, relative | 6.3e-15 |
| implicit Jacobian vs central differences | 8.6e-11 |
| θ vs tfscreen `mwc.mwc_theta` (no monomers, dimer U, MWC) | 2.3e-15 |
| θ vs the prototype tree, nested brentq, O_t = 1e-14 M | 3.3e-10 |

## How to run

From this directory (the last check needs the prototype path):

```bash
~/miniconda3/bin/python validate.py --prototype ~/work/Documents/writing/grants/2026-11-20_nsf-grant/code/thermo_model
```

```bash
~/miniconda3/bin/python rank_check.py --out_prefix rank
```

```bash
~/miniconda3/bin/python oii_folding.py --out_prefix oii
```

Step 3, one run per row of its table (about 10 minutes each; they can run
in parallel), then concatenate the `*_designs.csv` files with a `run`
column into `power_designs.csv`:

```bash
for s in 1 2 3 4; do ~/miniconda3/bin/python design_power.py --seed $s --out_prefix power_s$s; done
```

```bash
~/miniconda3/bin/python design_power.py --seed 1 --ln_k_oii 0 --out_prefix power_oii0
```

```bash
~/miniconda3/bin/python design_power.py --seed 1 --variant coupled --out_prefix power_coupled
```

Step 4, every case for seeds 1 and 2 (minutes each for the benign cases,
up to an hour for the harmful ones at the 2000-evaluation cap), then
concatenate the JSON records into `misspec_results.csv`:

```bash
for c in correct oii0_folded oii4_folded x_coupling c_coupling bg_offtarget mut_x_bg ligand_specific pair_coupling; do for s in 1 2; do ~/miniconda3/bin/python misspec.py --case $c --seed $s --out_prefix misspec; done; done
```

Step 5, four seeds in parallel (about 30 minutes each on a laptop), then
pool them into `remedies_results.csv` and print the median table:

```bash
for s in 1 2 3 4; do ~/miniconda3/bin/python remedies.py --seed $s > remedies_s$s.log 2>&1 & done; wait
~/miniconda3/bin/python remedies_summary.py
```

Step 5b, six seeds in parallel, then `tails_results.csv` and the median
tables:

```bash
for s in 1 2 3 4 5 6; do ~/miniconda3/bin/python remedies_tails.py --seed $s > tails_s$s.log 2>&1 & done; wait
~/miniconda3/bin/python tails_summary.py
```

Step 5c, four HMC chains each for no doubles and the 15 Å library on
seed 1 (about 20-30 minutes per chain with eight running at once), then
the comparison with the Laplace (`hmc_results.csv`, `hmc_param_sd.csv`):

```bash
for c in 15 0; do for k in 0 1 2 3; do ~/miniconda3/bin/python remedies_hmc.py --seed 1 --close $c --chain $k > hmc_c${c}_k$k.log 2>&1 & done; done; wait
~/miniconda3/bin/python remedies_hmc_summary.py
```

Chain 0 at 15 Å stuck on the first run (acceptance 0 from iteration 400
on, beside overflow warnings in the solver), so we set it aside and ran
chain 4 in its place; the summary globs whatever `hmc_s1_c15_k*.npz` are
present.

Step 5d, the three Gaussian variational fits for both libraries (about
20 minutes with six at once), then every method against the HMC chains
(`methods_results.csv`):

```bash
for c in 15 0; do for f in mean_field low_rank full_rank; do ~/miniconda3/bin/python remedies_vi.py --seed 1 --close $c --family $f > vi_c${c}_$f.log 2>&1 & done; done; wait
~/miniconda3/bin/python remedies_methods.py
```

The chain and variational draws (`hmc_s*.npz`, `vi_s*.npz`, about 80 MB)
are outputs, not committed; the CSVs hold the results.

`rank_check.py` takes about 15 minutes on a laptop; `validate.py` and
`oii_folding.py` under a minute. Seeds are fixed in the scripts.

## Step 1: structural rank (`rank_check.py`)

**Setup.** A condition is (background b, effector f, operator o, TF level t)
with 6 backgrounds (wt; H<->L, shifting `L1`; operator affinity of H, `oH`;
effector affinity of L, `eL`; effector affinity of H, `eH`;
dimerization and stability, `dH`, `dL`, `U`), 2 effectors, 2 operators and
3 TF levels (1, 0.3 and 3 times 1e-5 M protomer; level 0 known, the other
two latent unless stated). Each condition measures the background alone and
6 single mutations in it at 12 effector concentrations (0 and 1e-7 to
1e-2 M). Parameters: the variant's wt terms, each background's shift on its
target terms, effector 2's change to the effector terms and operator 2's to
the operator terms, the latent TF levels, and each mutation's shift on every
wt term, additive in every background, effector, operator and TF level. The
affine observable adds X = a + s θ per condition (or one map per operator
construct, `affine_by = operator`, the case where backgrounds, effectors and
TF levels share a reporter). Rank is the number of singular values of the
exact Jacobian above 1e-9 of the largest, at a generic point (monomers
populated, K_d 5 µM; second engagement near marginal, ln K_oii about 0; L
binds operator), best of two seeds. Every result below has a clean gap: kept
singular values at 1e-6 or more of the largest, dropped ones at 1e-16,
except `wt_only` (no gap; its count depends on the threshold) and the
off-target case noted below.

Designs: `full` (6 x 2 x 2 x 3 = 72 conditions); `star` (every background
at the base knobs, with effector 2, operator 2 and the two other TF levels
each crossed with every background one at a time: the 6x2 + 6x2 + 6x3
design, 30 unique conditions because the base condition appears in all
three arms); `wt_knobs` (every knob combination in the wt background, every
other background at the base knobs only, 17); `knobs_only` (every knob
combination, wt background only, 12); `base` (the six backgrounds at the
base knobs, 6); `wt_only` (1).

Variants: `mwc_dimer` (no monomers, dimer U, MWC: tfscreen's C12 U1 plus
Oii), `mwc` (adds monomers and monomer operator binding), `mwc_umono` (U as
a monomer, the prototype's form), `coupled` (`mwc` plus second-site
effector coupling `c` and effector-operator coupling `x`, so MWC
independence is no longer assumed).

Rank deficiency (number of unidentified directions), θ / affine per
condition / affine per operator:

| variant | params (θ, star) | full | star | wt_knobs | knobs_only | base | wt_only |
|---|---|---|---|---|---|---|---|
| mwc_dimer | 69 | 4 / 4 / 4 | 4 / 4 / 4 | 4 / 4 / 4 | 4 | 14 / 14 / 14 | 21 / 22 / 22 |
| mwc | 101 | 4 / 4 / 4 | 4 / 4 / 4 | 4 / 4 / 4 | 4 | 14 / 14 / 14 | 26 / 32 / 32 |
| mwc_umono | 101 | 4 / 4 / 4 | 4 / 4 / 4 | 4 / 4 / 4 | 4 | 14 / 14 / 14 | 27 / 33 / 33 |
| coupled | 133 | 4 / 4 / 4 | 4 / 4 / 4 | 4 / 4 / 4 | 4 | 14 / 14 / 14 | 53 / 57 / 57 |

What the deficiencies are (`rank_null_directions.json`):

- **Everything but the Oi/Oii split is identifiable** in every design with
  backgrounds or knobs, in every variant: H/L, dimerization in both
  conformations, unfolding in either form, monomer operator binding, and in
  `coupled` the second-site and effector-operator couplings. The six
  backgrounds alone (`base`) are enough.
- **The 4 left in the richer designs are all Oi/Oii.** Per conformation,
  only K_oi (1 + K_oii) is seen. With backgrounds only (`base`) the split is
  free for wt and for each mutation separately (2 + 12 = 14). Once operator
  2 is in the design and mutations act the same way on both operators, the
  per-mutation splits are tied together and only 2 global directions per
  conformation remain (wt's split and operator 2's).
- **An operator background that also shifts the second engagement**
  (`bg_terms = target_oii`, background 2 shifts `oH` and `oiiH`): with
  operator 2 in the design (`star`, `full`) the H split is resolved
  (deficiency 2, the L side only, which no background touches). Without
  operator 2 (`base`, deficiency 9) the per-mutation H splits are tied
  together, but two global H directions remain (wt's split and the
  background's), and every L split stays free. The information is the curvature of
  softplus(ln K_oii), so it is strong only when wt K_oii is near 1, which
  is the regime where folding Oii is costly (step 2).
- **Absolute θ is not needed for structural identifiability.** The affine
  map is identified from the sigmoid's curvature and the shared physics in
  every design except `wt_only`. What it costs in precision is a step 3
  question (singular-value ratios are not comparable across
  parameterizations: the affine offsets' columns change the largest one).
- **TF levels.** Known, latent ratios (level 0 known) and all latent give
  the same deficiency (`mwc`, `full` and `star`: 4 each); absolute P_t is
  structurally identifiable through the monomer-dimer equilibrium. The
  weakest identified directions in `mwc` are per-mutation `U` (dimer
  unfolding acts much like a lower TF level), at 1e-4 of the largest.
- **Mutation effects specific to the operator** (`ligand = specific`: a
  mutation's operator-term shifts differ between operators, and its
  effector-term shifts between effectors): deficiency 28 in `full` and
  `star`, all of it Oi/Oii again (each mutation's split under each
  operator). The effector-specific mutation effects are identified.
- **Background off-target effects** (`bg_terms = all`: every background
  shifts every term): full rank. In `mwc` the weakest directions are
  1e-7 to 1e-9 of the largest, all background off-target shifts (for
  example background 4's shifts of `oH` and `oiiH`): formally identifiable,
  practically not. In `coupled` they are 1e-6. Either way the backgrounds'
  own effects will need priors or their own characterization.
- **Designs do not differ structurally.** `full`, `star`, `wt_knobs` and
  `knobs_only` give the same deficiency. They differ in conditioning: the
  smallest kept singular value relative to the largest is 4e-4 (`star`,
  `mwc_dimer`), 2e-4 (`wt_knobs`) and 2e-6 (`knobs_only`), so the
  backgrounds carry much of the power even where the knobs alone are
  formally enough. That ratio ignores noise and the number of
  observations; the comparison that counts is step 3's.

Caveat: these are structural results at a generic point. With lac-like
constants (tight dimer, L barely binding operator, the second engagement
far from marginal) many identified directions become practically
unidentifiable; that is step 3.

## Step 2: folding Oii into one constant (`oii_folding.py`)

Truth: mutations shift the first engagement by d1 and the second by d2,
iid Normal(0, sigma) kT for operator-touching mutations. Folded model: one
constant per conformation, additive in its log. Singles fit exactly; a
double misses ln K_eff by eps = sp(x + a2 + b2) - sp(x + a2) - sp(x + b2) +
sp(x), with x the wt ln K_oii. The same eps is the apparent coupling of a
mutation to a background that shifts the second engagement. θ error: the
largest |θ_true - θ_folded| over a 22-point IPTG curve on the C12 U1
defaults, for 150 random doubles (2000 for eps), with the wt effective
constant held at 4.2e8 M^-1 for every x.

| sigma (kT) | wt ln K_oii | median abs eps | 95% abs eps | median max Δθ | 95% max Δθ | fraction Δθ > 0.02 |
|---|---|---|---|---|---|---|
| 0.5 | ±6 | 0.000 | 0.001-0.002 | 0.000 | 0.000 | 0.00 |
| 0.5 | 0 | 0.023 | 0.119 | 0.005 | 0.025 | 0.09 |
| 1.0 | -6 / 6 | 0.001 | 0.011 / 0.013 | 0.000 | 0.003 | 0.01 / 0.00 |
| 1.0 | -4 / 4 | 0.006 | 0.074 / 0.078 | 0.002 / 0.001 | 0.013 / 0.023 | 0.03 / 0.07 |
| 1.0 | -2 / 2 | 0.036 / 0.035 | 0.30 / 0.29 | 0.008 | 0.052 / 0.061 | 0.27 / 0.22 |
| 1.0 | 0 | 0.078 | 0.39 | 0.015 | 0.097 | 0.42 |
| 2.0 | 0 | 0.26 | 1.17 | 0.053 | 0.28 | 0.74 |
| 2.0 | ±6 | 0.002 | 0.15 / 0.17 | 0.001 / 0.000 | 0.046 / 0.032 | 0.09 / 0.07 |

The full grid (sigma 0.5, 1, 2; x from -6 to 6) is `oii_folding.csv`.
These are pairs in which both mutations touch the operator; for the whole
library scale by the square of that fraction.

Reading: folding is exact in practice when the wt second engagement is far
from marginal (|ln K_oii| >= 4 at 1 kT effects). Near ln K_oii = 0 it
produces structured spurious epistasis of the size the screen resolves (θ
error above 0.02 in 42% of operator x operator doubles at 1 kT). The model
should keep Oii as a softplus term rather than fold it; when K_oii is far
from 1 the term is unidentified and harmless, and when it is near 1 the
operator background crossed with operator 2 identifies it (step 1).

## Step 3: predicting doubles at lac-like constants (`design_power.py`)

**Criterion.** The goal is to predict combined effects, not to resolve
every term (Mike, 2026-10-31: never seeing monomers or L-operator binding
is fine if hidden changes do not come back when mutations are combined).
So the score is the posterior predictive SD of θ for doubles the design
never measures: all 190 pairs of 20 measured mutations, in the wt
background at the base effector, operator and TF level, from a Laplace
posterior at the truth (Fisher information plus Gaussian priors). An
unresolved term costs nothing unless a double's θ depends on it, and then
its prior variance shows up in the prediction.

**Truth and priors** (in the script's docstring): wt at tfscreen's C12 U1
defaults plus a tight dimer (1e10 M^-1), weak monomer operator binding and
ln K_oii = 4. Each mutation has one class: allosteric (`L1`), stability
(`U`), pocket (`eH`, `eL`), DNA-binding (operator terms) or interface
(`dH`, `dL`), main effects SD 1.5 kT, every other term SD 0.2. The fit
does not know the class: prior SD 1.5 on every term. Backgrounds shift
their target by 1.5-2.5 and everything else by SD 0.2 (prior SD 0.5).
TF levels are latent around measured ratios (prior SD 0.3). The fit model
is `mwc` (the truth's structure) unless stated. Noise: θ SD 0.03 per point
at one condition's depth (`--sigma_ref`). At a fixed total budget equal to
12 conditions' reads, a K-condition design has SD 0.03·sqrt(K/12).

**Context.** The true epistasis these doubles carry, as the largest
|θ_double − θ_a − θ_b + θ_wt| over the IPTG curve: median 0.06 to 0.17,
90th percentile 0.22 to 0.60, above 0.05 in 53-79% of doubles (by seed).
The prior alone predicts the doubles to SD 2.6-3.1 (max over the curve,
in θ units, so no information).

Results (`power_designs.csv`, `power_details.json`): largest predictive SD
over the curve per double, median and 90th percentile over doubles, for
four mutation draws (seeds 1-4), then seed 1 with ln K_oii = 0 and seed 1
fit with the `coupled` model.

| design (K) | budget | median, seeds 1-4 | q90, seeds 1-4 | q90, K_oii ≈ 1 | q90, coupled fit |
|---|---|---|---|---|---|
| wt_only (1) | fixed depth | 0.053-0.076 | 0.08-0.23 | 0.13 | 0.20 |
| wt_only (1) | fixed budget | 0.017-0.026 | 0.03-0.11 | 0.08 | 0.12 |
| base (6) | fixed budget | 0.018-0.019 | 0.024-0.043 | 0.052 | 0.034 |
| knobs_only (12) | fixed budget | 0.017-0.021 | 0.032-0.098 | 0.051 | 0.083 |
| wt_knobs (17) | fixed budget | 0.016-0.017 | 0.019-0.043 | 0.047 | 0.036 |
| star (30) | fixed budget | 0.015-0.018 | 0.018-0.036 | 0.036 | 0.029 |
| full (72) | fixed budget | 0.015-0.020 | 0.020-0.034 | 0.037 | 0.032 |
| greedy (6) | fixed budget | 0.013-0.017 | 0.015-0.030 | 0.031 | 0.030 |
| greedy (12) | fixed budget | 0.014-0.017 | 0.016-0.031 | 0.031 | 0.029 |
| star (30) | fixed depth | 0.010-0.012 | 0.011-0.027 | 0.025 | 0.021 |
| full (72) | fixed depth | 0.007-0.009 | 0.008-0.018 | 0.017 | 0.017 |

What this says:

- **The model predicts the epistasis.** Predictive SD for a double is
  about 0.015 at the median against true epistasis of 0.06-0.17, under
  any design that goes beyond one condition.
- **The design matters in the tail, not the median.** At a fixed budget
  the medians barely move. The 90th percentile is where designs differ:
  0.03-0.11 for every read in one condition, 0.015-0.03 for the greedy
  designs. The bad tail is the bite. In seed 1 every double in it contains
  mutation 6, a DNA-binding mutation that weakens H operator binding by
  2.5 kT. The singles sit near saturation (θ(0) ≈ 0.9995), which hides
  other mutations' effector-site shifts; paired with mutation 6 the curve
  moves where they show. Median SD of mutation 6's doubles: 0.094 with
  every read in one condition and 0.074 for `star`, both at fixed budget,
  and 0.037 for `full` at fixed depth.
- **What the remaining uncertainty is.** In every design and seed most of
  the doubles' predictive variance comes from the mutations' effector-site
  shifts (`eL`, `eH`), then `L1` and the operator terms, and in the
  coupled fit the operator and coupling terms (`variance_share_fixed_depth`
  in `power_details.json`; shares overlap, so they do not sum to 1). That
  is, the hidden physics that bites is effector binding hidden behind
  saturation, not monomers or L-operator binding, which never appear.
- **About six well-chosen conditions get most of the value at fixed
  budget.** Greedy designs of 6 and 12 match or beat `star` and `full`,
  and adding conditions past 6 changes little. Spreading a fixed budget
  over 72 conditions costs depth and is no better than the greedy 6.
- **What greedy chooses, across all six runs:** the low TF level (0.3x)
  in most conditions (it unsaturates θ at low IPTG, exposing the effects
  saturation hides); always the wt background and the H<->L background;
  effector 2 in a third to a half of conditions; operator 2 rarely unless
  the fit is `coupled` (where it resolves effector-operator coupling) or
  K_oii is near 1. Which other backgrounds it picks depends on the
  mutation draw. Backgrounds matter for the tail: `knobs_only` has the
  worst tail of the multi-condition designs (q90 up to 0.098).
- **Oii near marginal** (ln K_oii = 0) and **an unneeded coupled fit**
  each widen the greedy designs' tail by about a third (seed 1, q90 0.023
  to 0.031 and 0.030), and push them toward operator 2.

Caveats: a Laplace posterior at the truth (linearization, correct model,
no misspecification); θ noise Gaussian, the same at every θ, with an
assumed 0.03 at one condition's depth; doubles scored only in the base
condition; mutation effects shared across effectors and operators; the
TF levels are 0.3x, 1x and 3x only, so "lower is better" is not tested
below 0.3x. The greedy objective is the mean predictive variance, so it
does not target the tail directly.

## Step 4: misspecification (`misspec.py`)

**Method.** Noise-free data from a richer truth, fit by MAP (scipy
`least_squares`) with the reduced model of step 3. Without noise the MAP
is the pseudo-true point a misspecified fit converges to, so a double's
prediction error is the bias real data would carry. Each error is scored
against the Laplace SD the fit reports at that point. "Detect" is the
misfit's added chi-square over the measured data, in SDs of the noise-only
chi-square (sqrt(2n)): above about 3 the misfit should be visible in the
residuals. Truth, priors, noise (fixed budget of 12 conditions' reads)
and the 190 doubles follow step 3, two mutation draws. Designs:
`wt_only`, `rec6` (a six-condition design in the style of step 3's greedy
choice: base; low TF in the wt background; low TF in the H<->L and
dimer/stability backgrounds; effector 2 at low TF in the wt and H<->L
backgrounds), `star`, `full`. A vectorized solver (`solve_batch`, bisection
in ln e) replaced the per-row brentq; it matches brentq to 1e-14.

The cases are listed in the script's docstring. The recorded table
(`misspec_results.csv`) was fit with a cap of 200 evaluations; 15 of 72
fits hit it, all in the three harmful cases. Rerun with a cap of 2000
(the script's default now), three of those six case-seeds finished and
changed by at most 0.001 in the errors below; the ligand-specific fits do
not converge even then, on a flat misfit plateau.

Largest prediction error over the curve per double (median and 90th
percentile over doubles), 95% coverage of the reported intervals, and
detectability; ranges over the two seeds:

| case | design | error median | error q90 | coverage95 | detect |
|---|---|---|---|---|---|
| correct (prior shrinkage only) | any | 0.000-0.002 | 0.002-0.017 | 0.99-1.00 | < 0.03 |
| Oii folded, ln K_oii = 4 | any | 0.001-0.002 | 0.005-0.017 | 0.99-1.00 | < 0.2 |
| Oii folded, ln K_oii = 0 | rec6 / full | 0.001-0.005 | 0.008-0.023 | 0.95-1.00 | < 0.5 |
| effector-operator coupling, fit MWC | rec6 / full | 0.001-0.002 | 0.004-0.007 | 0.98-1.00 | < 0.06 |
| second-site coupling, fit independent | rec6 / full | 0.011-0.019 | 0.031-0.040 | 0.94-0.98 | 1.6-2.7 |
| background off-target held at 0 | rec6 | 0.005-0.006 | 0.013-0.015 | 0.98-1.00 | 0.7-1.0 |
| background off-target held at 0 | star / full | 0.038-0.083 | 0.084-0.118 | 0.47-0.70 | 10-21 |
| mutation x background coupling | rec6 | 0.008-0.009 | 0.031-0.061 | 0.96-0.98 | 0.7-2.8 |
| mutation x background coupling | star / full | 0.011-0.027 | 0.035-0.080 | 0.75-0.96 | 1.1-4.0 |
| ligand-specific mutation effects | rec6 / star / full | 0.037-0.084 | 0.14-0.23 | 0.61-0.85 | 13-71 |
| specific pair coupling (10% of doubles) | any | 0.001-0.003 | 0.002-0.022 | 0.96-0.98 | 0 |

`wt_only` is immune to every misspecification that lives in other
conditions (backgrounds, effector 2), with errors as in `correct`.

What this says:

- **Benign.** Folding Oii (in this draw, few doubles pair two DNA-binding
  mutations), and effector-operator coupling the fit ignores. The fit
  absorbs them without visible bias.
- **Harmful and visible.** Mutation effects on effector binding that
  differ between effectors, and off-target background effects held at 0.
  The damage is large (errors comparable to the true epistasis, coverage
  0.5-0.85) but the misfit to the measured data is 10-70 noise SDs, so a
  real fit would show it. Both are remedied in the model, not the design:
  the default step 3 fit already gave backgrounds free off-target shifts
  (prior SD 0.5), and step 1 found effector-specific mutation effects
  identifiable, so they can be fit with a prior that ties the two
  effectors together.
- **Harmful and hard to see.** Specific mutation x background coupling
  (Mike's residual): errors 0.01-0.03 at the median, 0.03-0.08 at the
  90th percentile, coverage down to 0.75, and detectability 0.7-4, so a
  fit might pass its residual check while carrying the bias. It enters
  only through the backgrounds, so its cost grows with how much the
  design leans on them (rec6 less than star and full). A residual term
  for it, a variance per mutation x background or a sparse prior, would
  widen the intervals to match; leaving it out makes them too narrow.
  Second-site coupling is the milder version of the same problem.
- **Invisible from singles, as expected.** Specific pair coupling between
  library mutations cannot be predicted by any design that measures only
  singles (detect 0): the 10% of doubles that carry it are 5% of doubles
  beyond 3 SD. That is the floor the physics model cannot remove.
- **The trade-off.** Each condition beyond the base adds precision (step
  3) and an assumption that the same mutation effects hold there. When the
  assumption fails it moves the shared parameters and every prediction
  with them. A small design that leans on few cross-condition assumptions
  (rec6) was the most robust multi-condition design to background
  misspecification; it is as exposed as the others to effector-specific
  effects, because two of its six conditions use effector 2.

Caveats: the remedies (free off-target shifts, effector-specific effects
with a tying prior, a mutation x background residual) are not yet tested;
two mutation draws; misspecification sizes are guesses (0.5 kT for the
specific couplings).

## Step 5: remedies, scored on five-mutants (`remedies.py`)

**Question.** Do the step 4 remedies restore honest predictions of
higher-order combinations, and what do measured close doubles add? Mike's
target is combinations of five or more mutations predicted from singles
and doubles.

**Truth.** Step 4's machinery (noise-free MAP, Laplace at the MAP, the
same budget and noise), with 15 library mutations, a truth that carries
both harmful effects of step 4, and synthetic geometry:

- A mutation's effect on the effector terms (eH, eL) differs under
  effector 2, and its effect on the operator terms differs under operator
  2. Everything else is shared. Deviation SD 0.5 kT for pocket and
  DNA-binding mutations respectively, 0.1 kT otherwise.
- The 15 library sites and the 5 background sites sit uniformly in a ball
  of radius 20 A. A pair is coupled with probability exp(-d/6 A), by
  Normal(0, 1) kT on one term either site acts on. Mutation m in
  background b is the double m+b, so mutation x background coupling is
  the same object as library pair coupling. Over four seeds, 8-16 of the
  180 site pairs were coupled, and the close pairs (d < 10 A, 14-20 of
  them) held 2-8 of those.

**Fits.**
- `shared`: step 4's model, with effects shared across effectors and
  operators and no pair terms.
- `lig`: adds a deviation for each mutation under effector 2 (on eH and
  eL) and under operator 2 (on the o terms), with prior Normal(0, 0.5).
- `pair`: `lig` plus a pair term on each term either site of a pair acts
  on. Its Gaussian prior SD is pair_sd*sqrt(exp(-d/6)/n_terms), which
  matches the truth's second moment; `x0.3` and `x3` rescale it.

The structure is assumed to say which function each site touches.

**Data.** Each design's singles screen, optionally plus the close doubles
in the wt background. `dbl1` measures them in the base condition only;
`dbl3` adds effector 2 and low TF. The doubles get their own budget of
three conditions' reads.

**Targets.** Unmeasured library doubles, 200 random triples and 200
random five-mutants, all at the base condition. Each target is classed
by the coupled pairs it contains: an unmeasured one (uhit), only measured
ones (mhit), or none.

**Medians over seeds 1-4, five-mutants.** Error is the largest over the
curve, taken at the 90th percentile over targets. z3 is the fraction of
targets with a point beyond 3 SD.

| design | fit | doubles | detect | error q90 | cov95 | z3 | cov95, mhit | cov95, uhit |
|---|---|---|---|---|---|---|---|---|
| wt_only | shared = lig | none | 0.02 | 0.48 | 0.82 | 0.29 | | 0.74 |
| wt_only | pair | none | 0.02 | 0.48 | 0.84 | 0.22 | | 0.79 |
| wt_only | pair | dbl1 | 0.06 | 0.26 | 0.90 | 0.16 | 0.97 | 0.85 |
| wt_only | lig (no pair term) | dbl3 | 0.61 | 0.43 | 0.87 | 0.21 | 0.92 | 0.76 |
| rec6 | shared | none | 2.9 | 0.47 | 0.72 | 0.39 | | 0.70 |
| rec6 | lig | none | 0.02 | 0.44 | 0.82 | 0.31 | | 0.70 |
| rec6 | pair | none | 0.01 | 0.44 | 0.86 | 0.24 | | 0.82 |
| rec6 | pair | dbl1 | 0.02 | 0.25 | 0.92 | 0.17 | 0.97 | 0.84 |
| rec6 | pair | dbl3 | 0.02 | 0.25 | 0.92 | 0.17 | 0.97 | 0.84 |
| rec6 | pair, prior x0.3 | dbl3 | 0.28 | 0.38 | 0.86 | 0.22 | 0.93 | 0.70 |
| rec6 | pair, prior x3 | dbl3 | 0.01 | 0.25 | 0.94 | 0.14 | 0.98 | 0.88 |
| rec6 | lig (no pair term) | dbl3 | 0.13 | 0.46 | 0.85 | 0.24 | 0.91 | 0.72 |
| star | shared | none | 4.5 | 0.52 | 0.78 | 0.29 | | 0.71 |
| star | lig | none | 0.22 | 0.45 | 0.79 | 0.30 | | 0.71 |
| star | pair | none | 0.01 | 0.44 | 0.86 | 0.23 | | 0.82 |
| star | pair | dbl3 | 0.02 | 0.29 | 0.92 | 0.17 | 0.96 | 0.84 |
| star | pair, prior x3 | dbl3 | 0.02 | 0.26 | 0.94 | 0.15 | 0.98 | 0.88 |

The uhit set is larger without doubles, since nothing is measured, so
compare the uhit columns only between runs that both have doubles. The
full table, with doubles, triples and the clean targets, is
`remedies_results.csv`.

**Reading.**

1. The ligand-specific remedy works. With backgrounds and knobs (rec6,
   star), the shared model is visibly wrong, with detect 3-4.5, and its
   coverage drops to 0.72-0.78. Per-effector and per-operator deviations
   remove the misfit. Alone they do not restore coverage, because the
   remaining error is pair coupling.
2. The pair term helps even with no doubles measured. It gives mutation x
   background coupling somewhere to go other than the singles, so
   unmeasured doubles drop from an error q90 of 0.013 to 0.004 on rec6.
   With the pair term in place, wt_only, rec6 and star predict
   five-mutants about equally well. Backgrounds no longer cost anything,
   and in this toy they add little.
3. Measured close doubles roughly halve the five-mutant error tail (q90
   0.44 to 0.25) and cut the z3 fraction from about 0.23 to 0.17. Targets
   whose coupled pairs were all measured are predicted almost exactly
   (coverage 0.96-0.97). One condition does as well as three here. The
   data are noise-free and the pair term's terms are given, so this is an
   upper bound on what one condition does.
4. Measuring doubles without a pair term is worse than not measuring them
   (wt_only, lig, dbl3: q90 0.43 and detect 0.6, against 0.26 with the
   pair term). The couplings in the doubles get pushed into the singles.
   Doubles and the pair term go together.
5. A pair prior that is too narrow hurts, and the misfit stays below
   detection (detect 0.3). A prior that is too wide costs almost nothing
   and covers best. The truth is sparse (spike and slab), so a Gaussian
   prior with the matching variance is too narrow in its tails. That
   argues for a heavy-tailed pair prior, horseshoe-like as in the PddG
   pairwise prior, or for erring wide.
6. The floor is coupled pairs that nobody measured. In this geometry,
   25-70% of random five-mutants contain one, depending on the seed. For
   those, coverage is 0.84-0.88 at best and the 90th-percentile error is
   about 0.5. How much of the coupling falls inside the doubles library
   sets the achievable accuracy, so it is the number the real structure
   has to supply.

**Caveats.**
- Noise-free data.
- Four seeds with few coupled pairs each (8-16).
- The coupling rate, its distance law and its 1 kT size are guesses.
- The geometry is synthetic.
- The fit knows each site's function and the true distance law. Only the
  scale was varied, not learned.
- 19 of the 96 fits hit the 400-evaluation cap. All of them are
  misspecified ones (shared, lig without the pair term, or a pair prior
  at x0.3), on the same flat plateau as step 4.
- No third-order energy coupling is planted.

## Step 5b: pair prior tails and doubles library size (`remedies_tails.py`)

**Question.** Does a heavy-tailed pair prior fix step 5's undercoverage,
and how large does the doubles library have to be?

**Setup.** Step 5's truth, geometry, rec6 singles screen and `pair` fit,
with the pair terms started at 0 rather than at the truth. The doubles
library is every site pair closer than a threshold, mutation plus
background-site pairs included, measured in the base condition. The pair
prior is Normal at the matched scale s, Normal at 3s, Student-t with 3
dof at s, or Cauchy at s. Intervals are sampled, 1000 draws through the
full model: the Laplace at the MAP for every parameter the data touch,
and the prior itself for any pair term no measured row touches. Targets
are 200 random five-mutants per seed, scored over all of them. Six
seeds; no fit hit the evaluation cap; the misfit statistic is below 0.1
everywhere, as it should be for the correct model.

**Results** (five-mutants, medians over six seeds; width is the largest
95% width over the curve, "pairs held" the same with every pair term at
the MAP, "curve" the median width over the curve; "unmeasured" is
coverage on targets holding a coupled pair outside the library, a
truth-selected subset):

| close (Å) | double rows | coupling inside | prior | error q90 | cov95 | width | pairs held | curve | unmeasured |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 0 | gauss | 0.34 | 0.95 | 0.93 | 0.90 | 0.63 | 0.92 |
| | | | gauss x3 | 0.35 | 1.00 | 0.98 | 0.91 | 0.87 | 1.00 |
| | | | t3 | 0.34 | 0.99 | 0.95 | 0.90 | 0.72 | 0.97 |
| | | | cauchy | 0.34 | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 |
| 10 | 15 | 0.38 | gauss | 0.22 | 0.97 | 0.91 | 0.89 | 0.61 | 0.89 |
| | | | gauss x3 | 0.23 | 1.00 | 0.96 | 0.89 | 0.78 | 1.00 |
| 15 | 39 | 0.61 | gauss | 0.14 | 1.00 | 0.90 | 0.88 | 0.58 | 1.00 |
| | | | gauss x3 | 0.13 | 1.00 | 0.95 | 0.88 | 0.73 | 1.00 |
| 20 | 84 | 0.88 | gauss | 0.02 | 1.00 | 0.88 | 0.87 | 0.54 | 0.55 |
| | | | gauss x3 | 0.01 | 1.00 | 0.92 | 0.87 | 0.65 | 0.93 |
| | | | t3 | 0.02 | 1.00 | 0.90 | 0.87 | 0.58 | 0.66 |
| | | | cauchy | 0.01 | 1.00 | 1.00 | 0.86 | 0.99 | 1.00 |

t3 and Cauchy at 10 and 15 Å follow the same pattern; every row is in
`tails_results.csv`.

**Reading.**
1. Library size is the lever for the point prediction. The share of
   coupled pairs inside the library sets the error: q90 0.34, 0.22, 0.14
   and 0.02 at 0, 38, 61 and 88% of the coupling. In this geometry that
   took 84 double rows for 15 sites. The real structure has to say what
   threshold captures most of the coupling and how many pairs that is.
2. The prior's tails do not move the point. They move the intervals.
   Cauchy at the matched scale is too wide: its median width over the
   curve is about 1, so it says nothing. t3 is barely wider than the
   matched Gaussian and does not protect the targets with an unmeasured
   coupled pair (0.66 at 20 Å). The Gaussian at 3s does (0.93) at a
   moderate cost in width. The scale matters more than the tails, which
   argues for learning it from the doubles.
3. The intervals overcover, and the width does not come from the pairs.
   Coverage is 0.95-1.0 everywhere, and the 95% width is about 0.9 even
   with every pair term held at the MAP, while the error q90 is 0.01-0.34.
   Step 5's undercoverage was mostly its linearization. The sampled width
   comes from the singles' parameters: in the draws the wt L1, dH and dL
   terms have SDs near 2.6 against a prior of 3, and each mutation's L1
   about 1 against 1.5. Holding one block at the MAP while the others
   keep their marginal draws breaks their correlations, so it does not
   cleanly split the width (holding L1, dH, dL and U widened it). Two
   readings remain open. The singles may leave sloppy directions that
   five stacked mutations extrapolate along, for instance dimerization,
   saturated at these TF levels for every single but perhaps not for a
   five-mutant. Or the Gaussian Laplace may put mass far out along a
   curved ridge the real posterior does not follow. The linearized
   intervals of step 5 and the sampled ones here disagree by a factor of
   several, so neither is trustworthy until checked against a real
   posterior. Step 5c did: the second reading is right.

**Caveats.**
- Noise-free data, and the non-pair terms start at the truth, so the
  point errors are optimistic.
- Six seeds, few coupled pairs each.
- The fit knows each site's function and the true distance law; only
  the pair scale and tails were varied.
- The sampled Laplace widths are wrong; see step 5c.

## Step 5c: the Laplace against HMC (`remedies_hmc.py`)

**Question.** Are step 5b's wide sampled intervals the posterior's, or
the Laplace approximation's?

**First check.** If the Laplace were right, -2 log p of its draws
relative to the MAP would follow a chi-square with as many degrees of
freedom as touched parameters, 581 ± 34 for seed 1 at 15 Å. The Laplace
draws gave a median of 3000 and a 90th percentile of 44,000, so most of
its mass sits where the posterior has none.

**Setup.** HMC on the same posterior: seed 1, the Gaussian pair prior at
the matched scale, no doubles and the 15 Å library. Coordinates are
whitened by the Laplace covariance at the MAP, the step size is tuned by
dual averaging over 400 warm-up iterations, and each chain keeps 600
draws. Untouched pair terms are drawn from their prior, as before. Four
chains per library: acceptance 0.75-0.81, -2 log p of the draws 490 and
581 against 492 and 581 degrees of freedom, split R-hat at most 1.07
over parameters and 1.03 over the five-mutant predictions.

**Results** (seed 1, the same 200 five-mutants as step 5b):

| library | method | cov95 | width | curve | width, pairs held |
|---|---|---|---|---|---|
| none | Laplace | 0.97 | 0.88 | 0.37 | |
| | HMC | 0.92 | 0.59 | 0.11 | 0.32 |
| 15 Å | Laplace | 1.00 | 0.84 | 0.32 | |
| | HMC | 1.00 | 0.34 | 0.057 | 0.19 |

Width is the largest 95% width over the curve, median over targets;
curve is the median width over the curve.

**Reading.**
1. The Laplace intervals are 2-5 times too wide. With the 15 Å library
   the real posterior's median five-mutant width over the curve is 0.057,
   against the Laplace's 0.32.
2. The marginals agree; the joint shape does not. Every parameter's
   posterior SD matches between HMC and the Laplace to within a few
   percent: wt L1, dH and dL near 1.8, 2.7 and 2.6 against a prior of 3,
   each mutation's L1, U, dH and dL near 1.0, 1.5, 1.4 and 1.4 against
   1.5. These are the dimerization and unfolding terms, and the singles
   do not resolve them. But the posterior is a curved ridge along which
   the five-mutant predictions barely move, and the Laplace's straight
   line leaves the ridge and moves them. So step 5b's first reading, that
   the singles leave directions five-mutants extrapolate along, is wrong
   for the posterior: the singles constrain the predictions better than
   they constrain the parameters.
3. With the right posterior, about half of the five-mutant width comes
   from the pair terms (0.19 of 0.34 at 15 Å; 0.32 of 0.59 with no
   doubles).
4. Without doubles the real posterior covers 0.92, slightly below 0.95.
   On noise-free data a calibrated posterior should over-cover, since the
   truth sits where the data are fit exactly, so this is the cost of a
   Gaussian pair prior on a sparse truth. The 15 Å library covers 1.00.
5. For this model Laplace intervals, linearized or sampled, are not
   usable for combinations of several mutations. Predictions need
   posterior draws from a sampler that follows the ridge, or at least
   Laplace draws reweighted by the real density, which here would need
   far more draws than are practical.

**Caveats.**
- One seed and two libraries; HMC is about 25 minutes per chain here.
- Noise-free data, so coverage is an upper bound on calibration: none of
  the noise-free coverages in steps 4 and 5 test calibration, only the
  undercoverage they found.
- One chain of eight stuck; the solver overflowed nearby, so the
  posterior has a region the sampler cannot cross at this step size.

## Step 5d: which approximation matches HMC (`remedies_vi.py`, `remedies_methods.py`)

**Question.** The production fit reports intervals by MAP plus a Laplace
(full or arrowhead, which is the same Gaussian), or by a Gaussian SVI
guide. Which of them gets five-mutant intervals right? The toy is the
one place this can be asked, since only here is the true posterior
(step 5c's HMC) available.

**Methods** (seed 1, both libraries, the same 200 five-mutants):
- the Laplace sampled through the model (step 5b);
- the Laplace propagated linearly (step 5);
- Gaussian variational fits by stochastic ELBO with reparameterized
  gradients and Adam (3000 steps, 4 particles), with mean-field,
  rank-20 low-rank and full-rank covariances;
- importance reweighting of the Laplace draws toward the true posterior.

The variational fits ran in coordinates whitened by the Laplace
covariance, so the mean-field fit here inherits the Laplace's
correlations. It is not the production mean-field guide, which is
diagonal in the raw parameters and should do worse. On a known
Gaussian target all three families recovered the answer to within 6%.

**Results** (ratio is the method's 95% width over HMC's, per target and
curve point; ELBO is -ELBO relative to the MAP's -log p, lower is
better; HMC's -2 log p median is 581 and 491):

| library | method | cov95 | ratio median | ratio q10-q90 | -ELBO | -2 log p median |
|---|---|---|---|---|---|---|
| 15 Å | HMC | 1.00 | 1 | | | 581 |
| | Laplace, sampled | 1.00 | 7.6 | 1.6-33 | | 3316 |
| | Laplace, linearized | 0.99 | 0.80 | 0.49-1.0 | | |
| | mean-field | 1.00 | 0.84 | 0.40-0.98 | 306 | 568 |
| | low-rank | 0.87 | 0.71 | 0.13-0.98 | 362 | 661 |
| | full-rank | 0.93 | 0.74 | 0.17-1.1 | 1381 | 2010 |
| none | HMC | 0.91 | 1 | | | 491 |
| | Laplace, sampled | 0.97 | 3.9 | 1.2-22 | | 1530 |
| | Laplace, linearized | 0.95 | 0.59 | 0.24-1.0 | | |
| | mean-field | 0.91 | 0.90 | 0.43-1.0 | 254 | 484 |
| | low-rank | 0.91 | 0.75 | 0.23-0.99 | 263 | 488 |
| | full-rank | 0.88 | 0.74 | 0.17-1.0 | 1042 | 1261 |

Importance reweighting of 2400 Laplace draws left an effective sample
size of 1.1 and 1.5.

**Reading.**
1. No Gaussian method reproduces HMC. The sampled Laplace is 4-8 times
   too wide at the median and up to 30 times in the tail. The linearized
   Laplace is too narrow, 0.6-0.8 at the median and down to a quarter in
   the tail. The truth lies between the two ways of using the same
   Gaussian, which is the signature of a curved posterior.
2. Importance reweighting cannot repair the Laplace; an effective sample
   size of about 1 means the Laplace and the posterior barely overlap.
3. The best variational fit, mean-field in whitened coordinates, is
   about 10-15% too narrow at the median and 2.5 times too narrow in its
   worst tenth. That is the usual cost of a mode-seeking Gaussian on a
   ridge.
4. The low-rank and full-rank fits did not converge: they nest the
   mean-field family and reached worse ELBOs, the full-rank by far. That
   is a failure of this optimizer on 2e4 and 3e5 parameters in 3000
   steps, not a verdict on the families. They have to be judged by a
   converged fit; it says the production low-rank guide needs its
   convergence checked against a simpler guide's ELBO.
5. For the ported model, multi-mutant intervals should come from a
   sampler (NUTS on the sites a prediction touches, or on the whole
   model where it fits), or from a guide checked against one. MAP plus
   Laplace samples through `tfs-predict-theta` would give combination
   intervals several times too wide.

**Caveats.**
- One seed, two libraries, noise-free data.
- The variational fits are not the production guides: different
  coordinates, a hand-written optimizer, no normalizing flow tried.

**What carries to the real pipeline.** The toy observes θ directly with
Gaussian noise. It has no growth map, counts, ln_cfu0, tube offsets or
congression, and 15 sites in a synthetic geometry with given functional
classes. Its structural results carry over, since the real pipeline can
only add uncertainty to them: identifiability (steps 1-2), the rules for
doubles (step 5: a pair term with them, per-effector and per-operator
effects on the e and o terms, the share of coupling inside the library
as the lever), and the curved posterior that defeats Gaussian intervals
for combinations (steps 5c-5d). Its numbers do not: widths, coverage,
the library size needed and the conditions to measure all have to be
re-measured in the full simulation, where the growth map's uncertainty
may change the balance. The general species model has to be ported into
the pipeline before that can happen.

## Next

**Next step (Mike, 2026-10-31): port the general species model into the
pipeline,** with PK/PnnC/PddG as priors on its terms, a pair term on the
constants with a distance-scaled sparse prior, per-effector and
per-operator deviations on the e and o terms, and posterior draws from a
sampler (or a guide checked against one) for combination predictions.
Then revalidate steps 3-5 in the full simulation. The items below are
the toy's open questions, mostly superseded by that.


The ultimate target (Mike, 2026-10-31) is predicting combinations of five
or more mutations from single and double measurements. Steps 3 and 4
score predictions of doubles from singles; in the target, doubles are
data, and they are the remedy for the higher-order inference. A
background screen is already a double screen (mutation m in background b
is the double m+b), so mutation x background coupling and pair coupling
are one object: a pair term on a physical constant. The planned
structurally designed double libraries (pairs close in 3D, a handful of
effector, operator or TF regimes, no extra backgrounds) calibrate that
term's distance-scaled sparse prior, which then supplies the many
unmeasured pairs inside each higher-order combination (ten in a
five-mutant). Third- and higher-order energy coupling becomes the floor
that neither singles nor doubles can see.

- Step 5 follow-ups:
  - set aside in favor of porting (Mike, 2026-10-31): rerunning step
    5b's comparison with HMC on noisy data, learning the pair scale;
    their numbers would need re-measuring in the full simulation anyway;
  - learn the pair prior's scale and length from the measured doubles,
    by evidence or a hierarchical fit, rather than giving them; step 5b
    found the scale matters more than the tails;
  - plant a little third-order coupling;
  - run with noise;
  - recheck step 3's preference for effector 2 under the per-effector
    model.
- A real noise model before trusting step 3's preference for low TF:
  θ precision from the count likelihood as a function of θ and reads, and
  the growth map (k, m) shared across TF levels, backgrounds and
  effectors, since m belongs to the marker. A lower TF compresses θ's
  range, and with θ-dependent noise or an estimated map that may cost
  more than step 3's constant-SD model allows.
- A tail-targeted design objective, and a weaker operator as an
  alternative to low TF for unsaturating θ.
- Before experiments (Mike, 2026-10-31): full simulations with
  structure-based ΔΔG per conformation (PddG-style) in place of the
  class-based mutation draws, fit by the real machinery rather than a
  Laplace linearization.
