# Is the general dimer model useful in the full pipeline

**Question.** Does `thermo.dimer` (PK, PddG, PnnC) work and help in the
full simulation, with growth, counts, congression and the production MAP
and Laplace, rather than the toy of `planning/studies/thermo-identifiability/`?

**Decision it feeds.** `planning/dimer-thermo-model.md`. Stage A here is
plan step 2: a laptop feasibility check, and any machinery problem it finds
is fixed before the data axes (plan step 3). The compute-heavy utility
grids (plan step 4) run later, on the axis-capable model.

**Date and commit.** Stage A, 2026-11-01, on `d9406fbb` plus the
configure census fix described below.

## Stage A setup

Two simulations from the BGMP structural-prior pilot v2 smoke config
(`inputs/base_smoke.yaml`): 100 + 100 singles at LacI 29-98, 50 doubles
and nine spikes from the 0.5.0 library design, 259 genotypes with growth
data, counts likelihood in 118 tubes (design `inputs/design.csv`), OD600
tube totals, spiked and library binding. The truth's per-state ΔΔG is the
AF3 state table (`inputs/2026-10-11_laci-state-ddg.csv`, kcal/mol at
310.15 K; built in the BGMP project, `structural-models/energetic-calcs/`)
plus Normal(0, 0.5 kcal/mol) per mutation and state. wt is held at each
model's defaults and the truth has no pair epistasis.

- **A1, nesting.** Truth `thermo.O2_C12_K5_U1_a.PddG`. Fit with
  `thermo.O2_C12_K5_U1_a.PK` (`O2PK`) and with `thermo.dimer.PK` with the
  second engagement off (`dimerPK_nested`), which is the same physics. No
  pair term.
- **A2, default variant.** Truth `thermo.dimer.PddG` (unfolded dimer and
  second engagement). Fit with `thermo.dimer.PK` and `thermo.dimer.PddG`
  (learned σ_s), pair term on.

Every fit: configure, staged MAP capped at 50,000 epochs per stage, full
Laplace with 500 draws, the MAP point, extract, predict theta, summarize.

## How to run

From this directory (each fit about four hours on the laptop with all four
running at once):

```bash
~/miniconda3/bin/python make_configs.py
```

```bash
for r in A1 A2; do (cd runs/$r && ~/miniconda3/bin/tfs-simulate tfs_sim_config.yaml --out_prefix sim/tfs_sim --no_write_raw); done
```

```bash
bash -c 'for ra in "A1 O2PK" "A1 dimerPK_nested" "A2 dimerPK" "A2 dimerPddG"; do set -- $ra; nohup bash run_fit.sh $1 $2 > runs/$1/nohup_$2.log 2>&1 & done'
```

```bash
~/miniconda3/bin/python score.py
```

`runs/` is untracked; `results/stageA_*.csv` hold the scores.

## Stage A results

**Cost.** 5,095 parameters for the A2 fits (3,002 theta, 784 of them pair
terms), 3,812 for A1. The staged MAP took about 4 h per fit with four
running at once (150,000 steps), peak 1.3 GB. The full Laplace took 9-12
min and 1.8-2.9 GB. No non-finite theta anywhere.

**No fit converged.** Stages 1 and 3 hit the 50,000-epoch cap in every
arm, stage 3 still falling 85-120 nats per 2,000 steps at step size 1e-4
(1e-5 for PddG), the slowest parameter `growth_inv_r` in the PK fits. The
O2 fit behaves the same, so this is the pipeline's MAP, not the port. All
numbers below are at equal steps and paired, but provisional.

**Nesting passes.** The O2 and dimer fits of A1 reached the same exact loss
at every stage: 220,858.5 against 220,854.7 (stage 1), 209,161.7 against
209,158.9 (stage 2), 183,766.70 against 183,766.88 (stage 3). Their MAP
points differ by RMSE 0.06 in theta and 0.18-0.33 kT in the constants
(r 0.96-0.99), against 0.12 and 0.9-2.4 kT from the truth: flat directions
in an unconverged fit, not a different model. Both score the same against
the truth.

**Hierarchical scales collapse under MAP.** A per-mutation scale whose term
the data do not see goes to its floor, and the Laplace then reports width 0
for every mutation in it, coverage 0:

| fit | collapsed to ~0 | kept (MAP) |
|---|---|---|
| O2PK, dimerPK_nested | σ_d of l_o, u | h_l, h_o, h_e, l_e: 3.9-4.5 |
| dimerPK (A2) | σ_d of l_o, u, l_oii; pair τ (1e-4) | h_l, h_o, h_oii, h_e, l_e: 3.6-4.2 |
| dimerPddG (A2) | σ_s of H, HO, L, LO, HE2; σ_d of l_oii | σ_s of LE2: 1.07 (truth 0.81); pair τ 149 |

The same happens in O2, so it is not new. For PK it costs nothing in theta
(those terms do not reach θ at this design; U in particular stays
negligible), but the reported intervals on those constants are wrong. The
kept σ_d are 3.6-4.5 against a HalfNormal(1) prior; the truth's spreads are
0.6-2.9 kT, so the prior is tight for realistic ΔΔG and the MAP overshoots.

**PddG: the prior becomes a constraint.** With σ_s at 0 for five of six
states the constants sit exactly at the table's projection. At this table
error (0.5 kcal/mol) that is a much better point than PK's:

| constant | PK RMSE | PddG RMSE | PK r | PddG r |
|---|---|---|---|---|
| d_ln_K_h_o | 2.51 | 1.17 | 0.66 | 0.91 |
| d_ln_K_h_l | 1.74 | 1.11 | 0.21 | 0.75 |
| d_ln_K_u | 4.50 | 0.79 | 0.12 | 0.96 |
| d_ln_K_l_o | 1.53 | 1.21 | 0.12 | 0.61 |
| d_ln_K_h_e | 1.10 | 0.54 | 0.37 | 0.69 |
| d_ln_K_l_e | 1.00 | 0.65 | 0.44 | 0.67 |

and theta RMSE 0.089 against 0.106. But its intervals are gone: constant
coverage 0, theta 95% width 0.04 and coverage 0.62 (PK 0.74 and 0.93). Its
exact loss is 13,500 nats below PK's because the collapsed MVN density is
unbounded (the limit in the `prior.sample_dlnK` docstring), not because it
fits better. And with the constants held, the pair term's global scale ran
to 149 although the truth has no epistasis. At a larger table error a
collapsed σ_s would pin the constants to wrong values with zero width.

**Configure census bug (fixed).** `tfs-configure-model` counted parameters
from the default priors, before `--set_priors` was applied, so the nested
dimer reported 2,264 theta parameters instead of 1,698. The fit always read
the edited priors. The census now reloads the written configuration
(`test_priors_edit.py::test_model_stats_count_the_edited_priors`).

## Reading

The port works in the pipeline: it nests O2_C12 U1 to the nat, runs at
realistic cost and gives finite theta. What does not work is MAP plus
Laplace on its hierarchical scales, in O2 as in the dimer. Collapsed scales
give zero-width intervals on unseen terms, and the PddG structural prior's
σ_s collapses into a hard constraint whose point happens to be good at 0.5
kcal/mol. This is a fitting-method problem that the axis version would
inherit, so it belongs before plan step 3. Candidates:

- a non-centered structural prior (Δln K = P·μ + L(σ_s)·z), whose MAP has
  no singular spike, as the pair horseshoe already is;
- a σ_d prior with its mode above 0 and room for 2-4 kT;
- SVI (low-rank guide) or a sampler in place of MAP plus Laplace, which
  plan step 5 needs anyway for combinations;
- a longer stage 3, to confirm the comparisons at convergence.

## Held scales and EM (2026-11-01)

Joint MAP cannot estimate the scales. The PK shifts are already
non-centered (d = offset · σ_d), and stage A shows what MAP does with that:
the kept σ_d ran to 3.6-4.5 while the MAP shifts had RMS 0.8-1.4 (truth
SDs 0.7-2.9), and σ_d of an unseen term fell to 0. So a non-centered
structural prior would trade PddG's collapse for the same runaway. We
instead hold every scale (`theta_sigma_d_{term}_fixed`,
`theta_ddG_sigma_{state}_fixed`, added to thermo.dimer for this) and update
the held values between fits.

`dimer_em.py` runs that loop on A2: configure with the held scales, MAP
from the previous point (30,000 epochs at step 1e-4), full Laplace, then
the EM update σ² = mean over mutations and draws of the squared shift (for
PddG, of the per-state residual P⁻¹(Δln K − Pμ)). Every scale started at
1 kT. We stopped it after two iterations: EM moved the informed scales
about 7% per iteration (PK h_o 1.00 → 1.10 → 1.17, truth 2.89), far too
slowly for a fit that costs an hour per iteration.

`direct_update.py` replaces the EM step with one maximization of the
Laplace-approximate marginal likelihood. It takes each mutation's
likelihood curvature from the Laplace with the held prior removed, keeps
only curvature beyond the Monte Carlo error of 500 draws, and adds the
structural prior's own LogNormal(0, 1.2) hyperprior on each scale. On the
iteration 1 posteriors:

| scale (kT) | held | direct update | truth SD |
|---|---|---|---|
| PK h_l | 1.08 | 1.61 | 1.68 |
| PK h_o | 1.10 | 1.62 | 2.89 |
| PK h_oii | 1.13 | 1.82 | 1.11 |
| PK h_e, l_e | 0.92, 0.95 | 0.80, 0.93 | 0.74, 0.82 |
| PK l_o, u, l_oii (unseen) | 1.00 | 0.82, 1.00, 1.00 | 1.6, 2.9, 0.6 |
| PddG σ_s H, HO, L, LO | 0.79-0.96 | 0.08-0.24 | 0.81 |
| PddG σ_s HE2, LE2 | 1.07, 1.23 | 1.37, 1.74 | 0.81 |

The PK update is usable: one step reaches the truth for most informed
terms, and the unseen terms stay at the hyperprior median. The PddG σ_s
update is not. The data barely resolve per-state residuals (posterior SD
0.6-0.9 kT against prior 0.8-1.0, correlation with the true residual
0.05-0.43), and the update sends four states to near 0 with a 147-nat
gain, more confidence than evidence that weak can carry. The per-mutation
block approximation (it ignores the Laplace's coupling across mutations
and to the shared growth parameters) is the likely cause; we did not
pursue it.

Scores of the iteration 1 fits, scales held near 1 kT
(`results/em_iter1_scores.csv`):

| | stage A PK | EM PK | stage A PddG | EM PddG |
|---|---|---|---|---|
| constant 95% coverage | 0-1.0 (0 on 3 terms) | 0.37-1.0 | 0-0.61 | 0.71-1.0 |
| constant RMSE, h_o / u | 2.51 / 4.50 | 2.50 / 4.49 | 1.17 / 0.79 | 1.10 / 0.72 |
| θ RMSE | 0.106 | 0.084 | 0.089 | 0.095 |
| θ 95% coverage | 0.93 | 0.44 | 0.62 | 0.63 |
| θ 95% width | 0.74 | 0.042 | 0.040 | 0.039 |

Holding the scales fixes the constants. Every PddG constant is now covered
at 0.71-1.0, from 0, with the stage A point accuracy kept or improved.
PK's unseen terms get prior-width intervals that cover (l_o 0.86, l_oii
1.0), except u (0.37), whose true spread of 2.9 kT is three times the held
1 kT. Stage A PK's θ coverage of 0.93 came from absurd widths (0.74 on a 0
to 1 scale, from σ_d ≈ 4).

**θ is overconfident in every fit**: 95% widths of 0.04 against a θ RMSE
of 0.08-0.10, coverage 0.44-0.63. That does not come from the scales, since
it is the same with them held. Candidates: the wt constants and growth
parameters (k, m, dk_geno) the full Laplace holds near a MAP short of its
optimum (every MAP here hit its epoch cap), and the per-genotype dk_geno
and ln_cfu0 trading with θ. It is the next thing to diagnose before plan
step 3, and it bears on O2 fits as much as on the dimer.
