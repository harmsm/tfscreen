---
title: Fit growth alone on a wt-relative scale, then measure the growth-binding map
status: idea
filed: 2026-09-26
area: tfmodel
revisit_when: >-
  When we want to test, rather than assume, that in vitro binding and in vivo
  growth report the same occupancy. Step 0 (model-free look) can be run any
  time on the existing real data.
related:
  - src/tfscreen/tfmodel/model_orchestrator.py
  - src/tfscreen/tfmodel/generative/model.py
  - src/tfscreen/tfmodel/scripts/configure_model_cli.py
  - src/tfscreen/tfmodel/scripts/prefit_calibration_cli.py
  - src/tfscreen/tfmodel/analysis/prediction.py
  - src/tfscreen/tfmodel/generative/components/theta/hill_geno.py
  - src/tfscreen/tfmodel/generative/components/theta_rescale/
  - planning/congression-physics-plan.md
  - planning/studies/congression-calibration/README.md
---

**Context:** While calibrating fits for congression, it became clear we want
to fit growth data without binding data, at least to check whether the two
are consistent. An investigation (2026-09-25) found the plumbing is small but
that growth data alone do not fix an absolute theta scale. Discussion then
reframed the goal: don't ask for absolute theta yet.

The joint fit currently asserts, at minimum:

1. In vitro binding equals in vivo occupancy (same effective [IPTG], same
   [protein], no genomic-site competition, ...).
2. Growth is linear in that occupancy: `g = k_c + dk_g + m_c·A_g·θ_g`.

It also overweights binding: each binding row is scaled by
`N_growth_rows / N_binding_rows` (`model_orchestrator.py`, `binding_weight`
auto-computation), so the binding set as a whole counts as much as the entire
growth dataset. Any binding/growth mismatch is resolved by distorting growth.

**Idea:** Infer a hidden, IPTG-dependent growth variable X from growth alone,
on a scale defined relative to wt. Then ask empirically what transfer function
maps X to the theta measured in vitro. Base case: linear. Likely case:
monotone but nonlinear. Worst case: no map. Only after that, build whatever
map we find into a joint fit.

*Why a relative scale is identifiable.* Growth data fix X only up to a global
affine map `X -> a·X + b` (m and k compensate); this is exactly the
scale/sign non-identifiability of theta without binding (m -> c·m,
θ -> θ/c; θ -> 1-θ, m -> -m, k -> k+m). Hill K and n are invariant under it,
as are ratios to wt (e.g. a genotype's dynamic range relative to wt's). Fix
the gauge by convention: `X_wt = 1` at no IPTG, `X_wt = 0` at saturating
IPTG. Then `k_c` = wt's growth at saturating IPTG and `m_c` = wt's growth
change across IPTG, read straight from abundant wt data; the sign of m falls
out per condition. No binding data and no binding-based prefit needed.
(Design choice: pin wt's Hill asymptotes, simplest, or its values at the
measured end concentrations, if wt does not saturate in range.)

*What stays assumed.*
- One X per genotype × [IPTG], shared across conditions and pre/sel phases.
  Within a condition "growth linear in X" is a definition; across conditions
  it is testable (kan and 4CP responses should be affine in each other per
  genotype).
- X is Hill-shaped in IPTG. A strongly nonlinear true map would show as lack
  of fit.
- dk_geno is independent of IPTG and condition. The baselines are the weakest
  quantity: a genotype's X at no IPTG trades against dk_geno and is separated
  only by selective-vs-control contrasts (control conditions have not ranked
  dk_geno well so far).
- Congression: the mixture's max-theta rule is invariant under any increasing
  map (`max f(θ) = f(max θ)`), so it carries over if X is oriented to
  increase with repression.

**Step 0 (model-free, do first).** For the genotypes with binding data:
regress ln_cfu on t_sel per genotype × replicate × condition × concentration
(slope = selection-phase growth rate, since t_pre is fixed within a sample);
subtract wt (cancels per-tube artifacts, leaving `dk_g + m_c·(X_g - X_wt)`);
plot against binding theta, interpolating binding through its Hill fits if
concentrations differ. Spiked genotypes are deep and congression-free. One
collapsing curve → linear/monotone case; genotype-specific scatter → the
worse case. This answers which case we are in before any component is built.

**Stage 2 design (the map).**
- The map is between curves, not points: `X_g(c) = f(θ_g(s·c))` with a global
  concentration scale s (in vivo effective [IPTG], assumption 1) and a value
  map f (assumption 2). Nested family: identity → affine f → monotone f, plus
  per-genotype deviations (activity, in-cell effects).
- Stratify spiked vs bulk: congression should show as bulk genotypes on a
  compressed version of the spiked curve.
- Errors in both variables; genotype is the unit, concentrations are
  within-genotype points.
- Risk: converged mean-field SVI is overconfident (bulk theta 95% coverage
  ~0.66 in the congression-calibration study), which would make a real
  monotone map look like "no map". Needs an extra dispersion term or better
  calibrated posteriors.
- Eventual formalization: `theta_rescale` is already a fixed transfer
  function slot (`passthrough`/`logit`); the joint model would make it
  learnable and add s, with binding at weight 1.

**Why not now:** Current work is the congression physics plan; this is a
change of strategy for what the fit asserts, and step 0 should decide whether
it is worth building.

**What it would take:**

*Growth-only plumbing* (found 2026-09-25; no model component reads
`data.binding`, and batching, `get_batch`, `batch_safety`, `RunInference`,
extraction, `predict-theta`/`-epistasis`, `model_stats` counts and config data
paths already cope with no binding):
- `configure_model_cli.py`: `binding_df` is a required positional that raises
  when None → make it `--binding_df`, require at least one of growth/binding
  in the body (like `--library_config`). Line ~389 writes a phantom
  `data.binding: binding.csv` when binding is absent. Breaks positional
  callers: `examples/simulate-and-analyze/run.sh`, `test_base_growth.py`,
  `test_configure_and_run.py` (which asserts `configure_model(None)` raises).
- `ModelOrchestrator`: joint `_initialize_data` builds binding tensors,
  `BindingData` and auto `binding_weight` unconditionally (first failure at
  `_read_binding_df`); gate them. `_setup_batching` with an empty binding set
  already gives `num_binding=0`. In `_initialize_classes`, drop
  `theta_binding_noise`, skip `observe_binding`, set `priors.binding=None`
  explicitly (`populate_dataclass(BindingPriors, {})` raises). Infer the mode
  from `binding_df is None` rather than adding a `growth_only` config key
  (every `components:` key becomes an orchestrator kwarg).
- `generative/model.py`: gate the binding prediction, noise, deterministics
  and observer on `data.binding is not None` (static structure, jit-safe).
- `analysis/prediction.py`: `orchestrator.binding_df.copy()` crashes, breaking
  `tfs-predict-growth`, `tfs-sample-prior` and summarize-fit trajectories.
- `summarize_fit_cli.py`: with no binding genotypes the trajectory plots fan
  out to every genotype (a CSV + PDF each); needs a default subset.
- Fail fast on `binding_weight` or non-`zero` `theta_binding_noise_model`
  without binding. Minor: `predict_growth_cli.py` survives only via a
  try/except; `model_stats` warning wording; setup-grid `auto` enumerates
  `theta_binding_noise`; `None` prior leaves round-trip through the priors CSV
  as NaN (pre-existing, binding-only has it too).
- Tests: none cover growth-only. `growth-smoke.csv` + `library-smoke.yaml`
  without binding is a ready fixture. Needs analogues in `test_model.py`,
  `test_model_orchestrator.py`, `test_batch_safety.py`, `test_model_stats.py`,
  `test_configure_and_run.py`, and a smoke test.

*Relative-X fit:*
- New theta component (e.g. relative Hill): hill_geno's curve form but with
  real-valued baselines (not logit-bounded to [0,1], since mutants can fall
  outside wt's range), wt baselines pinned to (1, 0), population priors
  re-centered on wt. Must follow the registry rules (library-sized plates,
  `return_population`, `{site}_loc(s)`/`{site}_scale(s)` naming).
- Force `activity=fixed` (redundant with X's amplitude), `passthrough`,
  `linear` growth; per-genotype curves only (hill_mut's logit additivity and
  the thermo components presuppose absolute theta).
- `tfs-prefit-calibration` (needs binding by construction) becomes
  unnecessary or a wt-only fit.
- Downstream: logit epistasis, `in_regime` and anything assuming θ in (0,1)
  do not apply to X; only additive-on-X epistasis until f is known.

*Stage 2 analysis:* a study script first (`planning/studies/`), a CLI only if
it earns one.

*Validation:* on one sim, compare joint vs relative-X fits;
`tfs-summarize-calibration` stratifies by `has_binding` from the simulated
binding file (not what the fit saw), so held-out comparison is free. A sim
with a known nonlinear f and an [IPTG] scale s would test whether stage 2
recovers them.

*Alternative considered (absolute theta without binding in the fit):* keep
absolute theta but take m from binding only through the prefit
(`tfs-prefit-calibration --binding_df` for a growth-only config, with
`--pin_m`), or copy a calibration from another run, or rely on priors. Set
aside because it still asserts assumptions 1 and 2; the out11 ablation also
found free m with ~20 in-library binding anchors beat the prefit clamp
(theta slope 0.91 vs 0.85; older machinery).

**Open questions:**
- Gauge: pin wt asymptotes or wt values at the measured end concentrations?
- Shared X across conditions, or per-condition X and test the relation?
- Do growth and binding concentrations coincide in the real data?
- Default genotype subset for summarize-fit trajectories without binding.
