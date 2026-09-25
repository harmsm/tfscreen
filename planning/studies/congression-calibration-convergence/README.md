# Congression calibration: SVI convergence review

**Question.** Four rounds of diagnosis on the step 3.5 calibration grid
(detection floor, binding weight, unanchored k/dk_geno slide, ELBO/Jensen)
each explained the mixture fits' endpoints, and none of the fixes changed
them. Are those endpoints converged fits at all? If they are not, what does
the mixture actually do during training?

**Decision it feeds.** What to do next in step 3.5 of
[`planning/congression-physics-plan.md`](../../congression-physics-plan.md):
whether to keep diagnosing the mixture model, or fix the optimization first.
The routes forward below are proposals, not decisions.

## Summary

- **None of the 3.5 fits converged**, `single` included. The stop rule
  measures loss change relative to the loss at the start of SVI, which is
  ~500 times the final loss, so runs stop while the loss is still falling
  20 to 30% per 1000 epochs and the guide scales are still shrinking.
- **The mixture's "k drift" is an early excursion that was still unwinding
  at the stop.** In the high-variance start of SVI (guide scales at prior
  width, nu collapsing to ~1.3), the mixture's `growth_k` overshoots the
  truth by +0.12 at lambda 0 and +0.05 at lambda 1, then comes back
  slowly. `single`, on the same data, holds `growth_k` at the truth
  throughout.
- **The simulated library is dominated by wt**, which is 39% of the
  co-resident pool, so most congressed classes are "genotype + wt". wt is
  also the only genotype with a pinned dk_geno.
- Nothing here points at the congression physics or the mixture's design.
  The earlier diagnoses should be re-judged on converged fits.

## How to run

The scripts read a grid pulled from the cluster (`fetch.sh` in
`planning/studies/congression-calibration/`), by default the anchored grid
(`grid_anchor.yaml`) inside that study directory. From this directory, with
tfscreen importable:

```bash
python trace_checkpoints.py                  # runs 0001, 0002, 0005
python coresident_pool.py                    # run 0002
GRID=/path/to/grid python trace_checkpoints.py run_0002 run_0003
```

Both print to stdout and write nothing. `trace_checkpoints.py` takes about
20 s on a laptop CPU.

## Inputs

- Anchored grid (`planning/studies/congression-calibration/grid_anchor.yaml`,
  set up on c7b228d, run 2026-09-24): binding weight 1, prefit
  `--k_scale_ceiling 0.005`, lambda 0 and 1.0, wide dk_geno spread, seed 1.
  Run 0001 is `single`, 0002 the mixture with the matched lambda prior at
  simulated lambda 0 (fitted lambda ~0.004), 0005 the mixture with the matched
  prior at lambda 1.0. Runs 0001 and 0002 were fit to the same simulated data.
- Each run's `checkpoints/*_checkpoint.pkl` (every 1000 epochs),
  `tfs_fit_model_checkpoint.pkl`, `tfs_fit_model_losses.txt`,
  `tfs_sim_growth_parameters.csv`, `tfs_configure_library.csv`,
  `tfs_sim_library.csv` and `tfs_sim_presplit.csv`.

## Commit

Scripts written and run on 2d86049 (2026-09-25).

## Results

### 1. The stop rule fires on the start-up transient

`RunInference._update_loss_deque`
(`src/tfscreen/tfmodel/inference/run_inference.py`) defines the metric as

```
|mean(old window) - mean(new window)| / (loss_start - loss_best)
```

with `loss_start` taken once, when the first two 10-epoch windows of SVI
fill. At that point the component guide's per-genotype scales are still at
their initial 1.0 and the initial values carry 0.1 jitter, so `loss_start` is
~8e7. The runs end near 1.5e5 to 2.3e5:

| run | `loss_start` | `loss_best` | stop threshold on the 10-epoch change (tolerance 1e-4) |
|---|---|---|---|
| 0001 `single` | 7.64e7 | 2.29e5 | 7,620 |
| 0002 mixture, lambda 0 | 8.93e7 | 1.68e5 | 8,910 |
| 0005 mixture, lambda 1 | 8.91e7 | 1.34e5 | 8,900 |

The threshold is effectively "the loss has fallen by 99.99% of its initial
value," not "the loss has stopped changing." At the stop the smoothed loss is
still falling 20 to 30% per 1000 epochs (run 0002: 3.6e5 at epoch 7000,
2.5e5 at 8000, 1.9e5 at 9000).

The original 54-run grid (default binding weight) ended at losses of 9e6 to
1.2e7, even further from converged.

### 2. The pre-MAP warm-up is close to a no-op

`_run_map` (`src/tfscreen/tfmodel/scripts/fit_model_cli.py`) sets the
learning-rate schedule to decay from `adam_step_size` to
`adam_final_step_size` (1e-3 to 1e-6) over `max_num_epochs`, and the pre-MAP
passes `max_num_epochs=pre_map_num_epoch` (1000). The step size therefore
collapses within the warm-up. Adam moves a parameter by roughly the step
size per step, so about 0.15 in total over the warm-up. The pre-MAP loss
goes from 6.5e7 to 5.3e7, against ~1.5e5 at the end of SVI. SVI starts far
from the mode, with prior-width guide scales (`dk_geno_offset_scales` and the
other per-genotype scales initialize to 1.0).

### 3. Parameter trajectories

`trace_checkpoints.py`. `growth_k` for kanR+kan (truth 0.0107) in the three
runs; guide scale, nu, sigma_k and loss from run 0002 (runs 0001 and 0005
follow the same pattern). nu and sigma_k are guide medians; "dk scale" is
the median per-genotype `dk_geno_offset_scales`.

| epoch | `single` | mixture, λ 0 | mixture, λ 1 | dk scale | nu | sigma_k | loss |
|---|---|---|---|---|---|---|---|
| 1000 | 0.007 | 0.012 | -0.003 | 1.00 | 19.6 | 0.04 | 6.0e7 |
| 2000 | 0.003 | 0.096 | 0.051 | 0.93 | 7.5 | 0.10 | 1.9e7 |
| 3000 | 0.013 | 0.125 | 0.057 | 0.85 | 3.0 | 0.25 | 5.0e6 |
| 4000 | 0.016 | 0.135 | 0.046 | 0.78 | 1.3 | 0.57 | 2.0e6 |
| 5000 | 0.011 | 0.128 | 0.042 | 0.65 | 1.3 | 1.08 | 9.7e5 |
| 6000 | 0.010 | 0.128 | 0.039 | 0.50 | 2.1 | 1.15 | 5.5e5 |
| 7000 | 0.011 | 0.118 | 0.039 | 0.36 | 3.5 | 1.06 | 3.6e5 |
| 8000 | | 0.107 | 0.039 | 0.26 | 5.2 | 0.91 | 2.5e5 |
| 9000 | | 0.094 | 0.034 | 0.20 | 6.4 | 0.77 | 1.9e5 |
| stop | 0.009 (7982) | 0.092 (9234) | 0.030 (9876) | 0.18 | 6.6 | 0.74 | 1.8e5 |

The other three conditions move the same way (run 0002 at the stop: 0.090,
0.109, 0.115 against truths 0.015, 0.021, 0.029), and `dk_geno`'s hyper-shift
mirrors them (0.024 at epoch 1000, -0.169 at 4000, -0.079 at the stop).
Lambda in run 0005 also overshoots and returns: 1.02, 1.27 (epoch 2000), 0.95
(8000), 0.98 at the stop.

What the table shows:

- Between epochs 1000 and 5000 the fit escapes the high guide variance by
  inflating the noise: nu falls from ~20 to ~1.3 (Cauchy-like tails) and
  sigma_k rises from 0.04 to ~1.1 ln units. `single` does the same (nu 1.1 at
  epoch 5000).
- In that phase nothing holds the k/dk_geno slide. Under near-Cauchy tails,
  wt's pinned dk_geno costs only logarithmically to abandon, and the prefit's
  tight k prior (scale 0.005; ~1,400 log units at the run 0002 peak) is small
  next to a loss of 1e6 to 1e7.
- `single` wanders by about ±0.01 and ends at the truth. The mixture's
  `growth_k` jumps within the first 1000 epochs of SVI and peaks near epoch
  4000. It then returns at about 0.01 per 1000 epochs as the guide scales
  shrink and nu recovers, and was still falling when the run stopped.

This also accounts for the earlier findings from the endpoints. Under the
mixture's own joint density, `single`'s solution beat the mixture's endpoint
by ~18,000, and the mean gradient at that endpoint pointed k back down
(`diag_basin.py`, `diag_gradient.py`). That is what a fit stopped
mid-recovery looks like, not a real mode or an overshoot.

### 4. Why the mixture's excursion is larger (hypothesis, untested)

In the high-variance phase, a guide draw often puts a genotype's dk_geno deep
in the lognormal's negative tail, so its clean class dies in that draw. Its
congressed classes rescue the draw, because dilution halves the focal dk.
Most congressed classes contain wt (result 5), whose dk_geno is pinned at 0.
Sliding k up by C and every other dk_geno down by C leaves clean classes
unchanged but speeds up every wt-containing class by C/2, which makes the
rescue more effective and improves the expected log-likelihood under the
guide.

The excursion sizes fit this: rescue classes weighted ~0.2% (lambda 0.004)
need a larger boost than classes weighted 42% (lambda 1), and the excursion
is +0.12 against +0.05. This is the Jensen/ELBO idea in a concrete form. It
acts in the start-up phase, when guide scales are 0.8 to 1.0. `diag_elbo.py`
evaluates at the endpoints, where the scales are 0.15 to 0.26 and the effect is
much weaker.

### 5. The simulated library is dominated by wt

`coresident_pool.py`:

| quantity | value |
|---|---|
| wt share of the co-resident pool | 0.392 (next largest: 0.023) |
| wt slots per genotype, `congression_sets` [12, 3, 1] | 8.2 of 21 |
| wt entries in `single-1` / `single-2` / `double-1-2` | 30 of 46 / 36 of 67 / 1080 of 3082 |
| wt share of presplit cells (true ln_cfu0), replicate 1 / 2 | 0.61 / 0.22 |

`LibraryManager._prepare_indiv_lib_blocks`
(`src/tfscreen/genetics/library_manager.py`) treats every codon of a tile as
a site, and a non-degenerate codon's only "mutant" is its wt codon. The study
config's `degen_sites` has one degenerate codon in tile 1's 31 codons and two
in tile 2's 37, so each single sub-library gets one wt entry per
non-degenerate codon. For example, `single-1` has 30 wt entries plus the 16
NNT codons at site 42. The simulator and the fit share the enumerator, so
this is not a sim/fit mismatch. It does make this simulation unrepresentative
of a library whose tiles are fully degenerate. Whether the enumeration is
intended is an open question.

### 6. Status of the earlier diagnoses

| diagnosis (study README) | status |
|---|---|
| Detection floor | The floor bias itself (0 reads: +1.26 ln) is a data-side fact and stands. The "congressed classes are worth 36,700" measurement was taken at an unconverged endpoint. |
| Binding weight | Weight 251 times a binding SD of 0.001 dominating the objective is a property of the objective and stands. Its role in the mixture's drift is undetermined; the weight-1 fits drifted the same way, as the transient predicts. |
| Unanchored slide and Student-t | Partly right: the slide is loose during the transient because nu collapses. "The mixture abandons wt by 13.6 ln units" is a mid-recovery snapshot. |
| ELBO/Jensen (`diag_elbo.py`) | Plausible as the driver of the early excursion (result 4), but testing it at the endpoints measures a much weaker effect. |
| Grid-level numbers (lambda pulled up, bulk theta coverage 0.65, mixture theta RMSE above `single`'s, `single`'s -0.02 k drift) | All from unconverged fits, `single`'s baseline included. None are yet properties of the models. |

## Is the congression architecture at fault?

Probably not:

- The forward model is verified: lambda -> 0 reproduces `single` to 1e-6.
- The estimator is accurate at the true parameters (step 3.0,
  `planning/studies/congression-estimator/`).
- The matched-prior lambda 1 fit recovered lambda (0.98).

Two structural points to keep in view:

- **Growth data barely inform lambda in the measured regime.** Congression
  there is ~0.05 ln RMS, and `single`'s errors did not change with the
  simulated lambda. The plan already sets such parameters from biology.
- **A mixture of exponentials is dominated at late times by its fastest
  class.** That mimics the detection-floor bias in dying genotypes and makes
  a genotype's likelihood bimodal (clean-dominated vs congressed-dominated).
  Inference is most fragile there. Look at it once fits converge.

## Routes forward (proposals)

In rough order of cost:

1. **Resume the anchored runs to real convergence.** No new simulations. A
   resumed fit re-bases the stop rule: `run_optimization` resets
   `loss_start`, so the denominator becomes the resumed run's own
   improvement. From a *copy* of a run directory (the resume overwrites
   `tfs_fit_model_checkpoint.pkl` and `tfs_fit_model_losses.txt`):

   ```bash
   tfs-fit-model tfs_configure_config.yaml --seed 1 \
       --checkpoint_file tfs_fit_model_checkpoint.pkl \
       --convergence_tolerance 0.0001
   ```

   Keep `--max_num_epochs` at its default. It also sets the step-size decay
   horizon, and the optimizer's step count carries over from the checkpoint,
   so shortening it drops the step size abruptly. Against the resumed run's
   smaller improvement, the loss noise will probably keep the stop rule from
   firing, so expect the full 100,000 epochs: about 20 min on the cluster
   GPU (these runs took ~12 s per 1000 epochs). Then run
   `trace_checkpoints.py` on the copy. This settles "unconverged" vs "a real
   pathology" for runs 0001, 0002 and 0005.
2. **Fix the start and the stop** (code changes):
   - run the pre-MAP to convergence, with a step-size schedule that does not
     collapse within the warm-up;
   - start the component guide's per-genotype scales small (for example 0.01
     to 0.1, like AutoNormal's `init_scale`) rather than at prior width;
   - measure convergence against the current loss or parameter movement,
     not the start-up transient.
3. **Profile lambda with MAP.** Fix lambda on a grid, optimize everything
   else to convergence, on the lambda 0 and lambda 1 data. This answers "does
   the data pull lambda up?" without SVI.
4. **Test self-consistency before realism.** Fit data generated by the
   numpyro model itself (fixed known parameters), `single` vs mixture at
   lambda 0. This separates inference from sim/fit mismatch. Then add the
   realism back one piece at a time: per-tube noise, pseudocount floor,
   binding.
5. **Consider pinning lambda** as a constant by default, with lambda as a
   latent an option. This follows the plan's rule for parameters that only a
   lambda series could identify.
6. **Re-judge the earlier diagnoses on converged fits** (result 6).
7. **Library.** Decide whether the wt enumeration is intended, and use a
   simulation library with a realistic wt share. Check the real library's wt
   share of the co-resident pool.

`grid_controls.yaml` (2d86049) is worth holding until item 2 is done:

- Its MAP arms use the same stop rule (MAP `loss_start` ~6.5e7) and will
  likely stop early too.
- Its `fixed0` arm (lambda ~1e-6) still has congressed classes, at weight
  ~5e-7, which a large enough early excursion could use. Drift there would
  not by itself mean a bug in the mixture code.

**Practice.** Before interpreting any fit, run `trace_checkpoints.py` (or
equivalent) on its `checkpoints/` and confirm that the key parameters have
stopped moving.
