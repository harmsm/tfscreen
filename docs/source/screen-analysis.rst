==================
Analyzing a screen
==================

This page takes a real screen from per-tube read counts to one table of
per-genotype response curves with 95% intervals. It is the route used for
the current manuscript. :doc:`pipeline` lists every command the package
has; this page is the subset that a screen analysis needs, in order, with
the reasons for each choice.

The result is a curve of *X* against IPTG for every genotype. *X* is an
occupancy-like variable gauged on wild type: wt's *X* is 1 at the lowest
IPTG concentration and 0 at the highest. Read `What X is`_ before using
the numbers.

.. contents:: On this page
   :local:
   :depth: 1

What you need
=============

- **One counts file per tube**, from ``tfs-process-fastq`` (see
  :doc:`process-raw`).
- **The tube table**: one row per sequenced tube with its library,
  replicate, pre-growth and selection conditions, times and IPTG
  concentration.
- **OD600 readings** for every tube and the **OD600 calibration** for the
  instrument and strain (``tfs-calibrate-od600``), plus the culture volume.
- **The library YAML**, the same file given to ``tfs-process-fastq``.
- **A corrections file** for the tube table, if any tube was mislabeled,
  swapped or failed (see `Correct the tube table`_). Start with none.
- **The measured co-transformation rate** lambda, as a mean and SD. The
  fit requires it. For the 2026 screen it was 0.357 (likelihood-ratio 95%
  CI 0.156-0.674), entered as 0.38 +- 0.15.
- **Loose growth priors**: one row per condition with a growth baseline
  ``k`` and slope ``m``. The values only need to be the right order of
  magnitude; ``examples/tfmodel/growth_priors_loose.csv`` is the file used
  for the 2026 screen.
- **A GPU**, for the fits. The whole analysis takes about a day of GPU
  time (see `Cost`_).

Correct the tube table
======================

Check the tubes before fitting. ``tfs-check-tubes`` scores each tube's
apparent IPTG concentration from its genotype composition and flags tubes
that are out of order in their row of the design (a replicate at one
time). Tube totals and read depth cancel out of the score, so it tests the
labels, not the OD600.

.. code-block:: bash

   tfs-process-counts tube_table.csv counts \
       --od600_file od600.csv \
       --od600_calibration_file od600_calibration.yaml \
       --tube_volume_mL 5 \
       --out_prefix growth_unchecked
   tfs-check-tubes growth_unchecked.csv --out_prefix tube_qc

Read the whole row of any flagged tube. A single swap is common and is
fixed with a ``swap``; a row whose order cannot be recovered is left out
with ``exclude``. Each correction records who decided it, when, why and
the evidence, and goes in a YAML file kept with the experiment's data
(format: :doc:`process-raw`, "Correcting the tube table:
tfs-correct-tubes"; ``examples/process_raw/corrections.yaml``). Then
apply it and make the growth table the fit reads:

.. code-block:: bash

   tfs-correct-tubes tube_table.csv corrections.yaml --out_prefix tube_table_corrected
   tfs-process-counts tube_table_corrected.csv counts \
       --od600_file od600.csv \
       --od600_calibration_file od600_calibration.yaml \
       --tube_volume_mL 5 \
       --out_prefix growth

The original tube table is never edited. ``tube_table_corrected_changes.csv``
lists every change.

For the 2026 screen one row is excluded: pheS+4CP, replicate 2, t_sel 82.
Its IPTG tubes were out of order in a way no single swap explains.

Configure the fit
=================

The fit is one command, ``tfs-fit-two-stage``, driven by one YAML file.
Its top-level keys are ``tfs-configure-model``'s arguments; the
``two_stage`` block holds the fit's own settings. Paths are relative to
the YAML. Start from ``examples/tfmodel/two_stage.yaml``:

.. code-block:: yaml

   growth_df: growth.csv
   library_config: library_config.yaml
   theta_model: hill_relative
   growth_shares_replicates: true
   batch_size: 4096
   allow_missing_spikes: true
   growth_priors: growth_priors_loose.csv
   set_priors:
     - sigma_fixed=0.17
     - theta_log_hill_n_hyper_scale_fixed=0.5
   skip_model_stats: true

   two_stage:
     transformation_lambda: [0.38, 0.15]   # the measured lambda, required
     seed: 1
     num_posterior_samples: 500
     consistency: true
     consistency_library: kanR
     predict_epistasis: true

Keep everything except the data paths, ``transformation_lambda`` and
``consistency_library`` as shown. These are the settings the validation
used. ``sigma_fixed`` holds the per-tube level offsets at an SD of 0.17
(left free, they absorbed the population's growth on real data), and the
held log Hill-n spread keeps genotypes from hiding their response below
the lowest IPTG concentration. Set ``predict_epistasis: true`` if you want
epistasis on *X*.

The command refuses settings it does not support: binding data, theta
noise models, the experimental joint-fit priors and a marker warp
(``m_warp``) in the growth priors.

What the fit does
=================

Each step writes its own directory under ``--out_dir`` and is skipped
when its outputs exist, so a stopped run resumes where it stopped.

1. **stage1**: wt, the spiked controls and the single mutants (about 950
   genotypes), fitted with the congression mixture and the measured
   lambda. This gives every condition's growth baseline *k* and slope *m*.
   The full library cannot be used for this: each double mutant pulls *m*
   a little steeper, and over 200,000 doubles the pulls compress *X* about
   1.45-fold.
2. **stage2**: the full library with every condition's *k* and *m* pinned
   at stage 1's values, then the Laplace approximation of the posterior.
3. **predict**: stage 2's *X* from the posterior draws, and its epistasis
   with ``predict_epistasis: true``.
4. **consistency**: stage 2 refitted on the kanR tubes alone, compared
   genotype by genotype with the fit on both markers.
5. **summary**: the results table, ``tfs_x.csv``.

Two more steps, **jackknife** and **kshift**, try to add *k*'s
uncertainty to the intervals. They are experimental and off by default
(``kshift: false``): on simulations the shift they measured was a third
or less of the true one (``planning/studies/two-stage-validation/``,
run 2).

Run it
======

On one machine with a GPU, everything in order:

.. code-block:: bash

   tfs-fit-two-stage two_stage.yaml --out_dir two_stage

On a SLURM cluster, submit each step as its own job. ``--step NAME``
runs one step. ``examples/tfmodel/two_stage_submit.sh`` submits the whole
chain with dependencies, using ``examples/tfmodel/two_stage_step.srun``
for each job:

.. code-block:: bash

   bash two_stage_submit.sh two_stage.yaml two_stage

It prints the job ids. Edit the account, partition and time in
``two_stage_step.srun`` for your cluster first. Every path the jobs use
is made absolute, so the command works from any directory.

Cost
----

On one GPU, for a library of about 200,000 genotypes: stage 1 takes 15-30
minutes, stage 2 two to five hours with its Laplace, and the consistency
refit, which runs beside prediction, about five hours. About half a day
in all.

Read the results
================

``{out_dir}/tfs_x.csv`` has one row per genotype and IPTG concentration:

``genotype``, ``num_mutations``, ``titrant_name``, ``titrant_conc``
   The row.
``q0.001`` ... ``q0.999``
   Quantiles of *X* (``q0.5`` the median, ``q0.025`` and ``q0.975`` the
   95% interval).
``k_shift``
   Blank unless the experimental ``kshift`` is on: then *X*'s shift per
   SD of *k* at this row, already included in the quantiles.
``in_training_data``
   1 if the genotype was measured at this concentration.
``theta_scale``
   ``X``.
``reads``, ``reads_kanR``, ``reads_pheS``
   The genotype's reads over all tubes, in total and per library.
``consistency_z``, ``inconsistent``
   How far the genotype's kanR-only *X* sits from its both-marker *X*,
   in SDs after the overall map between the two fits, and whether that
   exceeds 2 SD. Blank for wt and when the check did not run.

Which genotypes to use
----------------------

Use genotypes with at least 1,000 reads. Below that, *X* is mostly set by
the population prior: the intervals still cover the truth, but the point
estimates carry little information about the genotype. On a simulation
of the 2026 screen, the double mutants' *X* against the truth, by read
depth:

=========== =========== ====== ====== ============ ============
reads       genotypes   slope  RMSE   95% coverage median width
=========== =========== ====== ====== ============ ============
<= 100      37,707      0.04   2.60   0.98         5.9
100-1e3     89,022      0.39   0.54   0.90         1.18
1e3-1e4     64,362      0.88   0.17   0.88         0.48
1e4-1e5     15,283      1.02   0.07   0.90         0.20
> 1e5       1,225       1.02   0.05   0.76         0.07
=========== =========== ====== ====== ============ ============

The slope is the truth regressed on the fit; 1 means the fitted *X* is on
the true scale. The 2026 screen has about 77,000 double mutants above
1,000 reads.

Report genotypes flagged ``inconsistent`` separately rather than dropping
them silently. On the 2026 screen about 10% of deep genotypes are flagged,
against about 2% on simulations with no marker problem, so most flags are
a real disagreement between the markers whose cause is not known.

What X is
=========

Growth alone fixes occupancy only up to an affine map: any change of
scale and offset in occupancy is absorbed by each condition's *k* and
*m*. So the fit does not estimate occupancy. It estimates *X*, which is
the in vivo occupancy put on a scale where wt reads 1 at the lowest IPTG
and 0 at the highest. If wt is not fully repressed or fully induced in the
cell, *X* is still 1 and 0 there.

*X* is not in vitro binding. Compare the two after the fit, and expect
them to differ:

- Growth curves are broader than binding curves (Hill n 0.7-0.95 in *X*
  against 1.9-3.2 in binding at 20 uM protein).
- wt's growth midpoint (about 21 uM IPTG) is about 8 times its midpoint
  in binding at 2 uM protein, and matches its midpoint at 20 uM.
- Order can invert: M42I switches at a fifth of wt's IPTG in growth and
  needs 4-5 times wt's in binding.

No single protein concentration or monotone map carries binding onto
growth for every genotype, so binding is not used to set *X*'s scale.

Epistasis on *X* is defined only on the additive scale
(``tfs-predict-epistasis --scale add``), since *X* has no fixed bounds.

What the intervals carry
========================

Each genotype's interval comes from the Laplace approximation of stage 2,
with every condition's *k* and *m* held at stage 1's values. On
simulations of this design, the deep genotypes' 95% intervals covered the
truth 0.88-0.91 of the time up to 1e5 reads, and 0.76-0.81 above 1e5
(table above; the second simulation seed gave 0.91 and 0.81). Treat them
as somewhat narrow, most of all for the deepest genotypes.

What they leave out:

- **k's uncertainty.** Stage 1 estimates each condition's growth
  baseline with an error that shifts every non-wt genotype's *X* by about
  the same amount (one SD of *k* moved *X* by 0.03-0.04 on simulations).
  It matters most where the per-genotype interval is narrowest, which is
  why the deepest genotypes undercover. It cancels from differences
  between genotypes, rankings and epistasis, which are better calibrated
  than single curves.
- **Stage 1's small bias in m**, 2-6% steep on simulations. It stretches
  *X* by about the same factor. Resampling cannot cover a bias.
- **The pheS gap.** In the pool, pheS selection is about half as strong
  as wt's monoculture rate predicts. *X* is set mostly through kanR, so
  this need not distort it, but it could if the cause depends on
  genotype. The consistency flag is the check.
- **kanR's absolute slope.** The fit takes *m* from the tube totals.
  Within-tube comparisons, which do not use the totals, give a kanR slope
  about 2.5 times smaller. Binding at 2 uM protein supports the totals'
  value. If the within-tube value were right, *X*'s scale would change.
- **Congression's per-genotype error.** Stage 2 does not fit the
  congression mixture; for most genotypes the error is small next to the
  interval.
