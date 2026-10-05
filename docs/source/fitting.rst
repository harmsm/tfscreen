=================
Fitting the model
=================

A fit takes a processed growth table, binding data or both, and ends with
posterior draws of every model parameter. It runs in up to four commands:

1. ``tfs-configure-model`` checks the data and writes the model
   configuration.
2. ``tfs-prefit-calibration`` pins each condition's growth baseline and slope.
   It runs only for models with binding data.
3. ``tfs-fit-model`` finds the MAP point or fits a variational posterior.
4. ``tfs-sample-posterior`` turns the fit into posterior draws.

The model itself is described in :doc:`model` and its inputs in
:doc:`model-inputs`. :doc:`cli` lists every flag. What to do with the
posterior file is in :doc:`downstream`.

Recommended routes
==================

There are two routes in current use. Both are described in detail in the
sections that follow.

Joint growth and binding model
------------------------------

With binding data for some genotypes, *θ* is an absolute occupancy. The
configuration defaults are the current recommendations: the count likelihood,
one level offset per tube and one starting abundance per replicate,
pre-growth condition and genotype.

.. code-block:: bash

   tfs-configure-model --growth_df growth.csv --binding_df binding.csv \
       --library_config library_config.yaml
   tfs-prefit-calibration tfs_configure_config.yaml --seed 1
   tfs-fit-model tfs_configure_config.yaml --seed 1 --analysis_method map
   tfs-sample-posterior tfs_configure_config.yaml tfs_fit_model_checkpoint.pkl \
       --skip_growth_observations

Growth-only relative-X model
----------------------------

Without binding data, growth fixes occupancy only up to an affine map. The
``hill_relative`` occupancy component fits a wt-relative variable *X*
instead (see :doc:`model`). The pre-fit refuses a growth-only model, so the
per-condition growth priors come from ``--growth_priors`` or
``--growth_priors_wt_rates`` at configuration, and the population SDs that
ran away on real data are held with ``--set_priors``.

On a full library the route is a MAP fit followed by the arrowhead Laplace.
A MAP with level tube offsets runs in stages by default (see `Staged MAP`_
below), which keeps the offsets from settling in a mode where they carry the
population's growth.

.. code-block:: bash

   tfs-configure-model --growth_df growth.csv \
       --library_config library_config.yaml \
       --theta_model hill_relative \
       --growth_shares_replicates \
       --growth_priors growth_priors.csv \
       --set_priors theta_log_hill_n_hyper_scale_fixed=0.5 sigma_fixed=0.17
   tfs-fit-model tfs_configure_config.yaml --seed 1 --analysis_method map
   tfs-sample-posterior tfs_configure_config.yaml tfs_fit_model_checkpoint.pkl \
       --laplace arrowhead --skip_growth_observations

On a small library the low-rank guide also
recovers *X*:

.. code-block:: bash

   tfs-fit-model tfs_configure_config.yaml --seed 1 --analysis_method svi \
       --guide_type auto_low_rank_multivariate_normal --guide_rank 20

Configure
=========

``tfs-configure-model`` reads the data, checks it against the library,
resolves the model components and writes four files:

``{out_prefix}_config.yaml``
   The configuration every later step reads: data paths, the component
   choices and settings, the names of the priors and guesses files, the
   library provenance and a ``provenance:`` block recording the tfscreen
   version and command line.
``{out_prefix}_priors.csv``
   Every prior, one row per value.
``{out_prefix}_guesses.csv``
   Starting values for the fit.
``{out_prefix}_library.csv``
   The per-genotype library composition, for growth models.

The default prefix is ``tfs_configure``. Like every command that takes
``--out_prefix``, it also writes ``{out_prefix}_provenance.json``. A census of parameters and
observations is printed and written to ``{out_prefix}_model_stats.csv`` and
``{out_prefix}_model_stats.json`` unless ``--skip_model_stats`` is given. Do
not edit these files by hand. The flags below set everything that needs
setting.

Data
----

``--growth_df`` and ``--binding_df`` are both flags, and at least one is
required. With both, the joint model is configured. With only
``--binding_df``, a binding-only model infers *θ* from the binding
measurements alone. With only ``--growth_df``, a growth-only model infers
*θ*, or *X* for ``hill_relative``, from growth alone. ``--presplit_df`` and
``--base_growth_df`` add optional observations. :doc:`model-inputs`
describes each table.

``--library_config`` is required whenever ``--growth_df`` is given. It is the
library YAML handed to ``tfs-process-fastq``. ``tfs-configure-model`` fails
if any genotype in the data is missing from the library, which catches a
residue-numbering or wt-sequence mismatch at once. It also fails if a spiked
genotype named in the library has no growth data, since that usually means
the counts were called against a different library file.
``--allow_missing_spikes`` turns that error into a report, for an experiment
in which a spike truly dropped out.

Components
----------

Each ``--<axis>_model`` flag picks a component. :doc:`model` describes every
option. The defaults are ``linear`` condition growth, ``instant``
transition, ``hierarchical`` starting abundance, ``hierarchical_geno``
pleiotropy, ``fixed`` activity, ``hill_geno`` occupancy, ``single``
transformation, ``level`` tube offsets and the ``counts`` likelihood.
``--batch_size`` sets the mini-batch of genotypes used by the optimizer,
default 1024.

Priors
------

``--set_priors name=value ...`` sets any scalar prior in the priors file. A
name is a full row name, such as ``growth.sample_offset.sigma_fixed``, or any
unique dotted suffix of one, such as ``sigma_fixed``. An unknown or
ambiguous name is an error. The common uses hold a population SD that would
otherwise run away:

.. code-block:: bash

   --set_priors sigma_fixed=0.17 theta_log_hill_n_hyper_scale_fixed=0.5

``--growth_priors table.csv`` sets per-condition priors for ``linear``
condition growth. It has a ``condition_rep`` column, plus ``replicate`` when
replicates do not share conditions, and any of ``k_loc``, ``k_scale``,
``m_loc`` and ``m_scale``. Conditions and values it leaves out keep their
defaults.

.. code-block:: text

   condition_rep,k_loc,k_scale,m_loc,m_scale
   kanR+kan,0.015,0.01,0.0,0.01
   pheS+4CP,0.015,0.01,0.0,0.01

``--growth_priors_wt_rates rates.csv`` derives the priors from wt
monoculture growth rates, for ``hill_relative`` only. Its columns are
``condition_sel``, ``titrant_conc``, ``rate_mean``, ``rate_sd`` and
``num_replicates``. For each listed condition, *k* is wt's rate at the high
gauge concentration and *m* is its rate at the low one minus *k*. Each SD is
the standard error of the mean, floored at ``--growth_priors_sd_floor``,
default 0.002 per minute. List only the conditions where the monoculture
stands for the library's wt. These rows override ``--growth_priors`` for
the conditions they list.

Pre-fit calibration
===================

``tfs-prefit-calibration`` pins the per-condition growth parameters of a
model with binding data. Condition baselines *k* and genotype effects
*dk_g* trade off against each other, and without a pin the whole system can
slide by a constant (see :doc:`model`). The pre-fit breaks the tie.

It restricts the growth and binding data to the genotype and concentration
cells present in both and builds a reduced model. Occupancy comes straight
from the binding data, *dk_g* is 0, or pinned from ``--base_growth_df`` when
given, and the other hierarchies are held at their prior locations. Only
the ``condition_growth`` and ``growth_transition`` components are fit. After
a MAP fit it computes Hessian SDs at the MAP point and rewrites the
production priors and guesses files in place, after copying each to
``.bak``. Each condition's *k* and *m* location becomes an indexed row in the
priors file, and the scales are set from the Hessian between floors and
ceilings:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Meaning
   * - ``--k_scale_floor``
     - 0.002
     - Smallest prior SD on *k*, the day-to-day variation the Hessian
       cannot see.
   * - ``--m_scale_floor``
     - 0.001
     - Smallest prior SD on *m*.
   * - ``--k_scale_ceiling``
     - 0.1
     - Largest prior SD on *k*, so a flat Hessian direction never loosens
       the prior.
   * - ``--m_scale_ceiling``
     - 0.01
     - Largest prior SD on *m*.
   * - ``--pin_m``
     - off
     - Clamp *m* to the calibrated value instead of a soft prior.
   * - ``--hessian_chunk_size``
     - 64
     - Hessian rows per device batch. Lower it if the device runs out of
       memory.

``--pin_m`` exists because the growth likelihood can pull *m* far from even
a very tight soft prior. *k* always keeps a soft prior, since it carries real
tube-to-tube variation.

``--seed`` is required unless resuming with ``--checkpoint_file``. The
optimizer and convergence flags are those of ``tfs-fit-model``. Its own
outputs, with the default prefix ``tfs_prefit``, are diagnostics:
``tfs_prefit_params.npz``, ``tfs_prefit_checkpoint.pkl``,
``tfs_prefit_losses.txt`` and ``tfs_prefit_convergence.csv``.

The pre-fit refuses a growth-only configuration, because it has no binding
data to calibrate against. Set those priors at configuration instead.

Fit
===

``tfs-fit-model config_file`` fits the model. ``--seed`` is required for a
new fit. ``--analysis_method`` picks the method:

``map``
   Maximum a posteriori optimization. The result is a point.
   ``tfs-sample-posterior`` builds a Laplace approximation around it.
``svi`` (default)
   Stochastic variational inference. A MAP warm-up runs first and the
   variational guide starts at its point. ``tfs-sample-posterior`` samples
   the guide.
``nuts``
   The No-U-Turn sampler, started at the MAP warm-up's point. It is exact
   but only practical for small models. ``--nuts_num_warmup``,
   ``--nuts_num_samples``, ``--nuts_num_chains``,
   ``--nuts_target_accept_prob`` and ``--nuts_dense_mass`` control it.

Choosing a method
-----------------

``svi`` with the component guide is the default, but the evidence so far
favors the other options. On simulations of the growth-only relative model
(``planning/studies/relative-fit/``, 30 runs), the component guide gave
biased *X*, with 95% coverage of 0.06 under Poisson count noise and 0.38
under realistic noise. The low-rank guide at rank 20 recovered *X*, with
coverage of 0.81 and 0.76. MAP with the Laplace covered *X* and *k* and *m*
best, with the widest intervals and the noisiest point. The arrowhead
Laplace covered *X* at 0.97 and 0.94. On the full real library the low-rank
guide slid along the ridge between *m* and *X* and ended far worse than the
MAP, so MAP and the arrowhead Laplace is the full-library route. In every
study, every variational guide collapsed the uncertainty in *k* and *m*:
expect *X* intervals from a guide to undercover somewhat and *k* and *m*
intervals to be too narrow (``planning/studies/svi-overconfidence/``). These
are results on particular simulated designs and one real library, not
general guarantees.

Guides
------

For ``svi``, ``--guide_type`` picks the variational family:

``component`` (default)
   The mean-field guide assembled from the model components.
``auto_normal``, ``auto_diagonal_normal``
   NumPyro's mean-field normal autoguides.
``auto_multivariate_normal``
   A dense multivariate normal. Its memory grows with the square of the
   number of parameters, and a warning is printed above 4 GB.
``auto_low_rank_multivariate_normal``
   A normal with a low-rank covariance. ``--guide_rank`` sets the rank,
   numpyro's default when omitted. Rank 20 worked in the relative-fit
   study.
``delta``
   A point mass, which is MAP through the SVI path.

NumPyro class names such as ``AutoNormal`` also work. The guide flags are
refused with ``map`` and ``nuts``. A resumed fit must use the guide its
checkpoint was written with.

``--guide_init_scale``, default 1e-4, sets the guide's starting width in
each parameter's unconstrained units. For ``component`` it caps every guide
scale at the start, and for an autoguide it is numpyro's ``init_scale``. It
must be small. The same number applies to every parameter, and at 0.1 a
growth rate per minute starts with an SD of many ln units over a selection.
SVI then started about 1000 times above the warm-up's loss and descended
into other optima. SVI widens the scales itself. ``--init_param_jitter``,
default 0, multiplies the component guide's starting values by random noise.
Leave it off when starting from a MAP point, because 0.1 moves a starting
abundance near 15 by about 1.5 ln units.

Starting points
---------------

A fresh MAP starts at the configured guesses. ``svi`` and ``nuts`` first run
a MAP warm-up for at most ``--pre_map_num_epoch`` epochs, default 10000, which
stops early when it converges. Its files carry the ``{out_prefix}_premap``
prefix. ``--pre_map_num_epoch 0`` skips it.

``--init_from params.npz`` starts the fit at the point saved by an earlier
MAP fit, its ``{out_prefix}_params.npz``. Every site that file names starts
there, and every other site starts at its guess. The earlier fit may be of a
different model, such as a fit without tube offsets.

Staged MAP
^^^^^^^^^^

The starting point matters because Adam moves every parameter by about one
step size per step. On real data a cold MAP with level tube offsets carried
the offsets and *k* and *m* by whole units in its first window, the
step-size cuts then froze them there, and the offsets settled near ±2.8.
Starting from a MAP without offsets plus each tube's best offset was 1.2e5
nats better. ``tfs-fit-model`` builds that start itself. A fresh MAP of a
model with ``sample_offset: level`` runs three MAPs:

1. The tube offsets held at 0 (and a learned offset SD held at its prior
   scale, since with every offset at 0 its MAP is 0). Files
   ``{out_prefix}_stage1_*``.
2. Every other site held at stage 1's MAP, so the offsets are the only
   latents. Given the rest the tubes do not couple, so this is each tube's
   best offset. Files ``{out_prefix}_stage2_*``.
3. The joint MAP, started at stage 1's point plus stage 2's offsets, at
   ``--staged_step_size`` (default 1e-4). This writes the run's own
   ``{out_prefix}_*`` files.

Each stage writes its own checkpoint and convergence record. Rerunning the
same command reuses a stage whose ``_params.npz`` exists and resumes one
that was interrupted, so to redo a stage delete its files. To resume the
joint stage, pass its checkpoint with ``--checkpoint_file``.
``--stage_offsets`` is ``auto`` by default: it stages a fresh level-offset
MAP and nothing else (not SVI, not a resumed fit, not ``--init_from``).
``off`` fits in one go; ``on`` insists on staging.

On the dev data the offset mode is not only a trap for the optimizer. Run to
convergence from the hand-built start, a fit moved into it and scored 7.9e4
nats better than the staged MAP, with offsets that follow IPTG in the
selection conditions and unphysical *k* and *m*. The staged MAP keeps the
fit in the physical basin, but a longer or better fit can leave it.
``tfs-summarize-fit`` checks every level-offset fit for this: it reports the
tube offsets in prior SDs and flags a fit whose offsets trend with titrant or
time within a condition (see :doc:`summarize-fit`, "Tube offsets"). The cause
is filed for study (``planning/offset-mode-growth-transition.md``).

``--checkpoint_file`` resumes a fit from its ``{out_prefix}_checkpoint.pkl``
at the checkpoint's step size and convergence state. It cannot be combined
with ``--init_from``. A new fit refuses to overwrite an existing checkpoint,
so resume it, delete it or change ``--out_prefix``.

Outputs
-------

With the default prefix ``tfs_fit_model``:

``tfs_fit_model_checkpoint.pkl``
   The optimizer state, written every ``--checkpoint_interval`` epochs,
   default 10. It records the guide, step size, convergence state and
   provenance. ``tfs-sample-posterior`` reads it.
``tfs_fit_model_params.npz``
   The fitted parameters. For a MAP fit these are the ``{site}_auto_loc``
   arrays that ``--init_from`` reads.
``tfs_fit_model_convergence.csv``, ``tfs_fit_model_losses.txt``
   The convergence record, described below.
``checkpoints/``
   Numbered checkpoints every ``--epoch_checkpoint_interval`` epochs, default
   1000. 0 disables them.

A NUTS fit also writes its posterior file.

Convergence
===========

``tfs-fit-model`` for MAP and SVI, the MAP warm-up and
``tfs-prefit-calibration`` all stop the same way. Optimization runs in
windows of ``--convergence_window_steps`` optimizer steps, default 2000 and
never fewer than 10 epochs, so that with mini-batches every genotype is seen
several times per window. At the end of each window the monitor asks three
questions.

**Is the loss still improving?** A line through the medians of 20 blocks of
the window's losses gives the drop per window and its standard error. The
loss is improving if the drop exceeds ``--convergence_z`` standard errors,
default 3, and ``--loss_rtol`` of the loss, default 1e-6. The loss is judged
against its own noise, so a noisy SVI loss neither stops early nor runs
forever. A rising or oscillating loss is not improving. Before acting on
``--patience`` stalled windows, default 3, the monitor fits one line through
all of them. A slow descent hidden by noise in each window shows up there,
and a significant drop resets the stall count.

**Is any parameter still moving?** Each parameter's trend over the window is
measured in units of its posterior SD: the paired guide scale for SVI, the
prior SD for MAP, and log or logit units for positive or bounded
parameters. The posterior SD is floored at 1% of the prior SD, because a
mean-field guide can shrink a hierarchical scale's SD far below any honest
width, and a negligible creep in those units would never settle. Movement
explained by noise, or smaller than the optimizer can resolve at the current
step size, does not count. A parameter moving more than
``--param_tolerance``, default 0.05, is moving.

**Does the median represent the mean?** The loss being minimized is a mean,
but the test uses medians. If rare draws carry a huge penalty, the medians
ignore them. The window's skew, how many robust SDs the mean sits above the
median, exposes this. Benign noise stays below about 1, and the monitor
treats a skew above 3 as a hidden penalty.

The decisions follow from those answers. Above the final step size, after
``--patience`` windows in a row without loss improvement, the step size is
multiplied by ``--adam_step_size_cut``, default 0.1, down to
``--adam_final_step_size``, default 1e-6, starting from ``--adam_step_size``,
default 1e-3. Only the loss decides a cut. Some directions never settle, and
holding the step size high for them wastes time, while a smaller step does
not remove rare penalties. At the final step size the run converges after
``--patience`` windows in which the loss has not improved, no parameter has
moved and the skew is at most 3. A still-moving parameter or a skewed loss
keeps the run going and is named in the log.

**A MAP is judged by its exact loss.** A MAP's objective is a fixed
function of the current point, so the only noise in its window losses comes
from the mini-batches. On a full library that noise is large: at batch size
4096 of 218,000 genotypes the trend's SE was about 5e4 nats per window, so a
descent of 1e5 nats per window read as a stall and the step size was cut to
its floor while the fit was still descending. A MAP fit (``tfs-fit-model
--analysis_method map``, every stage of the staged MAP, the SVI pre-MAP and
``tfs-prefit-calibration``) therefore computes the exact full-batch loss,
``-log p``, at the end of each window: one forward pass over the library,
in chunks of the batch size, a few percent of the run's time. A line
through the last ``--patience`` + 1 exact losses gives the drop per window
and its SE, which now measures only the optimizer's jitter about its path.
The window is improving when the drop exceeds both ``--convergence_z``
times the SE and ``--loss_rtol`` times the loss. A steady descent passes
however slow it is; jitter about a level does not. The skew check is
skipped, since the exact loss is the objective itself. SVI keeps the
mini-batch test: its ELBO is noisy by nature.

A window whose loss falls below −1000 times the magnitude of the run's first
block ends the run as ``diverged``. That means the objective is unbounded,
from a density singularity or from float32 error in a dense guide's own
density. It is not convergence.

``--max_num_epochs``, default 100000, is a cap only. A hierarchical MAP
often has no finite optimum, because a hyperparameter scale can shrink
forever, so MAP runs often end at the cap. A run that reaches it says so and
names the largest remaining movement.

The record
----------

``{out_prefix}_convergence.csv`` has one row per window:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``step``, ``epoch``, ``step_size``
     - Where the window ended and the step size it ran at.
   * - ``loss``, ``loss_mean``
     - Median and mean mini-batch loss over the window.
   * - ``loss_exact``
     - The exact full-batch loss at the end of the window (MAP only). The
       loss test used it, and two MAP runs on the same model can be
       compared by it.
   * - ``loss_drop``, ``loss_drop_se``, ``loss_t``
     - Drop per window, its standard error and their ratio.
   * - ``loss_improving``
     - Whether the loss test passed.
   * - ``loss_skew``, ``loss_skewed``
     - The skew and whether it exceeds 3.
   * - ``worst_param``, ``worst_param_excess``, ``worst_param_drift``
     - The parameter moving most, its movement beyond noise and its raw
       drift.
   * - ``params_moving``
     - Whether any parameter exceeds ``--param_tolerance``.
   * - ``plateau``, ``plateau_count``
     - Whether this window stalled, and the run of stalled windows.
   * - ``pooled_loss_t``
     - The pooled trend's t statistic, when one was computed.
   * - ``decision``
     - ``continue``, ``cut``, ``converged`` or ``diverged``.

``{out_prefix}_losses.txt`` has columns ``epoch``, ``loss``, ``step`` and
``step_size``, one row per block of steps, where ``loss`` is the block's
median.

The optimizer
-------------

The optimizer is plain Adam. ``--adam_clip_norm`` opts into numpyro's
``ClippedAdam``, which clips each gradient element to that value. Leave it
off. These losses have gradients of 1e3 to 1e6 per element, so a clip
applies to every element on every step, and Adam then follows the sign of
each draw's gradient. Its fixed point is where those signs balance, not
where the expected gradient is zero. With a rare large penalty, such as a
steep, precisely measured binding curve that an occasional draw crosses,
that point sat in violation for 5 to 90% of draws on every
congression-calibration run. Without clipping the same fits converged.

``--elbo_num_particles``, default 2, sets the draws per SVI step.

Sample the posterior
====================

``tfs-sample-posterior config_file checkpoint_file`` writes
``{out_prefix}.h5``, default ``tfs_posterior.h5``, which the prediction and
extraction commands read. The checkpoint type decides what happens. An SVI
checkpoint draws ``--num_posterior_samples`` samples from the guide, default
10000. A NUTS checkpoint uses its saved samples. A MAP checkpoint, including
SVI with the ``delta`` guide, becomes a Laplace approximation chosen by
``--laplace``:

``auto`` (default)
   ``full`` up to ``--laplace_max_params`` MAP parameters, default 20000, and
   ``arrowhead`` above.
``full``
   The Laplace from the dense Hessian. Its memory grows with the square of the
   parameter count, which puts a full library out of reach.
``arrowhead``
   The same Gaussian computed one genotype at a time. Given the shared
   parameters the genotypes do not couple, so the Hessian is per-genotype
   blocks plus their coupling to the shared parameters: *k*, *m*,
   hyperparameters and tube offsets. The shared parameters are drawn from
   their marginal and each genotype from its conditional. It runs on a full
   library.
``blocks``
   Per-genotype blocks with the shared parameters held at the MAP. It leaves
   out the *k* and *m* uncertainty.
``point``
   The MAP point itself, one sample and no Hessian.

Each Hessian eigenvalue is floored at the prior's curvature along its
direction, so no direction of the Laplace is wider than the prior. A MAP that
stopped at its epoch cap is not exactly at an optimum, and the arrowhead
Laplace may find directions of the shared block with negative curvature.
Those directions are held at the MAP and written to
``{out_prefix}_held_directions.csv``, since they have no interval. Its
columns are ``direction``, ``eigenvalue``, ``parameter`` and ``loading``,
with the five largest loadings of each direction. On the relative-fit
simulations the arrowhead Laplace covered *k* and *m* at 0.92 and 1.0 under
Poisson noise and at 0.50 and 0.42 under realistic noise.

``--skip_growth_observations`` leaves the per-observation growth sites out
of the file. They are most of its size, about 100 GB per site at 500 draws on
a 200,000-genotype library, and ``tfs-predict-growth`` recomputes growth from
the parameters. ``--hessian_chunk_size`` and ``--genotype_chunk_size`` trade
speed for memory if the device runs out. ``--sampling_batch_size`` and
``--forward_batch_size`` do the same for sampling.

Per-genotype fits
=================

``tfs-fit-genotypes`` is a non-Bayesian alternative. It fits the growth model
to each genotype independently by maximum likelihood, holding each
condition's *k* and *m* fixed at a calibration, and writes per-genotype Hill
parameters and *θ* predictions. It applies no congression correction and no
pooling across genotypes.

.. code-block:: bash

   tfs-fit-genotypes growth.csv tfs_configure_priors.csv --num_workers -1

The calibration is a priors file written by ``tfs-prefit-calibration`` or a
wide table with columns ``condition_rep``, ``growth_k`` and ``growth_m``. The
default output prefix is ``tfs_fit_genotypes``.

Running on a cluster
====================

``examples/tfmodel/run.srun`` is a Jinja template that ``tfs-setup-grid``
renders into each run directory of a grid, not a standalone script. It shows
the joint route as a Slurm job. See :doc:`grid`.
