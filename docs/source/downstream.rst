===================
Downstream analysis
===================

This page covers everything after a posterior exists: parameter tables,
predicted occupancy and growth, epistasis, response-shape classification and
cross-run comparison. Every flag and default is listed in :doc:`cli`. The
steps that produce the posterior are in :doc:`fitting`, and the diagnostic
summary of a finished run is in :doc:`summarize-fit`.

The examples assume the default file names from the earlier steps:
``tfs_configure_config.yaml`` from ``tfs-configure-model``,
``tfs_fit_model_checkpoint.pkl`` from ``tfs-fit-model`` and
``tfs_posterior.h5`` from ``tfs-sample-posterior``.

Inputs and output columns
-------------------------

The model-based commands (``tfs-extract-params``, ``tfs-predict-theta``,
``tfs-predict-growth`` and ``tfs-predict-epistasis``) take two positional
arguments: the config YAML used for the fit and a parameter file. The
parameter file is normally the ``.h5`` written by ``tfs-sample-posterior``.

A MAP checkpoint (``.pkl``) is also accepted. It holds a single point, so the
output has one ``q0.5`` column and no uncertainty. The predict commands
convert the checkpoint to a one-draw posterior on the fly and leave it behind
as ``{out_prefix}_map_posterior.h5``. SVI and NUTS checkpoints are refused;
run ``tfs-sample-posterior`` on them first. For a MAP fit, run
``tfs-sample-posterior`` too if you want intervals: it builds the Laplace
approximation and writes an ``.h5`` like any other.

Every posterior summary is written as bare ``q<level>`` columns with no
feature-name prefix. A posterior file gives 17 levels:

.. code-block:: text

    q0.001  q0.005  q0.01  q0.025  q0.05  q0.1  q0.159  q0.25  q0.5
    q0.75   q0.841  q0.9   q0.95   q0.975 q0.99 q0.995  q0.999

``q0.5`` is the posterior median and the recommended point estimate.
``q0.025`` and ``q0.975`` bound the 95% credible interval. ``q0.159`` and
``q0.841`` bound the central 68% interval, so ``(q0.841 - q0.159) / 2`` is a
robust one-sigma width. The table-based tools below (``tfs-extract-epistasis``,
``tfs-cat-response``, ``tfs-compare-runs``) use ``q0.5`` and that half-width
as their default value and uncertainty columns, so the outputs of one command
feed the next without renaming.

Parameter tables
----------------

``tfs-extract-params`` writes one CSV per parameter group.

.. code-block:: bash

    tfs-extract-params tfs_configure_config.yaml tfs_posterior.h5

The files are named ``{out_prefix}_{parameter}.csv`` with the default prefix
``tfs_params``. Which files appear depends on the configured components. A
``hill_geno`` fit gives ``tfs_params_log_hill_K.csv``,
``tfs_params_hill_n.csv``, ``tfs_params_theta_low.csv`` and
``tfs_params_theta_high.csv``, keyed by ``genotype`` and ``titrant_name``.
``hill_mut`` adds per-mutation effects such as
``tfs_params_d_log_hill_K.csv`` (keyed by ``mutation``) and, with pairs,
epistasis terms such as ``tfs_params_epi_log_hill_K.csv`` (keyed by
``pair``). ``hill_relative`` writes ``X_low`` and ``X_high`` in place of the
theta baselines. Other components add their own files, for example
``tfs_params_dk_geno.csv``, ``tfs_params_growth_k.csv`` and
``tfs_params_growth_m.csv`` (keyed by ``condition_rep``, plus
``replicate`` when replicates have their own growth parameters),
``tfs_params_lam.csv`` for the ``mixture`` transformation, and
``tfs_params_k_ref.csv`` when base-growth data were supplied.

``log_hill_K`` is the natural log of the Hill constant in the units of
``titrant_conc``. It is not a base-10 log.

Given a ``.pkl`` MAP checkpoint, ``tfs-extract-params`` writes a single
``q0.5`` column. Deterministic sites such as ``ln_cfu0`` are absent from a
checkpoint, so their files are skipped with a warning. Run
``tfs-sample-posterior`` first to get them.

Predicted occupancy
-------------------

``tfs-predict-theta`` predicts operator occupancy as a function of titrant
concentration.

.. code-block:: bash

    tfs-predict-theta tfs_configure_config.yaml tfs_posterior.h5

The output is ``tfs_pred_theta.csv``, one row per ``genotype``,
``titrant_name`` and ``titrant_conc``, with the ``q<level>`` columns and an
``in_training_data`` column. ``in_training_data`` is 1 when that genotype,
titrant name and concentration triple was in the training data and 0
otherwise.

By default the command predicts every training genotype at every training
titrant point. Three plain-text files add to that grid, one value per line
with ``#`` comments allowed. ``--genotypes_file`` adds genotypes, written as
slash-separated mutations such as ``M42I/K84L`` or as ``wt``.
``--titrant_names_file`` and ``--titrant_concs_file`` add titrant points and
must be given together. They pair line by line, and a single name is
broadcast across every concentration. The added genotypes and points are
unioned with the training set. Pass ``--only_files`` to predict only at the
file inputs.

A genotype that was not in the training data needs a theta component that can
predict it. ``hill_mut`` assembles the genotype from its per-mutation effects.
``hill_geno`` and ``hill_relative`` have no per-mutation structure and give
every unseen genotype the population average for its titrant.
``--genotype_batch_size`` (default 2000) caps the memory used for unseen
genotypes. ``--num_samples N`` adds ``sample_0`` to ``sample_{N-1}`` columns
holding joint posterior draws.

A ``hill_relative`` fit predicts the wt-relative growth variable X rather
than an occupancy. Its output carries an extra ``theta_scale`` column with the
value ``X`` so that nothing downstream reads it as theta. X is 1 for wt at the
low gauge concentration and 0 for wt at the high one. See :doc:`model` for the
gauge.

Predicted growth
----------------

``tfs-predict-growth`` predicts ln(CFU) from the fitted model.

.. code-block:: bash

    tfs-predict-growth tfs_configure_config.yaml tfs_posterior.h5

The output is ``tfs_pred_growth.csv``, one row per genotype, replicate,
condition, titrant and time point, with the ``q<level>`` columns, the observed
``ln_cfu`` and ``ln_cfu_std`` where an observation exists, and
``in_training_data``.

``--genotypes_file`` and ``--titrant_concs_file`` extend the prediction grid
as in ``tfs-predict-theta``, and ``--only_files`` restricts it to the file
inputs. ``--titrant_names_file`` behaves differently here. It is a
restrict-only filter applied after prediction: it narrows which titrant names
appear in the output and never adds points.

Growth prediction over a full library is the most memory-hungry step in this
page. ``--genotype_batch_size`` sets how many genotypes go through the model
at once. Left unset, it is estimated from the available device memory. Each
batch costs one JAX recompilation. ``--num_marginal_samples`` limits how many
posterior draws are run through the model when computing quantiles; by
default all are used. ``--num_samples N`` adds ``sample_*`` columns as in
``tfs-predict-theta``.

For a quick check of the fit without a full sweep, ``--subset_genotypes``
predicts a single memory-sized block. The block always holds the binding
genotypes, the spiked genotypes and any genotypes from ``--genotypes_file``,
and the rest is a random draw from the other genotypes. ``--seed`` makes the
draw reproducible.

.. code-block:: bash

    tfs-predict-growth tfs_configure_config.yaml tfs_posterior.h5 \
        --subset_genotypes --seed 1 --out_prefix tfs_pred_growth_subset

Epistasis
---------

Two commands compute second-order epistasis on mutant cycles. Each cycle is a
double mutant, its two single-mutant parents and wt. They differ in how they
treat the posterior.

``tfs-predict-epistasis`` works from the joint posterior. It draws theta for
all four corners of each cycle from the same posterior sample, computes
epistasis within that draw and then reports quantiles across draws. The
interval therefore carries the posterior covariance between the corners.

.. code-block:: bash

    tfs-predict-epistasis tfs_configure_config.yaml tfs_posterior.h5

The output is ``tfs_pred_epistasis.csv`` with ``genotype`` (the double
mutant), ``titrant_name``, ``titrant_conc``, the ``q<level>`` columns and a
trailing ``in_regime`` column. The default ``--scale logit`` takes additive
epistasis of logit(theta). ``add`` takes ``(Y11 - Y10) - (Y01 - Y00)`` and
``mult`` takes ``(Y11 / Y10) / (Y01 / Y00)``. ``--scale_constant`` multiplies
the transformed value before the difference. Since logit(theta) equals
-dG/RT, passing ``--scale_constant -0.6159`` reports logit epistasis as an
interaction free energy in kcal/mol at 310.15 K. Only genotypes seen in
training are supported, and an unseen genotype in ``--genotypes_file`` is an
error.

``in_regime`` is 1 only when the theta posterior of all four corners lies
inside the resolvable band ``[regime_eps, 1 - regime_eps]``. The posterior is
judged by its central ``--regime_ci`` interval, 95% by default, and
``--regime_eps`` defaults to 0.01. Near 0 or 1, logit(theta) saturates and
growth constrains it weakly, so epistasis there leans on the theta model's
functional form and on cross-genotype covariance. Treat rows with
``in_regime`` 0 as model-conditional, even when their interval is narrow. The
flag checks posterior mass only. It does not test whether the growth signal
exceeds the growth noise. With a MAP checkpoint there is one draw, so the
check reduces to the point value.

A ``hill_relative`` fit predicts X rather than theta. Pass ``--scale add``
for it. The output then has no ``in_regime`` column.

``tfs-extract-epistasis`` works from any long-form table with one row per
genotype per condition. It computes epistasis from each corner's marginal
estimate and propagates the errors as if the corners were independent. Use
it for tables that do not come from the model, or for a quick look at a
``tfs-predict-theta`` table.

.. code-block:: bash

    tfs-extract-epistasis tfs_pred_theta.csv \
        --group_by titrant_name titrant_conc --scale logit

``--y_obs`` defaults to ``q0.5`` when that column exists, and ``--y_std`` to
``(q0.841 - q0.159) / 2``. ``--group_by`` names the columns that define a
condition. Epistasis is computed within each condition, so for a predict-theta
table pass both ``titrant_name`` and ``titrant_conc``. The default scale here
is ``add``, unlike ``tfs-predict-epistasis``. ``logit`` requires values in
(0, 1) and clamps them to ``[logit_eps, 1 - logit_eps]``. ``--scale_constant``
works as above. On a ``hill_relative`` table only ``add`` is allowed. The
output is ``tfs_epistasis.csv`` with one row per double mutant and
condition, holding ``ep_obs`` and, when an uncertainty column is available,
``ep_std``. ``--keep_extra`` keeps every input column.

Prefer ``tfs-predict-epistasis`` for model output. Corners of a cycle share
the fitted shared parameters, so their errors are correlated, and the
marginal calculation can be off in either direction.

Response shapes
---------------

``tfs-cat-response`` fits a family of empirical curve models to each group's
``y_obs`` versus ``x_obs`` curve. It answers two separate questions: what
shape the curve has, and whether the curve can be told apart from zero.

.. code-block:: bash

    tfs-cat-response tfs_pred_theta.csv titrant_conc --group_by titrant_name

The positional arguments are the data file and ``x_obs``, the name of the
independent-variable column. ``--y_obs`` is a flag. It defaults to ``q0.5``
when that column exists. ``--y_std`` defaults to ``(q0.841 - q0.159) / 2``
when both quantiles exist; otherwise the fit is unweighted. Groups are the
``genotype`` column plus any ``--group_by`` columns. The same command works on
epistasis tables, for example ``tfs-cat-response tfs_pred_epistasis.csv
titrant_conc --group_by titrant_name``. The per-group fits run in parallel;
``--num_workers`` defaults to -1, which uses all but one CPU.

**Model x-scale.** The models fall into two camps, and they must be handed
raw concentration in both cases. The Hill family (``repressor``,
``inducer``, ``hill_repressor``, ``hill_inducer``) and the ``biphasic_*``
models are written in raw concentration and take the log internally, so they
are already sigmoids or peaks in log concentration. ``bell_peak_log``,
``bell_dip_log`` and ``linear_log`` are shapes in log10 concentration and
also apply the transform themselves. Concentrations of 0 are placed two
decades below the lowest nonzero concentration before the log. The plain
``bell_peak``, ``bell_dip`` and ``linear`` models are shapes in raw x.
``flat`` does not depend on x. Never log-transform the x column yourself.

**Shape.** ``--select_by`` picks the model. The default, ``shape``, is a
liberal classifier meant for exploration. It first decides flat versus curvy
from the residual autocorrelation of the flat fit: the curve is curvy when the
autocorrelation p-value is below ``--curvy_cutoff`` (default 0.1). It then
names the curvy shape by best weighted R², preferring the simpler model when
two are close. In this mode the default model set is ``flat``, ``inducer``,
``repressor``, ``bell_peak_log``, ``bell_dip_log``, ``biphasic_peak`` and
``biphasic_dip``. ``curvy_cutoff`` is the knob to sweep; run a few values and
inspect the curves. ``aicc`` picks the lowest small-sample AIC on weighted
residuals. It is conservative and often calls a well-fit curve flat at small
n. ``adequacy`` keeps the AICc pick unless a runs test finds clustered
residuals, then moves to the best adequate model that is no simpler. It never
demotes a curve to a simpler model. Outside ``shape`` mode the default model
set is ``flat``, ``linear_log``, ``repressor``, ``inducer``,
``bell_peak_log`` and ``bell_dip_log``. ``--models`` reaches any model in the
library.

The selected model is reported as ``best_model``, with ``shape`` giving its
qualitative form: ``flat``, ``linear``, ``step``, ``peak``, ``dip`` or
``biphasic``. ``aicc_best_model`` records the AICc pick for comparison.
``shape_status`` reports the runs test on the selected model.

**Magnitude.** Whether a curve is distinguishable from zero is judged on the
observed data, not on the fitted curve. The test statistic is
``sum((y_obs / y_std)**2)`` against a chi-squared distribution with n degrees
of freedom. Its p-value, ``nonzero_p``, is corrected across curves by
Benjamini-Hochberg to give ``nonzero_q``. ``fittable`` is True when
``nonzero_q`` is below ``--alpha`` (default 0.05). ``all_equiv_zero`` is True
when every point's interval lies inside the region of practical equivalence
``[-rope_cutoff, rope_cutoff]``. By default ``rope_cutoff`` is
``--rope_multiplier`` (default 2) times the median observed ``y_std``. That
value scales with the noise, so whole intervals rarely fit inside it. Pass an
explicit ``--rope_cutoff`` with biological meaning to make the call useful.
The two flags together give three outcomes: ``fittable`` True is a real
response; ``fittable`` False with ``all_equiv_zero`` True is confidently flat
at zero; ``fittable`` False with ``all_equiv_zero`` False cannot be told.

Shape and magnitude are independent. The intended read is to filter on
``fittable`` and then look at ``shape``.

The outputs, with the default prefix ``tfs_cat_response``:

``tfs_cat_response.csv``
    One row per group: ``best_model``, ``aicc_best_model``, ``shape``,
    ``shape_status``, per-model fit statistics and parameters, and the
    magnitude rollups (``nonzero_p``, ``nonzero_q``, ``n_nonzero``,
    ``all_equiv_zero``, ``fittable``). ``omnibus_p`` and ``omnibus_q`` are a
    model-based test reported for reference only. They gate nothing.

``tfs_cat_response_assessment.csv``
    One row per group and x for the best model: ``model``, ``fittable``,
    ``x``, the observed ``y_obs`` and ``y_std``, the model's ``y_model`` and
    its propagated error ``y_model_std``, then ``z`` (``y_obs / y_std``) and
    the per-point ``sig_nonzero``.

``tfs_cat_response_predictions.csv``
    The best model's curve for each group. ``--write_all_predictions`` writes
    every model's curve.

``tfs_cat_response_{model}.csv``
    One file per fitted model with its parameter table and per-group fit
    statistics.

Comparing runs
--------------

``tfs-compare-runs`` measures how much N independent estimates of the same
quantity disagree, and whether each run's reported uncertainty explains the
disagreement. The runs might differ by seed or by which data were held out.
It accepts any long-form table with a point estimate: predict-theta, -growth
and -epistasis output, and ``tfs-extract-params`` files alike.

.. code-block:: bash

    tfs-compare-runs seed1/tfs_pred_theta.csv seed2/tfs_pred_theta.csv \
        seed3/tfs_pred_theta.csv

Two or more arguments are read as CSV paths. A single argument is read as a
manifest file listing one CSV path per line, unless it ends in ``.csv``.
``--reference full/tfs_pred_theta.csv`` switches to reference mode, where each
run is scored by its deviation from that run, as for k-fold dropouts against a
full-data fit. Without it, runs are compared to their cross-run mean.

Three sets of columns control the comparison, and all three are printed and
recorded in the metadata. The **match key** is what makes a row the same row
across runs. By default it is every column shared by all runs except the value
columns (``q<level>``, ``y_obs``, ``y_std``) and bookkeeping columns such as
``in_training_data`` and ``in_regime``. ``--match_by`` overrides it. The key
must be unique within each run. **index_by** is the entity being scored. It is
``genotype`` if present, else ``parameter``, else the only match-key column.
A growth-parameter file keyed by both ``replicate`` and ``condition_rep``
has no single entity column, so it needs ``--index_by condition_rep``. ``--group_by`` breaks each entity out
further. The report key is ``index_by`` plus ``group_by``, with one output row
per value. The match-key columns outside the report key are pooled over.

``--group_by`` is a statistical zoom. At the finest grouping only N - 1
degrees of freedom remain per row. Check ``n_rows`` and ``n_eff`` for the
pooling depth.

.. code-block:: bash

    tfs-compare-runs seed1/tfs_params_growth_k.csv seed2/tfs_params_growth_k.csv \
        --index_by condition_rep

The command applies no thresholds and assigns no grades. Every quantity is a
number, and any cutline belongs in the downstream analysis that uses it, such
as ``df["overdispersion"] > 2``. The main output, ``tfs_compare_runs.csv``,
has one row per report key with these columns: ``n_runs``, ``n_present``,
``n_rows``, ``n_eff``, ``mode``, ``spread_estimator``, ``rms_sd``,
``max_sd``, ``mean_value``, ``dynamic_range``, ``mean_reported_sigma``,
``chi2``, ``dof``, ``overdispersion``, ``overdispersion_p`` and
``overdispersion_q``.

``rms_sd`` and ``max_sd`` measure run-to-run spread in the estimate's own
units, so they cannot be compared across parameters. ``overdispersion`` is
chi-squared over its degrees of freedom. It is unit-free and asks whether the
spread exceeds what each run's sigma predicts; ``overdispersion_q`` is its
Benjamini-Hochberg q-value. The default sigma is ``(q0.841 - q0.159) / 2``.
That stays a good scale for spread but biases the overdispersion test for
skewed posteriors, such as ``theta_low`` or ``theta_high`` near 0 or 1 or
``hill_n`` near its bound. ``dynamic_range`` is the range of the target over
the pooled axes. It is NaN when nothing is pooled.

``tfs_compare_runs_aggregate.csv`` mixes the per-run posteriors as an
equal-weight mixture, row by row on the match key. Its quantiles fold in both
each run's width and the spread between runs, and they do not shrink with N.
It needs at least two shared ``q<level>`` columns and never includes a
reference run. It is the slow step on a large library, and ``--no_aggregate``
skips it. ``tfs_compare_runs_metadata.json`` records every resolved setting.

When the tables have a ``genotype`` column, rows come back in canonical
genotype order: wt, then singles, then doubles, by site. Otherwise the main
table is sorted by ``rms_sd``.
