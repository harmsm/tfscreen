========
Analysis
========

The ``tfscreen.tfmodel`` module provides a hierarchical Bayesian model that
infers per-genotype and operator occupancy
(*θ*) from bacterial growth data and direct binding measurements.

Standard Workflow
=================

A complete analysis run consists of the following steps. The example commands
assume all input files are in the current working directory and use the
default output-file naming convention (``tfs_configure_*``, ``tfs_fit_model_*``,
etc.). The :download:`example run.srun <../../examples/tfmodel/run.srun>`
shows a complete Slurm script for a cluster run.

Step 1: Configure Model (``tfs-configure-model``)
--------------------------------------------------

Validates the input data, maps categorical labels to numerical indices, selects
model components, and writes three configuration files:

* ``{out_prefix}_config.yaml`` — main configuration read by all downstream steps
* ``{out_prefix}_priors.csv`` — prior distribution settings for all parameters
* ``{out_prefix}_guesses.csv`` — initial-value guesses for array parameters

``binding_df`` (direct binding measurements) is the only required argument.
``growth_df`` is optional; omitting it configures a binding-only model. See
:doc:`model-inputs` for what each input contributes to the fit, how much
data is useful, and the exact file format for ``binding_df``, ``growth_df``,
``base_growth_df``, ``presplit_df``, and ``transformation_lambda``.

.. code-block:: bash

    tfs-configure-model binding.csv \
        --growth_df growth.csv \
        --out_prefix tfs_configure

Key component flags (see :ref:`model-components` for the full list):

* ``--condition_growth_model`` — how growth rate depends on TF occupancy (default: ``linear``)
* ``--growth_transition_model`` — pre-to-selection phase lag model (default: ``instant``)
* ``--activity_model`` — per-genotype TF activity prior (default: ``horseshoe_geno``)
* ``--theta_model`` — operator occupancy parameterisation (default: ``hill_geno``)
* ``--dk_geno_model`` — pleiotropic growth-effect prior (default: ``hierarchical_geno``)
* ``--transformation_model`` — multi-plasmid congression correction (default: ``single``, i.e. no correction)

Step 2: Pre-fit Calibration (``tfs-prefit-calibration``)
---------------------------------------------------------

Runs a fast MAP fit on a simplified version of the model to calibrate the priors
for the ``condition_growth`` and ``growth_transition`` components. The
calibration fit is restricted to the intersection of genotypes and titrant
conditions present in both the growth and binding data.

After convergence the script updates the production ``{out_prefix}_priors.csv``
and ``{out_prefix}_guesses.csv`` files in place (a ``.bak`` backup is written
first), giving the full production fit a warm start.

Diagnostic artefacts from the calibration MAP run are also written (default
prefix ``tfs_prefit``):

* ``tfs_prefit_params.npz`` — MAP point estimates in constrained space.
* ``tfs_prefit_checkpoint.pkl`` — optimizer checkpoint (can be passed to
  ``--checkpoint_file`` to resume an interrupted calibration run).
* ``tfs_prefit_losses.txt`` — per-epoch loss history.

.. code-block:: bash

    tfs-prefit-calibration tfs_configure_config.yaml \
        --seed 42 \
        --convergence_tolerance 0.00001

Step 3: Fit Model (``tfs-fit-model``)
--------------------------------------

Performs the main parameter estimation. Three inference methods are available
via ``--analysis_method``:

* **map** — Maximum A Posteriori optimisation (Adam). Fast; produces a point
  estimate. Use ``tfs-sample-posterior`` afterwards to obtain uncertainty
  estimates via a Laplace approximation.
* **svi** (default) — Stochastic Variational Inference. Automatically runs a
  short MAP pre-pass (``--pre_map_num_epoch``) before the full variational fit.
  Produces a full approximate posterior; posterior samples are drawn after
  convergence.
* **nuts** — No-U-Turn Sampler (exact MCMC). Slowest; most accurate.

The example below matches the MAP configuration used in the example ``run.srun``:

.. code-block:: bash

    tfs-fit-model \
        tfs_configure_config.yaml \
        --seed 42 \
        --analysis_method map \
        --adam_step_size 1e-6 \
        --convergence_check_interval 100 \
        --convergence_window 50 \
        --checkpoint_interval 100 \
        --max_num_epochs 100000000 \
        --pre_map_num_epoch 100000 \
        --convergence_tolerance 0.0005 \
        --patience 5

Key outputs (with default ``--out_prefix tfs_fit_model``):

* ``tfs_fit_model_checkpoint.pkl`` — optimizer checkpoint; resume with ``--checkpoint_file``
* ``tfs_fit_model_params.npz`` — MAP/SVI parameter point estimates

Step 4: Sample Posterior (``tfs-sample-posterior``)
----------------------------------------------------

Draws posterior samples from a checkpoint produced by ``tfs-fit-model``. The
checkpoint type is detected automatically:

* **MAP checkpoint** — constructs a Laplace (Hessian-based) Gaussian
  approximation at the MAP point, then draws samples.
* **SVI checkpoint** — draws directly from the fitted variational distribution
  (resumes with 0 additional optimisation epochs).
* **NUTS checkpoint** — reconstructs posteriors from the saved MCMC samples.

.. code-block:: bash

    tfs-sample-posterior \
        tfs_configure_config.yaml \
        tfs_fit_model_checkpoint.pkl \
        --num_posterior_samples 1000 \
        --sampling_batch_size 10 \
        --seed 42

Output (default ``--out_prefix tfs_posterior``):

* ``tfs_posterior.h5`` — posterior samples for all latent variables; passed to
  the prediction steps below.

Step 5: Extract Parameters (``tfs-extract-params``)
----------------------------------------------------

Extracts interpretable parameter summaries from a posterior ``.h5`` file and
writes one CSV per parameter group.

.. code-block:: bash

    tfs-extract-params \
        tfs_configure_config.yaml \
        tfs_posterior.h5

Outputs (default ``--out_prefix tfs_params``):

* ``tfs_params_log_hill_K.csv`` — per-genotype log₁₀(*Kd*) with posterior quantiles.
* ``tfs_params_theta_low.csv``, ``tfs_params_theta_high.csv`` — per-genotype
  lower and upper occupancy plateaux.
* ``tfs_params_dk_geno.csv`` — per-genotype pleiotropic growth effect.
* Additional CSVs depending on the components selected during configuration
  (e.g. ``tfs_params_d_log_hill_K.csv`` for per-mutation effects when using
  ``hill_mut``; ``tfs_params_epi_*.csv`` for epistasis terms when
  ``--epistasis`` is set).

**Quantile columns**

Each CSV contains metadata columns (typically ``genotype`` and
``titrant_name``) followed by 17 quantile columns named ``q{level}``:

.. code-block:: text

    q0.001  q0.005  q0.01  q0.025  q0.05  q0.1  q0.159  q0.25
    q0.5    q0.75   q0.841 q0.9    q0.95  q0.975 q0.99  q0.995  q0.999

``q0.5`` is the posterior median (the recommended point estimate).  ``q0.025``
and ``q0.975`` span the 95% credible interval; ``q0.159`` and ``q0.841``
span the ±1 σ interval under a normal approximation.

When ``tfs-extract-params`` is given a MAP checkpoint (``.pkl``) instead of
a posterior file, only a single ``q0.5`` column is written (the MAP point
estimate, not a true posterior quantile).

Step 6: Predict Growth (``tfs-predict-growth``)
------------------------------------------------

Predicts ln(CFU) from the fitted model. By default, predictions are produced at
every (genotype, replicate, condition, titrant_name, titrant_conc, time)
combination present in the training data.

.. code-block:: bash

    tfs-predict-growth \
        tfs_configure_config.yaml \
        tfs_posterior.h5

Output (default ``--out_prefix tfs_growth_pred``):

* ``tfs_growth_pred.csv`` — one row per prediction point; quantile columns
  (``median``, ``lower_95``, ``upper_95``, etc.) plus ``in_training_data``.

To add predictions at novel genotypes or concentrations, use
``--genotypes_file`` or ``--titrant_concs_file`` (plain-text files, one value
per line). Pass ``--only_files`` to skip training-data combinations and predict
only at the file-specified inputs.

Step 7: Predict Theta (``tfs-predict-theta``)
----------------------------------------------

Predicts operator occupancy *θ* as a function of titrant concentration.

.. code-block:: bash

    tfs-predict-theta \
        tfs_configure_config.yaml \
        tfs_posterior.h5 \
        --genotypes_file predict_genotypes.txt

Output (default ``--out_prefix tfs_theta_pred``):

* ``tfs_theta_pred.csv`` — one row per (genotype, titrant_name, titrant_conc)
  with posterior quantile columns and an ``in_training_data`` flag.

``predict_genotypes.txt`` is a plain-text file with one genotype per line
(e.g. ``M42I/K84L`` or ``wt``). Genotypes not seen during training can be
predicted using the mutation-additivity model when the chosen ``theta_model``
supports it (e.g. ``hill_mut``).

Step 8: Categorise Response (``tfs-cat-response``)
---------------------------------------------------

Fits categorical response curve models to the *θ*-vs-titrant output of
``tfs-predict-theta`` and selects the best-fitting model per
(genotype, titrant_name) pair by AIC weight.

.. code-block:: bash

    tfs-cat-response \
        tfs_theta_pred.csv \
        --workers 8

Output (default ``--out_prefix tfs_cat_response``):

* ``tfs_cat_response.csv`` — one row per (genotype, titrant_name) with
  ``best_model``, AIC weights, and fitted parameters for every model.

Step 9: Summarise Fit (``tfs-summarize-fit``)
----------------------------------------------

Collects the outputs of the completed run, computes prediction quality
statistics, and writes diagnostic plots and tables to a ``summary/``
subdirectory.  It can be run after Steps 5–7 are complete and does not
require the full posterior file to be retained.

.. code-block:: bash

    tfs-summarize-fit out/

For a full guide to every output file — including how to read the theta
correlation plots, calibration curves, and parameter-recovery scatter
plots — see :doc:`summarize-fit`.

---

.. _model-components:

Model Components
================

``tfs-configure-model`` accepts ``--<axis>_model`` flags to select from a
registry of pluggable sub-models for each aspect of the generative process.

Condition Growth (``--condition_growth_model``)
-----------------------------------------------

Maps operator occupancy to per-condition growth rates: *k = b + A·m·θ*.

* **linear** (default) — shared hierarchical prior for *m* and *b* across conditions
* **power** — power-law relationship; incompatible with ``--theta_rescale_model logit``
* **saturation** — saturating (Michaelis-Menten-like) relationship; incompatible with ``logit``

Growth Transition (``--growth_transition_model``)
--------------------------------------------------

Models the lag phase when bacteria switch from pre-selection to selection medium.

* **instant** (default) — no lag; genotypes immediately adopt the new growth rate
* **memory** — occupancy-dependent lag time
* **baranyi** — Baranyi–Roberts lag model
* **baranyi_k** — Baranyi model parameterised through growth rate *k*
* **baranyi_tau** — Baranyi model parameterised through lag time *τ*
* **two_pop** — two-population lag model

Initial Population (``--ln_cfu0_model``)
-----------------------------------------

Models the starting genotype frequencies in each replicate.

* **hierarchical** (default) — shared global prior on ln(CFU\ :sub:`0`)
* **hierarchical_factored** — factored hierarchical prior

Pleiotropic Growth Effect (``--dk_geno_model``)
------------------------------------------------

Models the growth-rate offset attributable to each genotype independent of TF
occupancy (*dk_geno*).

* **hierarchical_geno** (default) — per-genotype effects drawn from a global prior
* **fixed** — no genotype-specific pleiotropic effects
* **pinned** — genotype-specific effects pinned to externally supplied values

Activity (``--activity_model``)
---------------------------------

Models the per-genotype scalar *A* that multiplies the occupancy contribution
to growth.

* **horseshoe_geno** (default) — sparse horseshoe prior over genotypes
* **hierarchical_geno** — standard hierarchical prior over genotypes
* **horseshoe_mut** — sparse horseshoe prior, decomposed by mutation
* **hierarchical_mut** — hierarchical prior, decomposed by mutation
* **fixed** — all genotypes share the wildtype activity (*A* = 1)

Occupancy (``--theta_model``)
------------------------------

Parameterises fractional operator occupancy *θ* as a function of titrant
concentration.

* **hill_geno** (default) — Hill equation with per-genotype *Kd* and *n*
* **categorical_geno** — independent *θ* at each titrant concentration
* **hill_mut** — Hill equation with per-mutation additive effects on *Kd* and *n*
* **thermo.\*** — thermodynamic partition-function models (see
  ``configure_model_cli.py`` docstring for the full set of registry keys)

Transformation Correction (``--transformation_model``)
-------------------------------------------------------

Corrects for congression (multiple plasmids entering one cell during
transformation).

* **empirical** (default) — empirical correction curve
* **logit_norm** — logit-normal correction
* **single** — no correction (assumes exactly one plasmid per cell)

Theta Rescale (``--theta_rescale_model``)
------------------------------------------

Rescales *θ* before it enters the condition-growth model.

* **passthrough** (default) — identity; *θ* ∈ [0, 1]
* **logit** — maps *θ* → log(*θ*/(1−*θ*)); expands dynamic range at extremes.
  Incompatible with the ``power`` and ``saturation`` growth models.

Noise Models
------------

Additional noise components (default ``zero`` for all):

* ``--theta_growth_noise_model``: ``zero`` (default), ``beta``, ``logit_normal``
* ``--theta_binding_noise_model``: ``zero`` (default), ``beta``
* ``--growth_noise_model``: ``zero`` (default), ``normal_kt`` (learns a global
  growth-rate noise term *σ_k* in quadrature with *ln_cfu_std*)

---

.. _model-naming:

Model Naming Conventions
========================

Model component names follow a set of conventions that encode what each component does.

Level of Parameterisation
--------------------------

Models that infer one parameter per *genotype* carry a ``_geno`` suffix; models that
decompose effects at the *mutation* level carry a ``_mut`` suffix. Models with no
natural per-mutation alternative (e.g. ``fixed``) have no suffix.

Examples: ``hierarchical_geno``, ``horseshoe_mut``, ``hill_geno``, ``hill_mut``.

Thermodynamic Theta Models
---------------------------

Operator-occupancy (*θ*) models derived from an explicit partition function use a
three-part dot-separated name::

    thermo.{MODEL}.{PRIOR}

*MODEL* encodes the partition-function topology with four underscore-separated fields:

.. list-table::
   :header-rows: 1
   :widths: 10 60

   * - Field
     - Meaning
   * - ``O``
     - Oligomeric state (e.g. ``O2`` = homodimer)
   * - ``C``
     - Number of conformational states
   * - ``K``
     - Number of independent equilibrium constants
   * - ``U``
     - Unfolded state: ``U0`` = folded only, ``U1`` = folding equilibrium

A trailing letter (``a``, ``b``, …) disambiguates topologically distinct models
that share the same O/C/K/U counts. These letters carry no ordering; ``a`` simply
means "first registered variant."

Currently implemented models:

* ``O2_C4_K3_U0_a`` — four-state lac-repressor homodimer (no unfolding)
* ``O2_C4_K3_U1_a`` — same with an explicit folding/unfolding equilibrium
* ``O2_C12_K5_U0_a`` — full MWC two-state homodimer (no unfolding)
* ``O2_C12_K5_U1_a`` — same with an explicit folding/unfolding equilibrium

*PRIOR* describes how the equilibrium constants are parameterised:

.. list-table::
   :header-rows: 1
   :widths: 10 60

   * - Name
     - Description
   * - ``PK``
     - Independent normal prior on each log-*K*
   * - ``PddG``
     - Priors informed by estimated ΔΔG values
   * - ``PnnC``
     - Neural network predicting per-conformation ΔΔG values
   * - ``PnnK``
     - Neural network predicting log-*K* values directly (planned)

Full example names: ``thermo.O2_C4_K3_U0_a.PK``, ``thermo.O2_C12_K5_U1_a.PnnC``.
