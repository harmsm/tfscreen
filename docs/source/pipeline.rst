========
Pipeline
========

This page lists every step from sequencing reads to a posterior, the command
for each and the file each one hands to the next. The pages linked from each
step explain the choices. :doc:`cli` lists every flag.

Overview
--------

.. code-block:: text

   reads (one fastq pair per tube)
     │  tfs-process-fastq            library_config.yaml
     ▼
   counts/counts_<tube>.csv
     │  tfs-process-counts           tube table, OD600 table, OD600 calibration
     ▼                               (tfs-calibrate-od600 makes the calibration)
   growth.csv
     │  tfs-configure-model          library_config.yaml, optional binding.csv,
     ▼                               growth priors, held prior values
   tfs_configure_config.yaml + _priors.csv + _guesses.csv + _library.csv
     │  tfs-prefit-calibration       joint models only: pins growth k and m
     │  tfs-fit-model                MAP or SVI
     ▼
   tfs_fit_model_checkpoint.pkl + _params.npz
     │  tfs-sample-posterior         Laplace (MAP) or guide draws (SVI)
     ▼
   tfs_posterior.h5
     │  tfs-extract-params, tfs-predict-theta, tfs-predict-growth,
     │  tfs-predict-epistasis
     ▼
   per-parameter and per-prediction tables with q<level> columns

Every command prints its provenance when it starts: the tfscreen version, the
git commit with a flag for uncommitted changes, and the command line. A
command that takes ``--out_prefix`` also writes it to
``{out_prefix}_provenance.json``. The model configuration and every fit
checkpoint carry the same record.

1. Count reads
--------------

Run ``tfs-process-fastq`` once per tube. It calls each read pair against the
library described by the library YAML and writes ``counts_<f1 name>.csv``
with one row per library genotype, plus an ``__unknown__`` row for reads that
matched none.

.. code-block:: bash

   tfs-process-fastq library_config.yaml tube01_R1.fastq.gz tube01_R2.fastq.gz --out_dir counts

The same library YAML goes to ``tfs-configure-model`` later. Keep one per
experiment. See :doc:`process-raw`.

2. Tube totals and the growth table
-----------------------------------

The model needs the number of cells in each tube. The lab measures OD600, so
the totals come from an OD600 calibration made once per instrument and strain
with ``tfs-calibrate-od600``. ``tfs-process-counts`` reads the tube table, the
OD600 readings, the calibration and the culture volume, computes each tube's
total and its uncertainty, and turns the counts into the growth table. All
libraries go through in one call; the tube table's ``library`` column keeps
them apart.

.. code-block:: bash

   tfs-calibrate-od600 replicates.csv plate_counts.csv --out_prefix od600_calibration
   tfs-process-counts tube_table.csv counts \
       --od600_file od600.csv \
       --od600_calibration_file od600_calibration.yaml \
       --tube_volume_mL 5 \
       --out_prefix growth

Lab-specific cleanup, such as swapped samples, resequenced tubes pooled
together or failed tubes, belongs in the lab's own script, which writes the
clean tube table. ``examples/process_raw/`` has a synthetic tube table, OD600
table and count files to run these commands on. See :doc:`process-raw`.

3. Configure the model
----------------------

``tfs-configure-model`` reads the data, checks it against the library and
writes the model configuration. The defaults are the current recommendations:
the count likelihood, one level offset per tube and a separate starting
abundance per library. Priors that need values other than the defaults are
set here, never by editing the priors file.

A joint model, with binding data:

.. code-block:: bash

   tfs-configure-model --growth_df growth.csv --binding_df binding.csv \
       --library_config library_config.yaml

A growth-only model on the wt-relative X scale, with per-condition growth
priors and held population SDs:

.. code-block:: bash

   tfs-configure-model --growth_df growth.csv \
       --library_config library_config.yaml \
       --theta_model hill_relative \
       --growth_shares_replicates \
       --growth_priors growth_priors.csv \
       --set_priors sigma_fixed=0.17 theta_log_hill_n_hyper_scale_fixed=0.5

See :doc:`model-inputs` for the inputs and :doc:`model` for the components.

4. Fit
------

For a joint model, ``tfs-prefit-calibration`` first fits a small model to the
binding genotypes and pins each condition's growth baseline and slope in the
priors file. It refuses growth-only models; their growth priors come from step
3. Then ``tfs-fit-model`` runs MAP or SVI until the convergence monitor stops
it.

.. code-block:: bash

   tfs-prefit-calibration tfs_configure_config.yaml --seed 1
   tfs-fit-model tfs_configure_config.yaml --seed 1 --analysis_method map

On a full library, MAP is the route that has worked; the SVI guides slid
along the growth slope and X ridge of the growth-only model. A MAP with level
tube offsets runs in three stages by default, so the offsets cannot settle in
a mode where they carry the population's growth. See :doc:`fitting`.

5. Posterior
------------

``tfs-sample-posterior`` turns a checkpoint into posterior draws. For a MAP
checkpoint it builds a Laplace approximation. ``--laplace auto`` (the default)
uses the full Laplace up to 20,000 parameters and the arrowhead Laplace above,
which gives the same Gaussian computed one genotype at a time. Directions it
had to hold at the MAP are written to ``{out_prefix}_held_directions.csv``,
because they have no interval.

.. code-block:: bash

   tfs-sample-posterior tfs_configure_config.yaml tfs_fit_model_checkpoint.pkl \
       --skip_growth_observations

6. Results
----------

The posterior file feeds the extraction and prediction commands. Each writes
bare quantile columns (``q0.025``, ``q0.5``, ``q0.975`` and so on).

.. code-block:: bash

   tfs-extract-params tfs_configure_config.yaml tfs_posterior.h5
   tfs-predict-theta tfs_configure_config.yaml tfs_posterior.h5
   tfs-predict-epistasis tfs_configure_config.yaml tfs_posterior.h5

A growth-only relative fit predicts X rather than an occupancy, so its
epistasis is only defined on the additive scale: pass
``tfs-predict-epistasis --scale add``. ``tfs-cat-response`` classifies the
response curves, ``tfs-extract-epistasis``
computes epistasis from any long-form table, and ``tfs-compare-runs``
measures agreement between runs. See :doc:`downstream`.

7. Check the fit
----------------

``tfs-summarize-fit`` gathers a run's diagnostics from its directory: the
loss history, observed against predicted growth, and, for a fit with tube
offsets, the offsets in prior SDs with a flag for offsets that trend with
titrant or time. A flagged fit has offsets carrying growth the model leaves
out, so read its curves with care. On a simulation it also compares every
parameter and prediction with the truth. See :doc:`summarize-fit`.

.. code-block:: bash

   tfs-summarize-fit .

Simulations
-----------

``tfs-simulate`` writes a simulated experiment and its ground truth from one
YAML file. It writes the experiment in the raw formats a lab's data come in,
one counts file per tube, the tube table and, with an ``od600`` block, the
OD600 table and calibration, so a simulation enters the pipeline at step 2
through the same ``tfs-process-counts`` command as real data; the command is
printed. (Step 1 is replaced by the simulator's own sequencing model.) Its
growth table, written directly, is the same table and can enter at step 3.
A simulation can also follow a real experiment's tube table (the ``design``
key), so the same chain runs on a simulated copy of the experiment with only
the data paths changed. A full-size simulation of a real screen is large,
tens of GB of memory and over an hour for one seed. ``tfs-setup-sim-grid``
and ``tfs-setup-grid`` lay out grids of simulations and fits. See
:doc:`simulation` and :doc:`grid`.
