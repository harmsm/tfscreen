===========
Quick start
===========

This page runs the bundled example in ``examples/simulate-and-analyze/``
from start to finish. It simulates a small TF-library screen, fits the
hierarchical Bayesian model to the simulated data, and compares the fit with
the known truth.

The example library has 483 genotypes: wild type, the single mutants at
three NNT sites (M42 in tile 1, H74 and K84 in tile 2), the double mutants
between the two tiles, and nine spiked control sequences, three of which
are not in the bulk library.
It is grown under kanR and pheS selection at eight IPTG concentrations in
two replicates. The fit is the slow step. Expect tens of minutes on a
laptop CPU and a few minutes on a GPU.

Prerequisites
-------------

Install ``tfscreen`` and check that the commands are on your PATH:

.. code-block:: bash

    git clone https://github.com/harmslab/tfscreen
    cd tfscreen
    pip install -e .
    tfs-simulate --help

JAX and Numpyro are installed as dependencies. On a Linux machine with a
GPU, install the ``jaxlib`` wheel for your CUDA version.

Getting the example
-------------------

Copy the example directory to a working location:

.. code-block:: bash

    cp -r examples/simulate-and-analyze/ ~/tfscreen-example
    cd ~/tfscreen-example

It holds three files. ``simulate_config.yaml`` sets up the simulation:
library genetics, conditions, growth parameters, binding and presplit data.
It also serves as the library description for ``tfs-configure-model``.
``hill_params.csv`` holds the Hill parameters of wild type and three spiked
single mutants, which become their true *θ* curves and their binding data.
``run.sh`` is the pipeline.

Running the pipeline
--------------------

Run it from the example directory, since the config names
``hill_params.csv`` by a relative path:

.. code-block:: bash

    bash run.sh simulate_config.yaml out 1

The arguments are the simulate config, the output directory (created if
needed) and the random seed (default 1). The seed goes to the simulation,
the pre-fit and the fit.

On a laptop the script has JAX spread work over eight CPU devices
(``XLA_FLAGS="--xla_force_host_platform_device_count=8"``). On a cluster
with a GPU, comment that line out and uncomment ``module load cuda/...``.
The ``#SBATCH`` lines at the top let you submit the same script with
``sbatch``.

The script runs nine steps and prints a ``>>>`` header before each one.

1. ``tfs-simulate simulate_config.yaml --out_prefix out/tfs_sim --seed 1``
   builds the library, draws every genotype's *θ* curve from the
   ``hill_mut`` model with sparse epistasis, grows two replicates and
   sequences them. It writes the growth, binding and presplit tables plus
   the ground truth. The script then moves into ``out/``.

2. ``tfs-configure-model`` picks the model. The script passes the binding,
   growth and presplit tables and the simulate config as
   ``--library_config``, and chooses components that match how the data
   were simulated: ``linear`` condition growth, an ``instant`` growth
   transition, ``hierarchical`` ln_cfu0, ``hierarchical_geno`` dk_geno,
   ``fixed`` activity, ``hill_mut`` theta with ``--epistasis``, the
   ``single`` transformation (the simulation puts one plasmid in each
   cell), ``passthrough`` theta rescaling, ``logit_normal`` growth-side
   theta noise and ``zero`` binding-side noise. Growth is observed through
   the ``counts`` likelihood with a ``level`` offset per tube and ``zero``
   growth noise, the recommended default. ``--growth_shares_replicates``
   gives both replicates the same growth parameters.

3. ``tfs-prefit-calibration`` runs a MAP fit on a simplified model of the
   genotypes with binding data and writes each condition's growth baseline
   *k* and slope *m* into the priors and guesses CSVs.

4. ``tfs-fit-model --analysis_method svi`` runs a MAP warm-up, then fits the
   variational guide. This is the long step.

5. ``tfs-sample-posterior --skip_growth_observations`` draws 10,000
   posterior samples from the guide. The flag leaves the stored
   per-observation growth sites out of the file. They would be several GB
   here, and the later steps recompute growth from the parameter samples.

6. ``tfs-extract-params`` writes posterior quantiles for each parameter
   group.

7. ``tfs-predict-theta`` predicts *θ* at every genotype and concentration in
   the training data.

8. ``tfs-predict-growth --num_marginal_samples 500`` predicts every
   training ln_cfu from 500 posterior samples.

9. ``tfs-summarize-fit .`` compares the predictions with the simulated
   truth and writes plots and statistics to ``out/summary/``.

:doc:`pipeline` explains each step and :doc:`cli` lists every flag.

Expected outputs
----------------

Every command writes a ``*_provenance.json`` next to its outputs with the
tfscreen version, git commit and command line. After the run, ``out/``
holds:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Contents
   * - ``tfs_sim_growth.csv``
     - Simulated growth table, the format ``tfs-process-counts`` writes,
       with each row's true values.
   * - ``tfs_sim_binding.csv``
     - Simulated binding curves for the four genotypes in
       ``hill_params.csv``.
   * - ``tfs_sim_presplit.csv``
     - Simulated presplit abundances.
   * - ``tfs_sim_library.csv``
     - Every genotype in the library and its origin.
   * - ``tfs_sim_parameters.csv``
     - True per-genotype parameters (Hill parameters, ``dk_geno``,
       activity).
   * - ``tfs_sim_genotype_theta.csv``
     - True *θ* for every genotype and concentration.
   * - ``tfs_sim_growth_parameters.csv``
     - True growth *k* and *m* for each condition.
   * - ``tfs_sim_transformation_lam.csv``
     - The simulated congression rate.
   * - ``tfs_sim_input-config.yaml``
     - The simulate config as run.
   * - ``tfs_configure_config.yaml``
     - Model configuration read by every later step.
   * - ``tfs_configure_priors.csv``, ``tfs_configure_guesses.csv``
     - Priors and starting values, updated by the pre-fit (the originals
       are kept as ``.bak``).
   * - ``tfs_configure_library.csv``
     - Library composition resolved from the simulate config.
   * - ``tfs_configure_model_stats.csv``, ``tfs_configure_model_stats.json``
     - Parameter and observation census of the configured model.
   * - ``tfs_prefit_*``
     - Pre-fit diagnostics: checkpoint, parameters, losses, convergence.
   * - ``tfs_fit_model_checkpoint.pkl``
     - Fitted model checkpoint.
   * - ``tfs_fit_model_params.npz``
     - Fitted guide parameters.
   * - ``tfs_fit_model_losses.txt``, ``tfs_fit_model_convergence.csv``
     - Loss trace and one row per convergence window.
   * - ``tfs_fit_model_premap_*``
     - The MAP warm-up's checkpoint, parameters, losses and convergence.
   * - ``checkpoints/``
     - Numbered checkpoints written during the fit.
   * - ``tfs_posterior.h5``
     - Posterior samples.
   * - ``tfs_params_*.csv``
     - Posterior quantiles for each parameter group.
   * - ``tfs_pred_theta.csv``
     - Predicted *θ* with posterior quantiles.
   * - ``tfs_pred_growth.csv``
     - Predicted ln_cfu with posterior quantiles.
   * - ``summary/``
     - Plots and statistics from ``tfs-summarize-fit``.

Next steps
----------

:doc:`summarize-fit` walks through every plot and statistic in
``out/summary/``. :doc:`simulation` documents the simulate config, so you
can change the library, the conditions or the noise. :doc:`grid` runs a
sweep of settings. :doc:`process-raw` turns real FASTQ files into the growth
table that ``tfs-configure-model`` reads.
