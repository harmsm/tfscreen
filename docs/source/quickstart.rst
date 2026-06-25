===========
Quick Start
===========

This page walks through a complete simulate-and-analyze run using the
bundled example in ``examples/simulate-and-analyze/``.  By the end you
will have simulated a synthetic high-throughput TF-library screen, fitted
the hierarchical Bayesian model, and produced a full set of diagnostic
plots and statistics.

The example uses a small library (~600 genotypes across three tiles)
so the full pipeline finishes in roughly 20–60 minutes on a laptop CPU or a
few minutes on a GPU.

Prerequisites
-------------

Install ``tfscreen`` and verify the entry points are on your PATH:

.. code-block:: bash

    git clone https://github.com/harmslab/tfscreen
    cd tfscreen
    pip install -e .
    tfs-simulate --help   # should print usage

The example requires JAX and Numpyro, which are installed as dependencies.
On Apple Silicon you may need to install the Metal-accelerated ``jax-metal``
package separately; on a Linux cluster with a GPU, install the appropriate
``jaxlib`` wheel for your CUDA version.

Getting the example
-------------------

Copy the example directory to a working location:

.. code-block:: bash

    cp -r examples/simulate-and-analyze/ ~/tfscreen-example
    cd ~/tfscreen-example

The directory contains:

* ``simulate_config.yaml`` — simulation parameters (library genetics,
  growth conditions, binding data).
* ``hill_params.csv`` — per-genotype Hill binding parameters that set the
  ground-truth θ values used during simulation.
* ``run.sh`` — the pipeline script.

Running the pipeline
--------------------

.. code-block:: bash

    bash run.sh simulate_config.yaml out/ 1

The three positional arguments are:

1. The simulation config file.
2. The output directory (created automatically).
3. The random seed (``1`` reproduces the documented example outputs).

On a local machine the script uses JAX's CPU parallelism
(``XLA_FLAGS="--xla_force_host_platform_device_count=8"``).  On a
cluster, comment that line out and uncomment ``module load cuda/...``
instead.

The script runs nine steps in sequence, printing a ``>>>`` header before
each one:

.. list-table::
   :header-rows: 1
   :widths: 5 30 65

   * - Step
     - Command
     - What it does
   * - 1
     - ``tfs-simulate``
     - Builds the genotype library, draws per-genotype θ from the Hill
       model, converts θ to growth rates, and writes simulated sequencing
       data.
   * - 2
     - ``tfs-configure-model``
     - Validates inputs, selects model components, writes
       ``tfs_configure_config.yaml``.
   * - 3
     - ``tfs-prefit-calibration``
     - Fast MAP fit to calibrate the growth-linking-function priors.
   * - 4
     - ``tfs-fit-model``
     - Main SVI inference; writes ``tfs_fit_model_checkpoint.pkl``.
   * - 5
     - ``tfs-sample-posterior``
     - Draws posterior samples; writes ``tfs_posterior.h5`` (~5 GB).
   * - 6
     - ``tfs-extract-params``
     - Summarises the posterior into per-parameter CSV files.
   * - 7
     - ``tfs-predict-theta``
     - Predicts θ at every (genotype, titrant concentration) in the
       training data.
   * - 8
     - ``tfs-predict-growth``
     - Predicts ln(CFU) with posterior quantiles for all training
       observations.
   * - 9
     - ``tfs-summarize-fit``
     - Computes fit statistics and writes diagnostic plots and CSVs to
       ``out/summary/``.

Expected outputs
----------------

After the run completes, ``out/`` will contain:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - File / directory
     - Contents
   * - ``tfs_sim_library.csv``
     - All genotypes in the simulated library.
   * - ``tfs_sim_growth.csv``
     - Simulated ln(CFU) data (same format as ``tfs-process-counts`` output).
   * - ``tfs_sim_binding.csv``
     - Simulated binding curve observations.
   * - ``tfs_sim_parameters.csv``
     - Ground-truth per-genotype parameters (θ, dk_geno, etc.).
   * - ``tfs_configure_config.yaml``
     - Model configuration read by all downstream steps.
   * - ``tfs_fit_model_checkpoint.pkl``
     - Fitted model checkpoint.
   * - ``tfs_posterior.h5``
     - Posterior samples (~5 GB; not committed to the repo).
   * - ``tfs_params_*.csv``
     - Posterior summaries (quantiles) for each parameter group.
   * - ``tfs_pred_theta.csv``
     - Predicted θ with posterior quantiles.
   * - ``tfs_pred_growth.csv``
     - Predicted ln(CFU) with posterior quantiles.
   * - ``summary/``
     - Diagnostic plots and statistics from ``tfs-summarize-fit``.

Next steps
----------

* **Understand each analysis step** — see :doc:`analysis` for the full CLI
  reference covering every ``tfs-*`` command in the pipeline.
* **Interpret the diagnostic outputs** — see :doc:`summarize-fit` for a
  guided walkthrough of every plot and statistic in ``out/summary/``.
* **Simulate a parameter sweep** — see :doc:`grid` for how to run a
  Cartesian grid of configurations.
* **Use real data** — see :doc:`process-raw` for converting FASTQ reads
  into the ``growth.csv`` format that ``tfs-configure-model`` accepts.
