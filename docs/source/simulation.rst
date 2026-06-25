==========
Simulation
==========

The ``tfscreen.simulate`` module generates synthetic high-throughput TF-library
selection experiments. Starting from a thermodynamic model of TF binding, it
builds a genotype library, samples per-genotype occupancy (*θ*) values, applies
the growth model to convert occupancy to fitness, and then simulates the
sequencing step. The result is a ``growth.csv`` that has exactly the same
format as the output of ``tfs-process-counts``, so simulated data can be fed
directly into ``tfs-configure-model`` for end-to-end benchmarking.

Configuration File (simulate_config.yaml)
------------------------------------------

All simulation parameters are collected in a single YAML file. An
:download:`annotated example <../../examples/simulate/simulate_config.yaml>`
is provided in ``examples/simulate/``. The top-level sections are:

**Library genetics** — same keys as the ``run_config.yaml`` used by
``tfs-process-fastq`` (``reading_frame``, ``wt_seq``, ``degen_sites``,
``tiles``, ``expected_5p``, ``expected_3p``, ``tile_combos``,
``spiked_seqs``). This means a single config file can drive both simulation
and raw-data processing.

**Phenotype** — controls how genotype *θ* values are drawn:

* ``theta_component`` — registered theta-model name (e.g. ``hill_geno``,
  ``thermo.O2_C12_K5_U0_a.PK``).
* ``theta_rng_seed`` — JAX RNG seed for reproducible *θ* sampling.
* ``theta_priors`` (optional) — dict of hyperparameter overrides for the
  chosen component.
* ``thermo_data`` (optional) — path to structural/thermodynamic data required
  by ``PnnC`` and ``PddG`` theta components.

**Conditions** — ``condition_blocks`` list; each entry specifies:
``library``, ``titrant_name``, ``titrant_conc`` list, ``condition_pre``,
``t_pre``, ``condition_sel``, and ``t_sel`` list.

**Growth** — per-condition ``{m, b}`` parameters mapping *θ* to growth rate
(*k = b + A·m·θ*). ``dk_geno_hyper_*`` sets the pleiotropic growth-cost
distribution. ``activity_wt`` and ``activity_mut_scale`` control per-genotype
TF activity.

**Experimental parameters** — ``transform_sizes``, ``library_mixture``,
``lib_assembly_skew_sigma``, ``transformation_poisson_lambda``,
``multi_plasmid_combine_fcn``, ``cfu0``, ``tube_noise_sigma``,
``total_num_reads``, ``prob_index_hop``, ``random_seed``.

**Growth transition** (optional) — ``growth_transition`` list; one entry per
``condition_pre`` that has a detectable lag phase. Supported models: ``instant``,
``memory``.

**Binding data** (optional) — ``binding_data`` block; if present, a simulated
binding CSV is also written. Required sub-keys: ``genotypes``,
``titrant_name``, ``titrant_conc``, ``noise``.

tfs-simulate
------------

Runs a full simulation from a config file and writes output CSVs.

**Usage:**

.. code-block:: bash

    tfs-simulate <config_file> <output_dir> [options]

**Positional arguments:**

* ``config_file``: Path to the simulate YAML configuration file.
* ``output_dir``: Directory to write output files (created if absent).

**Optional arguments:**

* ``--output_prefix``: Prefix for all output filenames (default: ``tfscreen_``).
* ``--num_replicates``: Number of independent experimental replicates to
  simulate (default: 2).

**Outputs** (all written to ``output_dir``):

* ``{prefix}library.csv`` — all genotypes in the library.
* ``{prefix}phenotype.csv`` — ground-truth growth rates per genotype.
* ``{prefix}genotype_theta.csv`` — ground-truth *θ* per genotype and
  titrant concentration.
* ``{prefix}growth.csv`` — analysis-ready ln(CFU) data (same format as
  ``tfs-process-counts`` output).
* ``{prefix}binding.csv`` — simulated binding curve data (only written when
  ``binding_data`` is present in the config).

tfs-setup-sim-grid
------------------

Creates a directory grid for sweeping simulate parameters (see
:doc:`grid` for a full description of the grid format). Each subdirectory
receives its own ``tfs_sim_config.yaml`` derived from a base config, plus
an optional rendered shell script.

.. code-block:: bash

    tfs-setup-sim-grid simulate_grid.yaml --out_prefix my_sim_grid

An :download:`annotated example grid YAML <../../examples/simulate/simulate_grid.yaml>`
is provided in ``examples/simulate/``. After running the command, launch all
simulations with:

.. code-block:: bash

    for d in my_sim_grid/*/; do
        cd "$d" && bash run.sh && cd -
    done
