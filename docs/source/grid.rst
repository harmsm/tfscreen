====================================
Grid Setup and Summarisation
====================================

``tfscreen`` provides two scripts for setting up parameter sweeps over model
configurations or simulation settings, and one script for summarising the
results once runs are complete.

Overview
--------

Both grid-setup scripts share the same concept:

1. You write a **grid YAML** file describing a base configuration and a set of
   parameter axes, each with a list of variants.
2. The setup script takes the **Cartesian product** of all variant lists and
   creates one subdirectory per combination.
3. Inside each subdirectory the script writes a per-run configuration file and
   optionally renders a **Jinja2 template** (e.g. a Slurm submission script).
4. After all runs complete, ``tfs-summarize-grid`` collects results into a
   summary CSV.

.. code-block:: text

    my_grid/
    ├── grid_summary.json          ← written by tfs-setup-grid
    ├── linear__instant__seed0/
    │   ├── combo.json             ← variable assignments for this run
    │   ├── tfs_configure_config.yaml
    │   ├── tfs_configure_priors.csv
    │   ├── tfs_configure_guesses.csv
    │   └── run.srun               ← rendered Jinja2 template
    ├── linear__instant__seed1/
    │   └── ...
    └── ...

tfs-setup-grid
--------------

Sets up a grid of model-fitting runs. For each combination the script calls
``tfs-configure-model`` in the corresponding subdirectory, so all three
configuration files (``_config.yaml``, ``_priors.csv``, ``_guesses.csv``) are
already present when the grid is created.

.. code-block:: bash

    tfs-setup-grid grid.yaml --out_prefix my_grid

An :download:`annotated example <../../examples/tfmodel/grid.yaml>` is provided
in ``examples/tfmodel/``. Once the grid is set up, submit all jobs with:

.. code-block:: bash

    for d in my_grid/*/; do
        cd "$d" && sbatch run.srun && cd -
    done

Grid YAML format
^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # Directory name template (Jinja2); variables from both sections available.
    run_name: "{{ condition_growth }}__{{ growth_transition }}__seed{{ seed }}"

    # Jinja2 template rendered into each subdirectory (relative to this YAML).
    output_file: run.srun

    # --- configure_model blocks -----------------------------------------------
    # Variables here are forwarded to tfs-configure-model.
    # They are NOT injected into the Jinja2 template.

    configure_model:

      # Fixed arguments — single variant = always selected.
      - name: data
        variants:
          - binding_df: ../data/binding.csv
            growth_df:  ../data/growth.csv

      # 'auto' enumerates every registered component for an axis.
      # Incompatible combinations are skipped automatically.
      - name: condition_growth
        auto: condition_growth

      # Manual enumeration — list only the components you want.
      - name: growth_transition
        variants:
          - growth_transition: instant
          - growth_transition: baranyi

      # Joint (co-varying) block — keys in the same dict always move together.
      - name: theta_and_epistasis
        variants:
          - theta: hill_geno
            epistasis: false
          - theta: hill_mut
            epistasis: true

    # --- template blocks ------------------------------------------------------
    # Variables here are injected into the Jinja2 template only.
    # They are NOT forwarded to tfs-configure-model.

    template:
      - name: seed
        variants:
          - seed: 0
          - seed: 1

      - name: predict_genotypes
        variants:
          - predict_genotypes_file: /path/to/predict_genotypes.txt

Key rules:

* The Cartesian product is taken across **all** blocks (``configure_model`` +
  ``template``).
* ``configure_model`` variables accept any flag accepted by
  ``tfs-configure-model`` (without the ``--`` prefix and ``_model`` suffix for
  component axes; e.g. ``condition_growth`` rather than
  ``--condition_growth_model``).
* Relative paths in ``configure_model`` blocks (``binding_df``, ``growth_df``,
  ``thermo_data``) are resolved relative to the grid YAML and re-expressed
  relative to each subdirectory in the written config.
* The ``auto`` form enumerates every registered component for the given axis.
  Incompatible combinations (e.g. ``power`` growth + ``logit`` theta_rescale)
  are caught by ``tfs-configure-model``, skipped, and logged in
  ``grid_summary.json``.
* Use the ``basename`` Jinja2 filter to strip directory paths from
  file-valued variables in ``run_name``:
  ``"{{ binding_df | basename }}__{{ condition_growth }}"``.

tfs-setup-sim-grid
------------------

Sets up a grid of simulation runs. Each subdirectory receives a
``tfs_sim_config.yaml`` derived from a base config with per-run overrides
applied.

.. code-block:: bash

    tfs-setup-sim-grid simulate_grid.yaml --out_prefix my_sim_grid

An :download:`annotated example <../../examples/simulate/simulate_grid.yaml>`
is provided in ``examples/simulate/``.

Grid YAML format
^^^^^^^^^^^^^^^^

The simulate grid YAML follows the same structure as the model grid, with two
differences:

1. A ``base_config`` key is required, pointing to the base ``simulate_config.yaml``.
2. The blocks are named ``simulate`` (not ``configure_model``).

.. code-block:: yaml

    base_config: ../simulate_config.yaml

    run_name: "{{ theta_component }}__noise{{ tube_noise_sigma }}__seed{{ random_seed }}"
    output_file: run.sh

    simulate:
      - name: thermodynamic_model
        variants:
          - theta_component: thermo.O2_C12_K5_U0_a.PK
          - theta_component: hill_geno

      - name: noise
        variants:
          - tube_noise_sigma: 0.001
          - tube_noise_sigma: 0.005

      - name: seed
        variants:
          - random_seed: 0
          - random_seed: 42

    template:
      - name: num_replicates
        variants:
          - num_replicates: 3

Key rules:

* ``simulate`` variables override top-level keys in the base config. Nested
  keys are not supported — override the entire top-level key if needed.
* The ``auto`` form is not supported for simulate grids.

Jinja2 template variables
^^^^^^^^^^^^^^^^^^^^^^^^^^

The rendered template receives all variables from the ``template`` blocks.
``simulate`` / ``configure_model`` variables are **not** available in the
template — add them to ``template`` as well if they are needed in both places.

The ``run.srun`` and ``run.sh`` files in ``examples/tfmodel/`` and
``examples/simulate/`` show realistic Slurm and shell templates. Inside a
template, each run's configuration file is always named:

* ``tfs_configure_config.yaml`` (model grid)
* ``tfs_sim_config.yaml`` (simulate grid)

tfs-summarize-grid
------------------

Scans a grid directory for completed runs and writes a flat summary CSV.

.. code-block:: bash

    tfs-summarize-grid my_grid

Output (default: ``my_grid/grid_summary.csv``):

* One row per subdirectory that contains a ``combo.json``.
* Columns from the ``configure_model`` / ``template`` variable assignments.
* ``configure_complete`` — whether ``tfs-configure-model`` finished (i.e.
  ``tfs_configure_config.yaml`` is present).
* Flattened fit-summary statistics from ``*_fit_summary.json`` if present
  (e.g. ``theta_training_rmse``, ``growth_training_rmse``, ``final_loss``).
* Calibration statistics from ``*_calib_stats.json`` if present (prefixed
  with ``calib_``).

Use ``--out_prefix`` to write the CSV to a different location.
