=============
Grids of runs
=============

A grid is a sweep: the same pipeline run over every combination of a few
settings. ``tfs-setup-grid`` sets up a grid of model fits and
``tfs-setup-sim-grid`` a grid of simulations. ``tfs-summarize-grid`` and
``tfs-summarize-calibration`` collect the results once the runs finish.

How a grid works
----------------

A **grid YAML** lists blocks of variants. The setup command takes the
Cartesian product of every block and makes one run directory per
combination. In each directory it writes the run's configuration, renders a
Jinja2 template (a shell or Slurm script) with that run's template
variables, and records the combination in ``combo.json``. It also writes
``grid_summary.json`` at the top, listing every run.

Run directories are named ``run_{index:04d}_<run_name>``, where
``<run_name>`` is the rendered ``run_name`` template with unsafe characters
replaced by underscores. Without a ``run_name`` the suffix is built from the
variable values.

.. code-block:: text

    my_grid/
    ├── grid_summary.json
    ├── inputs/                  copies of every input file
    │   ├── binding.csv
    │   ├── growth.csv
    │   └── library_config.yaml
    ├── run_0001_linear__instant__hill_geno__seed0/
    │   ├── combo.json
    │   ├── tfs_configure_config.yaml
    │   ├── tfs_configure_priors.csv
    │   ├── tfs_configure_guesses.csv
    │   ├── tfs_configure_library.csv
    │   └── run.srun
    ├── run_0002_linear__instant__hill_geno__seed1/
    └── ...

A grid directory is self-contained, so it can be moved as a unit on and off
a cluster. Every input file a run needs is copied once into
``<out_dir>/inputs/``, and each run's config and script refer to it as
``../inputs/<name>``. Two different files with the same name are kept apart
by a numeric suffix. Relative paths in the grid YAML resolve against the
grid YAML's own directory. Setup checks every combination before writing
anything and fails on a missing input file, an input that is a directory,
a config value naming a file the grid does not know to copy, or a template
that does not render. Launch each run from inside its own directory.

Two kinds of variables go into the product. Variables in the
``configure_model`` or ``simulate`` blocks go into the run's configuration.
Variables in the ``template`` blocks go only into the rendered script. To
use a value in both places, list it in both. ``run_name`` can use either
kind. A variant with several keys keeps them together, so use one when
settings only make sense in combination. The ``basename`` filter strips the
directory from a file-valued variable:
``run_name: "{{ growth_df | basename }}__{{ condition_growth }}"``.

Model grids
-----------

``tfs-setup-grid`` calls ``tfs-configure-model`` in each run directory, so
the configuration files exist as soon as the grid does:

.. code-block:: bash

    tfs-setup-grid grid.yaml --out_dir my_grid

The :download:`annotated example <../../examples/tfmodel/grid.yaml>` and
its Slurm template :download:`run.srun <../../examples/tfmodel/run.srun>`
are in ``examples/tfmodel/``. The template runs the pre-fit, a MAP fit, a
Laplace posterior, parameter extraction, growth and *θ* prediction, and
``tfs-cat-response``. Submit every run with:

.. code-block:: bash

    for d in my_grid/run_*/; do (cd "$d" && sbatch run.srun); done

A shortened grid YAML:

.. code-block:: yaml

    run_name: "{{ condition_growth }}__{{ theta }}__seed{{ seed }}"
    output_file: run.srun

    configure_model:
      - name: data
        variants:
          - binding_df: data/binding.csv
            growth_df: data/growth.csv
            library_config: data/library_config.yaml

      - name: condition_growth
        auto: condition_growth

      - name: theta_and_epistasis
        variants:
          - theta: hill_geno
            epistasis: false
          - theta: hill_mut
            epistasis: true

    template:
      - name: seed
        variants:
          - seed: 0
          - seed: 1

A ``configure_model`` variable is any ``tfs-configure-model`` argument
without the leading ``--``. Component choices drop the ``_model`` suffix:
``condition_growth`` rather than ``condition_growth_model``, and likewise
``growth_transition``, ``ln_cfu0``, ``dk_geno``, ``activity``, ``theta``,
``transformation``, ``theta_rescale``, ``theta_growth_noise``,
``theta_binding_noise``, ``growth_noise`` and ``sample_offset``. The input
files (``binding_df``, ``growth_df``, ``presplit_df``, ``base_growth_df``,
``library_config``, ``thermo_data``) are copied into ``inputs/``.

``auto: <axis>`` in place of ``variants`` enumerates every registered
component on that axis. It is safe for small axes such as
``condition_growth`` or ``growth_transition``. For ``theta`` it also yields
``_simple``, a private component used only by the pre-fit, and every
thermodynamic model, so list theta variants by hand. A combination that
``tfs-configure-model`` refuses is skipped, and the reason is logged in
``grid_summary.json``. For example, the ``mixture`` transformation with its
default ``homodimer`` rule refuses any activity component but ``fixed``.

Each run's configuration files are always ``tfs_configure_config.yaml``,
``tfs_configure_priors.csv``, ``tfs_configure_guesses.csv`` and, for a
growth model, ``tfs_configure_library.csv``. The template refers to them by
those names.

Simulation grids
----------------

``tfs-setup-sim-grid`` writes a ``tfs_sim_config.yaml`` in each run
directory: a base simulate config with the run's overrides applied.

.. code-block:: bash

    tfs-setup-sim-grid simulate_grid.yaml --out_dir my_sim_grid
    for d in my_sim_grid/run_*/; do (cd "$d" && bash run.sh); done

The :download:`example grid <../../examples/simulate/simulate_grid.yaml>`
and its template :download:`run.sh <../../examples/simulate/run.sh>` are in
``examples/simulate/``. The grid YAML has the same form as a model grid,
with two differences. A ``base_config`` key names the base simulate config,
and the config blocks are called ``simulate``:

.. code-block:: yaml

    base_config: simulate_config.yaml
    run_name: "{{ theta_component }}__noise{{ tube_noise_sigma }}__seed{{ seed }}"
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
          - seed: 0
          - seed: 42

    template:
      - name: num_replicates
        variants:
          - num_replicates: 3

A ``simulate`` variable replaces a top-level key of the base config. Nested
keys cannot be overridden one at a time, so to change one key inside a
block such as ``binding_data``, give the whole block as the variant. The
keys must be valid simulate-config keys (see :doc:`simulation`); setup does
not check them, but ``tfs-simulate`` refuses an unknown key when the run
starts. The random seed is ``seed``. ``auto`` is not available in simulation
grids.

File-valued simulate keys are copied into ``inputs/`` like model-grid
inputs: ``thermo_data``, ``empirical.phenotype_model``,
``od600.calibration`` and any ``binding_data`` ``choose_by`` that names a
file. A relative path in the base config resolves against the base config's
directory; one in a ``simulate`` override resolves against the grid YAML.

A simulation grid becomes a calibration study when its template also
configures and fits each simulated data set and runs ``tfs-summarize-fit``.
Fit settings then go in the ``template`` blocks, since only the script
sees them.

Summarizing a model grid
------------------------

``tfs-summarize-grid`` collects a model grid into one table:

.. code-block:: bash

    tfs-summarize-grid my_grid

It writes ``my_grid/grid_summary.csv`` (``--out_prefix`` changes the path)
with one row per run directory that has a ``combo.json``. The columns are
the run name, the ``configure_model`` and ``template`` variables,
``configure_complete`` (whether ``tfs_configure_config.yaml`` exists), and,
when the run has a ``*_fit_summary.json`` from ``tfs-summarize-fit``, its
statistics flattened into columns such as ``theta_training_rmse``,
``theta_test_rmse``, ``growth_training_rmse`` and ``final_loss``.
``tfs-summarize-fit`` writes that file to ``summary/`` by default, and
``tfs-summarize-grid`` looks only in the run directory itself, so run it
with ``--out_prefix tfs_summarize`` inside the run directory if the grid
summary should pick it up.

Summarizing calibration across a simulation grid
------------------------------------------------

``tfs-summarize-calibration`` asks whether the posterior intervals of a
simulate-and-fit grid are honest. Each run directory needs its
``combo.json`` and, once finished, the ``tfs-summarize-fit`` outputs in
``summary/``, where the posterior quantiles are joined to the simulated
truth for *θ* and for every fitted parameter with ground truth.

.. code-block:: bash

    tfs-summarize-calibration my_sim_grid --baseline guide_type=component --facet_by guide_type

For every run, quantity and stratum it computes the coverage of central
intervals from 50% to 99%, the mean calibration error and bias (negative
means intervals too narrow), PIT uniformity, mean interval widths, RMSE and
Pearson r. Widths are there so a method cannot look calibrated just by being
vague. Genotype-level quantities are also split by whether the genotype has
binding data and by its purity (spike, bulk or mixed), and *θ* by whether
the true value is resolvable (inside ``[regime_eps, 1 - regime_eps]``) or
saturated. Runs that share every grid variable except the replicate keys
(``--replicate_keys``, default ``seed`` and ``fit_seed``) form an arm and
are averaged. With ``--baseline key=value ...`` every other run is paired
with the baseline run fit to the same simulated data, and the paired
differences are reported. As with ``tfs-compare-runs``, every value is a
raw number with no thresholds or grades.

Outputs, under ``--out_prefix`` (default ``tfs_calibration``):
``_runs.csv`` (per run, quantity and stratum), ``_run_status.csv`` (which
runs are finished and what is missing), ``_arms.csv`` (mean and SD over
replicates), ``_paired.csv`` and ``_paired_summary.csv`` (with
``--baseline``), a calibration-curve PDF for ``--plot_quantity`` (default
``theta_test``), and ``_metadata.json`` with the resolved settings.

See :doc:`cli` for every flag of these four commands.
