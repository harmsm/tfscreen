==========
Simulation
==========

The ``tfscreen.simulate`` module generates synthetic high-throughput screens
of TF libraries. It builds a genotype library, draws each genotype's
operator occupancy (*θ*) curve, converts *θ* to growth rates, grows the
library through the selection experiment and sequences it. The simulated
growth table has the same format as the output of ``tfs-process-counts``, so
it goes straight into ``tfs-configure-model``. Because the ground truth is
known, a simulation is how we benchmark a model choice or an inference
method. :doc:`quickstart` runs one end to end.

The simulation config
---------------------

One YAML file holds every simulation setting. The
:download:`annotated example <../../examples/simulate/simulate_config.yaml>`
in ``examples/simulate/`` comments every key. ``tfs-simulate`` refuses a key
it does not know, so a misspelled key fails at once rather than being
ignored. Times are in minutes and growth rates are per minute throughout.

Library genetics
^^^^^^^^^^^^^^^^

The keys ``reading_frame``, ``first_amplicon_residue``, ``wt_seq``,
``degen_sites``, ``tiles``, ``tile_combos``, ``spiked_seqs``,
``expected_5p`` and ``expected_3p`` describe the library. They are the same
keys ``tfs-process-fastq`` reads, and together with ``library_mixture`` they
are what ``tfs-configure-model --library_config`` reads. Keep one file per
experiment and hand that same file to ``tfs-simulate``, ``tfs-process-fastq``
and ``tfs-configure-model --library_config``. Each tool ignores the keys it
does not use. Nothing checks that ``library_mixture`` is the same in all
three places, so a single file is the only guard against drift.

Phenotypes
^^^^^^^^^^

``theta_component`` names the theta model that draws each genotype's *θ*
curve. ``hill_geno`` draws an independent Hill curve per genotype around a
wild-type reference. ``hill_mut`` draws per-mutation effects, adds them and,
when the library has double mutants, adds sparse pairwise epistasis from a
regularized horseshoe. The ``thermo.*`` keys are thermodynamic
partition-function models; ``PnnC`` and ``PddG`` variants also need
``thermo_data``, the structural ensemble (HDF5) or the per-mutation ΔΔG CSV.
``theta_sim_priors`` overrides the simulation priors of ``hill_geno`` and
``hill_mut`` (wild-type curve, perturbation widths, epistasis scale), and
``theta_priors`` overrides the hyperparameters of the other components. The
example config lists every key. ``theta_rescale`` (``passthrough``, the
default, or ``logit``) transforms *θ* before it enters the growth model, as
the fit-side component of the same name does.

The ``growth`` block maps *θ* to a growth rate for every condition named in
``condition_blocks``. Each entry is a dict whose optional ``model`` key picks
the form. The default ``linear`` takes ``b`` and ``m`` and gives
*k = b + A·m·θ + dk_geno*. ``power`` takes ``b``, ``a`` and ``n``, and
``saturation`` takes ``kmin`` and ``kmax``. These match the fit's
``condition_growth`` components, and ``tfs-simulate`` writes the values as
ground truth under the fit's names (``growth_k``, ``growth_m`` and so on).

Each genotype's pleiotropic growth effect is drawn as
``dk_geno = dk_geno_hyper_shift - exp(Normal(dk_geno_hyper_loc,
dk_geno_hyper_scale))``, with wild type at 0. The three ``dk_geno_hyper_*``
keys are required unless ``dk_geno_zero: true``, which sets every
genotype's ``dk_geno`` to 0.

TF activity *A* defaults to 1 for every genotype. With the default
``activity_component: fixed``, ``activity_wt`` sets wild type and
``activity_mut_scale`` > 0 draws each mutant's log *A* around it.
``activity_component`` can instead be ``hierarchical_geno`` or
``horseshoe_geno``, numpy versions of the fit's priors, tuned through
``activity_priors``.

``phenotype_source: empirical`` replaces all of this with phenotypes
resampled from a distribution fit to real data. See
`Empirical phenotypes`_ below.

Conditions
^^^^^^^^^^

``condition_blocks`` is a list. Each entry gives a ``library``, a
``titrant_name``, a ``titrant_conc`` list, a ``condition_pre`` and its
``t_pre``, and a ``condition_sel`` with a ``t_sel`` list. Every combination
of concentration and selection time is one tube. Every ``condition_pre`` and
``condition_sel`` must appear in ``growth``.

``growth_transition`` is an optional list with one entry per
``condition_pre``, describing the switch from pre-growth to selection.
The simulator supports ``instant`` (no lag), ``memory`` (``tau0``, ``k1``,
``k2``; a lag that depends on *θ*), ``baranyi`` (``tau_lag``, ``k_sharp``)
and ``two_pop`` (``k_trans``). Omit the block for an instant transition
everywhere. The fit has two more transition components, ``baranyi_k`` and
``baranyi_tau``, that the simulator does not have.

Transformation and congression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``transform_sizes`` gives the number of transformants for each entry in
``tile_combos`` plus ``spiked``. ``library_mixture`` gives the ratio in
which those sub-libraries are pooled. ``lib_assembly_skew_sigma`` spreads
the starting genotype frequencies log-normally; 0 makes them even.
``cfu0`` is the number of cells in each tube at the start of pre-growth.

``transformation_poisson_lambda`` gives each cell a zero-truncated Poisson
number of plasmids; 0 or null gives exactly one. A cell with several
plasmids splits its abundance among them and has one growth rate, built from
cell-level physics. ``congression_theta_rule`` sets the cell's *θ*:
``homodimer`` (the default) combines the plasmids' *θ* odds weighted by
share, ``heterodimer`` sits closer to the mean logit, and ``max`` lets the
highest-*θ* plasmid set *θ* and activity. ``homodimer`` and ``heterodimer``
need activity 1. ``congression_dk_rule`` sets the cell's ``dk_geno``:
``dilution`` (the default, the share-weighted mean), ``softmin`` with a
finite ``congression_dk_alpha``, or ``min``, where the worst plasmid sets
the cost. ``congression_dk_alpha`` is required by ``softmin`` and refused by
the other rules. The fit's ``mixture`` transformation uses the same rules.

``tube_noise_sigma`` adds one growth-rate offset per tube, shared by every
genotype in it.

Sequencing and sampling noise
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``total_num_reads`` is the total number of reads across all tubes, and
``prob_index_hop`` is the fraction of reads reassigned to a random genotype.
``seed`` makes the run reproducible; ``tfs-simulate --seed`` overrides it.

Without further settings, read counts are Poisson given each tube's
composition. Real counts vary 5 to 18 times more than that at 100 to 3,000
reads (``planning/studies/noise-anatomy/``). Five optional keys, all off by
default, add the physical sources one at a time. ``founder_sampling`` seeds
each tube with a Poisson number of cells from each transformant clone.
``demographic_growth`` then grows each clone from its founders with
birth-death noise and needs ``founder_sampling``. ``shared_transformation``
draws every replicate from one library assembly and transformation, as from
one glycerol stock, instead of redrawing them per replicate.
``pcr_template_molecules`` draws each tube's reads from that many template
molecules, and ``pcr_amplification_cv`` gives each template's amplification
that coefficient of variation. The count variance grows by about
(reads per tube / templates)(1 + cv²), so matching real data needs templates
on the order of a tenth of the reads per tube.

``condition_selector`` and ``library_selector`` name the columns that define
a growth condition and a library. The defaults are right for every config we
know of; leave them out.

OD600
^^^^^

An optional ``od600`` block gives every tube one OD600 reading.
``calibration`` is a calibration YAML from ``tfs-calibrate-od600`` (the one
in ``examples/od600/`` is synthetic). The simulator inverts it to get each
tube's true OD600 from its true total, then adds reading noise and applies
the detection threshold. ``tube_volume_mL`` (default 5.0) converts the
calibration's per-mL units to the tube. ``num_od_only_replicates`` adds
replicates that are read but not sequenced. With
``sample_cfu_from_od600: true`` the growth table gets each tube's total as
estimated from its reading, as in the lab, instead of the true total.

Binding data
^^^^^^^^^^^^

An optional ``binding_data`` block writes a simulated binding table for
``tfs-configure-model --binding_df``. ``titrant_name``, ``titrant_conc`` and
``noise`` describe the assay. Noise is Gaussian on ``theta_obs`` and is not
clipped to [0, 1], since the fit's binding likelihood is an unclipped
Normal. ``clip_theta_obs: true`` restores the old clipping.

Two sub-blocks pick the measured genotypes. ``spiked_binding`` draws from
``spiked_seqs``, the clean monoclonal controls. ``library_binding`` draws
from the rest of the library, so its genotypes grow under congression like
any bulk genotype, which is what lets the fit learn the growth slope *m*
from data. Each sub-block takes ``choose_by``: ``stratified`` (curves spread
across the prior), ``random``, or the path to a CSV of measured Hill
parameters (``genotype``, ``theta_low``, ``theta_high``, ``log_hill_K``,
``hill_n``) that become those genotypes' true phenotypes. ``num`` sets how
many genotypes to pick and is forbidden with a file. It is optional for
``spiked_binding`` (default all spikes) and required for a non-file
``library_binding``. ``library_binding`` picks its genotypes after the
growth simulation, from those that survived with growth data. A file of
measured parameters needs a Hill theta component (``hill_geno`` or
``hill_mut``).

Presplit and base-growth data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An optional ``presplit_data`` block writes a presplit table for
``tfs-configure-model --presplit_df``: one ``ln_cfu`` per replicate,
``condition_pre`` and genotype, sequenced before the culture is split, with
the true ``ln_cfu_0`` alongside. Its one optional key, ``noise``, adds
Gaussian noise on the ln scale. ``presplit_data:`` with nothing under it
turns it on with no extra noise.

An optional ``base_growth_data`` block writes direct growth-rate
measurements for ``tfs-configure-model --base_growth_df``. ``k_ref`` (the
wild-type rate) is required. ``genotypes`` (default ``[wt]``) lists genotypes
already in the library, each measured at ``k_ref`` plus its own
``dk_geno``. ``rates`` overrides the true rate for listed genotypes, and
``noise`` is the measurement SD, also reported as ``rate_std``. Set
``noise`` above 0 if the table will be fit: ``rate_std`` is a likelihood
scale, and a value of 0 crashes the pre-fit and the fit the first time the
model is traced.

tfs-simulate
------------

``tfs-simulate`` runs a simulation from a config file:

.. code-block:: bash

    tfs-simulate simulate_config.yaml --out_prefix sim/tfs_sim --num_replicates 2 --seed 1

It draws the ground-truth library and phenotypes once, then simulates
``--num_replicates`` independent replicates (default 2). Files are written as
``{out_prefix}_{name}.csv``; the default prefix is ``tfs_sim``, and a
directory part of the prefix is created if needed. ``tfs-simulate`` refuses
to overwrite existing outputs. See :doc:`cli` for every flag.

Every run writes these files:

``tfs_sim_growth.csv``
    The analysis-ready growth table, the same format as
    ``tfs-process-counts`` output, with the read counts and tube totals the
    count likelihood needs. It also carries each row's true ``theta``,
    ``dk_geno``, ``activity``, ``k_pre``, ``k_sel`` and ``ln_cfu_0``.
``tfs_sim_library.csv``
    Every genotype in the library and its origin.
``tfs_sim_parameters.csv``
    Ground-truth per-genotype parameters (the theta model's parameters,
    ``dk_geno``, activity).
``tfs_sim_genotype_theta.csv``
    Ground-truth *θ* per genotype and titrant concentration.
    ``tfs-summarize-fit`` finds this file and uses it as the reference for
    out-of-sample *θ*.
``tfs_sim_growth_parameters.csv``
    Ground truth for each condition's growth parameters, keyed by
    ``condition_rep``.
``tfs_sim_transformation_lam.csv``
    The configured congression rate, the ground truth for the fit's ``lam``.
``tfs_sim_input-config.yaml``
    The config as run, with ``--seed`` applied.
``tfs_sim_provenance.json``
    Version, git commit and command line of the run.

The optional blocks add ``tfs_sim_binding.csv`` (and, with
``library_binding``, ``tfs_sim_library_binding.csv`` listing the in-library
genotypes chosen), ``tfs_sim_presplit.csv``, ``tfs_sim_base_growth.csv``
with ``tfs_sim_k_ref.csv``, and ``tfs_sim_od600.csv``.

Empirical phenotypes
--------------------

Prior-predictive phenotypes test whether a fit recovers the truth when the
truth looks like the model's prior. To test it on phenotypes shaped like a
real library, fit the real data to a phenotype-generating distribution and
resample the simulated library from it. The ground truth stays known.

``tfs-build-empirical`` builds that distribution from a real growth table.
It configures a ``hill_geno`` model with linear growth, runs the pre-fit to
calibrate each condition's *k* and *m*, fits each genotype's growth curves
separately by maximum likelihood (Stage 1), then fits a multivariate normal
to those estimates with their estimation noise deconvolved (Stage 2):

.. code-block:: bash

    tfs-build-empirical growth.csv --binding_df binding.csv \
        --library_config library_config.yaml --seed 42 --out_prefix emp

It writes ``emp_phenotype_model.json``, the distribution, and
``emp_stage1_fits.csv``, the per-genotype fits, plus the configure and
pre-fit files under ``emp_configure_*`` and ``emp_prefit_*``. The pre-fit is
the slow step. Once a calibration exists, pass it with
``--growth_calibration_file`` (the pre-fit priors CSV, or a CSV with
``condition_rep``, ``growth_k`` and ``growth_m``) to skip configuring and
pre-fitting. ``--binding_df``, ``--library_config`` and ``--seed`` are then
not needed. ``--num_workers -1`` runs the Stage 1 fits in parallel.

The Stage 1 fits do not correct for congression, so the distribution carries
its small bias (about 0.05 ln units in the measured regime).

To simulate from the distribution, set two keys in an ordinary simulate
config:

.. code-block:: yaml

    phenotype_source: empirical
    empirical:
      phenotype_model: /abs/path/to/emp_phenotype_model.json

Every genotype then gets a resampled Hill curve and ``dk_geno``; wild type
keeps its own Stage 1 fit with ``dk_geno`` 0. ``theta_component`` is forced
to ``hill_geno``, activity is forced to 1, and ``theta_priors`` and
``theta_sim_priors`` are ignored. The ``growth`` block must hold the same
calibrated *k* and *m* the build used, or the simulated growth will not
match the distribution. ``binding_data`` still chooses which genotypes are
measured, and their curves come from the resampled phenotypes. The
:download:`empirical example config <../../examples/simulate-empirical/simulate_config.yaml>`
shows a complete file.

Prior-predictive datasets
-------------------------

``tfs-sample-prior`` draws synthetic datasets from a configured model's own
prior rather than from the simulator. It takes the YAML written by
``tfs-configure-model``:

.. code-block:: bash

    tfs-sample-prior tfs_configure_config.yaml --num_datasets 10 --seed 0 --out_prefix tfs_prior

Each dataset gives ``tfs_prior_NNN_growth.csv``, the configured growth table
with ``ln_cfu`` replaced by a prior draw, and
``tfs_prior_NNN_ground_truth.h5``, the latent values that generated it in
the format of a ``tfs-sample-posterior`` file. Observation noise from
``ln_cfu_std`` is added unless ``--no_noise`` is given. Fitting these
datasets checks the inference against the model it assumes, the basis of
simulation-based calibration. ``tfs-simulate`` is the stronger test, since
its physics is not the fit's model.

Simulation grids
----------------

``tfs-setup-sim-grid`` sets up a sweep of simulate settings, one directory
per combination, each with its own ``tfs_sim_config.yaml`` and a rendered
run script:

.. code-block:: bash

    tfs-setup-sim-grid simulate_grid.yaml --out_dir my_sim_grid

The :download:`example grid YAML <../../examples/simulate/simulate_grid.yaml>`
and its ``run.sh`` template are in ``examples/simulate/``. :doc:`grid`
describes the grid format, and ``tfs-summarize-calibration`` pools the
posterior calibration across a finished simulate-and-fit grid.
