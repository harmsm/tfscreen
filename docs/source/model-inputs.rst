============
Model inputs
============

``tfs-configure-model`` builds a model from data files and a few
experimental numbers, and writes the configuration ``tfs-fit-model``
reads. This page describes what each input contributes to the fit and how
to pass it. The commands that produce the data files are in
:doc:`process-raw`, every flag is in :doc:`cli`, and the model components
are in :doc:`model`.

At least one of ``--growth_df`` and ``--binding_df`` is required. Both are
flags. With both, the joint model is configured; with growth only, a
growth-only model; with binding only, a binding-only model.

.. list-table::
   :header-rows: 1
   :widths: 26 24 50

   * - Flag
     - When
     - What it anchors
   * - ``--growth_df``
     - The screen itself
     - ``ln_cfu0``, ``dk_geno``, activity and *θ* jointly, through growth
   * - ``--library_config``
     - Required with growth data
     - Spiked genotypes, purity and pool shares for the congression model
   * - ``--binding_df``
     - Optional
     - The absolute scale and shape of *θ*
   * - ``--presplit_df``
     - Optional
     - ``ln_cfu0`` directly, from the presplit tube
   * - ``--base_growth_df``
     - Optional
     - The growth baseline against ``dk_geno``
   * - ``--transformation_lambda``
     - Required by ``mixture``
     - The congression rate
   * - ``--theta_gauge_conc``
     - ``hill_relative`` only
     - Where wt's relative X is 1 and 0
   * - ``--thermo_data``
     - Thermodynamic *θ* models
     - Structural features of each mutation
   * - ``--set_priors``, ``--growth_priors``, ``--growth_priors_wt_rates``
     - Optional
     - Prior values, without editing the priors CSV

Growth data
-----------

``--growth_df`` is the table ``tfs-process-counts`` writes (or
``tfs-simulate`` for a simulated experiment). It is the high-throughput
part of the experiment: one sequencing run reports every genotype in a
tube, so a screen has hundreds of thousands of rows. Growth identifies each
genotype's starting abundance, its pleiotropic growth effect ``dk_geno``, its
activity and its occupancy *θ* through

``ln_cfu = ln_cfu0 + (k_pre + dk_geno + m_pre·A·θ)·t_pre + (k_sel + dk_geno + m_sel·A·θ)·t_sel``

The model reads these columns:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Column
     - Meaning
   * - ``genotype``
     - Genotype name.
   * - ``library``
     - The transformed library. Two libraries may share condition names.
   * - ``replicate``
     - Biological replicate; 1 when absent.
   * - ``condition_pre``, ``condition_sel``
     - Pre-selection and selection conditions.
   * - ``t_pre``, ``t_sel``
     - Minutes of pre-selection and selection growth.
   * - ``titrant_name``, ``titrant_conc``
     - Titrant and its concentration.
   * - ``ln_cfu``, ``ln_cfu_var``
     - The genotype's cells in the tube, in log form, and its variance.
       ``ln_cfu_std`` may be given instead of the variance, or ``cfu`` with
       ``cfu_std`` or ``cfu_var``.
   * - ``counts``
     - Reads of the genotype in the tube. Count likelihood only.
   * - ``sample_reads``
     - The tube's total reads, ``__unknown__`` included. Count likelihood
       only. Older files without it can supply ``adjusted_counts`` and
       ``frequency``, from which it is derived about 1% high, by the same
       factor for every genotype in a tube.
   * - ``sample_ln_cfu``
     - The tube's total cells, in log form. Count likelihood only.
       ``sample_cfu`` with ``sample_cfu_std`` or ``sample_cfu_var`` works
       too.

**Likelihood.** The default, ``--growth_likelihood counts``, observes the
reads directly. Each genotype's count in a tube is negative binomial with
mean equal to the tube's reads times the genotype's predicted frequency,
and a learned dispersion whose variance has a part proportional to the mean
and a quadratic part. There is no pseudocount, so a genotype with zero
reads is an observation rather than a floor. Real counts vary several times
more than Poisson, and the dispersion describes that. The count likelihood
needs ``--growth_noise_model zero``, the default. It pairs with the default
``--sample_offset_model level``: one ``ln_cfu`` offset per tube, shared by
every genotype in it, with a learned SD. A tube's total enters the count
likelihood as given, so the level offset is what absorbs an error in it.
The SD columns of the totals are not used. ``--set_priors sigma_fixed=0.17``
holds that SD instead of learning it. On simulations the count likelihood
cut the RMSE of *θ* by 35 to 60% against the alternative.

The alternative, ``--growth_likelihood lncfu``, observes each row's
``ln_cfu`` with a Student-t likelihood whose scale is the row's
``ln_cfu`` SD. It needs none of the count columns.

Binding data
------------

``--binding_df`` holds direct, low-throughput measurements of *θ* against
titrant concentration. Because they measure *θ* without going through
growth, they fix its absolute scale and shape and separate effects growth
alone cannot, such as *θ* against activity *A*. Useful fits rely on dozens
of curves, a handful of genotypes each measured at 5 to 10 concentrations.
Growth rows outnumber binding rows by orders of magnitude, so the binding
log-likelihood is weighted by ``--binding_weight``. By default that is the
number of growth rows over the number of binding rows.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Column
     - Meaning
   * - ``genotype``
     - Genotype name.
   * - ``titrant_name``
     - Titrant; it must match the growth table's naming.
   * - ``titrant_conc``
     - Titrant concentration.
   * - ``theta_obs``
     - Measured fractional occupancy.
   * - ``theta_std``
     - SD of ``theta_obs``.

In a joint model, binding rows whose genotype and titrant do not appear in
the growth table are dropped with a message.

.. code-block:: bash

    tfs-configure-model --binding_df binding.csv --growth_df growth.csv \
        --library_config library_config.yaml

Without binding data, growth fixes *θ* only up to an affine map: the growth
slope ``m`` and baseline ``k`` absorb any rescaling of *θ*, so its absolute
scale rests on the priors. A growth-only model has no binding likelihood,
so ``--theta_binding_noise_model`` stays ``zero`` and ``--binding_weight``
stays unset. ``tfs-prefit-calibration`` calibrates the growth link on the
binding genotypes, so it refuses a growth-only model. The growth-only route
is ``--theta_model hill_relative`` with its growth priors set from wt
monoculture rates; see `Prior inputs`_ and :doc:`fitting`.

Library description
-------------------

``--library_config`` is the library YAML, the same file given to
``tfs-process-fastq``; its keys are described in :doc:`process-raw`. It is
required with growth data and refused without it.

``tfs-configure-model`` resolves it into ``{out_prefix}_library.csv``, one row
per genotype with ``is_wt``, ``in_spiked_origin`` (encoded by a spiked
sequence), ``pool_fraction`` (its expected share of the pool) and
``bulk_fraction`` (the share of its cells that come from the bulk,
congression-prone sub-libraries rather than from a monoclonal spike).
``tfs-fit-model`` reads this snapshot, never the YAML, so the CSV is the
place to override the design's numbers. ``in_spiked_origin`` sets which
genotypes get the spiked ``ln_cfu0`` prior. ``bulk_fraction`` and
``pool_fraction`` feed the ``mixture`` congression model.

Two checks run against the data. Every genotype in the growth, presplit and
base-growth tables must be in the library; a mismatch usually means the
counts were called with a different config, and a residue-numbering
difference mismatches nearly every mutant. Every spiked genotype must have
growth data; a missing spike also usually means a different config.
``--allow_missing_spikes`` turns the second check into a warning, for a
spike that really dropped out.

``library_mixture`` cannot be checked: no data file ever saw it. It is
recorded as given in the configuration's ``library`` block, and that record
is its only defense. Give the best estimate of what went into the pool.

Presplit data
-------------

``--presplit_df`` holds the abundances measured in the presplit tube, before
the culture was divided into tubes (t = -t_pre). It constrains each covered
genotype's starting abundance ``ln_cfu0`` directly instead of leaving it to
the extrapolation of the growth fit. It is cheap, one tube per library,
replicate and pre-selection condition, so it can cover the whole library.

``tfs-process-counts --presplit`` writes it, with columns ``library``,
``replicate``, ``condition_pre``, ``genotype``, ``ln_cfu`` and ``ln_cfu_std``.
Rows for genotypes not in the growth table are dropped. Genotypes in the
growth table but not in the presplit table are kept and simply have no
presplit constraint.

.. code-block:: bash

    tfs-configure-model --growth_df growth.csv --library_config library_config.yaml \
        --presplit_df presplit.csv

Base growth data
----------------

``--base_growth_df`` holds direct growth-rate measurements in a reference
condition for a few genotypes, wt at minimum. The condition baselines ``k``
and the genotype effects ``dk_geno`` are identified only up to a shared
constant: ``k + C`` with ``dk_geno - C`` leaves the growth likelihood
unchanged. wt's ``dk_geno`` is fixed at 0, so a measurement of wt's rate
pins a reference rate ``k_ref`` through
``rate ~ Normal(k_ref + dk_geno, rate_std)``. This anchor is complementary;
the per-condition priors set by ``tfs-prefit-calibration``, or by
``--growth_priors``, are the main fix.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Column
     - Meaning
   * - ``genotype``
     - Genotype name. Rows not in the growth table are dropped, and wt must
       remain.
   * - ``rate``
     - Measured growth rate in the reference condition, per minute.
   * - ``rate_std``
     - SD of ``rate``. It must be greater than 0.

Several rows for one genotype are combined by inverse-variance weighting.

.. code-block:: bash

    tfs-configure-model --growth_df growth.csv --library_config library_config.yaml \
        --base_growth_df base_growth.csv

Congression rate
----------------

``--transformation_lambda MEAN STD`` is not a data file. It is a measured
congression rate, the tendency of a transformed cell to take up more than
one plasmid, as a mean and SD in linear space. It sets a LogNormal prior on
the ``mixture`` transformation's lambda. It is required when
``--transformation_model`` is ``mixture`` and refused when it is ``single``,
the default.

Lambda is the rate of a zero-truncated Poisson. A transformant, a cell that
took up at least one plasmid and survived selection, carries ``M`` plasmids
with ``P(M = m) = Poisson(m; lambda) / (1 - exp(-lambda))``. A measured mean
number of distinct plasmids per transformant,
``E[M] = lambda / (1 - exp(-lambda))``, has to be converted to lambda first.
This is the same lambda as the simulator's
``transformation_poisson_lambda``.

.. code-block:: bash

    tfs-configure-model --binding_df binding.csv --growth_df growth.csv \
        --library_config library_config.yaml \
        --transformation_model mixture --transformation_lambda 0.36 0.05

Relative-X gauge
----------------

``--theta_gauge_conc C_LO C_HI`` applies only to ``--theta_model
hill_relative``, which fits growth alone on a wt-relative scale X: wt's X is
1 at ``C_LO`` and 0 at ``C_HI``. The default is the lowest and highest
titrant concentration in the growth table, and the resolved values are
written to the configuration. Other theta models refuse the flag.

Structural data
---------------

``--thermo_data`` is required by the thermodynamic partition-function theta
models and ignored by the others. For ``PnnC`` models it is the HDF5 file
written by ``scripts/generate_struct_ensemble.py``; see
:doc:`ligandmpnn-features`. For ``PddG`` models it is a CSV with a ``mut``
column and one column of prior mean ΔΔG per structure.

Prior inputs
------------

``tfs-configure-model`` writes every prior at its component default into
``{out_prefix}_priors.csv``. Three flags change it at configure time, so the
file is never edited by hand. Growth priors are applied first, then
``--set_priors``.

**--set_priors name=value ...** sets scalar priors. A name is a full row
name of the priors CSV, such as
``growth.sample_offset.sigma_fixed``, or a dotted suffix that matches
exactly one row, such as ``sigma_fixed``. An unknown or ambiguous name is an
error, and per-condition priors are set with ``--growth_priors`` instead.
Common uses: ``sigma_fixed=0.17`` holds the SD of the level tube offsets;
``theta_log_hill_n_hyper_scale_fixed=0.5`` holds the population SD of
``log(hill_n)``; for ``hill_relative``, ``theta_X_low_hyper_scale_fixed``,
``theta_X_delta_hyper_scale_fixed`` and
``theta_log_hill_K_hyper_scale_fixed`` hold the population SDs of its other
parameters. A value of 0 leaves the SD learned.

**--growth_priors table.csv** sets per-condition priors for ``linear``
condition growth. The table has a ``condition_rep`` column, the condition
name as it appears in ``condition_pre`` or ``condition_sel``, and any of
``k_loc``, ``k_scale``, ``m_loc`` and ``m_scale``, in rates per minute. With
a ``replicate`` column each row applies to one replicate; without it, to
every replicate. Conditions and values the table leaves out keep the
defaults, and a condition the model does not have is an error.
``m_scale`` sets the scale for selective and non-selective conditions
alike. Use it where ``tfs-prefit-calibration`` does not run, such as a
growth-only model; the pre-fit overwrites ``k_loc`` and ``m_loc``.

.. code-block:: text

    condition_rep,k_loc,k_scale,m_loc,m_scale
    kanR+kan,0.015,0.002,-0.010,0.003
    kanR-kan,0.020,0.002,,

**--growth_priors_wt_rates rates.csv** derives those priors for
``hill_relative`` from wt monoculture growth rates. The columns are
``condition_sel``, ``titrant_conc``, ``rate_mean``, ``rate_sd`` and
``num_replicates``, with one row at each gauge concentration for every
listed condition. In the X gauge wt grows at ``k`` at ``C_HI`` and at
``k + m`` at ``C_LO``, so ``k`` is the rate at ``C_HI`` and ``m`` is the rate at
``C_LO`` minus ``k``. Each SD is the standard error of the replicate mean,
floored at ``--growth_priors_sd_floor`` (default 0.002 per minute). List only
the conditions where the monoculture stands for the library's wt. These
values override a ``--growth_priors`` table for the conditions they list,
and they cannot be combined with a per-replicate table.

.. code-block:: bash

    tfs-configure-model --growth_df growth.csv --library_config library_config.yaml \
        --theta_model hill_relative \
        --growth_priors_wt_rates wt_rates.csv \
        --set_priors sigma_fixed=0.17 theta_log_hill_n_hyper_scale_fixed=0.5
