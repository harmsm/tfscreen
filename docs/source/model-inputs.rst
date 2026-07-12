=================
Model Input Data
=================

``tfs-configure-model`` (see :doc:`analysis`, Step 1) accepts up to five
pieces of experimental input. Only ``binding_df`` is strictly required;
everything else is optional and adds additional constraints to the
model. This page describes what each input contributes to the fit, the
rough amount of data that is useful in practice, and how it is passed in.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Input
     - Required?
     - What it anchors
   * - ``binding_df``
     - Yes
     - Absolute scale/shape of *θ*, independent of growth
   * - ``growth_df``
     - No (binding-only model if omitted)
     - ln_cfu\ :sub:`0`, dk_geno, activity, and *θ* jointly, via growth rate
   * - ``base_growth_df``
     - No
     - The condition-growth baseline / dk_geno identifiability confound
   * - ``presplit_df``
     - No
     - ln_cfu\ :sub:`0` directly, from pre-split sequencing counts
   * - ``transformation_lambda``
     - No (required by some ``transformation_model`` choices)
     - The congression-correction prior

Binding Data (``binding_df``)
==============================

**Role**

Direct, low-throughput measurements of operator occupancy (*θ*) as a
function of titrant concentration, independent of any growth-rate
observation. Because it measures *θ* directly rather than through the
growth likelihood, binding data anchors the absolute scale and shape of
the occupancy curve — resolving degeneracies (e.g. between *θ* and
activity *A*) that growth data alone cannot separate. It is the only
required input to ``tfs-configure-model``: with ``growth_df`` omitted, a
binding-only model is configured that infers *θ* from ``binding_df``
alone.

**Scale**

Binding curves are comparatively expensive to collect, so useful fits
typically rely on **dozens** of measured genotype/titrant curves (e.g. a
handful of calibration genotypes each measured across ~5-10 titrant
concentrations) rather than a full library. Because growth data typically
outnumber binding rows by several orders of magnitude, the binding
log-likelihood is upweighted by ``binding_weight`` (auto-computed as
``N_growth_rows / N_binding_rows`` unless overridden) so each binding
observation carries a comparable gradient contribution to the average
growth observation.

**Format**

A CSV (or ``pd.DataFrame``) with one row per (genotype, titrant_name,
titrant_conc) measurement:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``genotype``
     - Genotype string (``wt``, ``M42I``, ``M42I/K84L``, ...)
   * - ``titrant_name``
     - Name of the titrant (matches ``growth_df`` naming when both are used)
   * - ``titrant_conc``
     - Titrant concentration (float)
   * - ``theta_obs``
     - Measured fractional occupancy, in [0, 1]
   * - ``theta_std``
     - Standard deviation / uncertainty of ``theta_obs``

Passed as the required positional argument:

.. code-block:: bash

    tfs-configure-model binding.csv --growth_df growth.csv

Growth Data (``growth_df``)
==============================

**Role**

High-throughput growth-rate observations (``ln_cfu`` over time, per
genotype/replicate/condition), produced by ``tfs-process-counts`` or
``tfs-simulate``. This is the primary data source: it is what makes the
approach *high-throughput*, and it jointly identifies ln_cfu\ :sub:`0`,
``dk_geno`` (pleiotropic growth effect), activity, and *θ* through the
growth likelihood
``ln_cfu = ln_cfu0 + (k_pre + dk_geno + m_pre·A·θ)·t_pre + (k_sel + dk_geno + m_sel·A·θ)·t_sel``.
Omitting it configures a binding-only model (see above).

**Scale**

Growth data is cheap per-genotype relative to binding data, since a
single sequencing run reports abundances for the whole library
simultaneously. Useful fits typically involve **hundreds of thousands**
of rows — the product of library size (thousands of genotypes),
replicates, conditions, titrant concentrations, and timepoints.

**Format**

A CSV with exactly the schema produced by ``tfs-process-counts`` /
``tfs-simulate``; see :doc:`process-raw` for the full column
specification (``genotype``, ``library``, ``replicate``,
``condition_pre``, ``condition_sel``, ``titrant_name``, ``titrant_conc``,
``t_pre``, ``t_sel``, ``ln_cfu``, ``ln_cfu_std``).

Passed via the optional flag:

.. code-block:: bash

    tfs-configure-model binding.csv --growth_df growth.csv

base_growth Data (``base_growth_df``)
========================================

**Role**

Direct, reference-condition growth-rate measurements for a small subset
of genotypes (``wt`` at minimum). The ``condition_growth`` components'
per-condition baseline (*k*) and the per-genotype pleiotropic effect
(``dk_geno``) are only jointly identified up to an additive constant —
``k += C, dk_geno -= C`` leaves the growth likelihood unchanged. Because
``dk_geno`` is fixed to 0 for ``wt``, a direct measurement of wt's growth
rate anchors the new ``k_ref`` scalar via
``rate_obs ~ Normal(k_ref + dk_geno, rate_std)``, resolving this
identifiability confound. See the **Per-condition growth priors** section
of ``CLAUDE.md`` for the full mechanism (this is the complementary,
insufficient-alone anchor; per-condition priors pinned during
``tfs-prefit-calibration`` are the primary fix).

**Scale**

Small. ``wt`` is required after filtering against ``growth_df``; beyond
that, ``base_growth_df`` may cover anywhere from just ``wt`` up to a
handful of additional genotypes. Multiple rows for the same genotype are
combined via inverse-variance weighting.

**Format**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``genotype``
     - Genotype string; rows not present in ``growth_df`` are dropped
   * - ``rate``
     - Measured reference-condition growth rate
   * - ``rate_std``
     - Standard deviation of ``rate``. **Must be strictly positive** — a
       value of 0 (e.g. from an unset ``noise`` in simulated data) causes
       a division-by-zero that surfaces as a NaN-prior crash the first
       time the model is traced (at ``tfs-prefit-calibration`` or
       ``tfs-fit-model``, not at ``tfs-configure-model`` time).

Passed via the optional flag:

.. code-block:: bash

    tfs-configure-model binding.csv --growth_df growth.csv \
        --base_growth_df base_growth.csv

Pre-split Data (``presplit_df``)
===================================

**Role**

Sequencing-derived abundance measurements taken *before* the library is
divided into separate selection conditions (t = -t_pre). Genotypes it
covers get a direct constraint on their initial population
(ln_cfu\ :sub:`0`) instead of relying solely on the extrapolation implicit
in the growth-rate fit.

**Scale**

Ideally the whole library, since it is collected from a single pooled
sample at one timepoint (cheap relative to a full growth time-course).
Genotypes present in ``growth_df`` but absent from ``presplit_df`` are
not an error — they are kept and their ln_cfu\ :sub:`0` is masked out of
this constraint.

**Format**

Produced by ``tfs-process-presplit``; see :doc:`process-raw` for the full
column specification (``library``, ``replicate``, ``condition_pre``,
``genotype``, ``ln_cfu``, ``ln_cfu_std``).

Passed via the optional flag:

.. code-block:: bash

    tfs-configure-model binding.csv --growth_df growth.csv \
        --presplit_df presplit.csv

transformation_lambda
========================

**Role**

Not a data file — a single experimentally measured ``(mean, std)`` pair
describing plasmid congression (multiple plasmids entering one cell
during transformation), in linear space (e.g. ``(0.36, 0.05)``). It is
used to moment-match a LogNormal prior for the transformation model's
lambda parameter, replacing the manual step of hand-editing the priors
and guesses CSVs with rescaled log-space values. See
:ref:`model-components`'s Transformation Correction section for what the
``empirical``/``logit_norm``/``single`` models each do with it.

**Scale**

A single measurement (one mean, one uncertainty) from an independent
congression-rate experiment — not a per-genotype or per-row dataset.

**Format**

Passed directly as two floats via ``--transformation_lambda``, required
when ``--transformation_model`` is ``empirical`` or ``logit_norm``, and
forbidden (must be omitted) when it is ``single``:

.. code-block:: bash

    tfs-configure-model binding.csv --growth_df growth.csv \
        --transformation_model empirical \
        --transformation_lambda 0.36 0.05
