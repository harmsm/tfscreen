===================
Processing raw data
===================

A screen grows bacteria carrying a plasmid library of transcription factor
variants under one or more selection conditions. Each time-point is its own
tube. Every tube is deep-sequenced, and its total cell count is estimated,
usually from OD600. Together these give the absolute abundance of every
genotype in every tube.

The raw-data half of the pipeline has four steps:

1. ``tfs-process-fastq`` calls the genotype of every read pair in one
   tube's FASTQ files and writes a count file for that tube.
2. ``tfs-calibrate-od600`` fits the lab's OD600-to-CFU calibration. This is
   done once per reader, plate type, volume and strain, not once per
   experiment.
3. You write a tube table that names every sequenced tube and gives its
   design: library, replicate, conditions, titrant and times.
4. ``tfs-process-counts`` combines the count files, the tube table and each
   tube's OD600 into the growth table that ``tfs-configure-model`` reads.
   With ``--presplit`` it writes the presplit table instead.

Every flag and default is listed in :doc:`cli`. The model's side of these
tables is described in :doc:`model-inputs`.

One tube per time-point
-----------------------

The protocol:

1. Transform the library.
2. Grow the transformed culture to a set OD.
3. Take the ``presplit`` sample and read its OD600.
4. Dilute the culture by the same factor into every tube, one per
   condition, titrant concentration and planned time-point. Every tube of a
   replicate therefore starts from the same known total: the presplit
   OD600 estimate divided by the dilution factor. For a titration this is a
   grid. Rows are time-points and columns are IPTG concentrations.
5. At each time-point, pull every tube in that row. Read each pulled tube's
   OD600, then spin the tubes down and freeze them at once. The spread in
   stop time within a row is seconds, which is negligible on the time scale
   of growth.
6. After the experiment, process all samples together. Thaw and extract DNA
   in batches of about 20, so three batches for 60 samples. Then run one
   PCR block for every sample, then sequence. Record each sample's
   extraction batch. For the current dataset it was not recorded; batches
   generally, but not reliably, followed processing order.

Pre-growth and selection both happen in the tubes, after the split. OD600
is read on three bioreplicates of the experiment and two are sequenced.

Each tube's total CFU comes from its OD600 through an OD-to-CFU
calibration. The calibration depends on the plate reader, plate, volume and
strain, so each lab makes its own with ``tfs-calibrate-od600``. A
calibration gives CFU per mL. The pipeline works in cells in the whole
tube, so ``tfs-process-counts`` multiplies by the tube's culture volume:
``ln_cfu`` is ``ln(cfu/mL x tube volume)``. Under the strongest selection a
tube may never reach the reader's detection threshold. Its OD600 then bounds
its total from above rather than measuring it.

Why tubes and not aliquots: pulling aliquots from a single tube over time
meant stopping the shaker, pipetting and returning the tube across many
IPTG conditions. That disturbed the growth trajectories more than the
tube-to-tube differences of this design do.

Consequences for the model:

- A sequenced sample *is* a tube, and the time-points of one condition come
  from different tubes. Anything that affects a whole tube is independent
  from one time-point to the next, not carried along a trajectory. That
  includes its growth environment over ``t_pre + t_sel``, its OD600 reading
  and its PCR. The model's per-tube offset (``sample_offset``) is indexed
  the same way, one offset per replicate x time x condition x titrant. The
  default, ``level``, is a constant shift of the tube's ``ln_cfu`` shared by
  every genotype in it, with a learned SD. It absorbs the tube's composition
  offset and any error in its supplied total. ``normal`` is the older
  alternative, a growth-rate offset scaled by ``t_pre + t_sel``, and
  ``zero`` adds no offset.
- A growth difference shared by every cell in a tube changes every
  genotype's abundance by the same factor, so it does not change the
  genotype *frequencies* the reads measure. It enters ``ln_cfu`` only
  through the tube's total CFU. This is why a total smoothed across
  bioreplicates gives much steadier ``ln_cfu`` trajectories than each tube's
  own OD600 estimate. Earlier analyses used the pooled mean per condition
  and time-point.
- Everything before the split, such as transformation and outgrowth, is
  shared by all tubes of a replicate. It sets each genotype's starting
  abundance, ``ln_cfu0``, which the ``presplit`` sample measures directly.
- No step is shared by the tubes of one row and not by the others. Rows
  are stopped within seconds and processed with everything else. The one
  batched step is DNA extraction. Extraction yield does not change a
  sample's genotype frequencies, so it should not shift ``ln_cfu``. If it
  matters at all, it is through the amount of template going into PCR,
  which changes how noisy the counts are.

The library config
------------------

One YAML file describes the screened library. Pass the same file to
``tfs-process-fastq`` and to ``tfs-configure-model --library_config``.
Nothing checks that the two saw the same file, and a genetics mismatch
shows up only later, when ``tfs-configure-model`` finds genotypes the
library does not contain. An example is
:download:`examples/process_raw/library_config.yaml
<../../examples/process_raw/library_config.yaml>`.

``reading_frame``
    Reading frame of the amplicon: 0, 1 or 2.
``first_amplicon_residue``
    Residue number of the first in-frame codon. It sets the numbering of
    every genotype name, so a wrong value mismatches every mutant.
``wt_seq``
    The wildtype DNA sequence of the amplicon.
``degen_sites``
    Degenerate codon pattern, the same length as ``wt_seq``: ``.`` for a
    wildtype base, or a code such as ``nnt`` or ``nnk`` at library codons.
``tiles``
    Contiguous blocks of library sites cloned together. ``.`` is wildtype
    and each other character names a tile. Blocks must be contiguous.
``expected_5p``, ``expected_3p``
    Flanking sequences immediately upstream and downstream of the amplicon.
    Read only by ``tfs-process-fastq``.
``tile_combos``
    Which variants the library holds: ``single-x`` is every single mutant in
    tile ``x`` and ``double-x-y`` every pairwise combination between tiles
    ``x`` and ``y``.
``spiked_seqs``
    DNA sequences of monoclonal controls spiked into the pool. Their
    genotypes may also occur in the bulk sub-libraries.
``library_mixture``
    Relative amount of each sub-library in the pool, keyed by the
    ``tile_combos`` entries plus ``spiked``. Only ratios matter. Read only by
    ``tfs-configure-model``, which uses it to compute each genotype's share
    of the pool and how much of it comes from the bulk libraries. Nothing in
    the data can check these numbers, so give the best estimate of what
    went into the pool, not just the design.

Any other key is ignored, so a ``tfs-simulate`` config can serve as the
library config of a simulated experiment.

Counting reads: tfs-process-fastq
---------------------------------

``tfs-process-fastq`` reads one tube's paired FASTQ files, trims each read
to the amplicon between its flanks, masks low-quality bases and calls the
protein genotype against every sequence the library config allows. The
library config comes first:

.. code-block:: bash

    tfs-process-fastq library_config.yaml \
        kanR_kanRpkan_0mM_t0_R1.fastq.gz kanR_kanRpkan_0mM_t0_R2.fastq.gz \
        --out_dir counts

It writes two files into ``--out_dir`` (default ``counts``), named after the
read-1 file:

- ``counts_<f1 name>.csv``, with columns ``genotype`` and ``counts``. It has
  one row for every genotype in the library, zeros included, plus one
  ``__unknown__`` row. ``__unknown__`` counts read pairs that oriented and
  trimmed cleanly but matched no library genotype, or matched more than
  one. It is not a genotype. ``tfs-process-counts`` adds it to the tube's
  read depth, the denominator of every frequency, and then drops it.
- ``stats_<f1 name>.csv``, with columns ``success``, ``result``, ``counts``
  and ``fraction``: how many read pairs ended in each outcome, such as a
  missing flank, a read too short after quality trimming, or a disagreement
  between the forward and reverse reads. Check it when a tube has few
  reads or a large ``__unknown__`` share.

``tfs-process-counts`` finds a tube's count file by globbing
``counts*<sample>*.csv``, so name the FASTQ files after the tubes. Each
tube's name must match exactly one file. A tube named ``t1`` next to one
named ``t10`` matches two files and is an error.

The main tuning flags are ``--phred_cutoff`` (default 10),
``--min_read_length`` (default 50), ``--allowed_num_flank_diffs`` (default 1)
and ``--allowed_diff_from_expected`` (default 2). ``--num_workers`` defaults to
-1, which uses every CPU but one, and ``--chunk_size`` (default 10000) is the
number of reads sent to a worker at a time.

The tube table
--------------

The tube table has one row per sequenced tube. It is a CSV, TSV or Excel
file.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Column
     - Meaning
   * - ``sample``
     - Unique tube name, matched to its count file.
   * - ``library``
     - The transformed library the tube came from. Optional; ``default``
       when absent. kanR and pheS libraries are transformed and grown up
       separately, so they are separate libraries.
   * - ``replicate``
     - Biological replicate.
   * - ``condition_pre``
     - Growth condition before selection, such as ``kanR-kan``.
   * - ``t_pre``
     - Minutes of pre-selection growth.
   * - ``condition_sel``
     - Selection condition, such as ``kanR+kan``.
   * - ``t_sel``
     - Minutes of selection growth.
   * - ``titrant_name``
     - The titrant, such as ``iptg``.
   * - ``titrant_conc``
     - Titrant concentration; 0 for none.
   * - ``od600``
     - The tube's OD600 reading. Optional; it can come from a separate
       ``--od600_file`` instead.

``tfs-process-counts`` itself needs only ``sample``, the tube totals and,
with ``--presplit``, ``replicate`` and ``condition_pre``. Every other column
is carried through to the output, and ``tfs-configure-model`` needs the
design columns above. A selection condition is one that never appears as a
``condition_pre`` within its library; the control arm repeats the
pre-selection condition as its ``condition_sel``.

Lab-specific cleanup belongs in the lab's own script, before the tube table
is written: samples swapped at the bench, tubes resequenced and pooled,
failed or empty tubes. tfscreen takes the tube table as the record of what
each tube is and does not try to repair it.

OD600 calibration
-----------------

``tfs-calibrate-od600`` fits the calibration from two small experiments, both
done with the same handling as production samples: same reader, plate type,
volume and strain.

**Repeated readings of a dilution series.** Read each dilution several
times, repeating the whole step each time: swirl the culture, pipette into
the plate, read. The file has one row per reading, with columns
``dilution`` (relative to the undiluted culture) and ``od600``. The largest
relative SD across dilutions is the reading noise. The detection threshold
is midway between the mean readings of the two most dilute samples, where
the series has flattened onto the reader's floor. ``--detection_threshold``
overrides it.

**Plate counts of cultures whose OD600 was read.** The file has one row per
plate, with columns ``od600``, ``colonies``, ``dilution`` (the total dilution
factor before plating), ``plated_volume_mL``, ``num_dilutions`` and
``plating_steps``. CFU/mL is ``colonies * dilution / plated_volume_mL``. Its
relative variance is ``1 / colonies`` for counting plus
``pipette_rel_error^2`` for each dilution and plating step
(``--pipette_rel_error``, default 0.02).

A polynomial of CFU/mL in OD600 is fit by weighted least squares
(``--degree``, default 2):

.. code-block:: bash

    cd examples/od600
    python make_example_data.py
    tfs-calibrate-od600 replicates.csv plate_counts.csv --out_prefix od600_calibration

``od600_calibration.yaml`` holds the coefficients and their full covariance,
the reading noise, the detection threshold and the calibrated OD600 range.
This is the file ``tfs-process-counts`` reads. ``od600_calibration.pdf``
shows the dilution series, the fit and the calibrated CFU's relative error.
The two CSVs hold the per-dilution noise and the per-plate fit with its
standardized residuals.

The covariance matters. The curve's error is one error shared by every tube
calibrated with it. It does not average out across tubes and must not be
treated as independent per-tube noise, so the pipeline keeps it apart from
the reading noise: ``tfscreen.process_raw.od600.cfu_per_mL_error_components``
returns the two parts separately, and ``tfs-process-counts`` writes them as
separate columns.

The files in ``examples/od600/`` are synthetic and show the formats. They
are not any instrument's calibration; do not use them for real data.

Counts to the growth table: tfs-process-counts
----------------------------------------------

``tfs-process-counts`` takes the tube table and the directory of count
files:

.. code-block:: bash

    tfs-process-counts tube_table.csv counts \
        --od600_file od600.csv \
        --od600_calibration_file od600_calibration.yaml \
        --tube_volume_mL 5 \
        --out_prefix growth

It writes ``growth.csv`` (default ``tfs_growth.csv``) and prints which count
file it matched to each tube. ``--no_verbose`` turns that listing off.

**Tube totals.** Every tube needs its total cells, given one of two ways.
The first is OD600. Give one reading per tube, either as ``--od600_file``
(columns ``sample`` and ``od600``) or as an ``od600`` column in the tube
table, together with ``--od600_calibration_file`` and ``--tube_volume_mL``.
The calibration gives CFU per mL and the volume turns that into cells in the
tube. These columns are then added:

- ``sample_cfu``: cells in the tube.
- ``sample_cfu_std``: its SD, with the curve and reading errors combined.
- ``sample_cfu_curve_std``: the calibration curve's part, shared by every
  tube read through the same calibration.
- ``sample_cfu_reading_std``: the reading's own part, independent per tube.
- ``od600_in_calibrated_range``: False for a reading outside the OD600 range
  the plate counts covered, where the curve is extrapolated.

A reading below the calibration's detection threshold is an error, since
its total is only an upper bound. Drop those tubes from the tube table.

The second way is to supply the totals yourself, as ``sample_cfu`` with
``sample_cfu_std`` (or ``sample_cfu_var``) in the tube table, or the log
forms ``sample_ln_cfu`` with ``sample_ln_cfu_std`` (or
``sample_ln_cfu_var``). Totals must be cells in the whole tube. Use one
route for the whole table. A tube table that supplies totals and OD600
together is an error.

**Several libraries.** All libraries go through one call. Genotypes are
filtered and filled in within each library: a genotype with fewer than
``--min_genotype_obs`` reads (default 10) summed over a library's tubes is
dropped from that library, and a genotype kept in a library gets a row with
zero counts in every tube of that library where it was not seen.

**Output.** One row per genotype and tube. It carries the tube table's
columns and the tube totals, plus:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Column
     - Meaning
   * - ``genotype``
     - Genotype name: ``wt``, ``M42I``, ``M42I/K84L``.
   * - ``counts``
     - Reads of the genotype in the tube, without pseudocount.
   * - ``sample_reads``
     - The tube's total reads, ``__unknown__`` included, without
       pseudocounts. The count likelihood uses it as the tube's depth.
   * - ``adjusted_counts``
     - ``counts + --pseudocount`` (default 1).
   * - ``frequency``
     - ``adjusted_counts`` over the tube's read depth plus one pseudocount
       per genotype.
   * - ``sample_ln_cfu``, ``sample_ln_cfu_std``
     - The tube's total in log form.
   * - ``ln_cfu``, ``ln_cfu_var``
     - ``ln(frequency) + sample_ln_cfu``, the genotype's cells in the tube,
       and its variance: binomial sampling of the frequency plus
       ``sample_ln_cfu_std^2``.
   * - ``cfu``, ``cfu_var``
     - The same in linear form.

The model takes this file as it is (``tfs-configure-model --growth_df``). By
default it observes ``counts`` against ``sample_reads`` and ``sample_ln_cfu``
rather than ``ln_cfu``; see :doc:`model-inputs`.

**Presplit.** With ``--presplit``, the tube table lists the presplit tubes,
one per library, replicate and ``condition_pre``, and must have
``replicate`` and ``condition_pre`` columns. The output, ``tfs_presplit.csv``
by default, has columns ``library``, ``replicate``, ``condition_pre``,
``genotype``, ``ln_cfu`` and ``ln_cfu_std``. Pass it to
``tfs-configure-model --presplit_df``.

A worked example
----------------

``examples/process_raw/`` holds a small synthetic experiment: two libraries
(kanR and pheS), each with a selection and a control condition, two IPTG
concentrations and three time-points, for 24 tubes. ``make_example_data.py``
writes ``tube_table.csv``, ``od600.csv`` and ``counts/``, the count files in
the format ``tfs-process-fastq`` writes. It uses the calibration in
``examples/od600/``:

.. code-block:: bash

    cd examples/process_raw
    python make_example_data.py
    tfs-process-counts tube_table.csv counts \
        --od600_file od600.csv \
        --od600_calibration_file ../od600/od600_calibration.yaml \
        --tube_volume_mL 5 \
        --out_prefix growth

``growth.csv`` is a growth table ready for the next step:

.. code-block:: bash

    tfs-configure-model --growth_df growth.csv --library_config library_config.yaml

The numbers are made up. They show the formats, not a realistic screen.
