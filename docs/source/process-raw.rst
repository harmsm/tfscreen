===================
Processing Raw Data
===================

A typical TF screen involves growing bacteria transformed with a plasmid-encoded
library of TF variants under one or more selection conditions (e.g. antibiotic resistance
driven by a TF-regulated promoter). Each time-point is a tube of its own
(see "One tube per time-point" below). Its total colony-forming units (CFU)
are estimated, typically from OD600 through a lab-specific calibration, and
it is deep-sequenced, so that the absolute abundance of every genotype can be
determined. The raw
inputs to the pipeline are paired-end FASTQ files (one pair per sample) and
a sample metadata table (``sample_df``) that links each sequenced tube to its
biological context and its measured CFU count. The pipeline converts read
counts into per-genotype log-CFU estimates (``ln_cfu``) that feed the
hierarchical Bayesian growth model.

One tube per time-point
-----------------------

The protocol:

1. Transform the library.
2. Grow the transformed culture to a set OD.
3. Take the ``presplit`` sample and read its OD600.
4. Dilute the culture by the same factor into every tube, one per
   condition, titrant concentration and planned time-point. Every tube of a
   replicate therefore starts from the same, known total (the presplit
   OD600 estimate divided by the dilution factor). For a titration this is a grid: rows are
   time-points, columns are IPTG concentrations.
5. At each time-point, pull every tube in that row. Each pulled tube's
   OD600 is read, and the tubes are spun down and frozen immediately; the spread in stop time within
   a row is seconds, negligible on the time scale of growth.
6. After the experiment, all samples are processed together: thaw and DNA
   extraction in batches of about 20 (three batches for 60 samples), then
   one PCR block for every sample, then sequencing. Record each sample's
   extraction batch; for the current dataset it was not recorded (batches
   generally, but not reliably, followed processing order).

Pre-growth and selection both happen in the tubes, after the split. OD600
is read on three bioreplicates of the experiment; two are sequenced.

Total CFU per tube comes from OD600 through an OD-to-CFU calibration. The
calibration depends on the plate reader, plate, volume and strain, so each
lab makes its own with ``tfs-calibrate-od600`` (see "OD600 calibration"
below); the pipeline takes the resulting CFU
estimate per tube (``sample_cfu`` and its uncertainty, or the log-space
equivalents). A calibration gives cfu/mL; the pipeline's ``ln_cfu`` is cells
in the 5 mL tube, ``ln(cfu/mL * 5 mL)``. Under the strongest selection a
tube may never reach the reader's detection threshold, so its OD600 bounds
its total from above rather than measuring it.

Why tubes and not aliquots: pulling aliquots from a single tube over time
meant stopping the shaker, pipetting and returning the tube across many
IPTG conditions, and that disturbed the growth trajectories more than the
tube-to-tube differences of this design do.

Consequences for the model:

- A sequenced sample *is* a tube, and the time-points of one condition come
  from different tubes. Anything that affects a whole tube (its growth
  environment over ``t_pre + t_sel``, its OD600 reading, its PCR) is independent
  from one time-point to the next, not carried along a trajectory. The
  model's per-tube offset (``sample_offset``) is indexed accordingly: one
  offset per replicate x time x condition x titrant, scaled by
  ``t_pre + t_sel``.
- A growth difference shared by every cell in a tube changes every
  genotype's abundance by the same factor, so it does not change the
  genotype *frequencies* the reads measure. It enters ``ln_cfu`` only
  through the tube's total CFU. This is why a total smoothed across
  bioreplicates (earlier analyses used the pooled mean per condition and
  time-point) gives much steadier ``ln_cfu`` trajectories than each tube's
  own OD600 estimate.
- Everything before the split (transformation, outgrowth) is shared by all
  tubes of a replicate. It sets each genotype's starting abundance,
  ``ln_cfu0``, which the ``presplit`` sample measures directly.
- No step is shared by the tubes of one row and not by the others: rows
  are stopped within seconds and processed with everything else. The one
  batched step is DNA extraction. Extraction yield does not change a
  sample's genotype frequencies, so it should not shift ``ln_cfu``; if it
  matters at all, it is through the amount of template going into PCR,
  which changes how noisy the counts are.

There are three primary scripts for processing raw data:

1. ``tfs-process-fastq``: Analyses paired-end FASTQ files to count the
   occurrence of each genotype.
2. ``tfs-process-counts``: Aggregates counts across multiple samples and
   computes adjusted log-counts (``ln_cfu``) for downstream modelling.
3. ``tfs-process-presplit``: Like ``tfs-process-counts``, but for the
   pre-split time-point (before the library is divided into separate
   selection conditions). The output anchors the initial genotype
   abundances used by the growth model.

OD600 calibration
-----------------

``tfs-calibrate-od600`` fits the calibration from two small experiments, both
done with the same handling as production samples (same reader, plate type,
volume and strain):

- **Repeated readings of a dilution series.** Read each dilution several
  times, repeating the whole step each time (swirl the culture, pipette into
  the plate, read). A CSV with one row per reading, columns ``dilution``
  (relative to the undiluted culture) and ``od600``. The largest relative SD
  across dilutions is the reading noise; the detection threshold is midway
  between the mean readings of the two most dilute samples, where the series
  has flattened onto the reader's floor (``--detection_threshold``
  overrides it).
- **Plate counts of cultures whose OD600 was read.** One row per plate:
  ``od600``, ``colonies``, ``dilution`` (total dilution factor before
  plating), ``plated_volume_mL``, ``num_dilutions`` and ``plating_steps``.
  CFU/mL is ``colonies * dilution / plated_volume_mL``, with relative
  variance ``1 / colonies`` (counting) plus ``pipette_rel_error^2`` per
  dilution and plating step.

A polynomial of CFU/mL in OD600 (``--degree``, default 2) is fit by weighted
least squares. The output ``{out_prefix}.yaml`` holds the coefficients and
their full covariance, the reading noise, the detection threshold and the
calibrated OD600 range. The covariance matters: the curve's error is one
error shared by every tube calibrated with it, so it does not average out
across tubes and must not be treated as independent per-tube noise
(``tfscreen.process_raw.od600.cfu_per_mL_error_components`` returns the curve
and reading parts separately). ``{out_prefix}.pdf`` shows the dilution
series, the fit and the calibrated CFU's relative error; the two CSVs hold
the per-dilution noise and the per-plate fit.

.. code-block:: bash

    tfs-calibrate-od600 replicates.csv plate_counts.csv --out_prefix od600

``examples/od600/`` has synthetic inputs in these formats
(``make_example_data.py``) and the calibration made from them.

Configuration File (run_config.yaml)
-------------------------------------

``tfs-process-fastq`` requires a ``run_config.yaml`` file describing the
library of expected sequences. You can view or download an
:download:`example run_config.yaml <../../examples/process_raw/library_config.yaml>` file. Expected fields:

* ``reading_frame``: Amino acid reading frame offset (0, 1, or 2).
* ``first_amplicon_residue``: Amino acid residue number for the first in-frame
  residue.
* ``wt_seq``: The wildtype nucleic acid sequence.
* ``degen_sites``: Degenerate codon pattern the same length as ``wt_seq``
  (e.g. ``NNT``, ``NNK``, or ``.`` for wildtype).
* ``tiles``: Contiguous blocks of library components cloned together.
  ``.`` indicates wildtype; each unique character besides ``.`` defines a
  tile (blocks must be contiguous).
* ``expected_5p`` / ``expected_3p``: Flanking sequences immediately upstream
  and downstream of the amplicon.
* ``tile_combos``: List of strings such as ``single-x`` or ``double-x-y``,
  where ``x`` and ``y`` match characters in ``tiles``. ``single-x``
  specifies all single-mutation variants in tile ``x``; ``double-x-y``
  specifies all pairwise combinations between tiles ``x`` and ``y``.
* ``spiked_seqs``: Specific nucleic acid sequences (not part of the combinatorial
  library) that should be identified as controls.

tfs-process-fastq
-----------------

Reads paired-end FASTQ files and counts the protein genotype observed in each
read pair. Each read is matched against the predefined library after quality
filtering and flanking-sequence detection.

**Outputs** (written to ``out_dir``):

* ``stats_{filename}.csv`` — overall read success/failure statistics.
* ``counts_{filename}.csv`` — raw counts for each expected genotype.

**Usage:**

.. code-block:: bash

    tfs-process-fastq <f1_fastq> <f2_fastq> <out_dir> <run_config> [options]

**Positional arguments:**

* ``f1_fastq``: Path to the read-1 FASTQ file (gzip-compressed accepted).
* ``f2_fastq``: Path to the read-2 FASTQ file.
* ``out_dir``: Directory to write output CSV files (created if absent).
* ``run_config``: Path to the library configuration YAML file.

**Optional arguments:**

* ``--phred_cutoff``: Minimum Phred quality score; bases below this threshold
  are replaced with ``N`` (default: 10).
* ``--min_read_length``: Discard reads shorter than this length (default: 50).
* ``--allowed_num_flank_diffs``: Allowed mismatches when locating 5′ and 3′
  flanks (default: 1).
* ``--allowed_diff_from_expected``: Allowed mismatches from library genotypes
  (default: 2).
* ``--print_raw_seq``: If set, prints sequence matches to stdout for debugging.
* ``--max_num_reads``: Stop after this many reads.
* ``--chunk_size``: Block size for multiprocessing batches.
* ``--num_workers``: Number of parallel workers (default: available CPUs − 1).

tfs-process-counts
------------------

Takes the per-sample count CSVs produced by ``tfs-process-fastq``, aggregates
them according to a sample metadata file, and converts raw counts into
ln(CFU) values using per-sample CFU estimates.

**Output:** A single CSV file (``output_file``) containing ``ln_cfu`` per
genotype across all samples, ready for hierarchical modelling.

**Usage:**

.. code-block:: bash

    tfs-process-counts <sample_df> <counts_csv_path> <output_file> [options]

**Positional arguments:**

* ``sample_df``: Path to a CSV file describing samples. Must contain a unique
  ``sample`` column (used as the row index). Required columns:

  * ``sample`` *(index)* — unique identifier for each sequenced tube; used
    to match this row to a counts CSV file.
  * ``library`` — name of the physical library this sample belongs to. Genotypes
    are filtered and frequency-normalised within each library.
  * ``sample_ln_cfu`` — natural log of the total colony-forming units (CFU)
    measured for this tube.
  * ``sample_ln_cfu_std`` — standard deviation of ``sample_ln_cfu``.

  Instead of ``sample_ln_cfu``/``sample_ln_cfu_std``, you may supply
  ``sample_ln_cfu_var``, or the linear-space ``sample_cfu`` with
  ``sample_cfu_std`` (or ``sample_cfu_var``); the log-space columns are then
  inferred. Log-space columns are used when both are present.

  The following columns are not used by ``tfs-process-counts`` itself but are
  carried through into the ``ln_cfu`` output and are **required by the growth
  model** (``tfs-fit-model``):

  * ``replicate`` — integer replicate index distinguishing biological
    replicates that share the same condition.
  * ``condition_pre`` — name of the pre-selection growth condition (e.g.
    ``-kan``). Identifies the baseline growth arm.
  * ``condition_sel`` — name of the selection condition (e.g. ``+kan``).
  * ``titrant_name`` — name of the chemical titrant applied (e.g.
    ``IPTG``). Use a constant placeholder (e.g. ``none``) when no titrant
    is present.
  * ``titrant_conc`` — concentration of the titrant (float; 0 for no titrant).
  * ``t_pre`` — duration (minutes) of pre-selection growth.
  * ``t_sel`` — duration (minutes) of selection growth.

* ``counts_csv_path``: Directory containing the per-sample count CSV files.
  Each file is found by globbing
  ``{counts_glob_prefix}*{sample}*.csv`` within this directory.
* ``output_file``: Path for the output ln_cfu CSV.

**Optional arguments:**

* ``--counts_glob_prefix``: File prefix used when globbing for count files
  (default: ``counts``).
* ``--min_genotype_obs``: Minimum total counts across all samples for a
  genotype to be retained (default: 10).
* ``--pseudocount``: Pseudocount added to zero counts before log transformation
  (default: 1).
* ``--verbose``: If set, prints a summary of matched samples and file paths.

tfs-process-presplit
--------------------

Processes count files from the pre-split time-point — the single pooled
sample collected **before** the library is divided into separate selection
arms. These abundances anchor the initial per-genotype CFU values (``ln_cfu0``)
in the growth model. The output CSV is passed to ``tfs-fit-model`` via the
``presplit_df`` argument.

The interface is identical to ``tfs-process-counts``, but ``sample_df`` must
also include ``replicate`` and ``condition_pre`` columns so that the output
can be matched to the correct growth conditions.

**Output:** A single CSV file (``output_file``) with columns ``replicate``,
``condition_pre``, ``genotype``, ``ln_cfu``, and ``ln_cfu_std``.

**Usage:**

.. code-block:: bash

    tfs-process-presplit <sample_df> <counts_csv_path> <output_file> [options]

**Positional arguments:**

* ``sample_df``: Path to a CSV file describing the pre-split samples. Must
  contain a unique ``sample`` column plus:

  * ``library`` — physical library name (optional; defaults to ``default`` when
    absent, but must be consistent with the growth ``sample_df``).
  * ``replicate`` — replicate index; matched to the growth data.
  * ``condition_pre`` — pre-selection condition name; matched to the growth
    data.
  * ``sample_ln_cfu`` — natural log of the total CFU for this tube.
  * ``sample_ln_cfu_std`` — standard deviation of ``sample_ln_cfu``.

  As for ``tfs-process-counts``, these may instead be inferred from
  ``sample_ln_cfu_var`` or from ``sample_cfu`` with ``sample_cfu_std`` (or
  ``sample_cfu_var``).

* ``counts_csv_path``: Directory containing the per-sample count CSV files.
* ``output_file``: Path for the output presplit CSV.

**Optional arguments:**

* ``--counts_glob_prefix``: File prefix for globbing count files (default:
  ``counts``).
* ``--min_genotype_obs``: Minimum total counts for a genotype to be retained
  (default: 10).
* ``--pseudocount``: Pseudocount added before log transformation (default: 1).
* ``--verbose``: If set, prints a summary of matched samples and file paths.
