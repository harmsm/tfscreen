===================
Processing Raw Data
===================

A typical TF screen involves growing bacteria transformed with a plasmid-encoded
library of TF variants under one or more selection conditions (e.g. antibiotic resistance
driven by a TF-regulated promoter). At each time-point an aliquot is
plated to measure total colony-forming units (CFU) and deep-sequenced so
that the absolute abundance of every genotype can be determined. The raw
inputs to the pipeline are paired-end FASTQ files (one pair per sample) and
a sample metadata table (``sample_df``) that links each sequenced tube to its
biological context and its measured CFU count. The pipeline converts read
counts into per-genotype log-CFU estimates (``ln_cfu``) that feed the
hierarchical Bayesian growth model.

There are three primary scripts for processing raw data:

1. ``tfs-process-fastq``: Analyses paired-end FASTQ files to count the
   occurrence of each genotype.
2. ``tfs-process-counts``: Aggregates counts across multiple samples and
   computes adjusted log-counts (``ln_cfu``) for downstream modelling.
3. ``tfs-process-presplit``: Like ``tfs-process-counts``, but for the
   pre-split time-point (before the library is divided into separate
   selection conditions). The output anchors the initial genotype
   abundances used by the growth model.

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
  * ``sample_cfu`` — total colony-forming units (CFU) measured for this tube.
  * ``sample_cfu_std`` — standard deviation of ``sample_cfu``.

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
  * ``sample_cfu`` — total CFU for this tube.
  * ``sample_cfu_std`` — standard deviation of ``sample_cfu``.

* ``counts_csv_path``: Directory containing the per-sample count CSV files.
* ``output_file``: Path for the output presplit CSV.

**Optional arguments:**

* ``--counts_glob_prefix``: File prefix for globbing count files (default:
  ``counts``).
* ``--min_genotype_obs``: Minimum total counts for a genotype to be retained
  (default: 10).
* ``--pseudocount``: Pseudocount added before log transformation (default: 1).
* ``--verbose``: If set, prints a summary of matched samples and file paths.
