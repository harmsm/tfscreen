===================
Processing Raw Data
===================

This section describes how to process raw sequencing reads into genotype
counts, and how to combine those counts into final values suitable for
downstream statistical modelling.

There are two primary scripts for processing raw data:

1. ``tfs-process-fastq``: Analyses paired-end FASTQ files to count the
   occurrence of each genotype.
2. ``tfs-process-counts``: Aggregates counts across multiple samples and
   computes adjusted log-counts (``ln_cfu``) for analysis.

Configuration File (run_config.yaml)
-------------------------------------

``tfs-process-fastq`` requires a ``run_config.yaml`` file describing the
library of expected sequences. You can view or download an
:download:`example run_config.yaml <run_config.yaml>` file. Expected fields:

* ``reading_frame``: Amino acid reading frame offset (0, 1, or 2).
* ``first_amplicon_residue``: Amino acid residue number for the first in-frame
  residue.
* ``wt_seq``: The wildtype nucleic acid sequence.
* ``degen_sites``: Degenerate codon pattern the same length as ``wt_seq``
  (e.g. ``NNT``, ``NNK``, or ``.`` for wildtype).
* ``sub_libraries``: Contiguous blocks of library components cloned together.
  ``.`` indicates wildtype; each unique character besides ``.`` defines a
  sub-library (blocks must be contiguous).
* ``expected_5p`` / ``expected_3p``: Flanking sequences immediately upstream
  and downstream of the amplicon.
* ``library_combos``: List of strings such as ``single-x`` or ``double-x-y``,
  where ``x`` and ``y`` match characters in ``sub_libraries``. ``single-x``
  specifies all single-mutation variants in sub-library ``x``; ``double-x-y``
  specifies all pairwise combinations between sub-libraries ``x`` and ``y``.
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
  ``sample`` column (used as the row index) plus ``sample_cfu`` and
  ``sample_cfu_std`` columns giving the total CFU and its standard deviation
  for each sample tube.
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
