"""
tfs-correct-tubes: apply documented corrections to a tube table before
tfs-process-counts.

See ``tfscreen.process_raw.tube_corrections`` for the corrections file and
why a swap moves only the design columns.
"""

from tfscreen.process_raw.tube_corrections import apply_corrections
from tfscreen.util.cli import generalized_main


def correct_tubes(tube_table_file,
                  corrections_file,
                  out_prefix="tfs_tubes_corrected"):
    """
    Apply documented corrections to a tube table, leaving the original as is.

    A correction is an entry in a YAML corrections file kept with the
    experiment's data: an ``id``, an ``action``, the ``tubes`` it names (by
    the tube table's ``sample``), a ``reason``, who decided it (``by``) and
    when (``date``), and optional ``evidence`` such as a QC table. Actions:
    ``swap`` (two tubes of one library exchange their design columns),
    ``relabel`` (one tube gets the design values in ``set``) and ``exclude``
    (tubes left out). The design columns are ``replicate``,
    ``condition_pre``, ``t_pre``, ``condition_sel``, ``t_sel``,
    ``titrant_name`` and ``titrant_conc``. Every other column (the sample
    name and so its count file, OD600, tube totals, barcodes) stays with the
    physical sample it was measured on, which is right whether the tubes
    were swapped at the bench or the sample sheet was. ``library`` never
    changes, and a swap across libraries is refused.

    Writes ``{out_prefix}.csv`` (the corrected tube table, with a
    ``correction_id`` column, for tfs-process-counts),
    ``{out_prefix}_changes.csv`` (one row per changed value or excluded
    tube, with its original and corrected value) and
    ``{out_prefix}_provenance.json`` (the corrections and the SHA-256 of
    the inputs, the evidence files and the output).

    Fails on an unknown tube, a tube named in two corrections, a swap
    across libraries or one that changes nothing, a relabel of a column that
    is not a design column, a correction that leaves two tubes with the same
    library and design, a missing or empty reason, by or date, or an output
    that would overwrite an input.

    Parameters
    ----------
    tube_table_file : str
        The original tube table (CSV, TSV or Excel), in tfs-process-counts
        format.
    corrections_file : str
        YAML corrections file.
    out_prefix : str
        Output prefix.
    """
    apply_corrections(tube_table_file, corrections_file, out_prefix=out_prefix)


def main():
    return generalized_main(correct_tubes)


if __name__ == "__main__":
    main()
