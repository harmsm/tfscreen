"""
``tfs-process-counts``: per-tube genotype counts to the growth (or presplit)
table that ``tfs-configure-model`` reads.
"""


from tfscreen.process_raw import counts_to_lncfu
from tfscreen.process_raw.counts_to_lncfu import get_sample_ln_cfu
from tfscreen.process_raw._counts_io import _prep_sample_df, _aggregate_counts
from tfscreen.process_raw.od600 import tube_totals_from_od600
from tfscreen.util.cli import generalized_main
from tfscreen.util.dataframe import check_columns, get_scaled_cfu
from tfscreen.util.io import read_dataframe

# Tube-total columns a tube table may supply instead of OD600.
_TOTAL_COLUMNS = ("sample_cfu", "sample_cfu_std", "sample_cfu_var",
                  "sample_ln_cfu", "sample_ln_cfu_std", "sample_ln_cfu_var")

PRESPLIT_COLUMNS = ["library", "replicate", "condition_pre", "genotype",
                    "ln_cfu", "ln_cfu_std"]


def _read_tube_table(sample_file):
    """The tube table as a DataFrame with a string ``sample`` column."""
    df = read_dataframe(sample_file)
    if df.index.name == "sample":
        df = df.reset_index()
    check_columns(df, required_columns=["sample"])
    df = df.copy()
    df["sample"] = df["sample"].astype(str)
    return df


def add_tube_totals(sample_df, od600_file=None, od600_calibration_file=None,
                    tube_volume_mL=None):
    """
    Give every tube its total CFU, from OD600 or as supplied.

    With an OD600 table (``od600_file``, or an ``od600`` column in the tube
    table) and a calibration, each tube's ``sample_cfu`` and its SD come from
    ``tube_totals_from_od600``. Otherwise the tube table must already carry
    the totals (``sample_cfu`` + ``sample_cfu_std``, or the ``sample_ln_cfu``
    forms; see ``counts_to_lncfu``).

    Raises
    ------
    ValueError
        If OD600 is given without a calibration or volume (or the reverse),
        if the tube table supplies totals as well as OD600, or if a tube has
        no OD600 reading.
    """
    sample_df = sample_df.copy()

    od600_df = None
    if od600_file is not None:
        od600_df = read_dataframe(od600_file)
        if od600_df.index.name == "sample":
            od600_df = od600_df.reset_index()
        if "od600" in sample_df.columns:
            raise ValueError("OD600 is given twice: as od600_file and as an "
                             "'od600' column of the tube table. Use one.")
    elif "od600" in sample_df.columns:
        od600_df = sample_df[["sample", "od600"]]

    if od600_df is None:
        if od600_calibration_file is not None:
            raise ValueError("od600_calibration_file was given, but there is no "
                             "OD600: pass od600_file or add an 'od600' column "
                             "to the tube table.")
        return sample_df

    if od600_calibration_file is None:
        raise ValueError("OD600 needs an OD600 calibration "
                         "(od600_calibration_file, made by tfs-calibrate-od600).")
    if tube_volume_mL is None:
        raise ValueError("OD600 needs the culture volume of a tube "
                         "(tube_volume_mL): the calibration gives CFU/mL and "
                         "the model wants the cells in the tube.")
    supplied = [c for c in _TOTAL_COLUMNS if c in sample_df.columns]
    if supplied:
        raise ValueError(f"The tube table supplies tube totals {supplied} and "
                         "OD600 is also given. Use one or the other.")

    totals = tube_totals_from_od600(od600_df, od600_calibration_file, tube_volume_mL)
    missing = sorted(set(sample_df["sample"]) - set(totals.index))
    if missing:
        raise ValueError(f"No OD600 reading for tube(s): {missing}")
    if "od600" in sample_df.columns:
        sample_df = sample_df.drop(columns="od600")
    return sample_df.merge(totals, left_on="sample", right_index=True,
                           how="left")


def process_counts(sample_file,
                   counts_dir,
                   out_prefix=None,
                   od600_file=None,
                   od600_calibration_file=None,
                   tube_volume_mL=None,
                   presplit=False,
                   counts_glob_prefix="counts",
                   min_genotype_obs=10,
                   pseudocount=1,
                   verbose=True):
    """
    Turn per-tube genotype counts into the table tfs-configure-model reads.

    The tube table has one row per sequenced tube: a unique ``sample`` name,
    its ``library``, and the design columns (``replicate``, ``condition_pre``,
    ``t_pre``, ``condition_sel``, ``t_sel``, ``titrant_name``,
    ``titrant_conc``), which are carried through to the output. Every library
    is processed in one call; genotypes are filtered and filled in within
    each library.

    Each tube needs its total CFU. Give its OD600 (``--od600_file``, or an
    ``od600`` column in the tube table) with ``--od600_calibration_file`` and
    ``--tube_volume_mL``, and the totals and their SDs are computed here:
    ``sample_cfu_curve_std`` is the calibration curve's error, shared by every
    tube, and ``sample_cfu_reading_std`` is the reading's own. Or supply them
    in the tube table as ``sample_cfu`` and ``sample_cfu_std``.

    Writes ``{out_prefix}.csv``. Without ``--presplit`` it is the growth table
    for ``tfs-configure-model --growth_df``: one row per genotype and tube
    with ``counts``, ``sample_reads``, ``frequency``, ``ln_cfu`` and
    ``ln_cfu_var``. With ``--presplit`` it is the presplit table for
    ``--presplit_df`` (``library``, ``replicate``, ``condition_pre``,
    ``genotype``, ``ln_cfu``, ``ln_cfu_std``).

    Parameters
    ----------
    sample_file : str
        Tube table (CSV, TSV or Excel), one row per sequenced tube.
    counts_dir : str
        Directory of tfs-process-fastq count files. Each tube's file must be
        the only one matching ``{counts_glob_prefix}*{sample}*.csv``.
    out_prefix : str, optional
        Output prefix. Default tfs_growth, or tfs_presplit with --presplit.
    od600_file : str, optional
        Table of one OD600 reading per tube (columns ``sample``, ``od600``).
    od600_calibration_file : str, optional
        OD600-to-CFU/mL calibration written by tfs-calibrate-od600.
    tube_volume_mL : float, optional
        Culture volume of one tube in mL. Required with OD600.
    presplit : bool
        Write the presplit table instead of the growth table. The tube table
        then needs ``replicate`` and ``condition_pre``.
    counts_glob_prefix : str
        File-name prefix of the count files.
    min_genotype_obs : int
        Drop genotypes with fewer total reads than this within a library.
    pseudocount : int
        Added to every genotype's count before taking frequencies.
    verbose : bool
        Print which count file was matched to each tube.
    """

    if out_prefix is None:
        out_prefix = "tfs_presplit" if presplit else "tfs_growth"

    sample_df = _read_tube_table(sample_file)
    if presplit:
        check_columns(sample_df, required_columns=["replicate", "condition_pre"])
    if "library" not in sample_df.columns:
        sample_df["library"] = "default"

    sample_df = add_tube_totals(sample_df,
                                od600_file=od600_file,
                                od600_calibration_file=od600_calibration_file,
                                tube_volume_mL=tube_volume_mL)

    # Indexed by sample, with an 'obs_file' column naming each counts file.
    sample_df = _prep_sample_df(sample_df,
                                counts_dir,
                                counts_glob_prefix,
                                verbose)

    sample_df = get_sample_ln_cfu(sample_df)

    counts_df = _aggregate_counts(sample_df)

    ln_cfu_df = counts_to_lncfu(sample_df,
                                counts_df,
                                min_genotype_obs=min_genotype_obs,
                                pseudocount=pseudocount)

    if presplit:
        ln_cfu_df = get_scaled_cfu(ln_cfu_df, need_columns=["ln_cfu", "ln_cfu_std"])
        ln_cfu_df = (ln_cfu_df[PRESPLIT_COLUMNS]
                     .sort_values(by=["library", "replicate", "condition_pre",
                                      "genotype"],
                                  ignore_index=True))
    else:
        ln_cfu_df = ln_cfu_df.drop(columns=["obs_file"], errors="ignore")

    out_file = f"{out_prefix}.csv"
    ln_cfu_df.to_csv(out_file, index=False)
    return out_file


def main():
    return generalized_main(process_counts,
                            manual_arg_types={"tube_volume_mL": float})


if __name__ == "__main__":
    main()
