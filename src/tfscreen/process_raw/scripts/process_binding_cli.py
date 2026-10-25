"""
tfs-process-binding: a lab binding table to a binding_df for
tfs-configure-model.

See ``tfscreen.process_raw.binding`` for the averaging and theta_std.
"""

import os

from tfscreen.process_raw.binding import binding_table, format_info
from tfscreen.util.cli import generalized_main


def process_binding(binding_file,
                    protein_conc=2.0,
                    protein_conc_units="uM",
                    genotype_col="genotype",
                    protein_col="protein_dimer_conc_molar",
                    titrant_col="iptg_conc_molar",
                    theta_col="theta",
                    conc_units="M",
                    replicate_cols="biorep,date",
                    titrant_name="iptg",
                    theta_std=None,
                    sig_digits=6,
                    out_prefix="tfs_binding"):
    """
    Average a lab binding table into a binding_df at one protein
    concentration.

    The table has one row per measurement (genotype, protein and titrant
    concentrations, theta, and the columns naming its replicate). Rows
    without theta are dropped; concentrations are rounded to sig_digits
    significant digits so replicates from different days line up;
    measurements within a replicate are averaged, then replicates per
    genotype and titrant concentration (theta_obs). theta_std is the
    standard error of that mean from one replicate SD pooled over every
    replicated point at this protein concentration, or from --theta_std.
    The titrant is written in mM.

    Writes {out_prefix}.csv (genotype, titrant_name, titrant_conc,
    theta_obs, theta_std; for tfs-configure-model --binding_df) and
    {out_prefix}_summary.csv (replicate and measurement counts and each
    point's own replicate SD).

    Parameters
    ----------
    binding_file : str
        The lab's binding table (CSV, TSV or Excel).
    protein_conc : float
        Protein concentration to keep, in protein_conc_units.
    protein_conc_units : str
        Unit of protein_conc: M, mM, uM or nM.
    genotype_col : str
        Genotype column.
    protein_col : str
        Protein concentration column.
    titrant_col : str
        Titrant concentration column.
    theta_col : str
        Measured theta column.
    conc_units : str
        Unit of the protein and titrant columns: M, mM, uM or nM.
    replicate_cols : str
        Comma-separated columns that identify one replicate; empty for one
        replicate per row.
    titrant_name : str
        titrant_name for every row; must match the growth table's.
    theta_std : float, optional
        Fixed per-replicate SD instead of the pooled one.
    sig_digits : int
        Significant digits the concentrations are rounded to.
    out_prefix : str
        Output prefix.
    """
    reps = [c.strip() for c in replicate_cols.split(",") if c.strip()]
    df, summary, info = binding_table(
        binding_file, protein_conc,
        protein_conc_units=protein_conc_units,
        genotype_col=genotype_col, protein_col=protein_col,
        titrant_col=titrant_col, theta_col=theta_col,
        conc_units=conc_units, replicate_cols=reps,
        titrant_name=titrant_name, theta_std=theta_std,
        sig_digits=sig_digits)

    out = f"{out_prefix}.csv"
    if os.path.abspath(out) == os.path.abspath(binding_file):
        raise ValueError(f"{out} would overwrite the input")
    print(format_info(info), flush=True)
    print(f"{len(df)} points, {df['genotype'].nunique()} genotypes", flush=True)
    df.to_csv(out, index=False)
    summary.to_csv(f"{out_prefix}_summary.csv", index=False)
    print(f"wrote {out} and {out_prefix}_summary.csv", flush=True)


def main():
    return generalized_main(process_binding,
                            manual_arg_types={"theta_std": float})


if __name__ == "__main__":
    main()
