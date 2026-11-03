"""
A lab binding table to a ``binding_df`` for tfs-configure-model.

The lab's binding table has one row per measurement: a genotype, a protein
concentration, a titrant (IPTG) concentration and the measured occupancy
theta, plus the columns that say which biological replicate and which day
the measurement belongs to. A ``binding_df`` has one row per genotype,
protein concentration and titrant concentration: ``genotype``,
``titrant_name``, ``tf_total_uM`` (the pipetted protein, µM protomer, the
unit ``tfmodel/contexts.py`` reads), ``titrant_conc`` (mM, the unit the
growth table uses), ``theta_obs`` and ``theta_std``. The fit then evaluates
each measurement at its own protein concentration, which is known, rather
than at the in vivo TF total.

``binding_table`` makes one from the other:

1. Drops rows with no theta (counted and reported).
2. Converts both concentrations from the table's unit and rounds them to
   ``sig_digits`` significant digits. The lab's concentrations carry
   floating-point residue that differs between days (6.000000000000001e-08
   against 6.000000000000002e-08), so without rounding the replicates of
   one point would not line up.
3. Keeps every protein concentration, or only the requested one. The
   protein column is converted to protomer units (twice a dimer
   concentration).
4. Averages the measurements within each replicate (the ``replicate_cols``,
   biological replicate and date by default), so technical repeats count
   once.
5. Averages the replicates of each (genotype, protein concentration,
   titrant concentration) for ``theta_obs``.
6. Sets ``theta_std`` to the standard error of that mean, from one replicate
   SD pooled over every replicated point at that protein concentration
   (``s_pooled / sqrt(n)``), or from a fixed per-replicate SD given as
   ``theta_std``. A protein concentration with no replicated point takes
   the SD pooled over every protein concentration. Pooling is needed
   because most points have a single replicate; it assumes the replicate
   scatter does not depend on genotype or theta.

Measurements of one replicate across titrant concentrations share that
day's normalization, so their errors are correlated. The binding likelihood
treats every row as independent; this is not corrected here.
"""

import numpy as np
import pandas as pd

import tfscreen

# Multiplier from each unit to molar.
CONC_UNITS = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9}

# The growth table's titrant unit.
TITRANT_OUT_UNITS = "mM"

BINDING_COLUMNS = ["genotype", "titrant_name", "titrant_conc",
                   "theta_obs", "theta_std"]

# The binding_df column with the protein concentration (µM protomer).
TF_TOTAL_COLUMN = "tf_total_uM"

# Protomers per unit of the protein column.
PROTEIN_SPECIES = {"dimer": 2.0, "protomer": 1.0}


def _unit_factor(units, what):
    if units not in CONC_UNITS:
        raise ValueError(f"{what} '{units}' is not one of {sorted(CONC_UNITS)}")
    return CONC_UNITS[units]


def _round_sig(values, sig_digits):
    """Round each value to sig_digits significant digits (0 stays 0)."""
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values)
    nz = values != 0
    mag = np.floor(np.log10(np.abs(values[nz])))
    scale = 10.0 ** (sig_digits - 1 - mag)
    out[nz] = np.round(values[nz] * scale) / scale
    return out


def _pool(reps, point):
    """Pooled replicate SD and its dof over the points of ``reps``."""
    g = reps.groupby(point, observed=True)["mean"]
    resid = reps["mean"] - g.transform("mean")
    n = g.transform("size")
    dof = int((g.size() - 1).sum())
    sd = (float(np.sqrt((resid[n > 1] ** 2).sum() / dof)) if dof > 0
          else float("nan"))
    return sd, dof


def binding_table(binding_wells,
                  protein_conc=None,
                  protein_conc_units="uM",
                  genotype_col="genotype",
                  protein_col="protein_dimer_conc_molar",
                  titrant_col="iptg_conc_molar",
                  theta_col="theta",
                  conc_units="M",
                  replicate_cols=("biorep", "date"),
                  titrant_name="iptg",
                  theta_std=None,
                  sig_digits=6,
                  protein_species="dimer",
                  write_tf_total=True):
    """
    Average a lab binding table into a binding_df, one block per protein
    concentration.

    Parameters
    ----------
    binding_wells : pandas.DataFrame or str
        The lab's table (or a path to it), one row per measurement.
    protein_conc : float, optional
        Keep only this protein concentration, in ``protein_conc_units`` and
        in the units of the protein column (``protein_species``). None
        (default): keep every concentration.
    protein_conc_units : str
        Unit of ``protein_conc``: 'M', 'mM', 'uM' or 'nM'.
    genotype_col, protein_col, titrant_col, theta_col : str
        Column names in ``binding_wells``.
    conc_units : str
        Unit of the protein and titrant columns.
    replicate_cols : sequence of str
        Columns that identify one replicate (measurements sharing them are
        averaged first). Empty: every row is a replicate.
    titrant_name : str
        The ``titrant_name`` written to every row; must match the growth
        table's.
    theta_std : float, optional
        A fixed per-replicate SD. None (default): pooled from the replicated
        points.
    sig_digits : int
        Significant digits the concentrations are rounded to.
    protein_species : str
        What the protein column counts: 'dimer' (default, as in
        ``protein_dimer_conc_molar``) or 'protomer'. ``tf_total_uM`` is in
        protomers, so a dimer concentration is doubled.
    write_tf_total : bool
        Write the ``tf_total_uM`` column (default). Without it the
        binding_df is the older one, which a curve θ model (``hill_geno``)
        needs; it then must hold one protein concentration.

    Returns
    -------
    binding_df : pandas.DataFrame
        ``BINDING_COLUMNS`` with ``tf_total_uM`` after ``titrant_name``,
        sorted by genotype, protein and titrant concentration.
    summary : pandas.DataFrame
        One row per binding_df row with ``n_replicates``, ``n_measurements``,
        the point's own replicate SD (``replicate_sd``, NaN below two
        replicates) and ``theta_std``.
    info : dict
        ``rows_in``, ``rows_no_theta``, ``rows_kept``; ``pooled_sd`` and
        ``pooled_dof`` over every protein concentration; ``by_protein``, a
        DataFrame with each protein concentration's ``tf_total_uM``,
        ``rows``, ``pooled_sd``, ``pooled_dof`` and ``replicate_sd_used``;
        ``replicate_sd_used`` and ``protein_conc_M`` when one protein
        concentration is kept.
    """

    df = tfscreen.util.io.read_dataframe(binding_wells)
    replicate_cols = list(replicate_cols)
    required = [genotype_col, protein_col, titrant_col, theta_col] + replicate_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"binding table is missing column(s) {missing}; "
                         f"it has {list(df.columns)}")
    if sig_digits < 1:
        raise ValueError("sig_digits must be at least 1")
    if protein_species not in PROTEIN_SPECIES:
        raise ValueError(f"protein_species '{protein_species}' is not one of "
                         f"{sorted(PROTEIN_SPECIES)}")
    if theta_std is not None and (not np.isfinite(theta_std) or theta_std <= 0):
        raise ValueError(f"theta_std must be positive, got {theta_std}")

    to_M = _unit_factor(conc_units, "conc_units")
    info = {"rows_in": len(df)}
    no_theta = df[theta_col].isna()
    info["rows_no_theta"] = int(no_theta.sum())
    df = df[~no_theta].copy()

    df["_protein_M"] = _round_sig(df[protein_col].to_numpy() * to_M, sig_digits)
    if protein_conc is not None:
        target_M = protein_conc * _unit_factor(protein_conc_units,
                                               "protein_conc_units")
        if not np.isfinite(target_M) or target_M <= 0:
            raise ValueError(f"protein_conc must be positive, got "
                             f"{protein_conc}")
        target_M = float(_round_sig([target_M], sig_digits)[0])
        keep = np.isclose(df["_protein_M"], target_M, rtol=1e-9, atol=0)
        if not keep.any():
            have = sorted(set(df["_protein_M"]))
            raise ValueError(
                f"no measurements at protein {protein_conc} "
                f"{protein_conc_units} ({target_M:g} M); the table has (M) "
                f"{have}")
        df = df[keep].copy()
    info["rows_kept"] = len(df)
    proteins = sorted(set(df["_protein_M"]))
    if not write_tf_total and len(proteins) > 1:
        raise ValueError(
            f"the table has {len(proteins)} protein concentrations (M) "
            f"{proteins}; without the tf_total_uM column the binding_df can "
            f"hold only one, so give protein_conc.")
    df[TF_TOTAL_COLUMN] = _round_sig(
        df["_protein_M"] * PROTEIN_SPECIES[protein_species] / 1e-6, sig_digits)

    titrant_M = _round_sig(df[titrant_col].to_numpy() * to_M, sig_digits)
    df["titrant_conc"] = _round_sig(titrant_M / CONC_UNITS[TITRANT_OUT_UNITS],
                                    sig_digits)
    df["genotype"] = df[genotype_col].astype(str)

    point = ["genotype", TF_TOTAL_COLUMN, "titrant_conc"]
    if not replicate_cols:
        df["_row"] = np.arange(len(df))
        replicate_cols = ["_row"]
    reps = (df.groupby(point + replicate_cols, observed=True)[theta_col]
            .agg(["mean", "size"]).reset_index())
    g = reps.groupby(point, observed=True)
    summary = g.agg(theta_obs=("mean", "mean"),
                    n_replicates=("mean", "size"),
                    n_measurements=("size", "sum"),
                    replicate_sd=("mean", "std")).reset_index()

    info["pooled_sd"], info["pooled_dof"] = _pool(reps, point)
    rows = []
    for tf, block in reps.groupby(TF_TOTAL_COLUMN, observed=True):
        sd_p, dof_p = _pool(block, point)
        if theta_std is not None:
            used = float(theta_std)
        elif dof_p > 0 and sd_p > 0:
            used = sd_p
        elif info["pooled_dof"] > 0 and info["pooled_sd"] > 0:
            used = info["pooled_sd"]
        else:
            raise ValueError(
                "no replicated points to pool a replicate SD from; give "
                "theta_std")
        rows.append({TF_TOTAL_COLUMN: float(tf),
                     "rows": int((df[TF_TOTAL_COLUMN] == tf).sum()),
                     "pooled_sd": sd_p, "pooled_dof": dof_p,
                     "replicate_sd_used": used})
    by_protein = pd.DataFrame(rows)
    info["by_protein"] = by_protein
    if len(by_protein) == 1:
        info["replicate_sd_used"] = float(by_protein["replicate_sd_used"].iloc[0])
        info["protein_conc_M"] = float(proteins[0])

    used = summary[TF_TOTAL_COLUMN].map(
        by_protein.set_index(TF_TOTAL_COLUMN)["replicate_sd_used"])
    summary["theta_std"] = used / np.sqrt(summary["n_replicates"])
    summary.insert(1, "titrant_name", titrant_name)
    summary = summary.sort_values(point, kind="stable").reset_index(drop=True)
    columns = (["genotype", "titrant_name", TF_TOTAL_COLUMN]
               + BINDING_COLUMNS[2:]) if write_tf_total else BINDING_COLUMNS
    return summary[columns].copy(), summary, info


def format_info(info):
    """A short printable report of binding_table's info dict."""
    lines = [
        f"binding rows: {info['rows_in']} in, {info['rows_no_theta']} without "
        f"theta dropped, {info['rows_kept']} kept",
        f"pooled replicate SD over every protein concentration: "
        f"{info['pooled_sd']:.4g} ({info['pooled_dof']} dof)",
    ]
    for _, r in info["by_protein"].iterrows():
        lines.append(
            f"  protein {r[TF_TOTAL_COLUMN]:g} uM protomer: {int(r['rows'])} "
            f"rows, pooled SD {r['pooled_sd']:.4g} ({int(r['pooled_dof'])} "
            f"dof), replicate SD used for theta_std "
            f"{r['replicate_sd_used']:.4g}")
    return "\n".join(lines)
