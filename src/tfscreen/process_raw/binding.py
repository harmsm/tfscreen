"""
A lab binding table to a ``binding_df`` for tfs-configure-model.

The lab's binding table has one row per measurement: a genotype, a protein
concentration, a titrant (IPTG) concentration and the measured occupancy
theta, plus the columns that say which biological replicate and which day
the measurement belongs to. A ``binding_df`` has one row per genotype and
titrant concentration at a single protein concentration: ``genotype``,
``titrant_name``, ``titrant_conc`` (mM, the unit the growth table uses),
``theta_obs`` and ``theta_std``.

``binding_table`` makes one from the other:

1. Drops rows with no theta (counted and reported).
2. Converts both concentrations from the table's unit and rounds them to
   ``sig_digits`` significant digits. The lab's concentrations carry
   floating-point residue that differs between days (6.000000000000001e-08
   against 6.000000000000002e-08), so without rounding the replicates of
   one point would not line up.
3. Keeps the rows at the requested protein concentration.
4. Averages the measurements within each replicate (the ``replicate_cols``,
   biological replicate and date by default), so technical repeats count
   once.
5. Averages the replicates of each (genotype, titrant concentration) for
   ``theta_obs``.
6. Sets ``theta_std`` to the standard error of that mean, from one replicate
   SD pooled over every replicated point at this protein concentration
   (``s_pooled / sqrt(n)``), or from a fixed per-replicate SD given as
   ``theta_std``. Pooling is needed because most points have a single
   replicate; it assumes the replicate scatter does not depend on genotype
   or theta.

Measurements of one replicate across titrant concentrations share that
day's normalization, so their errors are correlated. The binding likelihood
treats every row as independent; this is not corrected here.
"""

import numpy as np

import tfscreen

# Multiplier from each unit to molar.
CONC_UNITS = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9}

# The growth table's titrant unit.
TITRANT_OUT_UNITS = "mM"

BINDING_COLUMNS = ["genotype", "titrant_name", "titrant_conc",
                   "theta_obs", "theta_std"]


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


def binding_table(binding_wells,
                  protein_conc,
                  protein_conc_units="uM",
                  genotype_col="genotype",
                  protein_col="protein_dimer_conc_molar",
                  titrant_col="iptg_conc_molar",
                  theta_col="theta",
                  conc_units="M",
                  replicate_cols=("biorep", "date"),
                  titrant_name="iptg",
                  theta_std=None,
                  sig_digits=6):
    """
    Average a lab binding table into a binding_df at one protein
    concentration.

    Parameters
    ----------
    binding_wells : pandas.DataFrame or str
        The lab's table (or a path to it), one row per measurement.
    protein_conc : float
        The protein concentration to keep, in ``protein_conc_units``.
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

    Returns
    -------
    binding_df : pandas.DataFrame
        ``BINDING_COLUMNS``, sorted by genotype and titrant concentration.
    summary : pandas.DataFrame
        One row per binding_df row with ``n_replicates``, ``n_measurements``,
        the point's own replicate SD (``replicate_sd``, NaN below two
        replicates) and ``theta_std``.
    info : dict
        ``rows_in``, ``rows_no_theta``, ``rows_kept``, ``pooled_sd``,
        ``pooled_dof``, ``replicate_sd_used`` and ``protein_conc_M``.
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

    to_M = _unit_factor(conc_units, "conc_units")
    target_M = protein_conc * _unit_factor(protein_conc_units, "protein_conc_units")
    if not np.isfinite(target_M) or target_M <= 0:
        raise ValueError(f"protein_conc must be positive, got {protein_conc}")

    info = {"rows_in": len(df)}
    no_theta = df[theta_col].isna()
    info["rows_no_theta"] = int(no_theta.sum())
    df = df[~no_theta].copy()

    df["_protein_M"] = _round_sig(df[protein_col].to_numpy() * to_M, sig_digits)
    target_M = float(_round_sig([target_M], sig_digits)[0])
    keep = np.isclose(df["_protein_M"], target_M, rtol=1e-9, atol=0)
    if not keep.any():
        have = sorted(set(df["_protein_M"]))
        raise ValueError(
            f"no measurements at protein {protein_conc} {protein_conc_units} "
            f"({target_M:g} M); the table has (M) {have}")
    df = df[keep].copy()
    info["rows_kept"] = len(df)
    info["protein_conc_M"] = target_M

    titrant_M = _round_sig(df[titrant_col].to_numpy() * to_M, sig_digits)
    df["titrant_conc"] = _round_sig(titrant_M / CONC_UNITS[TITRANT_OUT_UNITS],
                                    sig_digits)
    df["genotype"] = df[genotype_col].astype(str)

    point = ["genotype", "titrant_conc"]
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

    resid = reps["mean"] - g["mean"].transform("mean")
    n = g["mean"].transform("size")
    dof = int((summary["n_replicates"] - 1).sum())
    info["pooled_dof"] = dof
    info["pooled_sd"] = (float(np.sqrt((resid[n > 1] ** 2).sum() / dof))
                         if dof > 0 else float("nan"))

    if theta_std is not None:
        if not np.isfinite(theta_std) or theta_std <= 0:
            raise ValueError(f"theta_std must be positive, got {theta_std}")
        sd = float(theta_std)
    elif dof > 0 and info["pooled_sd"] > 0:
        sd = info["pooled_sd"]
    else:
        raise ValueError(
            "no replicated points at this protein concentration to pool a "
            "replicate SD from; give theta_std")
    info["replicate_sd_used"] = sd

    summary["theta_std"] = sd / np.sqrt(summary["n_replicates"])
    summary.insert(1, "titrant_name", titrant_name)
    summary = summary.sort_values(point, kind="stable").reset_index(drop=True)
    return summary[BINDING_COLUMNS].copy(), summary, info


def format_info(info):
    """A short printable report of binding_table's info dict."""
    lines = [
        f"binding rows: {info['rows_in']} in, {info['rows_no_theta']} without "
        f"theta dropped, {info['rows_kept']} at protein "
        f"{info['protein_conc_M']:g} M",
        f"pooled replicate SD: {info['pooled_sd']:.4g} "
        f"({info['pooled_dof']} dof)",
        f"replicate SD used for theta_std: {info['replicate_sd_used']:.4g}",
    ]
    return "\n".join(lines)
