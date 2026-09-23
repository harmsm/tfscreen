"""
Expected composition of a pooled library from its design.

A screen library is pooled from several sub-libraries (``library_origin``:
``single-1``, ``double-1-2``, ``spiked``, ...). Each sub-library is transformed
separately and the transformed cultures are mixed by ``library_mixture``. The
same amino-acid genotype can come from more than one sub-library: wt and the
spiked single mutants are also encoded by codons in the bulk sub-libraries, so
spiked and bulk copies of one genotype can't be told apart by sequence.

Within a sub-library every codon-level sequence is equally likely (before
assembly skew), so a genotype's share of that sub-library is its codon
``degeneracy`` divided by the sub-library's total number of sequences. That is
the ``weight`` column written by ``LibraryManager.build_library_df``.
"""

import numpy as np
import pandas as pd

SPIKED_ORIGIN = "spiked"


def _check_origin_keys(library_df, mapping, name):
    """Fail fast unless ``mapping`` has exactly the library's origins."""
    origins = set(library_df["library_origin"].astype(str))
    keys = set(mapping)
    missing = sorted(origins - keys)
    extra = sorted(keys - origins)
    if missing or extra:
        err = f"{name} keys must match the library origins {sorted(origins)}."
        if missing:
            err += f" Missing: {missing}."
        if extra:
            err += f" Not in the library: {extra}."
        raise ValueError(err)


def _origin_sequence_counts(library_df):
    """Number of codon-level sequences in each library origin."""
    if "degeneracy" not in library_df.columns:
        raise ValueError(
            "library_df needs a 'degeneracy' column (as written by "
            "LibraryManager.build_library_df)."
        )
    counts = (library_df
              .assign(library_origin=library_df["library_origin"].astype(str))
              .groupby("library_origin")["degeneracy"]
              .sum())
    return counts.to_dict()


def expected_library_composition(library_df,
                                 library_mixture,
                                 spiked_origin=SPIKED_ORIGIN):
    """
    Expected pool share and bulk fraction of every genotype.

    Parameters
    ----------
    library_df : pandas.DataFrame
        One row per (library_origin, genotype) with columns
        ``library_origin``, ``genotype`` and ``weight`` (the genotype's share
        of its sub-library), as written by
        ``LibraryManager.build_library_df``.
    library_mixture : dict
        Relative amount of each transformed sub-library in the pool, keyed by
        library origin. Only ratios matter. Must name every origin in
        ``library_df`` and no others.
    spiked_origin : str, default "spiked"
        The origin holding monoclonal spiked controls. Every other origin is
        bulk.

    Returns
    -------
    pandas.DataFrame
        One row per genotype, with columns:

        - ``genotype``
        - ``pool_fraction``: expected fraction of the pooled cells carrying
          the genotype. Sums to 1.
        - ``bulk_fraction``: expected fraction of the genotype's cells that
          come from bulk sub-libraries (0 = pure spike, 1 = pure bulk).

    Notes
    -----
    This is the design expectation. It ignores library assembly skew and the
    finite number of transformants, both of which scatter the realized value
    around it. It also assumes a cell's plasmid copies are shared among the
    variants it carries, so a multi-plasmid cell contributes one cell's worth
    of each genotype's observable in total, and the fraction does not depend
    on the congression parameter lambda.

    Examples
    --------
    >>> import pandas as pd
    >>> library_df = pd.DataFrame({
    ...     "library_origin": ["single-1", "single-1", "spiked", "spiked"],
    ...     "genotype": ["wt", "M42I", "wt", "H74A"],
    ...     "weight": [0.25, 0.75, 0.5, 0.5],
    ... })
    >>> expected_library_composition(library_df,
    ...                              {"single-1": 3, "spiked": 1})
      genotype  pool_fraction  bulk_fraction
    0       wt         0.3125            0.6
    1     M42I         0.5625            1.0
    2     H74A         0.1250            0.0
    """

    for col in ("library_origin", "genotype", "weight"):
        if col not in library_df.columns:
            raise ValueError(f"library_df needs a '{col}' column.")
    _check_origin_keys(library_df, library_mixture, "library_mixture")

    df = library_df[["library_origin", "genotype", "weight"]].copy()
    df["library_origin"] = df["library_origin"].astype(str)
    df["genotype"] = df["genotype"].astype(str)

    mixture = {k: float(v) for k, v in library_mixture.items()}
    if any(v < 0 for v in mixture.values()) or sum(mixture.values()) <= 0:
        raise ValueError(
            "library_mixture values must be non-negative with a positive sum."
        )

    # Normalize each origin's weights so they are shares even if the caller
    # passed an unnormalized weight column.
    origin_total = df.groupby("library_origin")["weight"].transform("sum")
    df["share"] = np.where(origin_total > 0, df["weight"] / origin_total, 0.0)
    df["mass"] = df["library_origin"].map(mixture) * df["share"]
    df["is_bulk"] = df["library_origin"] != spiked_origin
    df["bulk_mass"] = df["mass"].where(df["is_bulk"], 0.0)

    # Keep genotypes in order of first appearance so output order is stable.
    order = pd.unique(df["genotype"])
    per_geno = (df.groupby("genotype", sort=False)[["mass", "bulk_mass"]]
                .sum()
                .reindex(order))

    total_mass = per_geno["mass"].sum()
    out = pd.DataFrame({
        "genotype": per_geno.index,
        "pool_fraction": (per_geno["mass"] / total_mass).to_numpy(),
        "bulk_fraction": np.divide(per_geno["bulk_mass"].to_numpy(),
                                   per_geno["mass"].to_numpy(),
                                   out=np.full(len(per_geno), np.nan),
                                   where=per_geno["mass"].to_numpy() > 0),
    })
    return out.reset_index(drop=True)


def scale_library_design(reference_library_df,
                         target_library_df,
                         library_mixture,
                         transform_sizes,
                         cfu0=None,
                         total_num_reads=None):
    """
    Scale a library design onto a smaller (or larger) library.

    Simulating a full-size library is expensive, so simulations use libraries
    with fewer degenerate sites. Copying the real ``library_mixture`` and
    ``transform_sizes`` onto a smaller library changes how abundant each
    genotype is and what fraction of a spiked genotype's cells come from the
    bulk. This function instead preserves, for every library origin, the
    quantities that are defined per codon-level sequence:

    - abundance of each sequence relative to every other sequence, so a
      genotype encoded by the same number of sequences in each origin in both
      libraries keeps its relative abundance and its bulk fraction,
    - transformants per sequence (the transformation bottleneck),
    - starting cells per sequence (``cfu0``) and reads per sequence
      (``total_num_reads``).

    Parameters
    ----------
    reference_library_df : pandas.DataFrame
        The library the design was made for (for example the real library),
        with ``library_origin`` and ``degeneracy`` columns, as written by
        ``LibraryManager.build_library_df``.
    target_library_df : pandas.DataFrame
        The library to scale onto (for example the simulated library), same
        columns and the same set of origins.
    library_mixture : dict
        Reference ``library_mixture``, keyed by origin.
    transform_sizes : dict
        Reference ``transform_sizes`` (transformants per origin).
    cfu0 : float, optional
        Reference starting cells for the whole pool.
    total_num_reads : int, optional
        Reference total sequencing reads.

    Returns
    -------
    dict
        Keys ``library_mixture`` (dict of float), ``transform_sizes`` (dict of
        int, at least 1), ``cfu0`` (float or None) and ``total_num_reads``
        (int or None), for the target library.

    Notes
    -----
    Mixture values keep the reference's units, so an origin with the same
    number of sequences in both libraries (typically ``spiked``) keeps its
    value. ``cfu0`` and ``total_num_reads`` scale by the ratio of total pool
    mass, which holds cells and reads per sequence fixed in every origin at
    once. Reads are assumed proportional to cells at the start of the
    experiment.
    """

    for name, df in (("reference_library_df", reference_library_df),
                     ("target_library_df", target_library_df)):
        if "library_origin" not in df.columns:
            raise ValueError(f"{name} needs a 'library_origin' column.")

    ref_seqs = _origin_sequence_counts(reference_library_df)
    tgt_seqs = _origin_sequence_counts(target_library_df)
    if set(ref_seqs) != set(tgt_seqs):
        raise ValueError(
            "Reference and target libraries must have the same origins: "
            f"{sorted(ref_seqs)} vs {sorted(tgt_seqs)}."
        )
    _check_origin_keys(reference_library_df, library_mixture, "library_mixture")
    _check_origin_keys(reference_library_df, transform_sizes, "transform_sizes")

    ratio = {k: tgt_seqs[k] / ref_seqs[k] for k in ref_seqs}

    new_mixture = {k: float(library_mixture[k]) * ratio[k] for k in ref_seqs}
    new_transform = {k: max(1, int(round(float(transform_sizes[k]) * ratio[k])))
                     for k in ref_seqs}

    mass_ratio = sum(new_mixture.values()) / sum(float(v) for v in library_mixture.values())
    new_cfu0 = None if cfu0 is None else float(cfu0) * mass_ratio
    new_reads = (None if total_num_reads is None
                 else max(1, int(round(float(total_num_reads) * mass_ratio))))

    return {"library_mixture": new_mixture,
            "transform_sizes": new_transform,
            "cfu0": new_cfu0,
            "total_num_reads": new_reads}


def estimate_library_mixture(library_df,
                             observed,
                             exclude_genotypes=("wt",)):
    """
    Estimate the realized ``library_mixture`` from observed abundances.

    A sample of the pooled library (for example a pre-split sample) gives each
    genotype's abundance. Under the design, a genotype's expected abundance is
    ``sum_L mixture_L * weight_gL``, which is linear in the unknown mixture.
    Genotypes are grouped by the set of origins that encode them (for example
    tile-1 singles are encoded by ``single-1`` and ``double-1-2``); summing
    within a group averages over assembly skew and sampling noise, and the
    mixture is the non-negative least-squares solution over the group sums.

    Parameters
    ----------
    library_df : pandas.DataFrame
        Columns ``library_origin``, ``genotype`` and ``weight``, as written by
        ``LibraryManager.build_library_df``.
    observed : pandas.Series or pandas.DataFrame
        Observed abundance (cells or reads, not logs) per genotype: a Series
        indexed by genotype, or a DataFrame with ``genotype`` and
        ``abundance`` columns. Library genotypes that are absent count as
        zero.
    exclude_genotypes : iterable of str, default ("wt",)
        Genotypes left out of the fit. wt is excluded by default because it is
        often over-represented for reasons outside the design (for example
        unmutated template carried through cloning).

    Returns
    -------
    mixture : dict
        Estimated mixture, keyed by origin, summing to 1 over the origins.
    groups : pandas.DataFrame
        One row per group: ``origins`` (tuple), ``n_genotypes``,
        ``observed_share`` and ``fitted_share`` (shares of the fitted
        abundance).
    unexplained_fraction : float
        Fraction of the total observed abundance on genotypes that are not in
        ``library_df`` (not designed), which the fit ignores.

    Notes
    -----
    Origins that no group can distinguish get a mixture of 0 or an arbitrary
    split, so check ``groups`` before trusting a value. An origin whose
    genotypes are all excluded or absent is estimated at 0.
    """

    from scipy.optimize import nnls

    for col in ("library_origin", "genotype", "weight"):
        if col not in library_df.columns:
            raise ValueError(f"library_df needs a '{col}' column.")

    if isinstance(observed, pd.DataFrame):
        for col in ("genotype", "abundance"):
            if col not in observed.columns:
                raise ValueError(f"observed needs a '{col}' column.")
        observed = observed.set_index(observed["genotype"].astype(str))["abundance"]
    observed = pd.Series(observed, dtype=float)
    observed.index = observed.index.astype(str)
    if observed.index.duplicated().any():
        dups = sorted(set(observed.index[observed.index.duplicated()]))[:5]
        raise ValueError(f"observed has duplicate genotypes, e.g. {dups}.")
    if (observed < 0).any() or not np.isfinite(observed).all():
        raise ValueError("observed abundances must be finite and non-negative.")

    df = library_df[["library_origin", "genotype", "weight"]].copy()
    df["library_origin"] = df["library_origin"].astype(str)
    df["genotype"] = df["genotype"].astype(str)
    origin_total = df.groupby("library_origin")["weight"].transform("sum")
    df["share"] = np.where(origin_total > 0, df["weight"] / origin_total, 0.0)

    origins = sorted(df["library_origin"].unique())
    weights = (df.pivot_table(index="genotype", columns="library_origin",
                              values="share", aggfunc="sum", fill_value=0.0)
               .reindex(columns=origins, fill_value=0.0))

    total_observed = observed.sum()
    if total_observed <= 0:
        raise ValueError("observed abundances sum to zero.")
    unexplained = observed[~observed.index.isin(weights.index)].sum() / total_observed

    exclude = set(map(str, exclude_genotypes or ()))
    weights = weights[~weights.index.isin(exclude)]
    obs = observed.reindex(weights.index).fillna(0.0)

    key = weights.gt(0).apply(lambda row: tuple(o for o in origins if row[o]), axis=1)
    group_w = weights.groupby(key).sum()
    group_obs = obs.groupby(key).sum()
    group_n = key.value_counts()

    A = group_w[origins].to_numpy()
    b = group_obs.reindex(group_w.index).to_numpy()
    scale = b.sum()
    if scale <= 0:
        raise ValueError("No observed abundance on the included library genotypes.")
    m, _ = nnls(A, b / scale)
    if m.sum() <= 0:
        raise ValueError("Could not estimate a mixture (all components zero).")

    fitted = A @ m
    groups = pd.DataFrame({
        "origins": list(group_w.index),
        "n_genotypes": group_n.reindex(group_w.index).to_numpy(),
        "observed_share": b / scale,
        "fitted_share": fitted / fitted.sum(),
    }).sort_values("observed_share", ascending=False).reset_index(drop=True)

    mixture = {o: float(v / m.sum()) for o, v in zip(origins, m)}
    return mixture, groups, float(unexplained)


LIBRARY_COMPOSITION_COLUMNS = ["genotype",
                               "is_wt",
                               "in_spiked_origin",
                               "pool_fraction",
                               "bulk_fraction",
                               "origins"]


def library_composition_table(library_config,
                              spiked_origin=SPIKED_ORIGIN):
    """
    Build the per-genotype library composition table from a library config.

    This is the bridge between the hand-written library description (the same
    YAML handed to ``tfs-process-fastq``) and the model, which never reads the
    YAML itself. ``tfs-configure-model`` writes the returned table to
    ``{out_prefix}_library.csv``; ``tfs-fit-model`` reads that snapshot.

    Parameters
    ----------
    library_config : str or dict
        Path to (or parsed contents of) the library YAML. Must carry the keys
        ``LibraryManager`` requires (``reading_frame``,
        ``first_amplicon_residue``, ``wt_seq``, ``degen_sites``, ``tiles``,
        ``tile_combos``), plus ``library_mixture``. ``spiked_seqs`` is
        optional but is what makes a genotype congression-free. Any other keys
        (a full ``tfs-simulate`` config, say) are ignored.
    spiked_origin : str, default "spiked"
        Library origin holding the monoclonal spiked controls.

    Returns
    -------
    pandas.DataFrame
        One row per amino-acid genotype in the library, with columns:

        - ``genotype``: standardized genotype name ("wt", "M42I", ...)
        - ``is_wt``: True for the wildtype genotype
        - ``in_spiked_origin``: True if the genotype is encoded by a spiked
          sequence. Note this is *not* the same as congression-free: wt and
          the spiked single mutants are also encoded in the bulk
          sub-libraries, which is what ``bulk_fraction`` records.
        - ``pool_fraction``: expected fraction of pooled cells carrying the
          genotype (sums to 1)
        - ``bulk_fraction``: expected fraction of the genotype's cells coming
          from bulk sub-libraries (0 = pure spike, 1 = pure bulk)
        - ``origins``: "|"-joined sub-libraries encoding the genotype

    Raises
    ------
    ValueError
        If ``library_mixture`` is missing, or if its keys do not exactly match
        the library's origins.

    Notes
    -----
    ``pool_fraction`` and ``bulk_fraction`` are *design* expectations. They
    ignore library assembly skew and the finite number of transformants, and
    nothing downstream checks them against the data. The realized mixture can
    depart from the design substantially, so the values in ``library_mixture``
    should be the best available estimate of what actually went into the pool.
    """

    from tfscreen.util import read_yaml
    from tfscreen.genetics.library_manager import LibraryManager

    config = read_yaml(library_config)
    if "library_mixture" not in config:
        raise ValueError(
            "The library config must have a 'library_mixture' key giving the "
            "relative amount of each sub-library in the pool (keys matching "
            "'tile_combos', plus 'spiked' if 'spiked_seqs' is defined)."
        )

    library_df = LibraryManager(config).build_library_df()
    composition = expected_library_composition(library_df,
                                               config["library_mixture"],
                                               spiked_origin=spiked_origin)

    library_df = library_df.copy()
    library_df["genotype"] = library_df["genotype"].astype(str)
    library_df["library_origin"] = library_df["library_origin"].astype(str)

    origins = (library_df
               .groupby("genotype", sort=False)["library_origin"]
               .apply(lambda v: "|".join(sorted(set(v)))))
    in_spiked = (library_df
                 .assign(_s=library_df["library_origin"] == spiked_origin)
                 .groupby("genotype", sort=False)["_s"]
                 .any())

    composition["genotype"] = composition["genotype"].astype(str)
    composition["is_wt"] = composition["genotype"] == "wt"
    composition["in_spiked_origin"] = (composition["genotype"]
                                       .map(in_spiked)
                                       .fillna(False)
                                       .astype(bool))
    composition["origins"] = composition["genotype"].map(origins)

    return composition[LIBRARY_COMPOSITION_COLUMNS].reset_index(drop=True)


def write_library_composition(composition, path):
    """
    Write a library composition table to a CSV snapshot.

    Parameters
    ----------
    composition : pandas.DataFrame
        Table returned by ``library_composition_table``.
    path : str
        Output CSV path.
    """

    missing = [c for c in LIBRARY_COMPOSITION_COLUMNS
               if c not in composition.columns]
    if missing:
        raise ValueError(f"composition is missing columns: {missing}.")

    composition[LIBRARY_COMPOSITION_COLUMNS].to_csv(path, index=False)


def read_library_composition(path):
    """
    Read a library composition table written by ``write_library_composition``.

    Parameters
    ----------
    path : str
        Path to the CSV snapshot (``{out_prefix}_library.csv``).

    Returns
    -------
    pandas.DataFrame
        The table, with ``is_wt`` and ``in_spiked_origin`` as bools.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing or genotypes are duplicated.
    """

    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Library composition file not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in LIBRARY_COMPOSITION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{path}' is missing columns {missing}. It should be the "
            f"{{out_prefix}}_library.csv written by tfs-configure-model."
        )

    df["genotype"] = df["genotype"].astype(str)
    if df["genotype"].duplicated().any():
        dups = sorted(set(df["genotype"][df["genotype"].duplicated()]))[:5]
        raise ValueError(f"'{path}' has duplicate genotypes, e.g. {dups}.")

    for col in ("is_wt", "in_spiked_origin"):
        df[col] = df[col].astype(bool)

    return df
