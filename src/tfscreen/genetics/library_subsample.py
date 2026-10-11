"""
Restrict a described library to a subset of its genotypes.

A realistic library is far too large for some studies (a full Laplace
posterior needs a dense Hessian over every genotype's parameters), so the
library YAML can name a subset to simulate and fit. Four optional keys, read
here by ``tfs-simulate`` and ``tfs-configure-model --library_config`` (never
by ``tfs-process-fastq``, which counts every designed genotype):

``library_genotypes``
    Path to a text file, one genotype per line (``#`` comments allowed),
    relative to the YAML's own directory. Every library genotype not listed
    is dropped from every origin that encodes it. A listed genotype the
    library does not encode is an error.
``library_subsample``
    ``{origin: N}``: keep ``N`` genotypes of that origin, drawn at random
    (after ``library_genotypes``, if both are given). Origins not named are
    kept whole. The ``spiked`` origin cannot be subsampled.
``library_subsample_seed``
    Integer seed of the random draw; required with ``library_subsample``.
    Each origin draws from its own stream (the seed and the origin name), as
    a prefix of one fixed permutation, so N = 500 keeps a subset of the
    genotypes N = 1000 keeps and subsampling a second origin does not change
    the first's draw.
``library_subsample_depth``
    ``full_library`` (default) or ``as_written``; see
    :func:`subsample_design`.

wt and every genotype a spiked sequence encodes are always kept, in every
origin that encodes them, and do not count toward ``N``. A kept genotype
keeps all its codon-level sequences (its degeneracy), and ``weight`` is
renormalized within each origin.
"""

import os
import zlib

import numpy as np

from .genotype_sorting import standardize_genotypes

SPIKED_ORIGIN = "spiked"

SUBSAMPLE_KEYS = ("library_genotypes", "library_subsample",
                  "library_subsample_seed", "library_subsample_depth")

SUBSAMPLE_DEPTHS = ("full_library", "as_written")

SUMMARY_COLUMNS = ["library_origin", "num_genotypes", "num_genotypes_kept",
                   "num_sequences", "num_sequences_kept"]


def has_subsample(config):
    """True if ``config`` restricts the library (a list or a subsample)."""
    return (config.get("library_genotypes") is not None
            or bool(config.get("library_subsample")))


def subsample_depth(config):
    """The validated ``library_subsample_depth`` (default ``full_library``)."""
    depth = config.get("library_subsample_depth")
    if depth is None:
        return "full_library"
    if depth not in SUBSAMPLE_DEPTHS:
        raise ValueError(f"library_subsample_depth must be one of "
                         f"{list(SUBSAMPLE_DEPTHS)}, not {depth!r}.")
    return depth


def _resolve(path, base_dir):
    path = str(path)
    if os.path.isabs(path) or base_dir is None:
        return path
    return os.path.join(base_dir, path)


def _read_genotype_list(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"library_genotypes file not found: {path}")
    with open(path) as fh:
        lines = [ln.strip() for ln in fh]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise ValueError(f"library_genotypes file {path} lists no genotypes.")
    return set(str(g) for g in standardize_genotypes(lines))


def _check_seed(seed):
    if seed is None:
        raise ValueError(
            "library_subsample needs library_subsample_seed, an integer. It "
            "is not taken from 'seed': tfs-configure-model reads the library "
            "YAML without tfs-simulate's --seed, and both must draw the same "
            "genotypes.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) \
            or seed < 0:
        raise ValueError(f"library_subsample_seed must be a non-negative "
                         f"integer, not {seed!r}.")
    return int(seed)


def _check_counts(spec, origins):
    if not isinstance(spec, dict):
        raise ValueError("library_subsample must map library origins to the "
                         "number of genotypes to keep, e.g. "
                         "{double-1-2: 1000}.")
    out = {}
    for origin, n in spec.items():
        origin = str(origin)
        if origin == SPIKED_ORIGIN:
            raise ValueError("library_subsample cannot subsample the spiked "
                             "origin; every spiked genotype is kept.")
        if origin not in origins:
            raise ValueError(f"library_subsample names origin '{origin}', "
                             f"which is not in the library. Origins: "
                             f"{sorted(origins)}.")
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) \
                or n < 1:
            raise ValueError(f"library_subsample['{origin}'] must be an "
                             f"integer >= 1, not {n!r}. To leave an origin "
                             f"out, remove it from tile_combos.")
        out[origin] = int(n)
    return out


def subsample_library_df(library_df, config, base_dir=None):
    """
    Apply the subsample keys of ``config`` to a library table.

    Parameters
    ----------
    library_df : pandas.DataFrame
        One row per (library_origin, genotype) with ``degeneracy`` and
        ``weight``, as written by ``LibraryManager.build_library_df``.
    config : dict
        The library (or simulate) config; see the module docstring.
    base_dir : str, optional
        Directory a relative ``library_genotypes`` path is read against (the
        YAML's own directory). None reads it against the working directory.

    Returns
    -------
    library_df : pandas.DataFrame
        The kept rows, ``weight`` renormalized within each origin. The input
        unchanged (not a copy) when ``config`` has no subsample.
    summary : pandas.DataFrame or None
        One row per origin (``SUMMARY_COLUMNS``), or None with no subsample.
    """
    subsample_depth(config)  # validated even when unused
    if not has_subsample(config):
        return library_df, None

    df = library_df.copy()
    geno = df["genotype"].astype(str)
    origin = df["library_origin"].astype(str)
    origins = set(origin)

    protected = set(geno[origin == SPIKED_ORIGIN]) | {"wt"}
    keep = np.ones(len(df), dtype=bool)

    list_path = config.get("library_genotypes")
    if list_path is not None:
        path = _resolve(list_path, base_dir)
        listed = _read_genotype_list(path)
        missing = sorted(listed - set(geno))
        if missing:
            raise ValueError(
                f"{len(missing)} genotype(s) in library_genotypes ({path}) "
                f"are not in the described library, e.g. {missing[:10]}. "
                f"Check the residue numbering (first_amplicon_residue) and "
                f"tile_combos.")
        keep &= geno.isin(listed | protected).to_numpy()

    counts = config.get("library_subsample")
    if counts:
        counts = _check_counts(counts, origins)
        seed = _check_seed(config.get("library_subsample_seed"))
        for o in sorted(counts):
            in_o = (origin == o).to_numpy()
            cand = sorted(set(geno[in_o & keep & ~geno.isin(protected)
                                   .to_numpy()]))
            n = counts[o]
            if n > len(cand):
                raise ValueError(
                    f"library_subsample['{o}'] = {n}, but origin '{o}' has "
                    f"only {len(cand)} genotypes to draw from (wt and spiked "
                    f"genotypes are kept anyway and not counted).")
            rng = np.random.default_rng([seed, zlib.crc32(o.encode())])
            chosen = {cand[i] for i in rng.permutation(len(cand))[:n]}
            keep[in_o] &= geno[in_o].isin(chosen | protected).to_numpy()

    full = (df.assign(_o=origin)
            .groupby("_o")
            .agg(num_genotypes=("genotype", "size"),
                 num_sequences=("degeneracy", "sum")))
    out = df.loc[keep].copy()
    kept = (out.assign(_o=out["library_origin"].astype(str))
            .groupby("_o")
            .agg(num_genotypes_kept=("genotype", "size"),
                 num_sequences_kept=("degeneracy", "sum")))
    summary = full.join(kept, how="left").fillna(0)
    empty = sorted(summary.index[summary["num_genotypes_kept"] == 0])
    if empty:
        raise ValueError(
            f"The library subsample leaves origin(s) {empty} with no "
            f"genotypes. library_genotypes lists every genotype to keep (wt "
            f"and spiked genotypes are added); list some from each origin in "
            f"tile_combos, or remove the origin from tile_combos.")
    summary = (summary.astype(int)
               .rename_axis("library_origin")
               .reset_index()[SUMMARY_COLUMNS])

    out["weight"] = (out["degeneracy"]
                     / out.groupby(out["library_origin"].astype(str),
                                   observed=True)["degeneracy"]
                     .transform("sum"))
    return out.reset_index(drop=True), summary


def build_library(config, base_dir=None):
    """
    The described library with the subsample of ``config`` applied.

    Parameters
    ----------
    config : str or dict
        Path to the library (or simulate) YAML, or its parsed contents.
    base_dir : str, optional
        Directory for a relative ``library_genotypes``. Default: the YAML's
        directory when ``config`` is a path, else the working directory.

    Returns
    -------
    library_df : pandas.DataFrame
        As ``LibraryManager.build_library_df``, restricted to the subsample.
    summary : pandas.DataFrame or None
        Per-origin counts (see :func:`subsample_library_df`).
    """
    from tfscreen.util.io import read_yaml
    from .library_manager import LibraryManager

    if isinstance(config, (str, os.PathLike)) and base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(config))
    config = read_yaml(config)
    library_df = LibraryManager(config).build_library_df()
    return subsample_library_df(library_df, config, base_dir=base_dir)


def subsample_design(config, summary):
    """
    The experiment settings a subsampled library is simulated (and fit) with.

    With ``library_subsample_depth: full_library`` (the default) the config's
    ``library_mixture``, ``transform_sizes``, ``total_num_reads`` and
    ``pcr_template_molecules`` describe the full described library, and each
    kept sequence keeps the pool share, transformants, reads and PCR templates
    it has there (the per-sequence rule of
    ``library_design.scale_library_design``): each origin's mixture weight
    and transformants are multiplied by its kept fraction of sequences, and
    the reads and templates by the kept share of the pool. ``cfu0`` is not
    changed, so tube totals (and OD600) stay those of the full experiment and
    each kept genotype has ``1 / kept share`` times the cells.

    With ``as_written`` the config's values apply to the subsampled library
    as given: the subsample is the library.

    Parameters
    ----------
    config : dict
        The library or simulate config.
    summary : pandas.DataFrame or None
        From :func:`subsample_library_df`.

    Returns
    -------
    dict
        The keys to replace in ``config`` (only those present in it), plus
        ``kept_pool_share``. Empty without a subsample or with ``as_written``.
    """
    if summary is None or subsample_depth(config) == "as_written":
        return {}
    frac = dict(zip(summary["library_origin"],
                    summary["num_sequences_kept"] / summary["num_sequences"]))
    out = {}
    mixture = config.get("library_mixture")
    share = None
    if mixture is not None:
        missing = sorted(set(frac) - set(map(str, mixture)))
        if missing:
            raise ValueError(f"library_mixture has no value for origin(s) "
                             f"{missing}.")
        new = {k: float(v) * frac.get(str(k), 1.0) for k, v in mixture.items()}
        total = sum(float(v) for v in mixture.values())
        share = sum(new.values()) / total if total > 0 else 1.0
        out["library_mixture"] = new
        out["kept_pool_share"] = share
    sizes = config.get("transform_sizes")
    if sizes is not None:
        out["transform_sizes"] = {
            k: max(1, int(round(float(v) * frac.get(str(k), 1.0))))
            for k, v in sizes.items()}
    if share is not None:
        for key in ("total_num_reads", "pcr_template_molecules"):
            if config.get(key) is not None:
                out[key] = max(1, int(round(float(config[key]) * share)))
    return out


def format_subsample_summary(config, summary, design):
    """Human-readable report of a subsample for the run log."""
    lines = [f"Library subsample (library_subsample_depth: "
             f"{subsample_depth(config)}"
             + (f", seed {config['library_subsample_seed']}"
                if config.get("library_subsample") else "")
             + (f", list {config['library_genotypes']}"
                if config.get("library_genotypes") is not None else "")
             + "):"]
    width = max(len("origin"), *(len(o) for o in summary["library_origin"]))
    lines.append(f"  {'origin':<{width}}  {'genotypes':>10}  {'kept':>8}  "
                 f"{'sequences':>10}  {'kept':>8}")
    for row in summary.itertuples(index=False):
        lines.append(f"  {row.library_origin:<{width}}  "
                     f"{row.num_genotypes:>10}  {row.num_genotypes_kept:>8}  "
                     f"{row.num_sequences:>10}  {row.num_sequences_kept:>8}")
    if design:
        share = design.get("kept_pool_share")
        if share is not None:
            lines.append(f"  kept share of the pool {share:.4g}: cells per "
                         f"kept genotype x{1 / share:.3g} (cfu0 unchanged)")
        for key in ("library_mixture", "transform_sizes", "total_num_reads",
                    "pcr_template_molecules"):
            if key in design:
                val = design[key]
                if isinstance(val, dict):
                    val = {k: (float(f"{v:.6g}") if isinstance(v, float)
                               else v) for k, v in val.items()}
                lines.append(f"  {key}: {config.get(key)} -> {val}")
    return "\n".join(lines)


def subsample_record(config):
    """The subsample keys of ``config`` for a provenance record (or None)."""
    if not has_subsample(config):
        return None
    rec = {k: config.get(k) for k in SUBSAMPLE_KEYS if config.get(k) is not None}
    if "library_genotypes" in rec:
        rec["library_genotypes"] = os.path.basename(str(rec["library_genotypes"]))
    rec["library_subsample_depth"] = subsample_depth(config)
    return rec


__all__ = ["SUBSAMPLE_KEYS", "SUBSAMPLE_DEPTHS", "build_library",
           "subsample_library_df", "subsample_design",
           "format_subsample_summary", "subsample_record", "has_subsample"]
