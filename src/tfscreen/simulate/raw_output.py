"""
A simulated experiment written in the raw formats real data come in.

``tfs-simulate`` builds its growth table in memory (``counts_to_lncfu`` on
the simulated counts). This module writes the same experiment the way a lab
hands it to the pipeline, so a simulation enters at ``tfs-process-counts``
exactly where real data do (pipeline plan step 6):

- ``{out_prefix}_counts/counts_<sample>.csv``: one file per sequenced tube in
  ``tfs-process-fastq``'s format, ``genotype`` and ``counts`` for every
  library genotype (zeros included) plus the ``__unknown__`` row (0: the
  simulator has no reads it cannot assign).
- ``{out_prefix}_tubes.csv``: the tube table, one row per sequenced tube with
  ``sample``, ``library`` and the design columns. Ground truth is left out.
- With an ``od600`` block, ``{out_prefix}_tube_od600.csv`` (``sample``,
  ``od600``) and a copy of the calibration
  (``{out_prefix}_od600_calibration.yaml``); ``tfs-process-counts`` turns the
  readings into tube totals. A tube read below the detection threshold is
  left out of the tube table, as a lab would drop it. Without an ``od600``
  block the tube table carries ``sample_cfu`` and ``sample_cfu_std`` (the
  simulator's totals).

Tubes keep the design's names when the simulation follows a design file
(``design_sample``) and no name is a substring of another; otherwise they
are fixed-width (``tube0001``). Either way the count-file glob
``counts*{sample}*.csv`` matches exactly one file per tube.
"""

import os
import shlex
import shutil

import numpy as np
import pandas as pd

from tfscreen.genetics import UNKNOWN_GENOTYPE

TUBE_COLUMNS = ["sample", "library", "replicate", "condition_pre", "t_pre",
                "condition_sel", "t_sel", "titrant_name", "titrant_conc"]


def sample_names(n):
    """Fixed-width tube names, so no name is a substring of another."""
    width = max(4, len(str(max(n, 1))))
    return [f"tube{i + 1:0{width}d}" for i in range(n)]


def _tube_names(tubes):
    """
    The design's tube names (``design_sample``) when every tube has one and no
    name is a substring of another (the count-file glob would match two
    files); else fixed-width ``tubeNNNN`` names.
    """
    if "design_sample" in tubes.columns:
        given = tubes["design_sample"]
        if given.notna().all():
            given = given.astype(str).tolist()
            clash = [a for a in given for b in given if a != b and a in b]
            if not clash and len(set(given)) == len(given):
                return given
            print(f"Design tube names cannot name count files uniquely "
                  f"(e.g. {clash[:3]}); using tube0001, ... instead.",
                  flush=True)
    return sample_names(len(tubes))


def write_raw_experiment(sample_df, counts_df, library_genotypes, out_prefix,
                         od600_config=None):
    """
    Write a simulated experiment in the raw formats (see the module docstring).

    Parameters
    ----------
    sample_df : pandas.DataFrame
        The simulated sequenced tubes, indexed by the simulator's sample id,
        with the design columns, ``sample_cfu``/``sample_cfu_std`` and, with
        an ``od600`` block, ``od600`` and ``od600_detectable``.
    counts_df : pandas.DataFrame
        ``sample`` (the simulator's id), ``genotype``, ``counts``.
    library_genotypes : sequence of str
        Every genotype of the library, the rows ``tfs-process-fastq`` writes.
    out_prefix : str
        Output prefix.
    od600_config : dict or None
        The simulate config's ``od600`` block (``calibration``,
        ``tube_volume_mL``), or None.

    Returns
    -------
    dict
        Paths written (``counts_dir``, ``tubes``, and with OD600 ``od600`` and
        ``calibration``), ``dropped`` (tubes left out for an undetectable
        reading) and ``command`` (the ``tfs-process-counts`` call that
        processes them).
    """
    tubes = sample_df.copy()
    tubes = tubes.sort_index()
    names = dict(zip(tubes.index, _tube_names(tubes)))
    tubes["sample"] = tubes.index.map(names)

    dropped = []
    if od600_config is not None and "od600_detectable" in tubes.columns:
        bad = ~tubes["od600_detectable"].astype(bool)
        dropped = tubes.loc[bad, "sample"].tolist()
        tubes = tubes.loc[~bad]

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    counts_dir = f"{out_prefix}_counts"
    os.makedirs(counts_dir, exist_ok=True)

    genotypes = list(dict.fromkeys(str(g) for g in library_genotypes))
    genotypes = [g for g in genotypes if g != UNKNOWN_GENOTYPE]
    counts = counts_df.copy()
    counts["genotype"] = counts["genotype"].astype(str)
    extra = sorted(set(counts["genotype"]) - set(genotypes) - {UNKNOWN_GENOTYPE})
    if extra:
        raise ValueError(f"Simulated counts name genotypes outside the library: "
                         f"{extra[:10]}")
    for sim_id, name in names.items():
        if name in dropped:
            continue
        c = (counts.loc[counts["sample"] == sim_id]
             .groupby("genotype", sort=False)["counts"].sum())
        out = pd.DataFrame({"genotype": genotypes + [UNKNOWN_GENOTYPE]})
        out["counts"] = (out["genotype"].map(c).fillna(0)
                         .astype(np.int64).to_numpy())
        out.to_csv(os.path.join(counts_dir, f"counts_{name}.csv"), index=False)

    paths = {"counts_dir": counts_dir, "tubes": f"{out_prefix}_tubes.csv",
             "dropped": dropped}
    columns = [c for c in TUBE_COLUMNS if c in tubes.columns]
    args = ["tfs-process-counts", paths["tubes"], counts_dir]
    if od600_config is not None:
        tubes[columns].to_csv(paths["tubes"], index=False)
        paths["od600"] = f"{out_prefix}_tube_od600.csv"
        tubes[["sample", "od600"]].to_csv(paths["od600"], index=False)
        paths["calibration"] = f"{out_prefix}_od600_calibration.yaml"
        shutil.copyfile(od600_config["calibration"], paths["calibration"])
        args += ["--od600_file", paths["od600"],
                 "--od600_calibration_file", paths["calibration"],
                 "--tube_volume_mL", f"{float(od600_config['tube_volume_mL']):g}"]
    else:
        tubes[columns + ["sample_cfu", "sample_cfu_std"]].to_csv(paths["tubes"],
                                                                 index=False)
    args += ["--out_prefix", f"{out_prefix}_processed_growth"]
    paths["command"] = " ".join(shlex.quote(a) for a in args)
    return paths
