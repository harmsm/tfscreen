"""
Documented corrections to a tube table, applied before tfs-process-counts.

A tube table is the record of what each sequenced tube is. When the data
show that a tube's label is wrong (its genotype composition says 1 mM IPTG
and its label says 0.1 uM, say), the fix is a correction, not an edit: an
entry in a corrections file that names the tubes, says what changes and
why, and who decided it when. ``apply_corrections`` applies the file to the
original table and writes a new one; the original is never touched. The
corrections file lives with the experiment's data, under version control.

**Design and measurement columns.** A tube table mixes two kinds of column.
The design columns (``DESIGN_COLUMNS``: replicate, conditions, times,
titrant) say what was meant to be in the tube. Every other column is a
measurement of, or a fact about, the physical sample: its ``sample`` name
(which picks its count file), its OD600 and tube totals, its barcodes, its
sequencing run. A correction moves only design columns. A swap exchanges
two tubes' design columns and leaves every measurement with the sample it
was made on. That is right for both ways a swap happens:

- the tubes were physically swapped at the bench (the tube labeled 0.1 uM
  held the 1 mM culture): the OD600, reads and counts were measured on the
  culture now known to be 1 mM, so they belong with the 1 mM label;
- the sample sheet mapped files or barcodes to the wrong tubes: the count
  file and its reads came from the culture whose design is the other label,
  and the OD600 reading was taken on that same culture when it was pulled.

Either way the sample's measurements stay together and only its label
changes. ``library`` is neither: kanR and pheS libraries are transformed
and grown separately, so a swap between libraries is refused, and a
relabel cannot change it.

**Actions.**

- ``swap``: exactly two tubes of one library exchange every design column
  the table has.
- ``relabel``: one tube gets new values for the design columns named in
  ``set``. Three relabels express a rotation of three tubes.
- ``exclude``: one or more tubes are left out of the corrected table.

No tube may appear in two corrections, so the corrections do not depend on
order: each one is applied to the original table.

**Corrections file** (YAML)::

    description: Corrections to the example screen's tube table.
    corrections:
      - id: pheS-t60-iptg-swap
        action: swap
        tubes: [pheS_pheSp4CP_0mM_t60, pheS_pheSp4CP_1mM_t60]
        reason: >
          Composition QC: the tube labeled 0 mM looks like a 1 mM tube
          and the one labeled 1 mM like a 0 mM tube.
        by: A. Researcher
        date: 2026-10-15
        evidence: qc/apparent_iptg.csv
      - id: kanR-t120-time
        action: relabel
        tubes: [kanR_kanRpkan_0mM_t120]
        set: {t_sel: 125}
        reason: The bench log says this row was pulled at 125 minutes.
        by: A. Researcher
        date: 2026-10-15

``id``, ``action``, ``tubes``, ``reason``, ``by`` and ``date`` are required;
``evidence`` (a string or list of strings, often a path relative to the
corrections file) is optional. Evidence that names an existing file is
hashed into the provenance record.
"""

import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd

from tfscreen.util.io import read_dataframe
from tfscreen.util.io.read_yaml import read_yaml
from tfscreen.util import provenance

# Columns that say what a tube was meant to be. A correction moves only
# these; every other column stays with the physical sample.
DESIGN_COLUMNS = ("replicate", "condition_pre", "t_pre", "condition_sel",
                  "t_sel", "titrant_name", "titrant_conc")

# Never changed by a correction: the sample's own name and its library.
TUBE_ID_COLUMN = "sample"
LIBRARY_COLUMN = "library"

ACTIONS = {"swap", "relabel", "exclude"}

_REQUIRED_KEYS = ("id", "action", "tubes", "reason", "by", "date")
_OPTIONAL_KEYS = ("evidence", "set")
_TOP_KEYS = ("corrections", "description")

# Columns apply_corrections adds; a table that has them was already corrected.
RECORD_COLUMNS = ("correction_id",)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _nonempty_str(value):
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if value is None:
        return ""
    return str(value).strip()


def read_corrections(corrections_file):
    """
    Read and check a corrections file.

    Parameters
    ----------
    corrections_file : str
        YAML file with a ``corrections`` list (and optional ``description``).

    Returns
    -------
    dict
        ``description`` (str or None) and ``corrections``: a list of dicts
        with ``id``, ``action``, ``tubes`` (list of str), ``reason``, ``by``,
        ``date`` (str), ``evidence`` (list of str) and ``set`` (dict, relabel
        only).

    Raises
    ------
    ValueError
        On an unknown key or action, a missing or empty required field, a
        duplicate id, the wrong number of tubes for the action, a tube named
        twice, or ``set`` on an action other than relabel.
    """
    cf = read_yaml(corrections_file)
    if not isinstance(cf, dict) or "corrections" not in cf:
        raise ValueError(f"{corrections_file}: expected a mapping with a "
                         "'corrections' list.")
    unknown = sorted(set(cf) - set(_TOP_KEYS))
    if unknown:
        raise ValueError(f"{corrections_file}: unknown top-level key(s) "
                         f"{unknown}; allowed: {list(_TOP_KEYS)}.")
    entries = cf["corrections"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{corrections_file}: 'corrections' must be a "
                         "non-empty list.")

    out = []
    seen_ids = set()
    tube_owner = {}
    for i, entry in enumerate(entries):
        where = f"{corrections_file}, correction {i + 1}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: each correction must be a mapping.")
        unknown = sorted(set(entry) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
        if unknown:
            raise ValueError(f"{where}: unknown key(s) {unknown}.")
        for key in _REQUIRED_KEYS:
            if key == "tubes":
                continue
            if not _nonempty_str(entry.get(key)):
                raise ValueError(f"{where}: '{key}' is required and must not "
                                 "be empty.")
        cid = _nonempty_str(entry["id"])
        where = f"{corrections_file}, correction '{cid}'"
        if cid in seen_ids:
            raise ValueError(f"{where}: duplicate id.")
        seen_ids.add(cid)

        action = _nonempty_str(entry["action"])
        if action not in ACTIONS:
            raise ValueError(f"{where}: unknown action '{action}'; "
                             f"allowed: {sorted(ACTIONS)}.")

        tubes = entry.get("tubes")
        if isinstance(tubes, (str, int, float)):
            tubes = [tubes]
        if not isinstance(tubes, list) or not tubes:
            raise ValueError(f"{where}: 'tubes' must name at least one tube.")
        tubes = [_nonempty_str(t) for t in tubes]
        if any(not t for t in tubes):
            raise ValueError(f"{where}: empty tube name in 'tubes'.")
        if len(set(tubes)) != len(tubes):
            raise ValueError(f"{where}: a tube is listed twice.")
        need = {"swap": 2, "relabel": 1}.get(action)
        if need is not None and len(tubes) != need:
            raise ValueError(f"{where}: {action} takes exactly {need} "
                             f"tube(s), got {len(tubes)}.")
        for t in tubes:
            if t in tube_owner:
                raise ValueError(f"{where}: tube '{t}' is also named by "
                                 f"correction '{tube_owner[t]}'. A tube may "
                                 "appear in only one correction; combine "
                                 "them.")
            tube_owner[t] = cid

        new_values = entry.get("set")
        if action == "relabel":
            if not isinstance(new_values, dict) or not new_values:
                raise ValueError(f"{where}: relabel needs a non-empty 'set' "
                                 "mapping of design columns to new values.")
            bad = sorted(set(new_values) - set(DESIGN_COLUMNS))
            if bad:
                raise ValueError(f"{where}: relabel can set only design "
                                 f"columns {list(DESIGN_COLUMNS)}; got {bad}.")
            new_values = dict(new_values)
        elif new_values is not None:
            raise ValueError(f"{where}: 'set' is only for relabel.")

        evidence = entry.get("evidence")
        if evidence is None:
            evidence = []
        elif not isinstance(evidence, list):
            evidence = [evidence]
        evidence = [_nonempty_str(e) for e in evidence if _nonempty_str(e)]

        out.append({"id": cid,
                    "action": action,
                    "tubes": tubes,
                    "reason": _nonempty_str(entry["reason"]),
                    "by": _nonempty_str(entry["by"]),
                    "date": _nonempty_str(entry["date"]),
                    "evidence": evidence,
                    "set": new_values})

    return {"description": cf.get("description"), "corrections": out}


def _same_value(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-12, atol=0.0))
    except (TypeError, ValueError):
        return str(a) == str(b)


def _coerce(value, series, where, column):
    """A relabel value in the dtype of its column (numbers stay numbers)."""
    if pd.api.types.is_numeric_dtype(series.dtype):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{where}: '{column}' is numeric in the tube "
                             f"table; got {value!r}.")
        if pd.api.types.is_integer_dtype(series.dtype):
            if not float(value).is_integer():
                raise ValueError(f"{where}: '{column}' holds integers in the "
                                 f"tube table; got {value!r}.")
            value = int(value)
        return value
    return str(value)


def correct_tube_table(tube_df, corrections):
    """
    Apply checked corrections to a tube table.

    Parameters
    ----------
    tube_df : pandas.DataFrame
        The original tube table, with a unique ``sample`` column.
    corrections : list of dict
        ``read_corrections(...)["corrections"]``.

    Returns
    -------
    corrected : pandas.DataFrame
        The tube table without excluded tubes, design columns corrected, in
        the original row order, plus ``correction_id`` (empty for tubes no
        correction touched).
    changes : pandas.DataFrame
        One row per changed cell or excluded tube: ``correction_id``,
        ``action``, ``sample``, ``column``, ``original``, ``corrected``
        (``column`` and ``corrected`` empty for an exclusion).

    Raises
    ------
    ValueError
        If the table has no or duplicate ``sample`` names, was already
        corrected, a correction names an unknown tube, a swap crosses
        libraries or would change nothing, a relabel sets a column the table
        lacks or changes nothing, or a correction leaves a tube with the same
        library and design as another tube.
    """
    if TUBE_ID_COLUMN not in tube_df.columns:
        raise ValueError("The tube table has no 'sample' column.")
    already = [c for c in RECORD_COLUMNS if c in tube_df.columns]
    if already:
        raise ValueError(f"The tube table already has {already}: it was "
                         "corrected before. Apply every correction to the "
                         "original table in one file.")

    df = tube_df.copy()
    df[TUBE_ID_COLUMN] = df[TUBE_ID_COLUMN].astype(str)
    dupes = sorted(df.loc[df[TUBE_ID_COLUMN].duplicated(), TUBE_ID_COLUMN])
    if dupes:
        raise ValueError(f"Duplicate tube name(s) in the tube table: {dupes}")
    df = df.set_index(TUBE_ID_COLUMN, drop=False)

    design = [c for c in DESIGN_COLUMNS if c in df.columns]
    original = df.copy()
    df["correction_id"] = ""
    excluded = []
    changes = []

    for corr in corrections:
        cid, action = corr["id"], corr["action"]
        where = f"correction '{cid}'"
        missing = [t for t in corr["tubes"] if t not in df.index]
        if missing:
            raise ValueError(f"{where}: no tube named {missing} in the tube "
                             "table.")

        if action == "swap":
            a, b = corr["tubes"]
            if LIBRARY_COLUMN in df.columns and not _same_value(
                    original.at[a, LIBRARY_COLUMN], original.at[b, LIBRARY_COLUMN]):
                raise ValueError(
                    f"{where}: tubes '{a}' and '{b}' are in different "
                    f"libraries ({original.at[a, LIBRARY_COLUMN]}, "
                    f"{original.at[b, LIBRARY_COLUMN]}). Libraries are "
                    "transformed and grown separately; a swap across them "
                    "is not a label swap.")
            differ = [c for c in design
                      if not _same_value(original.at[a, c], original.at[b, c])]
            if not differ:
                raise ValueError(f"{where}: tubes '{a}' and '{b}' have the "
                                 "same design; swapping them changes nothing.")
            for c in design:
                df.at[a, c] = original.at[b, c]
                df.at[b, c] = original.at[a, c]
            for t, other in ((a, b), (b, a)):
                for c in differ:
                    changes.append((cid, action, t, c, original.at[t, c],
                                    original.at[other, c]))

        elif action == "relabel":
            (t,) = corr["tubes"]
            absent = sorted(set(corr["set"]) - set(df.columns))
            if absent:
                raise ValueError(f"{where}: the tube table has no column(s) "
                                 f"{absent}.")
            changed = False
            for c, value in corr["set"].items():
                value = _coerce(value, original[c], where, c)
                if _same_value(original.at[t, c], value):
                    continue
                df.at[t, c] = value
                changes.append((cid, action, t, c, original.at[t, c], value))
                changed = True
            if not changed:
                raise ValueError(f"{where}: tube '{t}' already has those "
                                 "values; the relabel changes nothing.")

        else:  # exclude
            for t in corr["tubes"]:
                excluded.append(t)
                changes.append((cid, action, t, "", "", ""))

        for t in corr["tubes"]:
            df.at[t, "correction_id"] = cid

    df = df.drop(index=excluded)

    # A corrected tube must not end up as a copy of another tube's design.
    key = ([LIBRARY_COLUMN] if LIBRARY_COLUMN in df.columns else []) + design
    if key:
        touched = df["correction_id"] != ""
        dup = df.duplicated(subset=key, keep=False)
        clash = df.loc[dup & touched, TUBE_ID_COLUMN].tolist()
        if clash:
            groups = (df[dup].groupby(key, dropna=False)[TUBE_ID_COLUMN]
                      .apply(list).tolist())
            groups = [g for g in groups if set(g) & set(clash)]
            raise ValueError("After the corrections these tubes share a "
                             f"library and design: {groups}. Correct every "
                             "tube involved (three relabels for a rotation).")

    df = df.reset_index(drop=True)
    changes = pd.DataFrame(changes, columns=["correction_id", "action",
                                             "sample", "column", "original",
                                             "corrected"])
    return df, changes


def _evidence_record(evidence, base_dir):
    out = []
    for e in evidence:
        path = e if os.path.isabs(e) else os.path.join(base_dir, e)
        rec = {"evidence": e}
        if os.path.isfile(path):
            rec["path"] = os.path.abspath(path)
            rec["sha256"] = _sha256(path)
        out.append(rec)
    return out


def apply_corrections(tube_table_file, corrections_file,
                      out_prefix="tfs_tubes_corrected"):
    """
    Read a tube table and a corrections file and write the corrected table.

    Writes ``{out_prefix}.csv`` (the corrected tube table, for
    tfs-process-counts), ``{out_prefix}_changes.csv`` (one row per changed
    cell or excluded tube) and ``{out_prefix}_provenance.json`` (the run's
    provenance, the SHA-256 of both inputs and every evidence file, the
    checked corrections and the SHA-256 of the corrected table). The inputs
    are only read.

    Returns
    -------
    str
        Path of the corrected tube table.

    Raises
    ------
    ValueError
        If ``{out_prefix}.csv`` would overwrite an input, or on any problem
        ``read_corrections`` or ``correct_tube_table`` raises for.
    """
    out_file = f"{out_prefix}.csv"
    changes_file = f"{out_prefix}_changes.csv"
    for target in (out_file, changes_file):
        for src in (tube_table_file, corrections_file):
            if os.path.abspath(target) == os.path.abspath(src) or (
                    os.path.exists(target) and os.path.samefile(target, src)):
                raise ValueError(f"Output {target} would overwrite the input "
                                 f"{src}. Choose another out_prefix.")

    tube_hash = _sha256(tube_table_file)
    corr_hash = _sha256(corrections_file)

    cf = read_corrections(corrections_file)
    tube_df = read_dataframe(tube_table_file)
    if tube_df.index.name == TUBE_ID_COLUMN:
        tube_df = tube_df.reset_index()

    corrected, changes = correct_tube_table(tube_df, cf["corrections"])

    parent = os.path.dirname(os.path.abspath(out_file))
    os.makedirs(parent, exist_ok=True)
    corrected.to_csv(out_file, index=False)
    changes.to_csv(changes_file, index=False)

    base_dir = os.path.dirname(os.path.abspath(corrections_file))
    corrections = []
    for c in cf["corrections"]:
        rec = {k: v for k, v in c.items() if not (k == "set" and v is None)}
        rec["evidence"] = _evidence_record(c["evidence"], base_dir)
        corrections.append(rec)

    prov = provenance.get_provenance()
    prov["inputs"] = {
        "tube_table": {"path": os.path.abspath(tube_table_file),
                       "sha256": tube_hash},
        "corrections_file": {"path": os.path.abspath(corrections_file),
                             "sha256": corr_hash},
    }
    prov["description"] = cf["description"]
    prov["design_columns"] = list(DESIGN_COLUMNS)
    prov["corrections"] = corrections
    prov["outputs"] = {
        "tube_table": {"path": os.path.abspath(out_file),
                       "sha256": _sha256(out_file),
                       "num_tubes": int(len(corrected))},
        "changes": {"path": os.path.abspath(changes_file)},
    }
    prov_file = f"{out_prefix}_provenance.json"
    with open(prov_file, "w") as fh:
        json.dump(prov, fh, indent=2, default=str)
        fh.write("\n")

    n_ex = int((changes["action"] == "exclude").sum())
    print(f"Applied {len(cf['corrections'])} correction(s): "
          f"{changes['sample'].nunique() - n_ex} tube(s) relabeled, "
          f"{n_ex} excluded. Wrote {out_file}.")
    return out_file
