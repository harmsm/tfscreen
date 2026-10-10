"""
Load structural data for struct theta models.

Two loaders are provided:

``load_struct_ensemble()`` — reads the HDF5 file produced by
``scripts/generate_struct_ensemble.py``, builds the LigandMPNN feature
matrix and contact-distance arrays, and returns a dict whose keys map
directly to the ``struct_*`` fields of ``GrowthData`` / ``BindingData``.

``load_ddG_prior_csv()`` — reads a CSV file with a mutation column (``mut``,
``Mutation`` or ``mutation``) and one column per structure (e.g. ``H``,
``HO``, ``L``, ``LO``, ``HE2``, ``LE2``).  Returns the same dict shape but
with the (M, S) ddG values, in kT, stored in ``struct_features``;
``struct_n_chains`` and the contact arrays are None.  Used by the PddG
models.  Energy units, the sign convention, column mapping, residue
numbering and coverage checks are described in its docstring and in
``normalize_thermo_options()``.

``load_thermo_data()`` picks the loader from the theta component name.

HDF5 schema (written by generate_struct_ensemble.py):
    structure_names          str dataset: names of all structures in the file
    {name}/logP              float32 (L, 20): per-residue log-probabilities
    {name}/residue_nums      int32   (L,):   PDB residue numbers (sorted)
    {name}/dist_matrix       float32 (L, L): min Cα-Cα distances in Å
    {name}/n_chains_bearing_mut  int32 scalar: chains carrying the mutated residue
    {name}/wt_aa             str     (L,):   optional one-letter wt residues
"""

import warnings

import numpy as np
import h5py
import pandas as pd
from .features import build_feature_matrix, parse_mutation, _MUT_PATTERN


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_h5_structure(hf, name):
    """Read one structure group from an open HDF5 file and validate keys."""
    required = {'logP', 'residue_nums', 'dist_matrix', 'n_chains_bearing_mut'}
    if name not in hf:
        raise KeyError(
            f"Structure group {name!r} not found in HDF5 file. "
            f"Available groups: {sorted(k for k in hf.keys() if k != 'structure_names')}."
        )
    grp = hf[name]
    missing = required - set(grp.keys())
    if missing:
        raise KeyError(
            f"Structure group {name!r} is missing required dataset(s): "
            f"{sorted(missing)}. Found: {sorted(grp.keys())}."
        )
    out = {
        'logP':                 grp['logP'][:].astype(np.float32),
        'residue_nums':         grp['residue_nums'][:].astype(np.int32),
        'dist_matrix':          grp['dist_matrix'][:].astype(np.float32),
        'n_chains_bearing_mut': int(grp['n_chains_bearing_mut'][()]),
    }
    # Optional (L,) one-letter wt residues, written by newer
    # generate_struct_ensemble.py; used to check the residue numbering.
    if 'wt_aa' in grp:
        out['wt_aa'] = [a.decode() if isinstance(a, bytes) else str(a)
                        for a in grp['wt_aa'][:]]
    return out


def _check_h5_coverage(h5_path, struct_names, struct_data, mut_labels, offset):
    """
    Check that every library mutation's residue is in every structure and
    that the wt residues (when the file stores ``wt_aa``) agree.

    Unlike the ΔΔG CSV there is no neutral value for a missing residue's
    LigandMPNN features, so any missing residue is an error.
    """
    M = len(mut_labels)
    if M == 0:
        return
    source = f"Structural ensemble {h5_path!r}"
    lib_parsed = _parse_library_labels(mut_labels)
    for sname, sdata in zip(struct_names, struct_data):
        res_nums = sdata['residue_nums']
        present = set(res_nums.tolist())
        file_wt_by_pos = {}
        if 'wt_aa' in sdata:
            file_wt_by_pos = {int(r) - offset: a
                              for r, a in zip(res_nums, sdata['wt_aa'])
                              if a and a != 'X'}
            _check_wt_identity(lib_parsed, mut_labels, file_wt_by_pos, offset,
                               f"{source}, structure {sname!r}")
        missing = [lbl for lbl, (_, pos) in zip(mut_labels, lib_parsed)
                   if pos is not None and pos not in present]
        if missing:
            n_cov = M - len(missing)
            span = (f"{int(res_nums.min())}-{int(res_nums.max())}"
                    if res_nums.size else "none")
            raise ValueError(
                f"{source}, structure {sname!r}: residues for {len(missing)} "
                f"of {M} library mutations are absent ({n_cov}/{M} covered; "
                f"structure residues {span} after residue_offset {offset}): "
                f"{_truncated(missing)}. Every library residue needs "
                f"LigandMPNN features; check the residue numbering "
                f"(thermo_options residue_offset) or rebuild the ensemble."
                + _offset_hint(lib_parsed, file_wt_by_pos, offset)
            )
    if not any('wt_aa' in d for d in struct_data):
        print(f"{source}: no 'wt_aa' datasets, so wild-type identity is not "
              f"checked against the library.", flush=True)
    print(f"{source}: all {M} library mutations covered in "
          f"{len(struct_names)} structure(s).", flush=True)


def _build_contact_arrays(struct_data_list, mut_labels, pair_labels):
    """
    Build (P, 2) mutation-index array and (P, S) distance matrix for all pairs.

    For each pair label the two mutation indices are looked up in ``mut_labels``
    and the min Cα-Cα distance is read from each structure's dist_matrix at the
    corresponding residue positions.  Missing contacts (residue absent from a
    structure) default to 999.0 Å.

    Parameters
    ----------
    struct_data_list : list of dict
    mut_labels : sequence of str
    pair_labels : sequence of str
        Each element is "<MutA>+<MutB>" or "<MutA>/<MutB>".

    Returns
    -------
    pair_mut_idx  : np.ndarray (P, 2) int32
    distances     : np.ndarray (P, S) float32
    """
    P = len(pair_labels)
    S = len(struct_data_list)

    mut_pos = [parse_mutation(lbl)[1] for lbl in mut_labels]
    label_to_idx = {lbl: i for i, lbl in enumerate(mut_labels)}

    pair_mut_idx = np.zeros((P, 2), dtype=np.int32)
    distances    = np.full((P, S), 999.0, dtype=np.float32)

    for p_idx, pair_label in enumerate(pair_labels):
        sep = '+' if '+' in pair_label else '/'
        parts = pair_label.split(sep)
        if len(parts) != 2:
            raise ValueError(
                f"Cannot parse pair label {pair_label!r}: "
                f"expected '<MutA>+<MutB>' or '<MutA>/<MutB>' format."
            )
        mi = label_to_idx.get(parts[0])
        mj = label_to_idx.get(parts[1])
        if mi is None:
            raise ValueError(
                f"Mutation {parts[0]!r} from pair {pair_label!r} not found "
                f"in mut_labels."
            )
        if mj is None:
            raise ValueError(
                f"Mutation {parts[1]!r} from pair {pair_label!r} not found "
                f"in mut_labels."
            )
        pair_mut_idx[p_idx, 0] = mi
        pair_mut_idx[p_idx, 1] = mj

        pos_i = mut_pos[mi]
        pos_j = mut_pos[mj]

        for s_idx, sdata in enumerate(struct_data_list):
            res_nums    = sdata['residue_nums']
            dist_matrix = sdata['dist_matrix']
            hits_i = np.where(res_nums == pos_i)[0]
            hits_j = np.where(res_nums == pos_j)[0]
            if hits_i.size > 0 and hits_j.size > 0:
                distances[p_idx, s_idx] = dist_matrix[hits_i[0], hits_j[0]]

    return pair_mut_idx, distances


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_struct_ensemble(h5_path, mut_labels, pair_labels=None,
                         thermo_options=None):
    """
    Load a structural ensemble HDF5 file and build all arrays for DataClass
    struct fields.

    Structure names and ordering are read from the file itself
    (``structure_names`` dataset), so no external name list is required.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 ensemble file produced by
        ``scripts/generate_struct_ensemble.py``.
    mut_labels : list of str
        Mutation labels in "A42G" format.
    pair_labels : list of str or None
        Pair labels in "A42G+C18R" or "A42G/C18R" format.  If None or
        empty, contact arrays are not built and the corresponding return
        values are None.
    thermo_options : dict, optional
        Only ``residue_offset`` applies (see :func:`normalize_thermo_options`);
        it is added to every structure's ``residue_nums``.  Every library
        residue must be present in every structure, and when the file stores
        ``wt_aa`` its wt residues must match the library's.

    Returns
    -------
    dict with keys:
        struct_names            : tuple of str, length S
        struct_features         : np.ndarray (M, S, 60) float32
        struct_n_chains         : np.ndarray (S,) int32
        struct_contact_pair_idx : np.ndarray (P, 2) int32, or None
        struct_contact_distances: np.ndarray (P, S) float32, or None
    """
    with h5py.File(h5_path, 'r') as hf:
        if 'structure_names' not in hf:
            raise KeyError(
                f"HDF5 file {h5_path!r} is missing the 'structure_names' dataset. "
                "Was it created by generate_struct_ensemble.py?"
            )
        struct_names = [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in hf['structure_names'][:]
        ]
        struct_data = [_read_h5_structure(hf, name) for name in struct_names]

    offset = normalize_thermo_options(thermo_options)["residue_offset"]
    if offset:
        for d in struct_data:
            d['residue_nums'] = (d['residue_nums'] + offset).astype(np.int32)
    _check_h5_coverage(h5_path, struct_names, struct_data, mut_labels, offset)

    features = build_feature_matrix(struct_data, struct_names, mut_labels)
    n_chains  = np.array(
        [d['n_chains_bearing_mut'] for d in struct_data], dtype=np.int32
    )

    if pair_labels is not None and len(pair_labels) > 0:
        pair_mut_idx, distances = _build_contact_arrays(
            struct_data, mut_labels, pair_labels
        )
    else:
        pair_mut_idx = None
        distances    = None

    return {
        'struct_names':             tuple(struct_names),
        'struct_features':          features,
        'struct_n_chains':          n_chains,
        'struct_contact_pair_idx':  pair_mut_idx,
        'struct_contact_distances': distances,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Loader options: energy units, column mapping, numbering, coverage
# ──────────────────────────────────────────────────────────────────────────────

# Gas constant in the two supported molar energy units.
_R_KCAL = 1.987204259e-3   # kcal/(mol·K)
_R_KJ   = 8.314462618e-3   # kJ/(mol·K)

_UNIT_ALIASES = {
    "kt": "kT",
    "kcal/mol": "kcal/mol", "kcal": "kcal/mol", "kcal_per_mol": "kcal/mol",
    "kj/mol": "kJ/mol", "kj": "kJ/mol", "kj_per_mol": "kJ/mol",
}

# Defaults reproduce the behaviour before these options existed: ΔΔG values
# are used as-is (already in units of kT), file residue numbers are the
# library's, and the structure columns carry the model's names.
THERMO_OPTION_DEFAULTS = {
    "ddG_units": "kT",
    "temperature_K": 298.15,
    "column_map": None,
    "residue_offset": 0,
    "max_missing_fraction": 0.1,
}

# Valid mutation-column names in a ΔΔG CSV, in lookup order.
_MUT_COLUMN_ALIASES = ("mut", "Mutation", "mutation")

_MAX_LISTED = 10   # longest list of labels printed in one message


def _parse_column_map(value):
    """Turn a ``{model_name: csv_column}`` dict or ``"HD=HO,LE2=LOE2"`` string into a dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        out = {str(k): str(v) for k, v in value.items()}
    elif isinstance(value, str):
        out = {}
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    f"column_map entry {item!r} is not of the form "
                    f"'model_name=csv_column'."
                )
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    else:
        raise ValueError(
            f"column_map must be a dict or a 'model_name=csv_column,...' "
            f"string; got {type(value).__name__}."
        )
    return out or None


def normalize_thermo_options(options=None):
    """
    Validate a ``thermo_options`` mapping and fill in defaults.

    Keys (all optional):

    ``ddG_units``
        Energy unit of the ΔΔG values in a PddG CSV: ``kT`` (default; values
        are used as-is), ``kcal/mol`` or ``kJ/mol``.  Molar values are divided
        by RT once at load, so the model always works in kT (natural-log K).
    ``temperature_K``
        Temperature for that conversion (default 298.15 K).
    ``column_map``
        ``{model_structure_name: file_column}`` (or the string
        ``"HD=HO,LE2=LOE2"``) naming the CSV column that supplies a model
        structure.  ``mut`` may be mapped too.  Unmapped names are looked up
        under their own name.
    ``residue_offset``
        Integer added to every residue number in the file (CSV mutation labels
        or HDF5 ``residue_nums``) to give the library's numbering, e.g. 1 for a
        0-based file against a 1-based library.
    ``max_missing_fraction``
        Warn when more than this fraction of library mutations has no ΔΔG in
        the CSV (default 0.1).  Zero coverage is always an error.

    Returns
    -------
    dict
        A new dict with every key present and ``ddG_units`` canonicalised.
    """
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError(
            f"thermo_options must be a dict; got {type(options).__name__}."
        )
    unknown = set(options) - set(THERMO_OPTION_DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown thermo_options key(s) {sorted(unknown)}; "
            f"valid keys are {sorted(THERMO_OPTION_DEFAULTS)}."
        )
    out = dict(THERMO_OPTION_DEFAULTS)
    out.update({k: v for k, v in options.items() if v is not None})

    units = _UNIT_ALIASES.get(str(out["ddG_units"]).strip().lower())
    if units is None:
        raise ValueError(
            f"ddG_units {out['ddG_units']!r} not recognised; use 'kT', "
            f"'kcal/mol' or 'kJ/mol'."
        )
    out["ddG_units"] = units

    temperature = float(out["temperature_K"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature_K must be > 0 (Kelvin); got {temperature}.")
    out["temperature_K"] = temperature

    out["column_map"] = _parse_column_map(out["column_map"])

    offset = out["residue_offset"]
    if isinstance(offset, bool) or int(offset) != offset:
        raise ValueError(f"residue_offset must be an integer; got {offset!r}.")
    out["residue_offset"] = int(offset)

    frac = float(out["max_missing_fraction"])
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"max_missing_fraction must be in [0, 1]; got {frac}.")
    out["max_missing_fraction"] = frac
    return out


def ddG_to_kT_factor(ddG_units="kT", temperature_K=298.15):
    """
    Multiplier converting ΔΔG in ``ddG_units`` to units of kT (1/RT).

    ``kT`` gives exactly 1.0; ``kcal/mol`` at 298.15 K gives about 1.688.
    """
    units = normalize_thermo_options({"ddG_units": ddG_units,
                                      "temperature_K": temperature_K})
    if units["ddG_units"] == "kT":
        return 1.0
    R = _R_KCAL if units["ddG_units"] == "kcal/mol" else _R_KJ
    return 1.0 / (R * units["temperature_K"])


def _truncated(labels, n=_MAX_LISTED):
    labels = list(labels)
    text = ", ".join(str(x) for x in labels[:n])
    if len(labels) > n:
        text += f", ... ({len(labels) - n} more)"
    return text


def _parse_library_labels(mut_labels):
    """(wt_aa, resnum) per library label; (None, None) for an unparseable one."""
    out = []
    for lbl in mut_labels:
        m = _MUT_PATTERN.match(str(lbl))
        out.append((m.group(1), int(m.group(2))) if m else (None, None))
    return out


def _offset_hint(lib_parsed, file_wt_by_pos, current_offset, search=50):
    """
    Look for a residue offset under which every library wt residue matches the
    file.  ``file_wt_by_pos`` maps file-native residue number → wt letter.
    Returns a sentence (or "") suggesting ``residue_offset``.
    """
    lib = [(wt, pos) for wt, pos in lib_parsed if wt is not None]
    if not lib or not file_wt_by_pos:
        return ""
    best = None
    for k in range(-search, search + 1):
        compared = matched = 0
        for wt, pos in lib:
            fwt = file_wt_by_pos.get(pos - k)
            if fwt is None:
                continue
            compared += 1
            matched += (fwt == wt)
        if compared and matched == compared:
            if best is None or matched > best[1]:
                best = (k, matched)
    if best is None or best[0] == current_offset:
        return ""
    return (f" The library's wt residues all match the file if residue_offset "
            f"is {best[0]} (file residue + {best[0]} = library residue; "
            f"{best[1]} library mutations then line up); set "
            f"thermo_options residue_offset: {best[0]} if that numbering is "
            f"right.")


def _check_wt_identity(lib_parsed, mut_labels, file_wt_by_pos, offset, source):
    """Error if any library wt residue disagrees with the file at its position."""
    mismatches = []
    for lbl, (wt, pos) in zip(mut_labels, lib_parsed):
        if wt is None:
            continue
        fwt = file_wt_by_pos.get(pos - offset)
        if fwt is not None and fwt != wt:
            mismatches.append(f"{lbl} (file wt {fwt}{pos - offset})")
    if mismatches:
        raise ValueError(
            f"{source}: wild-type residues disagree with the library at "
            f"{len(mismatches)} of {len(mut_labels)} library mutations "
            f"({_truncated(mismatches)}). This usually means the file and the "
            f"library number residues differently."
            + _offset_hint(lib_parsed, file_wt_by_pos, offset)
        )


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch from the theta model name
# ──────────────────────────────────────────────────────────────────────────────

def required_struct_names(theta_name):
    """
    Structure names a thermo PddG/PnnC theta component needs, or None.

    Read from the component's ``STRUCTURE_NAMES`` so there is one source.
    """
    if theta_name is None:
        return None
    from tfscreen.tfmodel.generative.registry import model_registry
    module = model_registry["theta"].get(theta_name)
    names = getattr(module, "STRUCTURE_NAMES", None)
    return tuple(names) if names is not None else None


def load_thermo_data(path, mut_labels, pair_labels=None, theta_name=None,
                     thermo_options=None):
    """
    Load ``thermo_data`` for a thermo theta component.

    A ``PddG`` component reads a ΔΔG CSV via :func:`load_ddG_prior_csv`
    (keeping only its structures) and a ``PnnC`` component an HDF5 ensemble
    via :func:`load_struct_ensemble`.  For any other (or no) component the
    file extension decides: ``.csv`` is a ΔΔG CSV, anything else HDF5.
    """
    if theta_name is not None and theta_name.endswith(".PddG"):
        is_csv = True
    elif theta_name is not None and theta_name.endswith(".PnnC"):
        is_csv = False
    else:
        is_csv = str(path).lower().endswith(".csv")
    if is_csv:
        return load_ddG_prior_csv(path, mut_labels,
                                  struct_names=required_struct_names(theta_name),
                                  thermo_options=thermo_options)
    return load_struct_ensemble(path, mut_labels, pair_labels,
                                thermo_options=thermo_options)


# ──────────────────────────────────────────────────────────────────────────────
# ΔΔG CSV
# ──────────────────────────────────────────────────────────────────────────────

def load_ddG_prior_csv(csv_path, mut_labels, struct_names=None,
                       thermo_options=None):
    """
    Load per-mutation per-structure ΔΔG prior means from a CSV file.

    Format
    ------
    One row per single mutation.  The mutation column is ``mut`` (or
    ``Mutation`` / ``mutation``), with labels like ``A42G``.  Every other
    column is a structure.  An unnamed pandas index column (``Unnamed: 0``)
    is dropped.  When ``struct_names`` is given, each required name must be a
    column (or be mapped to one with ``column_map``); further columns are
    ignored with a note.  Without it, every remaining column is kept, in file
    order.  Rows whose label is not a single mutation are ignored.

    Units and sign convention
    -------------------------
    ``ddG[m, X]`` is the change mutation ``m`` makes to the free energy of
    structure ``X`` relative to the reference state (U, at ΔΔG_U = 0), so a
    positive value destabilises X.  For an equilibrium constant
    ``K_{A→B} = [B]/[A]`` the models use
    ``Δln K_{A→B} = ΔΔG_A − ΔΔG_B``, which needs ΔΔG in units of kT.  Values
    in ``kcal/mol`` or ``kJ/mol`` (``thermo_options['ddG_units']``) are
    divided by RT once here; the default ``kT`` uses them as-is.  A predictor
    that reports stability as ΔG_fold (negative = more stable) or reports
    ΔΔG with the opposite sign must be flipped before use.

    Coverage checks
    ---------------
    File labels are matched to ``mut_labels`` after adding
    ``residue_offset`` to the file's residue numbers.  The loader prints how
    many library mutations the file covers and raises if it covers none, or
    if the file's wild-type residue at any library position disagrees with
    the library (a numbering offset).  A library mutation missing from the
    file gets a prior mean of 0.0 (no structural prediction); when more than
    ``max_missing_fraction`` of them are missing a warning lists them.

    Parameters
    ----------
    csv_path : str
    mut_labels : list of str
        Mutation labels in the order expected by the model (e.g. "A42G").
    struct_names : sequence of str, optional
        Structures the theta model needs, returned in this order.
    thermo_options : dict, optional
        See :func:`normalize_thermo_options`.

    Returns
    -------
    dict with keys:
        struct_names            : tuple of str
        struct_features         : np.ndarray (M, S) float32, ΔΔG in kT
        struct_n_chains         : None
        struct_contact_pair_idx : None
        struct_contact_distances: None
    """
    opts = normalize_thermo_options(thermo_options)
    column_map = dict(opts["column_map"] or {})
    offset = opts["residue_offset"]
    source = f"ddG CSV {csv_path!r}"

    df = pd.read_csv(csv_path)
    unnamed = [c for c in df.columns
               if str(c).startswith("Unnamed:") or str(c).strip() == ""]
    df = df.drop(columns=unnamed)

    # ── Mutation column ──────────────────────────────────────────────────────
    if "mut" in column_map:
        mut_col = column_map.pop("mut")
        if mut_col not in df.columns:
            raise ValueError(
                f"{source}: column_map names mutation column {mut_col!r}, "
                f"which is not in the file; found columns: {list(df.columns)}."
            )
    else:
        present = [c for c in _MUT_COLUMN_ALIASES if c in df.columns]
        if not present:
            raise ValueError(
                f"{source} must have a 'mut' column (or 'Mutation'/'mutation'); "
                f"found columns: {list(df.columns)}."
            )
        if len(present) > 1:
            raise ValueError(
                f"{source} has more than one mutation column {present}; "
                f"keep one, or name it with column_map: {{mut: <column>}}."
            )
        mut_col = present[0]
    other_cols = [c for c in df.columns if c != mut_col]

    # ── Structure columns ────────────────────────────────────────────────────
    if struct_names is None:
        # No model given: keep every column; a column_map entry adds (or
        # re-sources) a structure from the named column.
        if not other_cols:
            raise ValueError(
                f"{source} has no structure columns (only {mut_col!r})."
            )
        out_names = list(other_cols)
        src_cols = list(other_cols)
        for s, c in column_map.items():
            if c not in df.columns:
                raise ValueError(
                    f"{source}: column_map maps {s!r} to column {c!r}, which "
                    f"is not in the file; found columns: {other_cols}."
                )
            if s in out_names:
                src_cols[out_names.index(s)] = c
            else:
                out_names.append(s)
                src_cols.append(c)
    else:
        out_names = list(struct_names)
        unknown_map = set(column_map) - set(out_names)
        if unknown_map:
            raise ValueError(
                f"{source}: column_map key(s) {sorted(unknown_map)} are not "
                f"structures of this model {tuple(out_names)}."
            )
        src_cols = [column_map.get(s, s) for s in out_names]
        missing = [f"{s}" + (f" (mapped to {c!r})" if c != s else "")
                   for s, c in zip(out_names, src_cols) if c not in df.columns]
        if missing:
            raise ValueError(
                f"{source} is missing structure column(s) {missing} needed by "
                f"this model, which uses {tuple(out_names)}. Found columns: "
                f"{other_cols}. Name the file column that supplies a structure "
                f"with thermo_options column_map (e.g. {{HD: HO}})."
            )
        ignored = [c for c in other_cols if c not in src_cols]
        if ignored:
            print(f"{source}: ignoring column(s) not used by this model: "
                  f"{ignored}.", flush=True)
        mapped = {s: c for s, c in zip(out_names, src_cols) if s != c}
        if mapped:
            print(f"{source}: reading structure(s) from mapped column(s) "
                  f"{mapped}.", flush=True)

    values = df[src_cols].apply(pd.to_numeric, errors="coerce")
    bad = [c for c in src_cols if values[c].isna().all() and df[c].notna().any()]
    if bad:
        raise ValueError(
            f"{source}: structure column(s) {bad} are not numeric."
        )

    # ── Parse file labels, apply the residue offset ──────────────────────────
    labels = df[mut_col].astype(str).str.strip()
    file_wt_by_pos = {}
    row_of = {}
    n_unparsed = 0
    duplicates = set()
    for i, lbl in enumerate(labels):
        m = _MUT_PATTERN.match(lbl)
        if m is None:
            n_unparsed += 1
            continue
        wt, pos, mt = m.group(1), int(m.group(2)), m.group(3)
        prev = file_wt_by_pos.setdefault(pos, wt)
        if prev != wt:
            raise ValueError(
                f"{source}: residue {pos} has two wild-type residues in the "
                f"file ({prev} and {wt}); every label at a position must share "
                f"its wt residue."
            )
        key = f"{wt}{pos + offset}{mt}"
        if key in row_of:
            duplicates.add(lbl)
        row_of[key] = i
    if duplicates:
        raise ValueError(
            f"{source} lists mutation(s) more than once: {_truncated(sorted(duplicates))}."
        )
    if n_unparsed:
        print(f"{source}: ignoring {n_unparsed} row(s) whose label is not a "
              f"single mutation like 'A42G'.", flush=True)

    # ── Assemble (M, S) and check coverage ───────────────────────────────────
    M = len(mut_labels)
    S = len(src_cols)
    scale = ddG_to_kT_factor(opts["ddG_units"], opts["temperature_K"])
    vals = values.to_numpy(dtype=np.float64)
    features = np.zeros((M, S), dtype=np.float32)
    missing_labels = []
    nan_labels = []
    for m_idx, mut in enumerate(mut_labels):
        r = row_of.get(mut)
        if r is None:
            missing_labels.append(mut)
            continue
        row = vals[r]
        if np.isnan(row).any():
            nan_labels.append(mut)
            continue
        features[m_idx] = (row * scale if scale != 1.0 else row).astype(np.float32)

    if M > 0:
        lib_parsed = _parse_library_labels(mut_labels)
        _check_wt_identity(lib_parsed, mut_labels, file_wt_by_pos, offset, source)
        uncovered = missing_labels + nan_labels
        n_cov = M - len(uncovered)
        if n_cov == 0:
            raise ValueError(
                f"{source} covers none of the {M} library mutations "
                f"(library e.g. {_truncated(mut_labels, 3)}; file e.g. "
                f"{_truncated(labels.head(3).tolist(), 3)}). Check the residue "
                f"numbering and the mutation-label format."
                + _offset_hint(lib_parsed, file_wt_by_pos, offset)
            )
        unit_note = ("" if opts["ddG_units"] == "kT" else
                     f"; converted {opts['ddG_units']} to kT at "
                     f"{opts['temperature_K']:g} K (x{scale:.4f})")
        print(f"{source}: ΔΔG for {n_cov}/{M} library mutations "
              f"({100.0 * n_cov / M:.1f}%){unit_note}.", flush=True)
        frac_missing = len(uncovered) / M
        if frac_missing > opts["max_missing_fraction"]:
            parts = []
            if missing_labels:
                parts.append(f"absent: {_truncated(missing_labels)}")
            if nan_labels:
                parts.append(f"blank values: {_truncated(nan_labels)}")
            warnings.warn(
                f"{source} has no ΔΔG for {len(uncovered)} of {M} library "
                f"mutations ({100.0 * frac_missing:.1f}% > max_missing_fraction "
                f"{opts['max_missing_fraction']:g}); they get a prior mean of "
                f"0 ({'; '.join(parts)}).",
                stacklevel=2,
            )

    return {
        'struct_names':             tuple(out_names),
        'struct_features':          features,
        'struct_n_chains':          None,
        'struct_contact_pair_idx':  None,
        'struct_contact_distances': None,
    }
