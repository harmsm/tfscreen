"""
Load structural ensemble data for struct theta models.

``load_struct_ensemble()`` reads the HDF5 file produced by
``scripts/generate_struct_ensemble.py``, builds the feature matrix and
contact-distance arrays, and returns a dict whose keys map directly to the
``struct_*`` fields of ``GrowthData`` / ``BindingData``.

HDF5 schema (written by generate_struct_ensemble.py):
    structure_names          str dataset: names of all structures in the file
    {name}/logP              float32 (L, 20): per-residue log-probabilities
    {name}/residue_nums      int32   (L,):   PDB residue numbers (sorted)
    {name}/dist_matrix       float32 (L, L): min Cα-Cα distances in Å
    {name}/n_chains_bearing_mut  int32 scalar: chains carrying the mutated residue
"""

import numpy as np
import h5py
from .features import build_feature_matrix, parse_mutation


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
    return {
        'logP':                 grp['logP'][:].astype(np.float32),
        'residue_nums':         grp['residue_nums'][:].astype(np.int32),
        'dist_matrix':          grp['dist_matrix'][:].astype(np.float32),
        'n_chains_bearing_mut': int(grp['n_chains_bearing_mut'][()]),
    }


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

def load_struct_ensemble(h5_path, mut_labels, pair_labels=None):
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
