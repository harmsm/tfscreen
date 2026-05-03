"""
Load structural ensemble data for struct theta models.

``load_struct_ensemble()`` reads per-structure NPZ files produced by
``scripts/generate_struct_ensemble.py``, builds the feature matrix and
contact-distance arrays, and returns a dict whose keys map directly to the
``struct_*`` fields of ``GrowthData`` / ``BindingData``.

NPZ format (one file per structure):
    logP               — float32 (L, 20): per-residue log-probabilities
                         (aggregated across symmetric chains)
    residue_nums       — int32   (L,):   PDB residue numbers
    dist_matrix        — float32 (L, L): min Cα-Cα distances in Å
    n_chains_bearing_mut — int scalar: chains carrying the mutated residue
"""

import numpy as np
from .features import build_feature_matrix, parse_mutation


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_npz(path):
    """Load a single structure NPZ and validate required keys."""
    data = np.load(path)
    required = {'logP', 'residue_nums', 'dist_matrix', 'n_chains_bearing_mut'}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(
            f"NPZ file {path!r} is missing required key(s): {sorted(missing)}. "
            f"Found: {sorted(data.keys())}."
        )
    return {
        'logP':                 data['logP'].astype(np.float32),
        'residue_nums':         data['residue_nums'].astype(np.int32),
        'dist_matrix':          data['dist_matrix'].astype(np.float32),
        'n_chains_bearing_mut': int(data['n_chains_bearing_mut']),
    }


def _build_contact_arrays(struct_data_list, mut_labels, pair_labels):
    """
    Build (P, 2) mutation-index array and (P, S) distance matrix for all pairs.

    For pair label "A42G+C18R", the two mutation indices are looked up in
    ``mut_labels`` and the min Cα-Cα distance is read from each structure's
    dist_matrix at the corresponding residue positions.  Missing contacts
    (residue not present in a structure) default to 999.0 Å.

    Parameters
    ----------
    struct_data_list : list of dict
    mut_labels : sequence of str
    pair_labels : sequence of str
        Each element is "<MutA>+<MutB>" or "<MutA>/<MutB>" (both accepted).

    Returns
    -------
    pair_mut_idx  : np.ndarray (P, 2) int32
    distances     : np.ndarray (P, S) float32
    """
    P = len(pair_labels)
    S = len(struct_data_list)

    # Residue number per mutation index
    mut_pos = [parse_mutation(lbl)[1] for lbl in mut_labels]
    label_to_idx = {lbl: i for i, lbl in enumerate(mut_labels)}

    pair_mut_idx = np.zeros((P, 2), dtype=np.int32)
    distances    = np.full((P, S), 999.0, dtype=np.float32)

    for p_idx, pair_label in enumerate(pair_labels):
        # Accept both "MutA+MutB" and "MutA/MutB" separators
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
            res_nums    = sdata['residue_nums']   # (L,)
            dist_matrix = sdata['dist_matrix']    # (L, L)

            hits_i = np.where(res_nums == pos_i)[0]
            hits_j = np.where(res_nums == pos_j)[0]
            if hits_i.size > 0 and hits_j.size > 0:
                distances[p_idx, s_idx] = dist_matrix[hits_i[0], hits_j[0]]

    return pair_mut_idx, distances


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_struct_ensemble(npz_paths, struct_names, mut_labels, pair_labels=None):
    """
    Load a structural ensemble and build all arrays for DataClass struct fields.

    Parameters
    ----------
    npz_paths : list of str, length S
        Paths to per-structure NPZ files from ``generate_struct_ensemble.py``.
    struct_names : list of str, length S
        Structure identifiers (e.g. ``['H', 'HD', 'L', 'LE2']``).
    mut_labels : list of str
        Mutation labels in "A42G" format.
    pair_labels : list of str or None
        Pair labels in "A42G+C18R" or "A42G/C18R" format.  If None or empty, contact arrays
        are not built and the corresponding return values are None.

    Returns
    -------
    dict with keys:
        struct_names            : tuple of str, length S
        struct_features         : np.ndarray (M, S, 60) float32
        struct_n_chains         : np.ndarray (S,) int32
        struct_contact_pair_idx : np.ndarray (P, 2) int32, or None
        struct_contact_distances: np.ndarray (P, S) float32, or None
    """
    S = len(struct_names)
    if len(npz_paths) != S:
        raise ValueError(
            f"npz_paths length {len(npz_paths)} != struct_names length {S}"
        )

    struct_data = [_load_npz(p) for p in npz_paths]

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
