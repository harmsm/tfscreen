"""
Feature extraction utilities for structural ensemble theta models.

Builds per-mutation per-structure feature vectors of shape (M, S, 60):

    [logP_s[pos, :20] | one_hot(wt_aa) | one_hot(mut_aa)]

where logP_s is the LigandMPNN per-residue log-probability matrix for
structure s (already aggregated across symmetric chains), and the one-hot
vectors encode wild-type and mutant amino-acid identity.

_ALPHABET, _AA_TO_IDX, and _MUT_PATTERN are defined here and re-exported
for use by other modules (e.g. io.py).
"""

import re
import numpy as np

# LigandMPNN standard alphabet (ProteinMPNN order; 20 canonical AAs).
_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
_AA_TO_IDX = {aa: i for i, aa in enumerate(_ALPHABET)}

_MUT_PATTERN = re.compile(r'^([A-Z])(\d+)([A-Z])$')

FEATURE_DIM = 60  # logP(20) + one_hot_wt(20) + one_hot_mut(20)


def parse_mutation(label):
    """
    Parse a mutation label such as "A42G" into constituent parts.

    Parameters
    ----------
    label : str
        Mutation string in <wt_aa><1-indexed PDB resnum><mut_aa> format.

    Returns
    -------
    wt_idx : int
        Index of wild-type amino acid in _ALPHABET.
    resnum : int
        1-indexed PDB residue number.
    mut_idx : int
        Index of mutant amino acid in _ALPHABET.

    Raises
    ------
    ValueError
        If the label cannot be parsed or contains an unknown amino acid.
    """
    m = _MUT_PATTERN.match(label)
    if m is None:
        raise ValueError(f"Cannot parse mutation label: {label!r}")
    wt_aa, resnum, mut_aa = m.group(1), int(m.group(2)), m.group(3)
    wt_idx = _AA_TO_IDX.get(wt_aa)
    mut_idx = _AA_TO_IDX.get(mut_aa)
    if wt_idx is None:
        raise ValueError(f"Unknown amino acid {wt_aa!r} in mutation label {label!r}")
    if mut_idx is None:
        raise ValueError(f"Unknown amino acid {mut_aa!r} in mutation label {label!r}")
    return wt_idx, resnum, mut_idx


def build_feature_matrix(struct_data_list, struct_names, mut_labels):
    """
    Build (M, S, 60) feature matrix from per-structure loaded NPZ data.

    Each feature vector for mutation m in structure s is:
        [logP_s[pos, :20]  |  one_hot(wt_aa)  |  one_hot(mut_aa)]

    Parameters
    ----------
    struct_data_list : list of dict, length S
        Each dict must have keys:
          'logP'         — float32 (L, 20): per-residue log-probabilities
          'residue_nums' — int32   (L,):   PDB residue numbers
    struct_names : sequence of str, length S
        Structure identifiers used in error messages.
    mut_labels : sequence of str
        Mutation strings in "A42G" format.

    Returns
    -------
    np.ndarray, shape (M, S, 60), dtype float32
    """
    M = len(mut_labels)
    S = len(struct_names)
    if len(struct_data_list) != S:
        raise ValueError(
            f"struct_data_list length {len(struct_data_list)} != "
            f"struct_names length {S}"
        )

    features = np.zeros((M, S, FEATURE_DIM), dtype=np.float32)

    parsed = [parse_mutation(lbl) for lbl in mut_labels]

    for s_idx, (sdata, sname) in enumerate(zip(struct_data_list, struct_names)):
        logP     = sdata['logP']          # (L, 20)
        res_nums = sdata['residue_nums']  # (L,)

        for m_idx, (wt_idx, resnum, mut_idx) in enumerate(parsed):
            hits = np.where(res_nums == resnum)[0]
            if hits.size == 0:
                raise ValueError(
                    f"Residue {resnum} not found in structure {sname!r}. "
                    f"Available residues: {sorted(set(res_nums.tolist()))}"
                )
            row = hits[0]
            features[m_idx, s_idx, :20]           = logP[row, :]
            features[m_idx, s_idx, 20 + wt_idx]  = 1.0
            features[m_idx, s_idx, 40 + mut_idx]  = 1.0

    return features
