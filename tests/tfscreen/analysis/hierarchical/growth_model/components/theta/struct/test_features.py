"""Tests for struct/features.py — feature extraction utilities."""

import numpy as np
import pytest

from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.features import (
    _AA_TO_IDX,
    _ALPHABET,
    FEATURE_DIM,
    build_feature_matrix,
    parse_mutation,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_struct_data(res_nums, logP=None):
    """
    Build a minimal struct-data dict.

    Parameters
    ----------
    res_nums : array-like of int
        Residue numbers (L,).
    logP : array-like or None
        (L, 20) log-probability matrix.  Defaults to zeros.
    """
    L = len(res_nums)
    if logP is None:
        logP = np.zeros((L, 20), dtype=np.float32)
    return {
        'logP':         np.asarray(logP, dtype=np.float32),
        'residue_nums': np.asarray(res_nums, dtype=np.int32),
    }


# ──────────────────────────────────────────────────────────────────────────────
# TestAlphabet
# ──────────────────────────────────────────────────────────────────────────────

class TestAlphabet:
    def test_length(self):
        assert len(_ALPHABET) == 20

    def test_aa_to_idx_length(self):
        assert len(_AA_TO_IDX) == 20

    def test_round_trip(self):
        for i, aa in enumerate(_ALPHABET):
            assert _AA_TO_IDX[aa] == i


# ──────────────────────────────────────────────────────────────────────────────
# TestParseMutation
# ──────────────────────────────────────────────────────────────────────────────

class TestParseMutation:
    def test_valid_simple(self):
        wt_idx, resnum, mut_idx = parse_mutation("A42G")
        assert wt_idx  == _AA_TO_IDX['A']
        assert resnum  == 42
        assert mut_idx == _AA_TO_IDX['G']

    def test_multidigit_resnum(self):
        _, resnum, _ = parse_mutation("K123R")
        assert resnum == 123

    def test_all_characters_accepted(self):
        for aa in _ALPHABET:
            wt_idx, _, mut_idx = parse_mutation(f"{aa}1{aa}")
            assert wt_idx == mut_idx == _AA_TO_IDX[aa]

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_mutation("abc")

    def test_lowercase_raises(self):
        with pytest.raises(ValueError):
            parse_mutation("a42G")

    def test_no_resnum_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_mutation("AG")

    def test_unknown_wt_aa_raises(self):
        with pytest.raises(ValueError, match="Unknown amino acid"):
            parse_mutation("B42G")

    def test_unknown_mut_aa_raises(self):
        with pytest.raises(ValueError, match="Unknown amino acid"):
            parse_mutation("A42B")


# ──────────────────────────────────────────────────────────────────────────────
# TestBuildFeatureMatrix
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_output_shape(self):
        sdata = _make_struct_data([10, 20, 30])
        features = build_feature_matrix([sdata], ["S0"], ["A10G", "V20T"])
        assert features.shape == (2, 1, FEATURE_DIM)

    def test_output_dtype(self):
        sdata = _make_struct_data([10])
        features = build_feature_matrix([sdata], ["S0"], ["A10G"])
        assert features.dtype == np.float32

    def test_multi_struct(self):
        s0 = _make_struct_data([5, 10])
        s1 = _make_struct_data([5, 10])
        features = build_feature_matrix([s0, s1], ["H", "L"], ["A5G", "C10R"])
        assert features.shape == (2, 2, FEATURE_DIM)

    def test_logP_row_copied(self):
        """First 20 elements of feature vector must equal the logP row for that residue."""
        logP = np.random.RandomState(0).randn(3, 20).astype(np.float32)
        res_nums = np.array([10, 20, 30])
        sdata = _make_struct_data(res_nums, logP)

        mut_labels = ["A20G"]
        features = build_feature_matrix([sdata], ["S0"], mut_labels)

        np.testing.assert_array_equal(features[0, 0, :20], logP[1, :])  # row index 1 = resnum 20

    def test_one_hot_wt_correct(self):
        """Bit at position 20+wt_idx should be 1, all others in [20:40] zero."""
        sdata = _make_struct_data([1])
        wt_aa = 'C'  # index 1
        features = build_feature_matrix([sdata], ["S0"], [f"{wt_aa}1G"])
        one_hot_wt = features[0, 0, 20:40]
        assert one_hot_wt[_AA_TO_IDX[wt_aa]] == 1.0
        assert one_hot_wt.sum() == 1.0

    def test_one_hot_mut_correct(self):
        """Bit at position 40+mut_idx should be 1, all others in [40:60] zero."""
        sdata = _make_struct_data([1])
        mut_aa = 'W'  # index 19
        features = build_feature_matrix([sdata], ["S0"], [f"A1{mut_aa}"])
        one_hot_mut = features[0, 0, 40:60]
        assert one_hot_mut[_AA_TO_IDX[mut_aa]] == 1.0
        assert one_hot_mut.sum() == 1.0

    def test_wt_to_same_aa_one_hot(self):
        """Both wt and mut one-hot should fire on the same index for a synonymous 'mutation'."""
        sdata = _make_struct_data([1])
        features = build_feature_matrix([sdata], ["S0"], ["A1A"])
        one_hot_wt  = features[0, 0, 20:40]
        one_hot_mut = features[0, 0, 40:60]
        np.testing.assert_array_equal(one_hot_wt, one_hot_mut)

    def test_missing_residue_raises(self):
        sdata = _make_struct_data([10, 20])
        with pytest.raises(ValueError, match="Residue 99 not found"):
            build_feature_matrix([sdata], ["S0"], ["A99G"])

    def test_wrong_list_length_raises(self):
        s0 = _make_struct_data([1])
        s1 = _make_struct_data([1])
        with pytest.raises(ValueError, match="length"):
            build_feature_matrix([s0, s1], ["S0"], ["A1G"])  # 2 data, 1 name

    def test_multiple_mutations_independent(self):
        """Each mutation row in the feature matrix should be independently computed."""
        logP = np.eye(20, dtype=np.float32)[:3, :]   # 3 residues, distinct logP rows
        sdata = _make_struct_data([1, 2, 3], logP=np.vstack([logP, np.zeros((0, 20))]))
        # simplify: build manually
        L_logP = np.zeros((3, 20), dtype=np.float32)
        L_logP[0, 5] = 1.0
        L_logP[1, 7] = 1.0
        L_logP[2, 9] = 1.0
        sdata2 = _make_struct_data([1, 2, 3], logP=L_logP)

        features = build_feature_matrix([sdata2], ["S0"], ["A1G", "A2G", "A3G"])
        assert features[0, 0, 5]  == 1.0   # logP row 0, col 5
        assert features[1, 0, 7]  == 1.0   # logP row 1, col 7
        assert features[2, 0, 9]  == 1.0   # logP row 2, col 9
