"""Tests for struct/io.py — structural ensemble loading."""

import numpy as np
import os
import pytest
import tempfile

from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.io import (
    _build_contact_arrays,
    _load_npz,
    load_struct_ensemble,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_npz(path, res_nums, logP=None, dist_matrix=None, n_chains=2):
    """Write a minimal valid structure NPZ file."""
    L = len(res_nums)
    if logP is None:
        logP = np.zeros((L, 20), dtype=np.float32)
    if dist_matrix is None:
        dist_matrix = np.zeros((L, L), dtype=np.float32)
    np.savez(
        path,
        logP=logP.astype(np.float32),
        residue_nums=np.asarray(res_nums, dtype=np.int32),
        dist_matrix=dist_matrix.astype(np.float32),
        n_chains_bearing_mut=np.int32(n_chains),
    )


def _make_dist_matrix(res_nums, pairs_close):
    """
    Build a distance matrix where specified pairs have distance 5.0 Å,
    all others are 999.0 Å.

    pairs_close : list of (resnum_i, resnum_j) tuples
    """
    L = len(res_nums)
    dm = np.full((L, L), 999.0, dtype=np.float32)
    np.fill_diagonal(dm, 0.0)
    r_to_idx = {r: i for i, r in enumerate(res_nums)}
    for (ri, rj) in pairs_close:
        i, j = r_to_idx[ri], r_to_idx[rj]
        dm[i, j] = dm[j, i] = 5.0
    return dm


# ──────────────────────────────────────────────────────────────────────────────
# TestLoadNpz
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadNpz:
    def test_valid_file_loads(self, tmp_path):
        p = str(tmp_path / "s.npz")
        _write_npz(p, [10, 20], n_chains=2)
        d = _load_npz(p)
        assert set(d.keys()) == {'logP', 'residue_nums', 'dist_matrix', 'n_chains_bearing_mut'}
        assert d['logP'].dtype == np.float32
        assert d['residue_nums'].dtype == np.int32
        assert d['dist_matrix'].dtype == np.float32
        assert isinstance(d['n_chains_bearing_mut'], int)

    def test_n_chains_value(self, tmp_path):
        p = str(tmp_path / "s.npz")
        _write_npz(p, [1], n_chains=4)
        d = _load_npz(p)
        assert d['n_chains_bearing_mut'] == 4

    def test_missing_key_raises(self, tmp_path):
        p = str(tmp_path / "bad.npz")
        np.savez(p, logP=np.zeros((2, 20)), residue_nums=np.array([1, 2]))
        with pytest.raises(KeyError, match="missing required key"):
            _load_npz(p)


# ──────────────────────────────────────────────────────────────────────────────
# TestBuildContactArrays
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildContactArrays:

    def _make_struct_data(self, res_nums, dist_matrix=None):
        L = len(res_nums)
        if dist_matrix is None:
            dist_matrix = np.zeros((L, L), dtype=np.float32)
        return {
            'logP':         np.zeros((L, 20), dtype=np.float32),
            'residue_nums': np.asarray(res_nums, dtype=np.int32),
            'dist_matrix':  dist_matrix.astype(np.float32),
            'n_chains_bearing_mut': 2,
        }

    def test_basic_shapes(self):
        dm = _make_dist_matrix([10, 20, 30], [(10, 20)])
        sdata = self._make_struct_data([10, 20, 30], dm)
        mut_labels  = ["A10G", "V20T", "C30R"]
        pair_labels = ["A10G+V20T"]
        pair_mut_idx, distances = _build_contact_arrays([sdata], mut_labels, pair_labels)
        assert pair_mut_idx.shape == (1, 2)
        assert distances.shape    == (1, 1)

    def test_pair_mut_idx_correct(self):
        sdata = self._make_struct_data([1, 2])
        mut_labels  = ["A1G", "C2R"]
        pair_labels = ["A1G+C2R"]
        pair_mut_idx, _ = _build_contact_arrays([sdata], mut_labels, pair_labels)
        assert pair_mut_idx[0, 0] == 0   # A1G is index 0
        assert pair_mut_idx[0, 1] == 1   # C2R is index 1

    def test_distance_value_from_matrix(self):
        dm = _make_dist_matrix([10, 20], [(10, 20)])
        sdata = self._make_struct_data([10, 20], dm)
        _, distances = _build_contact_arrays([sdata], ["A10G", "V20T"], ["A10G+V20T"])
        assert distances[0, 0] == pytest.approx(5.0)

    def test_missing_residue_defaults_to_999(self):
        """If a residue is absent from a structure, distance should be 999.0."""
        sdata_has_10 = self._make_struct_data([10])   # lacks residue 20
        # A20G can't be found in this structure
        sdata_has_both = self._make_struct_data([10, 20])
        _, distances = _build_contact_arrays(
            [sdata_has_10, sdata_has_both],
            ["A10G", "A20G"],
            ["A10G+A20G"],
        )
        assert distances[0, 0] == pytest.approx(999.0)  # struct 0 lacks residue 20
        assert distances[0, 1] == pytest.approx(0.0)    # struct 1: dist_matrix is zeros

    def test_multi_struct_multi_pair(self):
        dm1 = _make_dist_matrix([1, 2, 3], [(1, 2)])
        dm2 = _make_dist_matrix([1, 2, 3], [(2, 3)])
        s0 = self._make_struct_data([1, 2, 3], dm1)
        s1 = self._make_struct_data([1, 2, 3], dm2)
        pair_labels = ["A1G+C2R", "C2R+D3E"]
        pair_mut_idx, distances = _build_contact_arrays(
            [s0, s1], ["A1G", "C2R", "D3E"], pair_labels
        )
        assert pair_mut_idx.shape == (2, 2)
        assert distances.shape    == (2, 2)
        assert distances[0, 0] == pytest.approx(5.0)   # pair (1,2) close in s0
        assert distances[0, 1] == pytest.approx(999.0) # pair (1,2) not close in s1
        assert distances[1, 0] == pytest.approx(999.0) # pair (2,3) not close in s0
        assert distances[1, 1] == pytest.approx(5.0)   # pair (2,3) close in s1

    def test_invalid_pair_format_raises(self):
        sdata = self._make_struct_data([1])
        with pytest.raises(ValueError, match="Cannot parse pair label"):
            _build_contact_arrays([sdata], ["A1G"], ["A1G_bad"])

    def test_unknown_mutation_in_pair_raises(self):
        sdata = self._make_struct_data([1])
        with pytest.raises(ValueError, match="not found in mut_labels"):
            _build_contact_arrays([sdata], ["A1G"], ["A1G+C2R"])


# ──────────────────────────────────────────────────────────────────────────────
# TestLoadStructEnsemble
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadStructEnsemble:
    def test_output_keys(self, tmp_path):
        p = str(tmp_path / "s.npz")
        _write_npz(p, [10, 20], n_chains=2)
        result = load_struct_ensemble([p], ["H"], ["A10G"])
        assert set(result.keys()) == {
            'struct_names', 'struct_features', 'struct_n_chains',
            'struct_contact_pair_idx', 'struct_contact_distances',
        }

    def test_struct_names_is_tuple(self, tmp_path):
        p = str(tmp_path / "s.npz")
        _write_npz(p, [1], n_chains=2)
        result = load_struct_ensemble([p], ["H"], ["A1G"])
        assert isinstance(result['struct_names'], tuple)
        assert result['struct_names'] == ('H',)

    def test_feature_shape(self, tmp_path):
        p = str(tmp_path / "s.npz")
        _write_npz(p, [1, 2, 3], n_chains=2)
        result = load_struct_ensemble([p], ["H"], ["A1G", "C2R", "D3E"])
        assert result['struct_features'].shape == (3, 1, 60)

    def test_n_chains_value(self, tmp_path):
        p0 = str(tmp_path / "s0.npz")
        p1 = str(tmp_path / "s1.npz")
        _write_npz(p0, [1], n_chains=2)
        _write_npz(p1, [1], n_chains=1)
        result = load_struct_ensemble([p0, p1], ["H", "L"], ["A1G"])
        np.testing.assert_array_equal(result['struct_n_chains'], [2, 1])

    def test_no_pairs_returns_none(self, tmp_path):
        p = str(tmp_path / "s.npz")
        _write_npz(p, [1], n_chains=2)
        result = load_struct_ensemble([p], ["H"], ["A1G"])
        assert result['struct_contact_pair_idx']   is None
        assert result['struct_contact_distances']  is None

    def test_empty_pairs_returns_none(self, tmp_path):
        p = str(tmp_path / "s.npz")
        _write_npz(p, [1], n_chains=2)
        result = load_struct_ensemble([p], ["H"], ["A1G"], pair_labels=[])
        assert result['struct_contact_pair_idx']   is None
        assert result['struct_contact_distances']  is None

    def test_with_pairs(self, tmp_path):
        dm = _make_dist_matrix([1, 2], [(1, 2)])
        p = str(tmp_path / "s.npz")
        _write_npz(p, [1, 2], dist_matrix=dm, n_chains=2)
        result = load_struct_ensemble(
            [p], ["H"], ["A1G", "C2R"], pair_labels=["A1G+C2R"]
        )
        assert result['struct_contact_pair_idx'].shape   == (1, 2)
        assert result['struct_contact_distances'].shape  == (1, 1)
        assert result['struct_contact_distances'][0, 0] == pytest.approx(5.0)

    def test_path_count_mismatch_raises(self, tmp_path):
        p = str(tmp_path / "s.npz")
        _write_npz(p, [1], n_chains=2)
        with pytest.raises(ValueError, match="length"):
            load_struct_ensemble([p], ["H", "L"], ["A1G"])
