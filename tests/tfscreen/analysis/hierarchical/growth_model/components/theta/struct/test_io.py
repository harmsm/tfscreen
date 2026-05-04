"""Tests for struct/io.py — structural ensemble loading (HDF5)."""

import numpy as np
import pytest
import h5py

from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.io import (
    _build_contact_arrays,
    _read_h5_structure,
    load_struct_ensemble,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_h5(path, struct_specs, structure_names=None):
    """
    Write a minimal valid HDF5 ensemble file.

    struct_specs : dict  name -> dict with optional keys:
        res_nums    (list of int, default [1])
        logP        (ndarray, default zeros)
        dist_matrix (ndarray, default zeros)
        n_chains    (int, default 2)
    structure_names : list of str or None — defaults to list(struct_specs.keys())
    """
    if structure_names is None:
        structure_names = list(struct_specs.keys())
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('structure_names',
                          data=np.array(structure_names, dtype=h5py.string_dtype()))
        for name, spec in struct_specs.items():
            res_nums = spec.get('res_nums', [1])
            L = len(res_nums)
            logP        = spec.get('logP',        np.zeros((L, 20), dtype=np.float32))
            dist_matrix = spec.get('dist_matrix', np.zeros((L, L),  dtype=np.float32))
            n_chains    = spec.get('n_chains', 2)
            grp = hf.create_group(name)
            grp.create_dataset('logP',                 data=logP.astype(np.float32))
            grp.create_dataset('residue_nums',         data=np.asarray(res_nums, dtype=np.int32))
            grp.create_dataset('dist_matrix',          data=dist_matrix.astype(np.float32))
            grp.create_dataset('n_chains_bearing_mut', data=np.int32(n_chains))


def _make_dist_matrix(res_nums, pairs_close):
    """
    Build a distance matrix where specified pairs have distance 5.0 Å,
    all others 999.0 Å.
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
# TestReadH5Structure
# ──────────────────────────────────────────────────────────────────────────────

class TestReadH5Structure:
    def test_valid_group_loads(self, tmp_path):
        p = str(tmp_path / "e.h5")
        _write_h5(p, {'H': {'res_nums': [10, 20], 'n_chains': 2}})
        with h5py.File(p, 'r') as hf:
            d = _read_h5_structure(hf, 'H')
        assert set(d.keys()) == {'logP', 'residue_nums', 'dist_matrix', 'n_chains_bearing_mut'}
        assert d['logP'].dtype == np.float32
        assert d['residue_nums'].dtype == np.int32
        assert d['dist_matrix'].dtype == np.float32
        assert isinstance(d['n_chains_bearing_mut'], int)

    def test_n_chains_value(self, tmp_path):
        p = str(tmp_path / "e.h5")
        _write_h5(p, {'H': {'res_nums': [1], 'n_chains': 4}})
        with h5py.File(p, 'r') as hf:
            d = _read_h5_structure(hf, 'H')
        assert d['n_chains_bearing_mut'] == 4

    def test_missing_group_raises(self, tmp_path):
        p = str(tmp_path / "e.h5")
        _write_h5(p, {'H': {'res_nums': [1]}})
        with h5py.File(p, 'r') as hf:
            with pytest.raises(KeyError, match="not found"):
                _read_h5_structure(hf, 'HD')

    def test_missing_dataset_raises(self, tmp_path):
        p = str(tmp_path / "e.h5")
        with h5py.File(p, 'w') as hf:
            hf.create_dataset('structure_names', data=np.array(['H'], dtype=h5py.string_dtype()))
            grp = hf.create_group('H')
            grp.create_dataset('logP', data=np.zeros((2, 20), dtype=np.float32))
            # missing residue_nums, dist_matrix, n_chains_bearing_mut
        with h5py.File(p, 'r') as hf:
            with pytest.raises(KeyError, match="missing required dataset"):
                _read_h5_structure(hf, 'H')


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
        pair_mut_idx, distances = _build_contact_arrays(
            [sdata], ["A10G", "V20T", "C30R"], ["A10G+V20T"]
        )
        assert pair_mut_idx.shape == (1, 2)
        assert distances.shape    == (1, 1)

    def test_pair_mut_idx_correct(self):
        sdata = self._make_struct_data([1, 2])
        pair_mut_idx, _ = _build_contact_arrays(
            [sdata], ["A1G", "C2R"], ["A1G+C2R"]
        )
        assert pair_mut_idx[0, 0] == 0
        assert pair_mut_idx[0, 1] == 1

    def test_distance_value_from_matrix(self):
        dm = _make_dist_matrix([10, 20], [(10, 20)])
        sdata = self._make_struct_data([10, 20], dm)
        _, distances = _build_contact_arrays([sdata], ["A10G", "V20T"], ["A10G+V20T"])
        assert distances[0, 0] == pytest.approx(5.0)

    def test_missing_residue_defaults_to_999(self):
        sdata_has_10   = self._make_struct_data([10])
        sdata_has_both = self._make_struct_data([10, 20])
        _, distances = _build_contact_arrays(
            [sdata_has_10, sdata_has_both], ["A10G", "A20G"], ["A10G+A20G"]
        )
        assert distances[0, 0] == pytest.approx(999.0)
        assert distances[0, 1] == pytest.approx(0.0)

    def test_slash_separator_accepted(self):
        sdata = self._make_struct_data([1, 2])
        pair_mut_idx, _ = _build_contact_arrays(
            [sdata], ["A1G", "C2R"], ["A1G/C2R"]
        )
        assert pair_mut_idx[0, 0] == 0
        assert pair_mut_idx[0, 1] == 1

    def test_multi_struct_multi_pair(self):
        dm1 = _make_dist_matrix([1, 2, 3], [(1, 2)])
        dm2 = _make_dist_matrix([1, 2, 3], [(2, 3)])
        s0  = self._make_struct_data([1, 2, 3], dm1)
        s1  = self._make_struct_data([1, 2, 3], dm2)
        pair_mut_idx, distances = _build_contact_arrays(
            [s0, s1], ["A1G", "C2R", "D3E"], ["A1G+C2R", "C2R+D3E"]
        )
        assert pair_mut_idx.shape == (2, 2)
        assert distances.shape    == (2, 2)
        assert distances[0, 0] == pytest.approx(5.0)
        assert distances[0, 1] == pytest.approx(999.0)
        assert distances[1, 0] == pytest.approx(999.0)
        assert distances[1, 1] == pytest.approx(5.0)

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
        p = str(tmp_path / "e.h5")
        _write_h5(p, {'H': {'res_nums': [10, 20]}})
        result = load_struct_ensemble(p, ["A10G"])
        assert set(result.keys()) == {
            'struct_names', 'struct_features', 'struct_n_chains',
            'struct_contact_pair_idx', 'struct_contact_distances',
        }

    def test_struct_names_from_file(self, tmp_path):
        p = str(tmp_path / "e.h5")
        _write_h5(p, {'H': {'res_nums': [1]}, 'L': {'res_nums': [1]}})
        result = load_struct_ensemble(p, ["A1G"])
        assert isinstance(result['struct_names'], tuple)
        assert result['struct_names'] == ('H', 'L')

    def test_feature_shape(self, tmp_path):
        p = str(tmp_path / "e.h5")
        _write_h5(p, {'H': {'res_nums': [1, 2, 3]}})
        result = load_struct_ensemble(p, ["A1G", "C2R", "D3E"])
        assert result['struct_features'].shape == (3, 1, 60)

    def test_n_chains_values(self, tmp_path):
        p = str(tmp_path / "e.h5")
        _write_h5(p, {
            'H':  {'res_nums': [1], 'n_chains': 2},
            'L':  {'res_nums': [1], 'n_chains': 1},
        })
        result = load_struct_ensemble(p, ["A1G"])
        np.testing.assert_array_equal(result['struct_n_chains'], [2, 1])

    def test_no_pairs_returns_none(self, tmp_path):
        p = str(tmp_path / "e.h5")
        _write_h5(p, {'H': {'res_nums': [1]}})
        result = load_struct_ensemble(p, ["A1G"])
        assert result['struct_contact_pair_idx']  is None
        assert result['struct_contact_distances'] is None

    def test_empty_pairs_returns_none(self, tmp_path):
        p = str(tmp_path / "e.h5")
        _write_h5(p, {'H': {'res_nums': [1]}})
        result = load_struct_ensemble(p, ["A1G"], pair_labels=[])
        assert result['struct_contact_pair_idx']  is None
        assert result['struct_contact_distances'] is None

    def test_with_pairs(self, tmp_path):
        dm = _make_dist_matrix([1, 2], [(1, 2)])
        p  = str(tmp_path / "e.h5")
        _write_h5(p, {'H': {'res_nums': [1, 2], 'dist_matrix': dm}})
        result = load_struct_ensemble(p, ["A1G", "C2R"], pair_labels=["A1G+C2R"])
        assert result['struct_contact_pair_idx'].shape  == (1, 2)
        assert result['struct_contact_distances'].shape == (1, 1)
        assert result['struct_contact_distances'][0, 0] == pytest.approx(5.0)

    def test_missing_structure_names_raises(self, tmp_path):
        p = str(tmp_path / "e.h5")
        with h5py.File(p, 'w') as hf:
            grp = hf.create_group('H')
            grp.create_dataset('logP', data=np.zeros((1, 20), dtype=np.float32))
        with pytest.raises(KeyError, match="structure_names"):
            load_struct_ensemble(p, ["A1G"])
