"""Tests for struct/io.py — structural ensemble loading (HDF5 and CSV)."""

import numpy as np
import pandas as pd
import pytest
import h5py

from tfscreen.tfmodel.generative.components.theta.thermo.io import (
    _build_contact_arrays,
    _read_h5_structure,
    load_struct_ensemble,
    load_ddG_prior_csv,
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


# ──────────────────────────────────────────────────────────────────────────────
# TestLoadDdGPriorCsv
# ──────────────────────────────────────────────────────────────────────────────

def _write_ddG_csv(path, rows, columns=None):
    """Write a minimal CSV in the expected format."""
    if columns is None:
        columns = ["mut", "H", "HO", "L", "LO", "HE2", "LE2"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)


class TestLoadDdGPriorCsv:

    _MUT_LABELS = ["A10G", "C20R", "D30E"]
    _STRUCTS    = ["H", "HO", "L", "LO", "HE2", "LE2"]

    def _full_csv_rows(self):
        return [
            ["A10G", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ["C20R", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            ["D30E", -1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ]

    def test_output_keys(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, self._full_csv_rows())
        result = load_ddG_prior_csv(p, self._MUT_LABELS)
        assert set(result.keys()) == {
            'struct_names', 'struct_features', 'struct_n_chains',
            'struct_contact_pair_idx', 'struct_contact_distances',
        }

    def test_struct_names_from_csv_columns(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, self._full_csv_rows())
        result = load_ddG_prior_csv(p, self._MUT_LABELS)
        assert isinstance(result['struct_names'], tuple)
        assert result['struct_names'] == tuple(self._STRUCTS)

    def test_features_shape(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, self._full_csv_rows())
        result = load_ddG_prior_csv(p, self._MUT_LABELS)
        M, S = len(self._MUT_LABELS), len(self._STRUCTS)
        assert result['struct_features'].shape == (M, S)

    def test_features_dtype_float32(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, self._full_csv_rows())
        result = load_ddG_prior_csv(p, self._MUT_LABELS)
        assert result['struct_features'].dtype == np.float32

    def test_feature_values_match_csv(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, self._full_csv_rows())
        result = load_ddG_prior_csv(p, self._MUT_LABELS)
        # First mutation row (A10G): 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        np.testing.assert_allclose(
            result['struct_features'][0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rtol=1e-6
        )

    def test_mut_order_matches_mut_labels(self, tmp_path):
        """Rows in the CSV are in reversed order; result must follow mut_labels order."""
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, list(reversed(self._full_csv_rows())))
        result = load_ddG_prior_csv(p, self._MUT_LABELS)
        # A10G should still be at index 0
        np.testing.assert_allclose(
            result['struct_features'][0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rtol=1e-6
        )
        # D30E should still be at index 2
        np.testing.assert_allclose(
            result['struct_features'][2], [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], rtol=1e-6
        )

    def test_missing_mutation_defaults_to_zero(self, tmp_path):
        """Mutations in mut_labels but absent from CSV should get 0.0."""
        p = str(tmp_path / "ddG.csv")
        # Only include two of the three mutations
        rows = [
            ["A10G", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ["D30E", -1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ]
        _write_ddG_csv(p, rows)
        result = load_ddG_prior_csv(p, self._MUT_LABELS)
        # C20R (index 1) is absent → should be all zeros
        np.testing.assert_allclose(result['struct_features'][1], 0.0)
        # Others should still have their values
        np.testing.assert_allclose(
            result['struct_features'][0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rtol=1e-6
        )

    def test_extra_mutations_in_csv_are_ignored(self, tmp_path):
        """Rows in the CSV that are not in mut_labels should be silently ignored."""
        p = str(tmp_path / "ddG.csv")
        rows = self._full_csv_rows() + [["EXTRA", 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]]
        _write_ddG_csv(p, rows)
        result = load_ddG_prior_csv(p, self._MUT_LABELS)
        assert result['struct_features'].shape == (len(self._MUT_LABELS), len(self._STRUCTS))

    def test_struct_column_order_preserved(self, tmp_path):
        """struct_names should reflect the column order in the CSV."""
        cols = ["mut", "LE2", "HE2", "LO", "L", "HO", "H"]   # reversed
        rows = [["A10G"] + list(range(6, 0, -1))]
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, rows, columns=cols)
        result = load_ddG_prior_csv(p, ["A10G"])
        assert result['struct_names'] == ("LE2", "HE2", "LO", "L", "HO", "H")

    def test_n_chains_is_none(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, self._full_csv_rows())
        result = load_ddG_prior_csv(p, self._MUT_LABELS)
        assert result['struct_n_chains'] is None

    def test_contact_arrays_are_none(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, self._full_csv_rows())
        result = load_ddG_prior_csv(p, self._MUT_LABELS)
        assert result['struct_contact_pair_idx']  is None
        assert result['struct_contact_distances'] is None

    def test_missing_mut_column_raises(self, tmp_path):
        p = str(tmp_path / "bad.csv")
        pd.DataFrame({"H": [1.0], "L": [2.0]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="'mut' column"):
            load_ddG_prior_csv(p, ["A1G"])

    def test_no_structure_columns_raises(self, tmp_path):
        p = str(tmp_path / "bad.csv")
        pd.DataFrame({"mut": ["A1G"]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="no structure columns"):
            load_ddG_prior_csv(p, ["A1G"])

    def test_empty_mut_labels_gives_zero_rows(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, self._full_csv_rows())
        result = load_ddG_prior_csv(p, [])
        assert result['struct_features'].shape == (0, len(self._STRUCTS))
