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


# ──────────────────────────────────────────────────────────────────────────────
# Units, column handling, numbering and coverage checks
# ──────────────────────────────────────────────────────────────────────────────

import warnings

from tfscreen.tfmodel.generative.components.theta.thermo.io import (
    THERMO_OPTION_DEFAULTS,
    ddG_to_kT_factor,
    load_thermo_data,
    normalize_thermo_options,
    required_struct_names,
)

_C4 = ("H", "HD", "L", "LE2")
_C12 = ("H", "HO", "L", "LO", "HE2", "LE2")

# First 60 residues of LacI as they appear (0-based, V0 = residue 1) in the
# 12-state ensemble CSV this loader has to handle.
_LACI_60 = "VKPVTLYDVAEYAGVSYQTVSRVVNQASHVSAKTREKVEAAMAELNYIPNRVAQQLAGKQS"
_ENSEMBLE_COLS = ["unfolded", "L", "HO", "LE2", "HOE2", "LE", "HOE", "H",
                  "LO", "LOE", "LOE2", "HE", "HE2"]


def _ensemble_like_csv(path, n_pos=60):
    """Mimic notebooks/other-data/ddG_ensemble_clean_with_effector.csv:
    pandas index column, 'Mutation', 0-based numbering, 12 states + unfolded."""
    rng = np.random.default_rng(0)
    labels = [f"{wt}{i}{mt}" for i, wt in enumerate(_LACI_60[:n_pos])
              for mt in "ACDEFGHIKLMNPQRSTVWY"]
    df = pd.DataFrame(rng.normal(2, 1, (len(labels), len(_ENSEMBLE_COLS))),
                      columns=_ENSEMBLE_COLS)
    df["unfolded"] = 0.0
    df.insert(0, "Mutation", labels)
    df.to_csv(path)            # writes the 'Unnamed: 0' index column
    return str(path)


def _lib_label(pos1, mt):
    """Library (1-based) label for LacI residue pos1."""
    return f"{_LACI_60[pos1 - 1]}{pos1}{mt}"


class TestThermoOptions:

    def test_defaults(self):
        assert normalize_thermo_options(None) == THERMO_OPTION_DEFAULTS
        assert normalize_thermo_options({}) == THERMO_OPTION_DEFAULTS

    def test_kT_factor_is_exactly_one(self):
        assert ddG_to_kT_factor("kT") == 1.0

    def test_kcal_factor(self):
        assert ddG_to_kT_factor("kcal/mol", 298.15) == pytest.approx(
            1.0 / (1.987204259e-3 * 298.15), rel=1e-12)
        assert ddG_to_kT_factor("kcal/mol", 298.15) == pytest.approx(1.6878, abs=1e-4)

    def test_kJ_factor_and_temperature(self):
        assert ddG_to_kT_factor("kJ/mol", 310.0) == pytest.approx(
            1.0 / (8.314462618e-3 * 310.0), rel=1e-12)

    def test_unit_aliases(self):
        assert normalize_thermo_options({"ddG_units": "KCAL"})["ddG_units"] == "kcal/mol"
        assert normalize_thermo_options({"ddG_units": "kj/mol"})["ddG_units"] == "kJ/mol"

    @pytest.mark.parametrize("opts, match", [
        ({"ddG_units": "eV"}, "not recognised"),
        ({"temperature_K": -1}, "temperature_K"),
        ({"residue_offset": 1.5}, "residue_offset"),
        ({"max_missing_fraction": 2}, "max_missing_fraction"),
        ({"units": "kT"}, "Unknown thermo_options"),
        ({"column_map": "HD"}, "model_name=csv_column"),
    ])
    def test_bad_options_raise(self, opts, match):
        with pytest.raises(ValueError, match=match):
            normalize_thermo_options(opts)

    def test_column_map_string(self):
        opts = normalize_thermo_options({"column_map": "HD=HO, LE2=LOE2"})
        assert opts["column_map"] == {"HD": "HO", "LE2": "LOE2"}

    def test_required_struct_names(self):
        assert required_struct_names("thermo.O2_C4_K3_U1_a.PddG") == _C4
        assert required_struct_names("thermo.O2_C12_K5_U0_a.PnnC") == _C12
        assert required_struct_names("thermo.O2_C4_K3_U0_a.PK") is None
        assert required_struct_names("hill_geno") is None
        assert required_struct_names(None) is None


class TestDdGCsvUnits:

    _ROWS = [["A10G", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
             ["C20R", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
             ["D30E", -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]]
    _MUTS = ["A10G", "C20R", "D30E"]

    def test_default_uses_values_as_is(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, self._ROWS)
        raw = np.array([r[1:] for r in self._ROWS], dtype=np.float32)
        for opts in (None, {}, {"ddG_units": "kT", "temperature_K": 350.0}):
            out = load_ddG_prior_csv(p, self._MUTS, thermo_options=opts)
            np.testing.assert_array_equal(out["struct_features"], raw)

    def test_kcal_converted_to_kT(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, self._ROWS)
        out = load_ddG_prior_csv(p, self._MUTS,
                                 thermo_options={"ddG_units": "kcal/mol"})
        RT = 1.987204259e-3 * 298.15
        expected = np.array([r[1:] for r in self._ROWS]) / RT
        np.testing.assert_allclose(out["struct_features"], expected, rtol=1e-6)

    def test_kJ_converted_to_kT(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        _write_ddG_csv(p, self._ROWS)
        out = load_ddG_prior_csv(p, self._MUTS, thermo_options={
            "ddG_units": "kJ/mol", "temperature_K": 310.0})
        RT = 8.314462618e-3 * 310.0
        expected = np.array([r[1:] for r in self._ROWS]) / RT
        np.testing.assert_allclose(out["struct_features"], expected, rtol=1e-6)


class TestDdGCsvColumns:

    def _csv(self, tmp_path, columns, rows, index=False):
        p = str(tmp_path / "ddG.csv")
        pd.DataFrame(rows, columns=columns).to_csv(p, index=index)
        return p

    def test_mutation_alias_and_unnamed_index(self, tmp_path):
        p = self._csv(tmp_path, ["Mutation", "H", "HD", "L", "LE2"],
                      [["A10G", 1.0, 2.0, 3.0, 4.0]], index=True)
        assert "Unnamed: 0" in pd.read_csv(p).columns
        out = load_ddG_prior_csv(p, ["A10G"])
        assert out["struct_names"] == _C4
        np.testing.assert_array_equal(out["struct_features"][0], [1, 2, 3, 4])

    def test_lowercase_mutation_alias(self, tmp_path):
        p = self._csv(tmp_path, ["mutation", "H"], [["A10G", 1.0]])
        assert load_ddG_prior_csv(p, ["A10G"])["struct_names"] == ("H",)

    def test_two_mutation_columns_raise(self, tmp_path):
        p = self._csv(tmp_path, ["mut", "Mutation", "H"], [["A10G", "A10G", 1.0]])
        with pytest.raises(ValueError, match="more than one mutation column"):
            load_ddG_prior_csv(p, ["A10G"])

    def test_extra_columns_ignored_and_reordered(self, tmp_path, capsys):
        p = self._csv(tmp_path, ["mut", "LE2", "extra", "L", "HD", "H", "unfolded"],
                      [["A10G", 4.0, 9.0, 3.0, 2.0, 1.0, 0.0]])
        out = load_ddG_prior_csv(p, ["A10G"], struct_names=_C4)
        assert out["struct_names"] == _C4
        np.testing.assert_array_equal(out["struct_features"][0], [1, 2, 3, 4])
        assert "ignoring column(s)" in capsys.readouterr().out

    def test_missing_required_columns_listed(self, tmp_path):
        p = self._csv(tmp_path, ["mut", "H", "L"], [["A10G", 1.0, 3.0]])
        with pytest.raises(ValueError, match=r"missing structure column\(s\) \['HD', 'LE2'\]"):
            load_ddG_prior_csv(p, ["A10G"], struct_names=_C4)

    def test_column_map(self, tmp_path):
        p = self._csv(tmp_path, ["mut", "H", "HO", "L", "LE2"],
                      [["A10G", 1.0, 2.0, 3.0, 4.0]])
        for cmap in ({"HD": "HO"}, "HD=HO"):
            out = load_ddG_prior_csv(p, ["A10G"], struct_names=_C4,
                                     thermo_options={"column_map": cmap})
            np.testing.assert_array_equal(out["struct_features"][0], [1, 2, 3, 4])

    def test_column_map_to_absent_column(self, tmp_path):
        p = self._csv(tmp_path, ["mut", "H", "L", "LE2"], [["A10G", 1.0, 3.0, 4.0]])
        with pytest.raises(ValueError, match="mapped to 'HO'"):
            load_ddG_prior_csv(p, ["A10G"], struct_names=_C4,
                               thermo_options={"column_map": {"HD": "HO"}})

    def test_column_map_unknown_structure(self, tmp_path):
        p = self._csv(tmp_path, ["mut", "H", "HD", "L", "LE2"],
                      [["A10G", 1.0, 2.0, 3.0, 4.0]])
        with pytest.raises(ValueError, match="not structures of this model"):
            load_ddG_prior_csv(p, ["A10G"], struct_names=_C4,
                               thermo_options={"column_map": {"HO": "H"}})

    def test_non_numeric_column_raises(self, tmp_path):
        p = self._csv(tmp_path, ["mut", "H"], [["A10G", "big"]])
        with pytest.raises(ValueError, match="not numeric"):
            load_ddG_prior_csv(p, ["A10G"])

    def test_duplicate_labels_raise(self, tmp_path):
        p = self._csv(tmp_path, ["mut", "H"], [["A10G", 1.0], ["A10G", 2.0]])
        with pytest.raises(ValueError, match="more than once"):
            load_ddG_prior_csv(p, ["A10G"])


class TestDdGCsvCoverage:

    def _csv(self, tmp_path, labels):
        p = str(tmp_path / "ddG.csv")
        df = pd.DataFrame({"mut": labels, "H": np.arange(len(labels), dtype=float)})
        df.to_csv(p, index=False)
        return p

    def test_full_coverage_reported(self, tmp_path, capsys):
        p = self._csv(tmp_path, ["A10G", "C20R"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_ddG_prior_csv(p, ["A10G", "C20R"])
        assert "ΔΔG for 2/2 library mutations (100.0%)" in capsys.readouterr().out

    def test_missing_above_threshold_warns_and_zero_fills(self, tmp_path):
        p = self._csv(tmp_path, ["A10G"])
        with pytest.warns(UserWarning, match=r"no ΔΔG for 2 of 3.*C20R, D30E"):
            out = load_ddG_prior_csv(p, ["A10G", "C20R", "D30E"])
        np.testing.assert_array_equal(out["struct_features"][1:], 0.0)

    def test_threshold_is_configurable(self, tmp_path):
        p = self._csv(tmp_path, ["A10G"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_ddG_prior_csv(p, ["A10G", "C20R"],
                               thermo_options={"max_missing_fraction": 0.5})
        with pytest.warns(UserWarning, match="max_missing_fraction 0.4"):
            load_ddG_prior_csv(p, ["A10G", "C20R"],
                               thermo_options={"max_missing_fraction": 0.4})

    def test_warning_list_is_truncated(self, tmp_path):
        p = self._csv(tmp_path, ["A1G"])
        muts = ["A1G"] + [f"A{i}G" for i in range(2, 30)]
        with pytest.warns(UserWarning, match=r"\(18 more\)"):
            load_ddG_prior_csv(p, muts)

    def test_zero_coverage_raises(self, tmp_path):
        p = self._csv(tmp_path, ["A10G", "C20R"])
        with pytest.raises(ValueError, match="covers none of the 2 library mutations"):
            load_ddG_prior_csv(p, ["E40K", "F50L"])

    def test_wt_mismatch_raises_with_offset_hint(self, tmp_path):
        # File numbered one lower than the library: library H29 is file H28.
        p = self._csv(tmp_path, ["H28A", "V29A", "S27A"])
        with pytest.raises(ValueError, match=r"wild-type residues disagree.*"
                                             r"H29A \(file wt V29\).*"
                                             r"residue_offset is 1"):
            load_ddG_prior_csv(p, ["H29A", "S28A"])

    def test_residue_offset_lines_up_numbering(self, tmp_path):
        p = self._csv(tmp_path, ["H28A", "V29A", "S27A"])
        out = load_ddG_prior_csv(p, ["H29A", "S28A"],
                                 thermo_options={"residue_offset": 1})
        np.testing.assert_array_equal(out["struct_features"][:, 0], [0.0, 2.0])

    def test_inconsistent_file_wt_raises(self, tmp_path):
        p = self._csv(tmp_path, ["A10G", "C10R"])
        with pytest.raises(ValueError, match="two wild-type residues"):
            load_ddG_prior_csv(p, ["A10G"])

    def test_blank_values_count_as_missing(self, tmp_path):
        p = str(tmp_path / "ddG.csv")
        pd.DataFrame({"mut": ["A10G", "C20R"], "H": [1.0, np.nan]}).to_csv(p, index=False)
        with pytest.warns(UserWarning, match="blank values: C20R"):
            out = load_ddG_prior_csv(p, ["A10G", "C20R"])
        assert out["struct_features"][1, 0] == 0.0


class TestEnsembleLikeCsv:
    """The 12-state ensemble CSV layout (see _ensemble_like_csv)."""

    _LIB = [_lib_label(29, "A"), _lib_label(30, "G"), _lib_label(52, "D")]  # H29A ...

    def test_c4_names_missing_HD(self, tmp_path):
        p = _ensemble_like_csv(tmp_path / "ens.csv")
        with pytest.raises(ValueError, match=r"missing structure column\(s\) \['HD'\]"):
            load_thermo_data(p, self._LIB, theta_name="thermo.O2_C4_K3_U0_a.PddG")

    def test_c12_numbering_offset_detected(self, tmp_path):
        p = _ensemble_like_csv(tmp_path / "ens.csv")
        with pytest.raises(ValueError, match=r"H29A \(file wt V29\).*residue_offset is 1"):
            load_thermo_data(p, self._LIB, theta_name="thermo.O2_C12_K5_U0_a.PddG")

    def test_c12_loads_with_offset(self, tmp_path):
        p = _ensemble_like_csv(tmp_path / "ens.csv")
        out = load_thermo_data(p, self._LIB, theta_name="thermo.O2_C12_K5_U1_a.PddG",
                               thermo_options={"residue_offset": 1})
        assert out["struct_names"] == _C12
        df = pd.read_csv(p).set_index("Mutation")
        np.testing.assert_allclose(out["struct_features"][0],
                                   df.loc["H28A", list(_C12)].to_numpy(), rtol=1e-6)

    def test_c4_loads_with_offset_and_map(self, tmp_path):
        p = _ensemble_like_csv(tmp_path / "ens.csv")
        out = load_thermo_data(p, self._LIB, theta_name="thermo.O2_C4_K3_U0_a.PddG",
                               thermo_options={"residue_offset": 1,
                                               "column_map": {"HD": "HO"}})
        df = pd.read_csv(p).set_index("Mutation")
        np.testing.assert_allclose(out["struct_features"][2],
                                   df.loc["V51D", ["H", "HO", "L", "LE2"]].to_numpy(),
                                   rtol=1e-6)

    def test_real_file_if_present(self):
        import os
        real = os.path.expanduser(
            "~/work/programming/git-clones/tfscreen/notebooks/other-data/"
            "ddG_ensemble_clean_with_effector.csv")
        if not os.path.exists(real):
            pytest.skip("real ensemble CSV not on this machine")
        with pytest.raises(ValueError, match=r"\['HD'\]"):
            load_thermo_data(real, self._LIB, theta_name="thermo.O2_C4_K3_U0_a.PddG")
        with pytest.raises(ValueError, match="residue_offset is 1"):
            load_thermo_data(real, self._LIB, theta_name="thermo.O2_C12_K5_U0_a.PddG")
        out = load_thermo_data(real, self._LIB, theta_name="thermo.O2_C12_K5_U0_a.PddG",
                               thermo_options={"residue_offset": 1})
        assert out["struct_features"].shape == (3, 6)


class TestStructEnsembleChecks:

    def test_missing_residue_reports_coverage(self, tmp_path):
        p = str(tmp_path / "ens.h5")
        _write_h5(p, {"H": {"res_nums": [10, 20]}})
        with pytest.raises(ValueError, match=r"1 of 2 library mutations are absent "
                                             r"\(1/2 covered.*D30E"):
            load_struct_ensemble(p, ["A10G", "D30E"])

    def test_residue_offset_applied(self, tmp_path):
        p = str(tmp_path / "ens.h5")
        logP = np.arange(40, dtype=np.float32).reshape(2, 20)
        _write_h5(p, {"H": {"res_nums": [9, 19], "logP": logP}})
        out = load_struct_ensemble(p, ["C20R"], thermo_options={"residue_offset": 1})
        np.testing.assert_array_equal(out["struct_features"][0, 0, :20], logP[1])

    def test_wt_aa_mismatch_raises(self, tmp_path):
        p = str(tmp_path / "ens.h5")
        _write_h5(p, {"H": {"res_nums": [9, 10, 11]}})
        with h5py.File(p, "a") as hf:
            hf["H"].create_dataset("wt_aa", data=["A", "V", "C"],
                                   dtype=h5py.string_dtype())
        with pytest.raises(ValueError, match=r"A10G \(file wt V10\).*residue_offset is 1"):
            load_struct_ensemble(p, ["A10G", "C12R"])

    def test_wt_aa_match_passes(self, tmp_path):
        p = str(tmp_path / "ens.h5")
        _write_h5(p, {"H": {"res_nums": [10, 11]}})
        with h5py.File(p, "a") as hf:
            hf["H"].create_dataset("wt_aa", data=["A", "X"], dtype=h5py.string_dtype())
        out = load_struct_ensemble(p, ["A10G", "C11R"])
        assert out["struct_features"].shape == (2, 1, 60)


class TestPddGStructPerm:

    @pytest.mark.parametrize("module, names", [
        ("O2_C4_K3_U0_a", _C4), ("O2_C4_K3_U1_a", _C4),
        ("O2_C12_K5_U0_a", _C12), ("O2_C12_K5_U1_a", _C12),
    ])
    def test_extras_ignored_missing_listed(self, module, names):
        import importlib
        import types
        mod = importlib.import_module(
            f"tfscreen.tfmodel.generative.components.theta.thermo.{module}.PddG")
        data = types.SimpleNamespace(struct_names=("unfolded",) + tuple(reversed(names)))
        perm = mod._get_struct_perm(data)
        assert [data.struct_names[i] for i in perm] == list(names)
        data = types.SimpleNamespace(struct_names=names[1:])
        with pytest.raises(ValueError, match=rf"lacks \['{names[0]}'\]"):
            mod._get_struct_perm(data)
