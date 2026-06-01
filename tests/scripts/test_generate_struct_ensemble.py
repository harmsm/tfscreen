"""
Unit tests for scripts/generate_struct_ensemble.py.

Tests cover: _parse_pdb_ca, _build_min_dist_matrix, _aggregate_logP,
_parse_resnum, _load_config.  LigandMPNN subprocess calls and file I/O
are not tested here; those require the external binary.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Import module under test from the scripts/ directory (not a package).
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).parent.parent.parent / 'scripts' / 'generate_struct_ensemble.py'
_spec = importlib.util.spec_from_file_location('generate_struct_ensemble', _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_parse_pdb_ca         = _mod._parse_pdb_ca
_build_min_dist_matrix = _mod._build_min_dist_matrix
_aggregate_logP        = _mod._aggregate_logP
_parse_resnum          = _mod._parse_resnum
_load_config           = _mod._load_config


# ---------------------------------------------------------------------------
# Helper: construct a valid PDB ATOM record for a Cα atom.
# ---------------------------------------------------------------------------

def _ca_line(chain, resnum, x, y, z, serial=1, resname='ALA'):
    """
    Return one PDB ATOM line for a Cα atom.

    Column layout (0-indexed):
      0-3   ATOM
      6-10  serial
      12-15 atom name ( CA )
      17-19 residue name
      21    chain ID
      22-25 residue seq number
      30-37 x (8.3f)
      38-45 y (8.3f)
      46-53 z (8.3f)
    """
    return (
        f"ATOM  {serial:5d}  CA  {resname:3s} {chain}{resnum:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
    )


def _write_pdb(tmp_path, lines, name='test.pdb'):
    p = tmp_path / name
    p.write_text('\n'.join(lines) + '\n')
    return str(p)


# ===========================================================================
# _parse_resnum
# ===========================================================================

class TestParseResnum:
    def test_standard(self):
        assert _parse_resnum('A42') == 42

    def test_chain_b(self):
        assert _parse_resnum('B100') == 100

    def test_single_digit(self):
        assert _parse_resnum('A1') == 1

    def test_negative(self):
        assert _parse_resnum('A-1') == -1

    def test_large_number(self):
        assert _parse_resnum('C1234') == 1234

    def test_insertion_code_stripped(self):
        # "A42A" — insertion code is part of the label; only the integer is returned
        assert _parse_resnum('A42A') == 42

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match='Cannot parse'):
            _parse_resnum('42')   # no leading chain letter

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_resnum('')


# ===========================================================================
# _parse_pdb_ca
# ===========================================================================

class TestParsePdbCa:
    def test_monomer_single_residue(self, tmp_path):
        pdb = _write_pdb(tmp_path, [_ca_line('A', 1, 1.0, 2.0, 3.0)])
        result = _parse_pdb_ca(pdb)
        assert list(result.keys()) == ['A']
        np.testing.assert_allclose(result['A'][1], [1.0, 2.0, 3.0], atol=1e-3)

    def test_monomer_two_residues(self, tmp_path):
        lines = [
            _ca_line('A', 1, 0.0, 0.0, 0.0, serial=1),
            _ca_line('A', 2, 3.0, 4.0, 0.0, serial=2),
        ]
        result = _parse_pdb_ca(_write_pdb(tmp_path, lines))
        assert set(result['A'].keys()) == {1, 2}
        np.testing.assert_allclose(result['A'][2], [3.0, 4.0, 0.0], atol=1e-3)

    def test_homodimer_two_chains(self, tmp_path):
        lines = [
            _ca_line('A', 1, 0.0, 0.0, 0.0, serial=1),
            _ca_line('B', 1, 5.0, 0.0, 0.0, serial=2),
        ]
        result = _parse_pdb_ca(_write_pdb(tmp_path, lines))
        assert set(result.keys()) == {'A', 'B'}
        np.testing.assert_allclose(result['A'][1], [0.0, 0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(result['B'][1], [5.0, 0.0, 0.0], atol=1e-3)

    def test_non_ca_atoms_skipped(self, tmp_path):
        ca_record  = _ca_line('A', 1, 1.0, 2.0, 3.0)
        cb_record  = ca_record[:12] + ' CB ' + ca_record[16:]   # swap atom name
        pdb = _write_pdb(tmp_path, [ca_record, cb_record])
        result = _parse_pdb_ca(pdb)
        assert list(result['A'].keys()) == [1]   # only CA was parsed

    def test_hetatm_skipped(self, tmp_path):
        hetatm = _ca_line('A', 1, 9.0, 9.0, 9.0).replace('ATOM  ', 'HETATM')
        ca     = _ca_line('A', 2, 1.0, 2.0, 3.0, serial=2)
        result = _parse_pdb_ca(_write_pdb(tmp_path, [hetatm, ca]))
        assert list(result['A'].keys()) == [2]   # HETATM skipped

    def test_alt_conf_keeps_first(self, tmp_path):
        first  = _ca_line('A', 1, 1.0, 0.0, 0.0, serial=1)
        second = _ca_line('A', 1, 9.0, 0.0, 0.0, serial=2)   # duplicate resnum
        result = _parse_pdb_ca(_write_pdb(tmp_path, [first, second]))
        np.testing.assert_allclose(result['A'][1], [1.0, 0.0, 0.0], atol=1e-3)

    def test_no_ca_atoms_raises(self, tmp_path):
        cb_record = _ca_line('A', 1, 0.0, 0.0, 0.0).replace('  CA  ', '  CB  ')
        with pytest.raises(ValueError, match='No Cα atoms'):
            _parse_pdb_ca(_write_pdb(tmp_path, [cb_record]))


# ===========================================================================
# _build_min_dist_matrix
# ===========================================================================

class TestBuildMinDistMatrix:
    def test_monomer_diagonal_is_zero(self):
        chains = {'A': {1: np.array([0., 0., 0.]), 2: np.array([3., 4., 0.])}}
        mat = _build_min_dist_matrix(chains, [1, 2])
        assert mat[0, 0] == pytest.approx(0.0)
        assert mat[1, 1] == pytest.approx(0.0)

    def test_monomer_off_diagonal(self):
        chains = {'A': {1: np.array([0., 0., 0.]), 2: np.array([3., 4., 0.])}}
        mat = _build_min_dist_matrix(chains, [1, 2])
        # distance = sqrt(3^2 + 4^2) = 5.0
        assert mat[0, 1] == pytest.approx(5.0, abs=1e-3)
        assert mat[1, 0] == pytest.approx(5.0, abs=1e-3)

    def test_symmetric(self):
        chains = {
            'A': {1: np.array([0., 0., 0.]), 2: np.array([3., 4., 0.])},
            'B': {1: np.array([1., 0., 0.]), 2: np.array([4., 4., 0.])},
        }
        mat = _build_min_dist_matrix(chains, [1, 2])
        np.testing.assert_allclose(mat, mat.T, atol=1e-5)

    def test_dimer_inter_chain_contact_captured(self):
        """
        Intra-chain distance between residues 1 and 2 is 20 Å, but the
        inter-chain distance (B1 to A2) is only 5 Å.  The matrix must
        return 5 Å for that pair.
        """
        chains = {
            'A': {1: np.array([0.,  0., 0.]),
                  2: np.array([20., 0., 0.])},
            'B': {1: np.array([15., 0., 0.]),   # B1 is 5 Å from A2
                  2: np.array([35., 0., 0.])},
        }
        mat = _build_min_dist_matrix(chains, [1, 2])
        # intra-chain: 20 Å; inter-chain B1→A2: 5 Å
        assert mat[0, 1] == pytest.approx(5.0, abs=1e-3)

    def test_missing_residue_gets_999(self):
        chains = {'A': {1: np.array([0., 0., 0.])}}
        mat = _build_min_dist_matrix(chains, [1, 2])   # residue 2 not in PDB
        assert mat[0, 1] == pytest.approx(999.0)
        assert mat[1, 0] == pytest.approx(999.0)

    def test_output_shape(self):
        chains = {'A': {i: np.zeros(3) for i in [1, 2, 3, 4, 5]}}
        mat = _build_min_dist_matrix(chains, [1, 2, 3, 4, 5])
        assert mat.shape == (5, 5)

    def test_output_dtype_float32(self):
        chains = {'A': {1: np.array([0., 0., 0.], dtype=np.float32)}}
        mat = _build_min_dist_matrix(chains, [1])
        assert mat.dtype == np.float32


# ===========================================================================
# _aggregate_logP
# ===========================================================================

class TestAggregateLogP:
    def _uniform_logP(self, resnums, n_chains=1, seed=0):
        """Return (L_raw, 20) logP matrix and res_nums_raw for testing."""
        rng = np.random.default_rng(seed)
        unique = sorted(set(resnums))
        rows = {}
        for r in unique:
            rows[r] = rng.standard_normal(20).astype(np.float32)
        logP_raw  = np.array([rows[r] for r in resnums for _ in range(n_chains)],
                             dtype=np.float32)
        nums_raw  = np.array([r for r in resnums for _ in range(n_chains)],
                             dtype=np.int32)
        # Interleave chains (chain A then B per residue)
        logP_interleaved = np.array([rows[r] for r in resnums for _ in range(n_chains)],
                                    dtype=np.float32)
        nums_interleaved = np.array([r for r in resnums for _ in range(n_chains)],
                                    dtype=np.int32)
        return logP_interleaved, nums_interleaved, rows

    def test_single_chain_identity(self):
        """Single chain: aggregated output should equal the input."""
        rng = np.random.default_rng(1)
        logP_raw = rng.standard_normal((3, 20)).astype(np.float32)
        res_nums = np.array([10, 20, 30], dtype=np.int32)
        logP_agg, unique_nums = _aggregate_logP(logP_raw, res_nums)
        np.testing.assert_allclose(logP_agg, logP_raw, atol=1e-6)
        np.testing.assert_array_equal(unique_nums, [10, 20, 30])

    def test_two_chains_identical_values_unchanged(self):
        """Two chains with the same logP: average equals input."""
        rng = np.random.default_rng(2)
        row = rng.standard_normal(20).astype(np.float32)
        # Both chain copies have identical logP
        logP_raw = np.stack([row, row])      # (2, 20)
        res_nums = np.array([5, 5], dtype=np.int32)
        logP_agg, unique_nums = _aggregate_logP(logP_raw, res_nums)
        assert logP_agg.shape == (1, 20)
        np.testing.assert_allclose(logP_agg[0], row, atol=1e-6)
        np.testing.assert_array_equal(unique_nums, [5])

    def test_two_chains_different_values_averaged(self):
        """Two chains with different logP: result is the arithmetic mean."""
        rowA = np.array([1.0] * 20, dtype=np.float32)
        rowB = np.array([3.0] * 20, dtype=np.float32)
        logP_raw = np.stack([rowA, rowB])    # (2, 20)
        res_nums = np.array([7, 7], dtype=np.int32)
        logP_agg, _ = _aggregate_logP(logP_raw, res_nums)
        np.testing.assert_allclose(logP_agg[0], 2.0, atol=1e-6)

    def test_unique_resnums_are_sorted(self):
        """Output residue numbers must be in ascending sorted order."""
        logP_raw = np.zeros((3, 20), dtype=np.float32)
        res_nums = np.array([30, 10, 20], dtype=np.int32)
        _, unique_nums = _aggregate_logP(logP_raw, res_nums)
        assert list(unique_nums) == [10, 20, 30]

    def test_multiple_residues_two_chains(self):
        """Dimer with two residues: each position averaged independently."""
        rowA1 = np.full(20, 2.0, dtype=np.float32)
        rowA2 = np.full(20, 4.0, dtype=np.float32)
        rowB1 = np.full(20, 6.0, dtype=np.float32)
        rowB2 = np.full(20, 8.0, dtype=np.float32)
        # LigandMPNN order: A1, A2, B1, B2 (both chains sequentially)
        logP_raw = np.stack([rowA1, rowA2, rowB1, rowB2])
        res_nums = np.array([1, 2, 1, 2], dtype=np.int32)
        logP_agg, unique_nums = _aggregate_logP(logP_raw, res_nums)
        assert logP_agg.shape == (2, 20)
        np.testing.assert_allclose(logP_agg[0], 4.0, atol=1e-6)  # (2+6)/2
        np.testing.assert_allclose(logP_agg[1], 6.0, atol=1e-6)  # (4+8)/2
        np.testing.assert_array_equal(unique_nums, [1, 2])

    def test_output_dtype_float32(self):
        logP_raw = np.ones((2, 20), dtype=np.float64)
        res_nums = np.array([1, 2], dtype=np.int32)
        logP_agg, _ = _aggregate_logP(logP_raw, res_nums)
        assert logP_agg.dtype == np.float32

    def test_unique_resnums_dtype_int32(self):
        logP_raw = np.ones((2, 20), dtype=np.float32)
        res_nums = np.array([1, 2], dtype=np.int32)
        _, unique_nums = _aggregate_logP(logP_raw, res_nums)
        assert unique_nums.dtype == np.int32


# ===========================================================================
# _load_config
# ===========================================================================

class TestLoadConfig:
    def _write_yaml(self, tmp_path, obj, name='cfg.yaml'):
        p = tmp_path / name
        p.write_text(yaml.dump(obj))
        return str(p)

    def _touch_pdb(self, tmp_path, name):
        p = tmp_path / name
        p.write_text('REMARK minimal\n')
        return str(p)

    def test_valid_config(self, tmp_path):
        pdb = self._touch_pdb(tmp_path, 'H.pdb')
        cfg = {'n_chains_bearing_mut': 2, 'structures': {'H': pdb}}
        structures, n_chains = _load_config(self._write_yaml(tmp_path, cfg))
        assert n_chains == 2
        assert 'H' in structures
        assert structures['H'] == str(Path(pdb).resolve())

    def test_default_n_chains_is_1(self, tmp_path):
        pdb = self._touch_pdb(tmp_path, 'H.pdb')
        cfg = {'structures': {'H': pdb}}
        _, n_chains = _load_config(self._write_yaml(tmp_path, cfg))
        assert n_chains == 1

    def test_multiple_structures(self, tmp_path):
        pdbs = {name: self._touch_pdb(tmp_path, f'{name}.pdb')
                for name in ('H', 'HD', 'L', 'LE2')}
        cfg = {'n_chains_bearing_mut': 2,
               'structures': {name: path for name, path in pdbs.items()}}
        structures, _ = _load_config(self._write_yaml(tmp_path, cfg))
        assert set(structures.keys()) == {'H', 'HD', 'L', 'LE2'}

    def test_old_flat_format_raises(self, tmp_path):
        pdb = self._touch_pdb(tmp_path, 'H.pdb')
        cfg = {'H': pdb}   # old format: no 'structures' key
        with pytest.raises(ValueError, match="structures"):
            _load_config(self._write_yaml(tmp_path, cfg))

    def test_missing_pdb_raises(self, tmp_path):
        cfg = {'structures': {'H': '/nonexistent/path/H.pdb'}}
        with pytest.raises(FileNotFoundError):
            _load_config(self._write_yaml(tmp_path, cfg))

    def test_paths_are_absolute(self, tmp_path):
        pdb = self._touch_pdb(tmp_path, 'H.pdb')
        cfg = {'structures': {'H': pdb}}
        structures, _ = _load_config(self._write_yaml(tmp_path, cfg))
        assert Path(structures['H']).is_absolute()

    def test_non_dict_yaml_raises(self, tmp_path):
        p = tmp_path / 'bad.yaml'
        p.write_text('- a\n- b\n')
        with pytest.raises(ValueError, match='mapping'):
            _load_config(str(p))

    def test_empty_structures_raises(self, tmp_path):
        cfg = {'structures': {}}
        with pytest.raises(ValueError, match='non-empty'):
            _load_config(self._write_yaml(tmp_path, cfg))
