"""
Unit tests for generate_ligandmpnn_features.py

_parse_resnum, _score_structure, and main() are tested with subprocess and
torch mocked throughout — no GPU or LigandMPNN installation is required.
"""
import os
import subprocess
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Import-isolation regression test
# ---------------------------------------------------------------------------

def test_no_jax_or_numpyro_imported():
    """
    Importing generate_ligandmpnn_features must not pull in jax or numpyro. This
    is because (as of right now) LigandMPNN uses a python 3.11 environment that
    is incompatible with the jax version we rely on; version hell. 

    This guards against regressions where an eager import in a parent
    __init__.py re-introduces the dependency chain that broke the LigandMPNN
    environment (which has no JAX).  The check runs in a subprocess so that
    jax/numpyro already loaded in the test process do not mask the problem.
    """
    code = (
        "import sys; "
        "import tfscreen.analysis.hierarchical.growth_model.scripts.generate_ligandmpnn_features; "
        "bad = [m for m in sys.modules if m == 'jax' or m == 'numpyro']; "
        "sys.exit(1) if bad else sys.exit(0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        "Importing generate_ligandmpnn_features pulled in jax or numpyro.\n"
        f"stderr: {result.stderr}"
    )

from tfscreen.analysis.hierarchical.growth_model.scripts.generate_ligandmpnn_features import (
    _parse_resnum,
    _score_structure,
    _ALPHABET_20,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOD = "tfscreen.analysis.hierarchical.growth_model.scripts.generate_ligandmpnn_features"

def _make_fake_torch_output(res_names, probs_by_res):
    """
    Build the dict that torch.load would return from a LigandMPNN .pt file.

    Parameters
    ----------
    res_names : list[str]
        Residue identifier strings, e.g. ["A1", "A2"].
    probs_by_res : list[dict[str, float]]
        Per-residue dicts mapping AA letter → probability (need not sum to 1
        for test purposes).
    """
    residue_names = {i: name for i, name in enumerate(res_names)}
    mean_of_probs = {name: probs for name, probs in zip(res_names, probs_by_res)}
    return {"residue_names": residue_names, "mean_of_probs": mean_of_probs}


def _score_structure_args(score_py="/fake/score.py",
                          pdb_path="/fake/struct.pdb",
                          out_dir="/tmp/out",
                          model_type="ligand_mpnn",
                          checkpoint=None,
                          num_batches=3,
                          seed=42):
    return score_py, pdb_path, out_dir, model_type, checkpoint, num_batches, seed


# ---------------------------------------------------------------------------
# _parse_resnum
# ---------------------------------------------------------------------------

class TestParseResnum:

    def test_standard_residue(self):
        assert _parse_resnum("A42") == 42

    def test_chain_b(self):
        assert _parse_resnum("B100") == 100

    def test_large_residue_number(self):
        assert _parse_resnum("A999") == 999

    def test_negative_residue_number(self):
        assert _parse_resnum("A-3") == -3

    def test_insertion_code_ignored(self):
        # "A42A" — insertion code suffix — parses as residue 42
        assert _parse_resnum("A42A") == 42

    def test_insertion_code_B(self):
        assert _parse_resnum("Z10B") == 10

    def test_lowercase_chain_letter(self):
        # regex allows [A-Za-z] for chain
        assert _parse_resnum("a5") == 5

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse residue name"):
            _parse_resnum("bad")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Cannot parse residue name"):
            _parse_resnum("")


# ---------------------------------------------------------------------------
# _score_structure
# ---------------------------------------------------------------------------

class TestScoreStructure:
    """
    All tests mock subprocess.run and torch.load; no LigandMPNN required.
    torch itself is injected via sys.modules patching so the lazy `import torch`
    inside _score_structure finds a MagicMock regardless of installation state.
    """

    # Minimal two-residue LigandMPNN output --------------------------------

    _RES_NAMES = ["A1", "A2"]
    _PROBS = [
        {"A": 0.8, "C": 0.1},   # residue A1 — all other AAs get 0.0 → clamped
        {"G": 0.5, "W": 0.3},   # residue A2
    ]

    def _run(self, tmp_path, model_type="ligand_mpnn", checkpoint=None,
             primary_exists=True):
        """
        Common runner: mocks subprocess + torch, creates the .pt file at the
        expected path, and calls _score_structure.
        """
        score_py  = str(tmp_path / "score.py")
        pdb_path  = str(tmp_path / "struct.pdb")
        out_dir   = str(tmp_path / "out")
        os.makedirs(os.path.join(out_dir, "score"), exist_ok=True)

        pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]
        pt_primary  = os.path.join(out_dir, "score", f"{pdb_stem}.pt")
        pt_fallback = os.path.join(out_dir, f"{pdb_stem}.pt")

        # Create a real (empty) file at whichever location we want to be "found"
        if primary_exists:
            open(pt_primary, "wb").close()
        else:
            open(pt_fallback, "wb").close()

        fake_output = _make_fake_torch_output(self._RES_NAMES, self._PROBS)
        mock_torch = MagicMock()
        mock_torch.load.return_value = fake_output

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("subprocess.run") as mock_run:
                logP, res_nums = _score_structure(
                    score_py, pdb_path, out_dir,
                    model_type=model_type,
                    checkpoint=checkpoint,
                    num_batches=3,
                    seed=99,
                )
        return logP, res_nums, mock_run, mock_torch

    # --- subprocess command construction -----------------------------------

    def test_subprocess_called_once(self, tmp_path):
        _, _, mock_run, _ = self._run(tmp_path)
        assert mock_run.call_count == 1

    def test_command_contains_required_flags(self, tmp_path):
        _, _, mock_run, _ = self._run(tmp_path)
        cmd = mock_run.call_args[0][0]
        assert "--single_aa_score" in cmd
        assert "1" in cmd
        assert "--use_sequence" in cmd
        assert "--number_of_batches" in cmd
        assert "3" in cmd
        assert "--seed" in cmd
        assert "99" in cmd

    def test_command_cwd_is_ligandmpnn_dir(self, tmp_path):
        score_py = str(tmp_path / "score.py")
        pdb_path = str(tmp_path / "struct.pdb")
        out_dir  = str(tmp_path / "out")
        os.makedirs(os.path.join(out_dir, "score"), exist_ok=True)
        pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]
        open(os.path.join(out_dir, "score", f"{pdb_stem}.pt"), "wb").close()

        fake_output = _make_fake_torch_output(self._RES_NAMES, self._PROBS)
        mock_torch = MagicMock()
        mock_torch.load.return_value = fake_output

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("subprocess.run") as mock_run:
                _score_structure(score_py, pdb_path, out_dir,
                                 model_type="ligand_mpnn",
                                 checkpoint=None, num_batches=1, seed=0)

        kwargs = mock_run.call_args[1]
        assert kwargs["cwd"] == os.path.dirname(score_py)

    def test_no_checkpoint_flag_when_none(self, tmp_path):
        _, _, mock_run, _ = self._run(tmp_path, checkpoint=None)
        cmd = mock_run.call_args[0][0]
        assert "--checkpoint_ligand_mpnn" not in cmd
        assert "--checkpoint_protein_mpnn" not in cmd

    def test_checkpoint_flag_ligand_mpnn(self, tmp_path):
        _, _, mock_run, _ = self._run(
            tmp_path, model_type="ligand_mpnn", checkpoint="/ckpt/weights.pt"
        )
        cmd = mock_run.call_args[0][0]
        assert "--checkpoint_ligand_mpnn" in cmd
        assert "/ckpt/weights.pt" in cmd

    def test_checkpoint_flag_protein_mpnn(self, tmp_path):
        score_py = str(tmp_path / "score.py")
        pdb_path = str(tmp_path / "struct.pdb")
        out_dir  = str(tmp_path / "out")
        os.makedirs(os.path.join(out_dir, "score"), exist_ok=True)
        pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]
        open(os.path.join(out_dir, "score", f"{pdb_stem}.pt"), "wb").close()

        fake_output = _make_fake_torch_output(self._RES_NAMES, self._PROBS)
        mock_torch = MagicMock()
        mock_torch.load.return_value = fake_output

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("subprocess.run") as mock_run:
                _score_structure(score_py, pdb_path, out_dir,
                                 model_type="protein_mpnn",
                                 checkpoint="/ckpt/prot.pt",
                                 num_batches=1, seed=0)

        cmd = mock_run.call_args[0][0]
        assert "--checkpoint_protein_mpnn" in cmd
        assert "--checkpoint_ligand_mpnn" not in cmd

    # --- .pt file location -------------------------------------------------

    def test_primary_pt_path_used_when_present(self, tmp_path):
        logP, res_nums, _, mock_torch = self._run(tmp_path, primary_exists=True)
        # If primary was used, torch.load was called and we got valid arrays
        assert logP.shape == (2, 20)
        assert mock_torch.load.call_count == 1

    def test_fallback_pt_path_used_when_primary_absent(self, tmp_path):
        logP, res_nums, _, mock_torch = self._run(tmp_path, primary_exists=False)
        assert logP.shape == (2, 20)
        assert mock_torch.load.call_count == 1

    def test_file_not_found_error_when_no_pt_file(self, tmp_path):
        score_py = str(tmp_path / "score.py")
        pdb_path = str(tmp_path / "struct.pdb")
        out_dir  = str(tmp_path / "out")
        os.makedirs(out_dir, exist_ok=True)
        # Neither primary nor fallback .pt exists

        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("subprocess.run"):
                with pytest.raises(FileNotFoundError, match="LigandMPNN output not found"):
                    _score_structure(score_py, pdb_path, out_dir,
                                     model_type="ligand_mpnn",
                                     checkpoint=None, num_batches=1, seed=0)

    # --- output array correctness ------------------------------------------

    def test_logP_shape(self, tmp_path):
        logP, _, _, _ = self._run(tmp_path)
        assert logP.shape == (2, 20)
        assert logP.dtype == np.float32

    def test_res_nums_shape_and_dtype(self, tmp_path):
        _, res_nums, _, _ = self._run(tmp_path)
        assert res_nums.shape == (2,)
        assert res_nums.dtype == np.int32

    def test_res_nums_values(self, tmp_path):
        _, res_nums, _, _ = self._run(tmp_path)
        # "A1" → 1, "A2" → 2
        np.testing.assert_array_equal(res_nums, [1, 2])

    def test_logP_known_value(self, tmp_path):
        logP, _, _, _ = self._run(tmp_path)
        aa_idx = {aa: i for i, aa in enumerate(_ALPHABET_20)}
        # residue A1 (row 0): prob("A")=0.8 → log(0.8)
        assert logP[0, aa_idx["A"]] == pytest.approx(np.log(0.8), rel=1e-5)
        # residue A2 (row 1): prob("G")=0.5 → log(0.5)
        assert logP[1, aa_idx["G"]] == pytest.approx(np.log(0.5), rel=1e-5)

    def test_missing_aa_clamped_to_1e10(self, tmp_path):
        """AAs absent from mean_of_probs get prob=0.0, clamped to 1e-10."""
        logP, _, _, _ = self._run(tmp_path)
        aa_idx = {aa: i for i, aa in enumerate(_ALPHABET_20)}
        # residue A1: "W" not in probs → clamped
        assert logP[0, aa_idx["W"]] == pytest.approx(np.log(1e-10), rel=1e-5)

    def test_torch_scalar_unwrapped(self, tmp_path):
        """Prob values that are torch tensors (have .item()) are unwrapped."""
        score_py = str(tmp_path / "score.py")
        pdb_path = str(tmp_path / "struct.pdb")
        out_dir  = str(tmp_path / "out")
        os.makedirs(os.path.join(out_dir, "score"), exist_ok=True)
        pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]
        open(os.path.join(out_dir, "score", f"{pdb_stem}.pt"), "wb").close()

        # Wrap one prob value as a fake torch scalar
        fake_tensor = MagicMock()
        fake_tensor.item.return_value = 0.6
        fake_output = _make_fake_torch_output(
            ["A1"], [{"A": fake_tensor}]
        )
        mock_torch = MagicMock()
        mock_torch.load.return_value = fake_output

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("subprocess.run"):
                logP, _ = _score_structure(score_py, pdb_path, out_dir,
                                           model_type="ligand_mpnn",
                                           checkpoint=None, num_batches=1, seed=0)

        aa_idx = {aa: i for i, aa in enumerate(_ALPHABET_20)}
        assert logP[0, aa_idx["A"]] == pytest.approx(np.log(0.6), rel=1e-5)
        fake_tensor.item.assert_called_once()

    def test_torch_import_error_propagated(self, tmp_path):
        """If torch is not installed, ImportError with helpful message is raised."""
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="PyTorch is required"):
                _score_structure("/p/score.py", "/p/a.pdb", str(tmp_path),
                                 model_type="ligand_mpnn",
                                 checkpoint=None, num_batches=1, seed=0)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:

    def _write_yaml(self, path, content):
        import yaml
        with open(path, "w") as f:
            yaml.dump(content, f)

    def _run_main(self, argv):
        with patch.object(sys, "argv", argv):
            main()

    # --- happy path --------------------------------------------------------

    def test_happy_path_writes_npz(self, tmp_path):
        """main() calls _score_structure for each structure and saves an NPZ."""
        pdb1 = tmp_path / "A.pdb"
        pdb2 = tmp_path / "B.pdb"
        pdb1.touch()
        pdb2.touch()
        yaml_path = tmp_path / "structs.yaml"
        self._write_yaml(str(yaml_path), {"H": str(pdb1), "HD": str(pdb2)})

        score_py = tmp_path / "score.py"
        score_py.touch()
        out_npz = tmp_path / "out.npz"

        fake_logP     = np.zeros((5, 20), dtype=np.float32)
        fake_res_nums = np.arange(1, 6, dtype=np.int32)

        with patch(f"{_MOD}._score_structure",
                   return_value=(fake_logP, fake_res_nums)) as mock_score:
            self._run_main([
                "prog",
                str(yaml_path),
                "--out", str(out_npz),
                "--ligandmpnn_dir", str(tmp_path),
            ])

        assert mock_score.call_count == 2
        assert out_npz.exists()

        arrays = np.load(str(out_npz))
        assert "H"              in arrays
        assert "H_residue_nums" in arrays
        assert "HD"             in arrays
        assert "HD_residue_nums" in arrays
        np.testing.assert_array_equal(arrays["H"], fake_logP)
        np.testing.assert_array_equal(arrays["H_residue_nums"], fake_res_nums)

    def test_score_structure_receives_correct_args(self, tmp_path):
        """CLI options are forwarded correctly to _score_structure."""
        pdb = tmp_path / "A.pdb"
        pdb.touch()
        yaml_path = tmp_path / "structs.yaml"
        self._write_yaml(str(yaml_path), {"H": str(pdb)})
        (tmp_path / "score.py").touch()
        out_npz = tmp_path / "out.npz"

        fake_logP     = np.zeros((3, 20), dtype=np.float32)
        fake_res_nums = np.array([1, 2, 3], dtype=np.int32)

        with patch(f"{_MOD}._score_structure",
                   return_value=(fake_logP, fake_res_nums)) as mock_score:
            self._run_main([
                "prog",
                str(yaml_path),
                "--out", str(out_npz),
                "--ligandmpnn_dir", str(tmp_path),
                "--model_type", "protein_mpnn",
                "--checkpoint", "/ckpt/w.pt",
                "--num_batches", "7",
                "--seed", "123",
            ])

        _, kwargs = mock_score.call_args
        assert kwargs["model_type"] == "protein_mpnn"
        assert kwargs["checkpoint"] == "/ckpt/w.pt"
        assert kwargs["num_batches"] == 7
        assert kwargs["seed"] == 123

    # --- input validation errors -------------------------------------------

    def test_missing_yaml_raises_system_exit(self, tmp_path):
        with pytest.raises(SystemExit):
            self._run_main([
                "prog", str(tmp_path / "missing.yaml"),
                "--out", "out.npz",
                "--ligandmpnn_dir", str(tmp_path),
            ])

    def test_empty_yaml_raises_system_exit(self, tmp_path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("{}")
        (tmp_path / "score.py").touch()
        with pytest.raises(SystemExit):
            self._run_main([
                "prog", str(yaml_path),
                "--out", "out.npz",
                "--ligandmpnn_dir", str(tmp_path),
            ])

    def test_non_mapping_yaml_raises_system_exit(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("- item1\n- item2\n")
        (tmp_path / "score.py").touch()
        with pytest.raises(SystemExit):
            self._run_main([
                "prog", str(yaml_path),
                "--out", "out.npz",
                "--ligandmpnn_dir", str(tmp_path),
            ])

    def test_missing_score_py_raises_system_exit(self, tmp_path):
        pdb = tmp_path / "A.pdb"
        pdb.touch()
        yaml_path = tmp_path / "structs.yaml"
        self._write_yaml(str(yaml_path), {"H": str(pdb)})
        # No score.py created → parser.error()
        with pytest.raises(SystemExit):
            self._run_main([
                "prog", str(yaml_path),
                "--out", "out.npz",
                "--ligandmpnn_dir", str(tmp_path),
            ])

    def test_missing_pdb_raises_system_exit(self, tmp_path):
        yaml_path = tmp_path / "structs.yaml"
        self._write_yaml(str(yaml_path), {"H": str(tmp_path / "missing.pdb")})
        (tmp_path / "score.py").touch()
        with pytest.raises(SystemExit):
            self._run_main([
                "prog", str(yaml_path),
                "--out", "out.npz",
                "--ligandmpnn_dir", str(tmp_path),
            ])
