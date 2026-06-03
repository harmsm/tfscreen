"""Tests for tfs-simulate CLI (run_simulation_from_config)."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from tfscreen.simulate.scripts.run_simulation_cli import run_simulation_from_config
from tfscreen.util.cli.generalized_main import generalized_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_dfs():
    lib_df = pd.DataFrame({"genotype": ["wt"], "sub_library": ["spiked"]})
    pheno_df = pd.DataFrame({"genotype": ["wt"]})
    theta_df = pd.DataFrame({"genotype": ["wt"]})
    params_df = pd.DataFrame({"genotype": ["wt"], "dk_geno": [0.0], "activity": [1.0]})
    sample_df = pd.DataFrame({"sample": [0]}, index=[0])
    counts_df = pd.DataFrame({"sample": [0], "genotype": ["wt"], "counts": [1]})
    growth_df = pd.DataFrame({"genotype": ["wt"]})
    return lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df


# ---------------------------------------------------------------------------
# --seed CLI argument
# ---------------------------------------------------------------------------

def test_seed_cli_parsed_as_int():
    """--seed is registered with type=int so integer strings are accepted."""
    captured = {}

    def fake_run(config_file, output_dir, output_prefix="tfs_sim_",
                 num_replicates=1, seed=None):
        captured["seed"] = seed

    generalized_main(
        fake_run,
        argv=["config.yaml", "out_dir", "--seed", "42"],
        manual_arg_types={"seed": int},
    )
    assert captured["seed"] == 42
    assert isinstance(captured["seed"], int)


def test_seed_cli_defaults_to_none():
    """Omitting --seed leaves seed as None."""
    captured = {}

    def fake_run(config_file, output_dir, output_prefix="tfs_sim_",
                 num_replicates=1, seed=None):
        captured["seed"] = seed

    generalized_main(
        fake_run,
        argv=["config.yaml", "out_dir"],
        manual_arg_types={"seed": int},
    )
    assert captured["seed"] is None


# ---------------------------------------------------------------------------
# seed override behaviour in run_simulation_from_config
# ---------------------------------------------------------------------------

@pytest.fixture()
def patched_simulation(tmp_path):
    """Patch all I/O so run_simulation_from_config can run without real data."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    with patch("tfscreen.util.read_yaml", return_value={"random_seed": 99}) as mock_yaml, \
         patch("tfscreen.simulate.scripts.run_simulation_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df)), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv"):
        yield mock_yaml, tmp_path


def test_seed_overrides_config(patched_simulation):
    """When seed is given it replaces random_seed before seeding the RNG."""
    mock_yaml, tmp_path = patched_simulation

    with patch("tfscreen.simulate.scripts.run_simulation_cli.np.random.default_rng",
               wraps=lambda s: MagicMock()) as mock_rng:
        run_simulation_from_config("config.yaml", str(tmp_path), seed=7)

    mock_rng.assert_called_once_with(7)


def test_seed_none_preserves_config(patched_simulation):
    """When seed=None the config's random_seed is left unchanged."""
    mock_yaml, tmp_path = patched_simulation

    with patch("tfscreen.simulate.scripts.run_simulation_cli.np.random.default_rng",
               wraps=lambda s: MagicMock()) as mock_rng:
        run_simulation_from_config("config.yaml", str(tmp_path), seed=None)

    mock_rng.assert_called_once_with(99)


# ---------------------------------------------------------------------------
# Output file names
# ---------------------------------------------------------------------------

def test_writes_parameters_not_phenotype(tmp_path):
    """run_simulation_from_config must write parameters.csv, not phenotype.csv."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    written_paths = []

    def capture_csv(self_df, path, **kwargs):
        written_paths.append(str(path))

    with patch("tfscreen.util.read_yaml", return_value={"random_seed": 0}), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df)), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", str(tmp_path))

    written = "\n".join(written_paths)
    assert "parameters" in written, "parameters.csv must be written"
    assert "phenotype" not in written, "phenotype.csv must NOT be written"


def test_output_file_names_include_expected_stems(tmp_path):
    """library, parameters, genotype_theta, and growth CSVs are all written."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    written_paths = []

    def capture_csv(self_df, path, **kwargs):
        written_paths.append(str(path))

    with patch("tfscreen.util.read_yaml", return_value={"random_seed": 0}), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df)), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", str(tmp_path))

    written = "\n".join(written_paths)
    for stem in ("library", "parameters", "genotype_theta", "growth"):
        assert stem in written, f"Expected '{stem}' CSV to be written"
