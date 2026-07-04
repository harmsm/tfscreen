"""Tests for tfs-simulate CLI (run_simulation_from_config)."""

import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from tfscreen.simulate.scripts.simulate_cli import run_simulation_from_config
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
                 num_replicates=2, seed=None):
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
                 num_replicates=2, seed=None):
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

    with patch("tfscreen.util.read_yaml", return_value={"seed": 99}) as mock_yaml, \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv"):
        yield mock_yaml, tmp_path


def test_seed_overrides_config(patched_simulation):
    """When seed is given it replaces seed before seeding the RNG."""
    mock_yaml, tmp_path = patched_simulation

    with patch("tfscreen.simulate.scripts.simulate_cli.np.random.default_rng",
               wraps=lambda s: MagicMock()) as mock_rng:
        run_simulation_from_config("config.yaml", str(tmp_path), seed=7)

    mock_rng.assert_called_once_with(7)


def test_seed_none_preserves_config(patched_simulation):
    """When seed=None the config's seed is left unchanged."""
    mock_yaml, tmp_path = patched_simulation

    with patch("tfscreen.simulate.scripts.simulate_cli.np.random.default_rng",
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

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
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

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", str(tmp_path))

    written = "\n".join(written_paths)
    for stem in ("library", "parameters", "genotype_theta", "growth"):
        assert stem in written, f"Expected '{stem}' CSV to be written"


# ---------------------------------------------------------------------------
# base_growth_data: written only when configured, using parameters_df's dk_geno
# ---------------------------------------------------------------------------

def test_base_growth_not_written_without_config(tmp_path):
    """No base_growth CSV is written when 'base_growth_data' is absent."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    written_paths = []

    def capture_csv(self_df, path, **kwargs):
        written_paths.append(str(path))

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", str(tmp_path))

    basenames = "\n".join(os.path.basename(p) for p in written_paths)
    assert "base_growth" not in basenames


def test_base_growth_written_when_configured(tmp_path):
    """A 'base_growth_data' block produces a base_growth CSV computed from
    parameters_df's dk_geno column."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    written = {}

    def capture_csv(self_df, path, **kwargs):
        written[str(path)] = self_df.copy()

    cfg = {"seed": 0, "base_growth_data": {"k_ref": 0.025}}

    with patch("tfscreen.util.read_yaml", return_value=cfg), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", str(tmp_path))

    base_growth_paths = [p for p in written if "base_growth" in os.path.basename(p)]
    assert len(base_growth_paths) == 1
    result = written[base_growth_paths[0]]
    assert set(result["genotype"]) == {"wt"}
    assert float(result.loc[result["genotype"] == "wt", "rate"].iloc[0]) == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# _generate_binding_data: uses pre-computed theta from library_prediction
# ---------------------------------------------------------------------------

def test_generate_binding_data_uses_precomputed_theta():
    """_generate_binding_data must use theta_true from binding_theta_df,
    not re-sample from the prior."""
    import numpy as np
    from tfscreen.simulate.scripts.simulate_cli import _generate_binding_data

    binding_cfg = {
        "genotypes": ["wt", "M1A"],
        "titrant_name": "iptg",
        "titrant_conc": [0.0, 1.0],
        "noise": 0.0,
    }
    rng = np.random.default_rng(0)
    binding_theta_df = pd.DataFrame([
        {"genotype": "wt",  "titrant_conc": 0.0, "theta_true": 0.1},
        {"genotype": "wt",  "titrant_conc": 1.0, "theta_true": 0.9},
        {"genotype": "M1A", "titrant_conc": 0.0, "theta_true": 0.2},
        {"genotype": "M1A", "titrant_conc": 1.0, "theta_true": 0.8},
    ])

    result = _generate_binding_data(binding_cfg, rng, binding_theta_df)

    assert set(result.columns) >= {"genotype", "titrant_name", "titrant_conc",
                                   "theta_obs", "theta_std"}
    wt_row = result[(result["genotype"] == "wt") & (result["titrant_conc"] == 1.0)]
    assert float(wt_row["theta_obs"].iloc[0]) == pytest.approx(0.9)


def test_generate_binding_data_missing_genotype_raises():
    """Raises ValueError when a genotype/conc pair is absent from binding_theta_df."""
    import numpy as np
    from tfscreen.simulate.scripts.simulate_cli import _generate_binding_data

    binding_cfg = {
        "genotypes": ["wt", "A2V"],   # A2V is valid but not in binding_theta_df
        "titrant_name": "iptg",
        "titrant_conc": [1.0],
        "noise": 0.0,
    }
    rng = np.random.default_rng(0)
    binding_theta_df = pd.DataFrame([
        {"genotype": "wt", "titrant_conc": 1.0, "theta_true": 0.5},
    ])

    with pytest.raises(ValueError, match="No pre-computed theta"):
        _generate_binding_data(binding_cfg, rng, binding_theta_df)


# ---------------------------------------------------------------------------
# Rejecting the old theta_rng_seed key
# ---------------------------------------------------------------------------

def test_theta_rng_seed_rejected_as_unknown_key(tmp_path):
    """A config containing theta_rng_seed must be rejected with an error
    (it is no longer a recognized key)."""
    from tfscreen.simulate.selection_experiment import _check_cf

    cf = {
        "theta_component": "hill_geno",
        "theta_rng_seed": 0,       # old key — must now be unknown
        "condition_blocks": [],
        "growth": {},
        "transform_sizes": {},
        "library_mixture": {},
        "lib_assembly_skew_sigma": 0.0,
        "transformation_poisson_lambda": 1,
        "multi_plasmid_combine_fcn": "gmean",
        "cfu0": 1e7,
        "tube_noise_sigma": 0.0,
        "total_num_reads": 1000,
        "prob_index_hop": 0.0,
        "seed": 0,
    }
    with pytest.raises(Exception):   # check_unknown_keys raises ValueError
        _check_cf(cf)


# ---------------------------------------------------------------------------
# input-config.yaml output
# ---------------------------------------------------------------------------

def test_writes_input_config_yaml(tmp_path):
    """run_simulation_from_config must write an input-config.yaml using yaml.dump."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    import tfscreen.simulate.scripts.simulate_cli as cli_mod

    dumped = {}

    def capture_dump(data, fh, **kwargs):
        dumped["data"] = data

    with patch("tfscreen.util.read_yaml", return_value={"seed": 5}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv"), \
         patch("tfscreen.simulate.scripts.simulate_cli.yaml.dump", capture_dump):
        run_simulation_from_config("config.yaml", str(tmp_path), output_prefix="test_")

    assert "data" in dumped, "yaml.dump was not called"
    assert dumped["data"]["seed"] == 5


def test_input_config_yaml_existence_check(tmp_path):
    """If input-config.yaml already exists, FileExistsError is raised before any work."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    existing_yaml = tmp_path / "tfs_sim_input-config.yaml"
    existing_yaml.write_text("seed: 0\n")

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)) as mock_lib:
        with pytest.raises(FileExistsError, match="input-config.yaml"):
            run_simulation_from_config("config.yaml", str(tmp_path))

    mock_lib.assert_not_called()
