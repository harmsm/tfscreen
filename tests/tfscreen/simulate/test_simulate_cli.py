"""Tests for tfs-simulate CLI (run_simulation_from_config)."""

import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from tfscreen.simulate.scripts.simulate_cli import (
    replicate_read_totals,
    run_simulation_from_config,
)
from tfscreen.util.cli.generalized_main import generalized_main

_RAW = "tfscreen.simulate.scripts.simulate_cli.write_raw_experiment"


@pytest.fixture(autouse=True)
def _stub_raw_output(request):
    """The mocked frames here carry no tube totals; the raw writer has its
    own tests (test_raw_output.py). Tests that check the call patch it."""
    if "raw_call" in request.keywords:
        yield None
        return
    with patch(_RAW, return_value={"tubes": "t.csv", "counts_dir": "c",
                                   "dropped": [], "command": "x"}) as m:
        yield m


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

    def fake_run(config_file, out_prefix="tfs_sim",
                 num_replicates=2, seed=None):
        captured["seed"] = seed

    generalized_main(
        fake_run,
        argv=["config.yaml", "--seed", "42"],
        manual_arg_types={"seed": int},
    )
    assert captured["seed"] == 42
    assert isinstance(captured["seed"], int)


def test_seed_cli_defaults_to_none():
    """Omitting --seed leaves seed as None."""
    captured = {}

    def fake_run(config_file, out_prefix="tfs_sim",
                 num_replicates=2, seed=None):
        captured["seed"] = seed

    generalized_main(
        fake_run,
        argv=["config.yaml"],
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

    with patch("tfscreen.util.read_yaml", return_value={"seed": 99, "growth": {}}) as mock_yaml, \
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
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"), seed=7)

    mock_rng.assert_called_once_with(7)


def test_seed_none_preserves_config(patched_simulation):
    """When seed=None the config's seed is left unchanged."""
    mock_yaml, tmp_path = patched_simulation

    with patch("tfscreen.simulate.scripts.simulate_cli.np.random.default_rng",
               wraps=lambda s: MagicMock()) as mock_rng:
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"), seed=None)

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

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0, "growth": {}}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    written = "\n".join(written_paths)
    assert "parameters" in written, "parameters.csv must be written"
    assert "phenotype" not in written, "phenotype.csv must NOT be written"


def test_output_file_names_include_expected_stems(tmp_path):
    """library, parameters, genotype_theta, and growth CSVs are all written."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    written_paths = []

    def capture_csv(self_df, path, **kwargs):
        written_paths.append(str(path))

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0, "growth": {}}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    written = "\n".join(written_paths)
    for stem in ("library", "parameters", "genotype_theta", "growth"):
        assert stem in written, f"Expected '{stem}' CSV to be written"


# ---------------------------------------------------------------------------
# growth_parameters: always written, from cf['growth']
# ---------------------------------------------------------------------------

def test_growth_parameters_csv_always_written(tmp_path):
    """growth_parameters CSV is written unconditionally (cf['growth'] is
    always required, unlike the optional *_data blocks)."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    written = {}

    def capture_csv(self_df, path, **kwargs):
        written[str(path)] = self_df.copy()

    cfg = {"seed": 0, "growth": {"kanR+kan": {"model": "linear", "b": 0.005, "m": -0.01}}}

    with patch("tfscreen.util.read_yaml", return_value=cfg), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    matches = [p for p in written if "growth_parameters" in os.path.basename(p)]
    assert len(matches) == 1
    result = written[matches[0]]
    assert result.loc[0, "condition_rep"] == "kanR+kan"
    assert result.loc[0, "growth_k"] == pytest.approx(0.005)
    assert result.loc[0, "growth_m"] == pytest.approx(-0.01)


def test_growth_parameters_csv_written_for_real(tmp_path):
    """Real (unmocked) to_csv write produces a readable growth_parameters.csv."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    cfg = {"seed": 0, "growth": {"M9+kan": {"model": "saturation", "kmin": 0.001, "kmax": 0.04}}}

    with patch("tfscreen.util.read_yaml", return_value=cfg), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df):
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    growth_params_path = tmp_path / "tfs_sim_growth_parameters.csv"
    assert growth_params_path.exists()
    df = pd.read_csv(growth_params_path)
    assert df.loc[0, "condition_rep"] == "M9+kan"
    assert df.loc[0, "growth_min"] == pytest.approx(0.001)
    assert df.loc[0, "growth_max"] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# transformation_lam: always written, from cf['transformation_poisson_lambda']
# ---------------------------------------------------------------------------

def test_transformation_lam_csv_always_written(tmp_path):
    """transformation_lam CSV is written unconditionally, mirroring
    growth_parameters (transformation_poisson_lambda is a top-level,
    always-present-but-possibly-None config key, not an optional *_data
    block)."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    written = {}

    def capture_csv(self_df, path, **kwargs):
        written[str(path)] = self_df.copy()

    cfg = {"seed": 0, "growth": {}, "transformation_poisson_lambda": 1.5}

    with patch("tfscreen.util.read_yaml", return_value=cfg), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    matches = [p for p in written if "transformation_lam" in os.path.basename(p)]
    assert len(matches) == 1
    result = written[matches[0]]
    assert result.loc[0, "parameter"] == "lam"
    assert result.loc[0, "ref"] == pytest.approx(1.5)


def test_transformation_lam_csv_written_for_real(tmp_path):
    """Real (unmocked) to_csv write produces a readable transformation_lam.csv."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    cfg = {"seed": 0, "growth": {}, "transformation_poisson_lambda": None}

    with patch("tfscreen.util.read_yaml", return_value=cfg), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df):
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    lam_path = tmp_path / "tfs_sim_transformation_lam.csv"
    assert lam_path.exists()
    df = pd.read_csv(lam_path)
    assert df.loc[0, "parameter"] == "lam"
    assert df.loc[0, "ref"] == pytest.approx(0.0)


def test_transformation_lam_participates_in_existence_guard(tmp_path):
    """Pre-existing transformation_lam CSV triggers the same FileExistsError
    guard as the other always-written outputs."""
    (tmp_path / "tfs_sim_transformation_lam.csv").write_text("parameter,ref\nlam,1.5\n")

    cfg = {"seed": 0, "growth": {}, "transformation_poisson_lambda": 1.5}
    with patch("tfscreen.util.read_yaml", return_value=cfg):
        with pytest.raises(FileExistsError, match="transformation_lam"):
            run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))


# ---------------------------------------------------------------------------
# base_growth_data: written only when configured, using parameters_df's dk_geno
# ---------------------------------------------------------------------------

def test_base_growth_not_written_without_config(tmp_path):
    """No base_growth CSV is written when 'base_growth_data' is absent."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    written_paths = []

    def capture_csv(self_df, path, **kwargs):
        written_paths.append(str(path))

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0, "growth": {}}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    basenames = "\n".join(os.path.basename(p) for p in written_paths)
    assert "base_growth" not in basenames


def test_base_growth_written_when_configured(tmp_path):
    """A 'base_growth_data' block produces a base_growth CSV computed from
    parameters_df's dk_geno column."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    written = {}

    def capture_csv(self_df, path, **kwargs):
        written[str(path)] = self_df.copy()

    cfg = {"seed": 0, "growth": {}, "base_growth_data": {"k_ref": 0.025}}

    with patch("tfscreen.util.read_yaml", return_value=cfg), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    base_growth_paths = [p for p in written if "base_growth" in os.path.basename(p)]
    assert len(base_growth_paths) == 1
    result = written[base_growth_paths[0]]
    assert set(result["genotype"]) == {"wt"}
    assert float(result.loc[result["genotype"] == "wt", "rate"].iloc[0]) == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# k_ref: written only alongside base_growth_data, echoing its k_ref value
# ---------------------------------------------------------------------------

def test_k_ref_not_written_without_base_growth_config(tmp_path):
    """No k_ref CSV is written when 'base_growth_data' is absent."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    written_paths = []

    def capture_csv(self_df, path, **kwargs):
        written_paths.append(str(path))

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0, "growth": {}}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    basenames = "\n".join(os.path.basename(p) for p in written_paths)
    assert "k_ref" not in basenames


def test_k_ref_written_when_base_growth_configured(tmp_path):
    """A 'base_growth_data' block also produces a single-row k_ref CSV
    echoing the configured k_ref value."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    written = {}

    def capture_csv(self_df, path, **kwargs):
        written[str(path)] = self_df.copy()

    cfg = {"seed": 0, "growth": {}, "base_growth_data": {"k_ref": 0.031}}

    with patch("tfscreen.util.read_yaml", return_value=cfg), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv", capture_csv):
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    k_ref_paths = [p for p in written if "k_ref" in os.path.basename(p)]
    assert len(k_ref_paths) == 1
    result = written[k_ref_paths[0]]
    assert result.loc[0, "parameter"] == "k_ref"
    assert result.loc[0, "ref"] == pytest.approx(0.031)


# ---------------------------------------------------------------------------
# presplit_data: written only when configured
# ---------------------------------------------------------------------------

def test_run_simulation_writes_presplit_csv(tmp_path):
    """presplit CSV is written when presplit_data block is in the config."""
    lib_df   = pd.DataFrame({"genotype": ["wt", "A1V"]})
    pheno_df = pd.DataFrame({"genotype": ["wt", "A1V"]})
    theta_df = pd.DataFrame({"genotype": ["wt", "A1V"]})
    params_df = pd.DataFrame({"genotype": ["wt", "A1V"],
                               "dk_geno": [0.0, 0.0], "activity": [1.0, 1.0]})

    # Sample/counts for one (replicate, condition_pre, t_sel) combo
    # Real _simulate_library_group returns sample_df with "sample" as a
    # regular column and an unnamed integer index.
    sample_df = pd.DataFrame([{
        "sample": 0, "replicate": 1, "library": "lib", "condition_pre": "kanR",
        "t_sel": 60.0, "sample_cfu": 1e8, "sample_cfu_std": 5e6,
    }], index=[0])
    counts_df = pd.DataFrame([
        {"sample": 0, "genotype": "wt",  "counts": 500, "ln_cfu_0": 10.0},
        {"sample": 0, "genotype": "A1V", "counts": 500, "ln_cfu_0": 10.0},
    ])
    growth_df = pd.DataFrame({"genotype": ["wt", "A1V"], "ln_cfu": [10.0, 9.9]})

    cf = {
        "seed": 1,
        "growth": {},
        "cfu0": 1e8,
        "total_num_reads": 10_000_000,
        "prob_index_hop": None,
        "presplit_data": {"noise": 0.0},
    }

    with patch("tfscreen.util.read_yaml", return_value=cf), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df):
        run_simulation_from_config("fake_config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    presplit_path = tmp_path / "tfs_sim_presplit.csv"
    assert presplit_path.exists(), "presplit CSV was not written"
    presplit_df = pd.read_csv(presplit_path)
    for col in ["library", "replicate", "condition_pre", "genotype",
                "ln_cfu", "ln_cfu_std"]:
        assert col in presplit_df.columns


def test_run_simulation_no_presplit_without_config(tmp_path):
    """presplit CSV is NOT written when presplit_data is absent from config."""
    lib_df   = pd.DataFrame({"genotype": ["wt"]})
    pheno_df = pd.DataFrame({"genotype": ["wt"]})
    theta_df = pd.DataFrame({"genotype": ["wt"]})
    params_df = pd.DataFrame({"genotype": ["wt"], "dk_geno": [0.0], "activity": [1.0]})
    sample_df = pd.DataFrame([{"sample": 0, "replicate": 1, "library": "lib",
                                "condition_pre": "kanR", "t_sel": 60.0,
                                "sample_cfu": 1e8, "sample_cfu_std": 5e6}],
                              index=[0])
    counts_df = pd.DataFrame([{"sample": 0, "genotype": "wt",
                                "counts": 1000, "ln_cfu_0": 10.0}])
    growth_df = pd.DataFrame({"genotype": ["wt"], "ln_cfu": [10.0]})

    cf = {"seed": 1, "growth": {}, "cfu0": 1e8, "total_num_reads": 1_000_000,
          "prob_index_hop": None}

    with patch("tfscreen.util.read_yaml", return_value=cf), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df):
        run_simulation_from_config("fake_config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    presplit_path = tmp_path / "tfs_sim_presplit.csv"
    assert not presplit_path.exists(), "presplit CSV should not be written without config block"


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

    with patch("tfscreen.util.read_yaml", return_value={"seed": 5, "growth": {}}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=growth_df), \
         patch.object(pd.DataFrame, "to_csv"), \
         patch("tfscreen.simulate.scripts.simulate_cli.yaml.dump", capture_dump):
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "test"))

    assert "data" in dumped, "yaml.dump was not called"
    assert dumped["data"]["seed"] == 5


def test_input_config_yaml_existence_check(tmp_path):
    """If input-config.yaml already exists, FileExistsError is raised before any work."""
    lib_df, pheno_df, theta_df, params_df, sample_df, counts_df, growth_df = _make_mock_dfs()

    existing_yaml = tmp_path / "tfs_sim_input-config.yaml"
    existing_yaml.write_text("seed: 0\n")

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0, "growth": {}}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df, None)) as mock_lib:
        with pytest.raises(FileExistsError, match="input-config.yaml"):
            run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))

    mock_lib.assert_not_called()


# ---------------------------------------------------------------------------
# od600 block: per-tube readings and OD-only replicates (roadmap step 4)
# ---------------------------------------------------------------------------

def _od_run(tmp_path, cf):
    lib_df = pd.DataFrame({"genotype": ["wt"]})
    params_df = pd.DataFrame({"genotype": ["wt"], "dk_geno": [0.0],
                              "activity": [1.0]})
    calls = []

    def fake_selection(rep_cf, library_df, phenotype_df, shared_state=None,
                       sequence=True):
        calls.append({"sequence": sequence, "shared_state": shared_state,
                      "reads": rep_cf.get("total_num_reads")})
        rep = int(phenotype_df["replicate"].iloc[0])
        sample_df = pd.DataFrame([{
            "sample": 0, "replicate": rep, "library": "lib",
            "condition_sel": "kanR+kan", "t_sel": 60.0, "sample_cfu": 1e8,
            "sample_cfu_std": 1e6, "sample_cfu_true": 1e8, "od600": 0.3,
            "od600_detectable": True, "od600_in_range": True}], index=[0])
        if not sequence:
            return sample_df, pd.DataFrame(columns=["sample", "genotype"])
        counts_df = pd.DataFrame([{"sample": 0, "genotype": "wt",
                                   "counts": 10, "ln_cfu_0": 1.0}])
        return sample_df, counts_df

    with patch("tfscreen.util.read_yaml", return_value=cf), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, lib_df.copy(), lib_df.copy(), params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               side_effect=fake_selection), \
         patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu",
               return_value=pd.DataFrame({"genotype": ["wt"]})):
        run_simulation_from_config("fake_config.yaml", out_prefix=str(tmp_path / "tfs_sim"),
                                   num_replicates=2)
    return calls


def test_od600_written_with_od_only_replicates(tmp_path):
    cf = {"seed": 1, "growth": {}, "cfu0": 1e8, "total_num_reads": 100,
          "od600": {"calibration": "c.yaml", "num_od_only_replicates": 1}}
    calls = _od_run(tmp_path, cf)

    assert [c["sequence"] for c in calls] == [True, True, False]
    # One shared_state dict passed to every replicate.
    assert all(c["shared_state"] is calls[0]["shared_state"] for c in calls)

    od = pd.read_csv(tmp_path / "tfs_sim_od600.csv")
    assert list(od["replicate"]) == [1, 2, 3]
    assert list(od["sequenced"]) == [True, True, False]
    for col in ("od600", "od600_detectable", "od600_in_range",
                "sample_cfu_true", "sample_cfu"):
        assert col in od.columns


def test_total_reads_split_across_replicates(tmp_path):
    """total_num_reads is the total over all sequenced tubes, not per
    replicate (each replicate used to get the full total)."""
    cf = {"seed": 1, "growth": {}, "cfu0": 1e8, "total_num_reads": 1000,
          "od600": {"calibration": "c.yaml", "num_od_only_replicates": 1}}
    calls = _od_run(tmp_path, cf)
    assert [c["reads"] for c in calls if c["sequence"]] == [500, 500]


def test_replicate_read_totals_by_design_tube_count():
    design = pd.DataFrame({"replicate": [1] * 60 + [2] * 56 + [3] * 2})
    reads = replicate_read_totals(118_000, [1, 2, 3], 3, design)
    assert reads == {1: 60_000, 2: 56_000, 3: 2_000}
    assert replicate_read_totals(90, [1, 2, 3], 3) == {1: 30, 2: 30, 3: 30}
    # OD600-only replicates get no reads
    assert replicate_read_totals(90, [1, 2, 3], 2) == {1: 45, 2: 45}


def test_no_od600_file_without_block(tmp_path):
    cf = {"seed": 1, "growth": {}, "cfu0": 1e8, "total_num_reads": 100}
    calls = _od_run(tmp_path, cf)
    assert [c["sequence"] for c in calls] == [True, True]
    assert not (tmp_path / "tfs_sim_od600.csv").exists()


# ---------------------------------------------------------------------------
# Raw-format output
# ---------------------------------------------------------------------------

def test_raw_output_written_by_default(patched_simulation, _stub_raw_output):
    mock_yaml, tmp_path = patched_simulation
    run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))
    assert _stub_raw_output.call_count == 1
    args, kwargs = _stub_raw_output.call_args
    assert args[3] == str(tmp_path / "tfs_sim")
    assert list(args[2]) == ["wt"]
    assert kwargs["od600_config"] is None


def test_raw_output_can_be_skipped(patched_simulation, _stub_raw_output):
    mock_yaml, tmp_path = patched_simulation
    run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"),
                               write_raw=False)
    assert _stub_raw_output.call_count == 0


def test_growth_table_can_be_skipped(patched_simulation):
    mock_yaml, tmp_path = patched_simulation
    with patch("tfscreen.simulate.scripts.simulate_cli.counts_to_lncfu") as c2l:
        run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"),
                                   write_growth=False)
    # nothing needs the in-memory growth table, so it is not built
    assert c2l.call_count == 0


# ---------------------------------------------------------------------------
# Designs and raw-output messages
# ---------------------------------------------------------------------------

def _one_replicate_design():
    return pd.DataFrame({
        "sample": ["k-1", "k-2"], "library": "kanR", "replicate": 4,
        "condition_pre": "kanR-kan", "t_pre": 30, "condition_sel": "kanR+kan",
        "t_sel": [100, 120], "titrant_name": "iptg", "titrant_conc": 0.0})


class _Stop(Exception):
    pass


def test_design_sets_num_replicates(tmp_path, capsys):
    """A design's replicates replace num_replicates, with a message."""
    design = _one_replicate_design()
    design.to_csv(tmp_path / "design.csv", index=False)
    cf = {"seed": 1, "growth": {}, "cfu0": 1e8,
          "design": str(tmp_path / "design.csv")}
    lib_df = pd.DataFrame({"genotype": ["wt"]})
    pheno = design.drop(columns=["sample", "replicate"]).assign(genotype="wt")
    params_df = pd.DataFrame({"genotype": ["wt"], "dk_geno": [0.0],
                              "activity": [1.0]})
    seen = {}

    def fake_selection(rep_cf, library_df, phenotype_df, shared_state=None,
                       sequence=True):
        seen["replicate"] = phenotype_df["replicate"].unique().tolist()
        seen["t_sel"] = sorted(phenotype_df["t_sel"])
        raise _Stop

    with patch("tfscreen.util.read_yaml", return_value=cf), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, pheno, lib_df.copy(), params_df, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               side_effect=fake_selection):
        with pytest.raises(_Stop):
            run_simulation_from_config("cfg.yaml", out_prefix=str(tmp_path / "s"),
                                       num_replicates=2)
    assert "Design has 1 replicate(s) [4]; num_replicates (2) is ignored." \
        in capsys.readouterr().out
    assert seen == {"replicate": [4], "t_sel": [100, 120]}


def test_design_refuses_od_only_replicates(tmp_path):
    _one_replicate_design().to_csv(tmp_path / "design.csv", index=False)
    cf = {"seed": 1, "growth": {}, "cfu0": 1e8,
          "design": str(tmp_path / "design.csv"),
          "od600": {"calibration": "c.yaml", "num_od_only_replicates": 1}}
    with pytest.raises(ValueError, match="num_od_only_replicates cannot be"):
        _od_run(tmp_path, cf)


def test_raw_output_reports_dropped_tubes(patched_simulation, _stub_raw_output,
                                          capsys):
    mock_yaml, tmp_path = patched_simulation
    _stub_raw_output.return_value = {"tubes": "t.csv", "counts_dir": "c",
                                     "od600": "o.csv", "calibration": "cal.yaml",
                                     "dropped": ["tube0003"], "command": "x"}
    run_simulation_from_config("config.yaml", out_prefix=str(tmp_path / "tfs_sim"))
    out = capsys.readouterr().out
    assert "Wrote the raw experiment: t.csv, c/, o.csv, cal.yaml" in out
    assert "OD600 below the detection threshold): ['tube0003']" in out
