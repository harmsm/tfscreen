"""
Tests for the library-composition interface to ModelOrchestrator and
configure_model, which replaced the hand-supplied --spiked list.

The gate on the whole change is TestEquivalence: a library file whose spiked
origin is {G} must produce exactly the masks the old spiked_genotypes=[G]
produced.
"""

import os

import numpy as np
import pandas as pd
import pytest
import yaml

from tfscreen.genetics import library_composition_table, write_library_composition
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.configuration_io import write_configuration, read_configuration
from tfscreen.tfmodel.scripts.configure_model_cli import (
    configure_model,
    check_genotypes_in_library,
)


GENOTYPES = ["wt", "A2L", "A2C"]


@pytest.fixture
def library_config_dict():
    """Tiny library: wt and A2L are spiked AND in the bulk; A2C is bulk only."""
    return {
        "reading_frame": 0,
        "first_amplicon_residue": 1,
        "wt_seq":      "atggcaaaaccggaatgc",
        "degen_sites": "...nnt......nnt...",
        "tiles":       "...111......222...",
        "tile_combos": ["single-1", "single-2", "double-1-2"],
        "spiked_seqs": ["..................",     # wt
                        "...ctt............"],    # A2L
        "library_mixture": {"single-1": 10,
                            "single-2": 10,
                            "double-1-2": 100,
                            "spiked": 1},
    }


@pytest.fixture
def library_config_file(library_config_dict, tmp_path):
    path = tmp_path / "library.yaml"
    with open(path, "w") as f:
        yaml.dump(library_config_dict, f)
    return str(path)


@pytest.fixture
def library_file(library_config_dict, tmp_path):
    path = str(tmp_path / "tfs_library.csv")
    write_library_composition(library_composition_table(library_config_dict),
                              path)
    return path


def _growth_df(genotypes=GENOTYPES):
    rows = []
    for g in genotypes:
        for t in (0.0, 10.0):
            rows.append({"library": "lib",
                         "replicate": 1,
                         "time": t,
                         "genotype": g,
                         "ln_cfu": 1.0,
                         "ln_cfu_std": 0.1,
                         "condition_pre": "pre-cond",
                         "condition_sel": "sel+cond",
                         "t_pre": 1.0,
                         "t_sel": 10.0,
                         "titrant_name": "iptg",
                         "titrant_conc": 0.0})
    return pd.DataFrame(rows)


def _binding_df(genotypes=GENOTYPES):
    return pd.DataFrame([{"genotype": g,
                          "titrant_name": "iptg",
                          "titrant_conc": 0.0,
                          "theta_obs": 0.5,
                          "theta_std": 0.05}
                         for g in genotypes])


def _masks(orchestrator):
    idx = orchestrator.growth_tm.tensor_dim_names.index("genotype")
    labels = list(orchestrator.growth_tm.tensor_dim_labels[idx])
    return (labels,
            np.array(orchestrator.data.growth.congression_mask),
            np.array(orchestrator.data.growth.ln_cfu0_spiked_mask))


class TestEquivalence:
    """library_file must reproduce the legacy spiked_genotypes behavior."""

    def test_masks_match_legacy_spiked_list(self, library_file):
        growth_df, binding_df = _growth_df(), _binding_df()

        legacy = ModelOrchestrator(growth_df, binding_df,
                                   spiked_genotypes=["wt", "A2L"])
        from_library = ModelOrchestrator(growth_df, binding_df,
                                         library_file=library_file)

        legacy_labels, legacy_cong, legacy_spiked = _masks(legacy)
        lib_labels, lib_cong, lib_spiked = _masks(from_library)

        assert legacy_labels == lib_labels
        np.testing.assert_array_equal(legacy_cong, lib_cong)
        np.testing.assert_array_equal(legacy_spiked, lib_spiked)

    def test_spiked_genotypes_are_derived(self, library_file):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         library_file=library_file)
        assert sorted(orchestrator.settings["spiked_genotypes"]) == ["A2L", "wt"]
        assert orchestrator.settings["library_file"] == library_file

    def test_library_df_exposed(self, library_file):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         library_file=library_file)
        table = orchestrator.library_df
        # The whole library, not just the genotypes with growth data.
        assert len(table) > len(GENOTYPES)
        assert set(["pool_fraction", "bulk_fraction"]).issubset(table.columns)

    def test_no_library_file_leaves_everything_congressed(self):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df())
        _, congression, spiked = _masks(orchestrator)
        assert congression.all()
        assert not spiked.any()


class TestSpikedGenotypesAbsentFromData:
    """A library-derived spiked genotype with no growth data is not fatal."""

    def test_absent_spike_is_reported_not_raised(self, library_file, capsys):
        growth_df = _growth_df(["wt", "A2C"])       # A2L never observed
        orchestrator = ModelOrchestrator(growth_df, _binding_df(["wt", "A2C"]),
                                         library_file=library_file)

        assert orchestrator.settings["spiked_genotypes"] == ["wt"]
        assert "A2L" in capsys.readouterr().out

        labels, congression, spiked = _masks(orchestrator)
        assert not congression[labels.index("wt")]
        assert congression[labels.index("A2C")]

    def test_legacy_list_still_raises(self):
        with pytest.raises(ValueError, match="not found in the growth"):
            ModelOrchestrator(_growth_df(["wt", "A2C"]),
                              _binding_df(["wt", "A2C"]),
                              spiked_genotypes=["A2L"])


class TestMutualExclusion:

    def test_both_raises(self, library_file):
        with pytest.raises(ValueError, match="only one"):
            ModelOrchestrator(_growth_df(), _binding_df(),
                              spiked_genotypes=["wt"],
                              library_file=library_file)


class TestContentCheck:

    def test_passes_for_library_genotypes(self, library_config_dict):
        table = library_composition_table(library_config_dict)
        check_genotypes_in_library(set(table["genotype"]), _growth_df(),
                                   "growth_df")

    def test_unknown_sentinel_ignored(self, library_config_dict):
        table = library_composition_table(library_config_dict)
        growth_df = _growth_df(["wt", "__unknown__"])
        check_genotypes_in_library(set(table["genotype"]), growth_df,
                                   "growth_df")

    def test_undesigned_genotype_fails(self, library_config_dict):
        table = library_composition_table(library_config_dict)
        growth_df = _growth_df(["wt", "Q3W"])
        with pytest.raises(ValueError, match="Q3W"):
            check_genotypes_in_library(set(table["genotype"]), growth_df,
                                       "growth_df")

    def test_numbering_shift_fails(self, library_config_dict):
        """The presplit failure mode: residue numbering off by one."""
        table = library_composition_table(library_config_dict)
        # Same mutations, every residue number one lower.
        growth_df = _growth_df(["A1L", "A1C"])
        with pytest.raises(ValueError, match="not in the library"):
            check_genotypes_in_library(set(table["genotype"]), growth_df,
                                       "growth_df")


class TestConfigureModel:

    def _configure(self, tmp_path, library_config_file, **kwargs):
        growth_path = str(tmp_path / "growth.csv")
        binding_path = str(tmp_path / "binding.csv")
        _growth_df().to_csv(growth_path, index=False)
        _binding_df().to_csv(binding_path, index=False)

        out_prefix = str(tmp_path / "tfs_configure")
        configure_model(binding_df=binding_path,
                        growth_df=growth_path,
                        library_config=library_config_file,
                        out_prefix=out_prefix,
                        skip_model_stats=True,
                        **kwargs)
        return out_prefix

    def test_writes_snapshot_and_records_it(self, tmp_path, library_config_file):
        out_prefix = self._configure(tmp_path, library_config_file)

        library_path = f"{out_prefix}_library.csv"
        assert os.path.exists(library_path)

        with open(f"{out_prefix}_config.yaml") as f:
            config = yaml.safe_load(f)

        assert config["library_file"] == os.path.basename(library_path)
        assert config["library"]["source"] == library_config_file
        assert config["library"]["library_mixture"]["spiked"] == 1
        # One source of truth: the derived list is not also written out.
        assert "spiked_genotypes" not in config["components"]

    def test_round_trip_reproduces_masks(self, tmp_path, library_config_file):
        out_prefix = self._configure(tmp_path, library_config_file)
        orchestrator, _ = read_configuration(f"{out_prefix}_config.yaml")
        assert sorted(orchestrator.settings["spiked_genotypes"]) == ["A2L", "wt"]
        _, congression, spiked = _masks(orchestrator)
        assert spiked.any()
        assert not congression.all()

    def test_requires_library_config_with_growth(self, tmp_path):
        growth_path = str(tmp_path / "growth.csv")
        binding_path = str(tmp_path / "binding.csv")
        _growth_df().to_csv(growth_path, index=False)
        _binding_df().to_csv(binding_path, index=False)

        with pytest.raises(ValueError, match="library_config is required"):
            configure_model(binding_df=binding_path,
                            growth_df=growth_path,
                            out_prefix=str(tmp_path / "tfs_configure"),
                            skip_model_stats=True)

    def test_binding_only_rejects_library_config(self, tmp_path,
                                                 library_config_file):
        binding_path = str(tmp_path / "binding.csv")
        _binding_df().to_csv(binding_path, index=False)

        with pytest.raises(ValueError, match="binding-only"):
            configure_model(binding_df=binding_path,
                            library_config=library_config_file,
                            out_prefix=str(tmp_path / "tfs_configure"),
                            skip_model_stats=True)

    def test_undesigned_genotype_fails_configure(self, tmp_path,
                                                 library_config_file):
        growth_path = str(tmp_path / "growth.csv")
        binding_path = str(tmp_path / "binding.csv")
        _growth_df(["wt", "Q3W"]).to_csv(growth_path, index=False)
        _binding_df(["wt", "Q3W"]).to_csv(binding_path, index=False)

        with pytest.raises(ValueError, match="growth_df"):
            configure_model(binding_df=binding_path,
                            growth_df=growth_path,
                            library_config=library_config_file,
                            out_prefix=str(tmp_path / "tfs_configure"),
                            skip_model_stats=True)


class TestLegacyConfig:

    def test_config_with_spiked_genotypes_still_loads(self, tmp_path):
        """Configs written before this change must keep working."""
        growth_path = str(tmp_path / "growth.csv")
        binding_path = str(tmp_path / "binding.csv")
        _growth_df().to_csv(growth_path, index=False)
        _binding_df().to_csv(binding_path, index=False)

        orchestrator = ModelOrchestrator(growth_path, binding_path,
                                         spiked_genotypes=["wt", "A2L"])
        out_prefix = str(tmp_path / "legacy")
        write_configuration(orchestrator=orchestrator,
                            out_prefix=out_prefix,
                            growth_df_path=growth_path,
                            binding_df_path=binding_path)

        with open(f"{out_prefix}_config.yaml") as f:
            config = yaml.safe_load(f)
        assert config["components"]["spiked_genotypes"] == ["wt", "A2L"]
        assert "library_file" not in config

        reloaded, _ = read_configuration(f"{out_prefix}_config.yaml")
        assert reloaded.settings["spiked_genotypes"] == ["wt", "A2L"]
