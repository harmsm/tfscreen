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
            ~np.array(orchestrator.data.growth.ln_cfu0_spiked_mask),
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


def _bulk_fraction(orchestrator):
    idx = orchestrator.growth_tm.tensor_dim_names.index("genotype")
    labels = list(orchestrator.growth_tm.tensor_dim_labels[idx])
    return dict(zip(labels, np.array(orchestrator.data.growth.bulk_fraction)))


class TestBulkFraction:
    """Congression purity (bulk_fraction) is carried separately from the
    ln_cfu0 prior class (ln_cfu0_spiked_mask)."""

    def test_from_library_table(self, library_file):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         library_file=library_file)
        bf = _bulk_fraction(orchestrator)
        table = orchestrator.library_df.set_index("genotype")["bulk_fraction"]
        for g in GENOTYPES:
            assert bf[g] == pytest.approx(table[g])
        # wt and A2L are spiked *and* in the bulk; A2C is bulk only.
        assert 0 < bf["wt"] < 1
        assert 0 < bf["A2L"] < 1
        assert bf["A2C"] == pytest.approx(1.0)

    def test_ln_cfu0_class_unchanged(self, library_file):
        """Spiked-origin genotypes keep the spiked ln_cfu0 class even though
        they are mostly bulk."""
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         library_file=library_file)
        labels, _, spiked = _masks(orchestrator)
        assert spiked[labels.index("wt")] and spiked[labels.index("A2L")]
        assert not spiked[labels.index("A2C")]

    def test_legacy_spiked_list_is_binary(self):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         spiked_genotypes=["wt", "A2L"])
        assert _bulk_fraction(orchestrator) == {"wt": 0.0, "A2L": 0.0,
                                                "A2C": 1.0}

    def test_no_spike_information_is_all_bulk(self):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df())
        assert set(_bulk_fraction(orchestrator).values()) == {1.0}

    def test_unknown_bucket_is_bulk(self, library_file):
        growth_df = _growth_df(["wt", "A2C", "__unknown__"])
        orchestrator = ModelOrchestrator(growth_df,
                                         _binding_df(["wt", "A2C"]),
                                         library_file=library_file)
        assert _bulk_fraction(orchestrator)["__unknown__"] == 1.0

    def test_genotype_missing_from_table_raises(self, library_file):
        table = pd.read_csv(library_file)
        table[table["genotype"] != "A2C"].to_csv(library_file, index=False)
        with pytest.raises(ValueError, match="missing from the library"):
            ModelOrchestrator(_growth_df(), _binding_df(),
                              library_file=library_file)

    def test_bad_value_raises(self, library_file):
        table = pd.read_csv(library_file)
        table.loc[table["genotype"] == "A2C", "bulk_fraction"] = 1.5
        table.to_csv(library_file, index=False)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ModelOrchestrator(_growth_df(), _binding_df(),
                              library_file=library_file)

    def test_library_sized_under_batching(self, library_file):
        from tfscreen.tfmodel.tensors.batch import get_batch
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         library_file=library_file)
        full = np.array(orchestrator.data.growth.bulk_fraction)
        batch = get_batch(orchestrator.data, np.array([2, 0]))
        np.testing.assert_array_equal(np.array(batch.growth.bulk_fraction), full)


def _labels(orchestrator):
    idx = orchestrator.growth_tm.tensor_dim_names.index("genotype")
    return list(orchestrator.growth_tm.tensor_dim_labels[idx])


class TestCoresidentSets:
    """Fixed co-resident sets for the congression mixture (step 3.3b)."""

    def test_shapes_and_strata(self, library_file):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         library_file=library_file)
        g = orchestrator.data.growth
        idx = np.asarray(g.coresident_idx)
        n = np.asarray(g.coresident_n)

        assert idx.shape == (g.num_genotype, 16, 3)
        np.testing.assert_array_equal(n, [1] * 12 + [2] * 3 + [3])
        # Set k has exactly n[k] co-residents; the rest of its slots are -1.
        filled = (idx >= 0).sum(axis=-1)
        np.testing.assert_array_equal(filled, np.broadcast_to(n, filled.shape))
        assert idx.max() < g.num_genotype

    def test_pool_is_bulk_share_of_genotypes_with_data(self, library_file):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         library_file=library_file)
        mask = ~np.array(orchestrator.data.growth.ln_cfu0_spiked_mask)
        pool = orchestrator._coresident_pool(mask)
        table = orchestrator.library_df.set_index("genotype")
        weights = np.array([table.loc[g, "pool_fraction"]
                            * table.loc[g, "bulk_fraction"]
                            for g in _labels(orchestrator)])
        np.testing.assert_allclose(pool, weights / weights.sum())

    def test_draws_follow_pool(self, library_file):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         library_file=library_file,
                                         congression_sets=[4000])
        mask = ~np.array(orchestrator.data.growth.ln_cfu0_spiked_mask)
        pool = orchestrator._coresident_pool(mask)
        idx = np.asarray(orchestrator.data.growth.coresident_idx)
        counts = np.bincount(idx[idx >= 0], minlength=len(pool))
        np.testing.assert_allclose(counts / counts.sum(), pool, atol=0.01)

    def test_unknown_never_drawn(self, library_file):
        growth_df = _growth_df(["wt", "A2C", "__unknown__"])
        orchestrator = ModelOrchestrator(growth_df,
                                         _binding_df(["wt", "A2C"]),
                                         library_file=library_file,
                                         congression_sets=[500])
        unknown = _labels(orchestrator).index("__unknown__")
        idx = np.asarray(orchestrator.data.growth.coresident_idx)
        assert not np.any(idx == unknown)

    def test_legacy_pool_uniform_over_non_spiked(self):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         spiked_genotypes=["wt"])
        mask = ~np.array(orchestrator.data.growth.ln_cfu0_spiked_mask)
        pool = orchestrator._coresident_pool(mask)
        labels = _labels(orchestrator)
        assert pool[labels.index("wt")] == 0.0
        np.testing.assert_allclose(pool[[labels.index("A2L"),
                                         labels.index("A2C")]], [0.5, 0.5])

    def test_empty_pool_gives_empty_sets(self):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         spiked_genotypes=GENOTYPES)
        assert np.all(np.asarray(orchestrator.data.growth.coresident_idx) == -1)

    def test_mixture_refuses_empty_pool_with_bulk_genotypes(self,
                                                            library_file):
        """Bulk genotypes but nothing drawable: a mixture model cannot build
        congressed cells, so it refuses; 'single' does not care."""
        table = pd.read_csv(library_file)
        table["pool_fraction"] = 0.0
        table.to_csv(library_file, index=False)

        with pytest.raises(ValueError, match="co-resident pool is empty"):
            ModelOrchestrator(_growth_df(), _binding_df(),
                              library_file=library_file,
                              transformation="mixture")
        ModelOrchestrator(_growth_df(), _binding_df(),
                          library_file=library_file, transformation="single")

    def test_seed_reproduces_and_changes_draws(self, library_file):
        def draw(seed_value):
            o = ModelOrchestrator(_growth_df(), _binding_df(),
                                  library_file=library_file,
                                  congression_sets=[50, 10],
                                  congression_seed=seed_value)
            return np.asarray(o.data.growth.coresident_idx)

        np.testing.assert_array_equal(draw(3), draw(3))
        assert not np.array_equal(draw(3), draw(4))

    def test_settings_record_sets_and_seed(self, library_file):
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         library_file=library_file,
                                         congression_sets=[5, 2],
                                         congression_seed=7)
        assert orchestrator.settings["congression_sets"] == [5, 2]
        assert orchestrator.settings["congression_seed"] == 7

    @pytest.mark.parametrize("bad", [[], [0, 0], [-1, 2], [1.5], "abc", None])
    def test_bad_sets_raise(self, bad):
        with pytest.raises(ValueError, match="congression_sets"):
            ModelOrchestrator(_growth_df(), _binding_df(),
                              congression_sets=bad)

    def test_library_sized_under_batching(self, library_file):
        from tfscreen.tfmodel.tensors.batch import get_batch
        orchestrator = ModelOrchestrator(_growth_df(), _binding_df(),
                                         library_file=library_file)
        full = np.asarray(orchestrator.data.growth.coresident_idx)
        batch = get_batch(orchestrator.data, np.array([2, 0]))
        np.testing.assert_array_equal(np.asarray(batch.growth.coresident_idx),
                                      full)


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

    def test_round_trip_reproduces_coresident_sets(self, tmp_path,
                                                   library_config_file):
        out_prefix = self._configure(tmp_path, library_config_file)
        with open(f"{out_prefix}_config.yaml") as f:
            config = yaml.safe_load(f)
        assert config["components"]["congression_sets"] == [12, 3, 1]
        assert config["components"]["congression_seed"] == 0

        orchestrator, _ = read_configuration(f"{out_prefix}_config.yaml")
        direct = ModelOrchestrator(_growth_df(), _binding_df(),
                                   library_file=f"{out_prefix}_library.csv")
        assert _labels(orchestrator) == _labels(direct)
        np.testing.assert_array_equal(
            np.asarray(orchestrator.data.growth.coresident_idx),
            np.asarray(direct.data.growth.coresident_idx))

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
