"""
Tests for library_composition_table / read/write_library_composition.

The fixture library is deliberately tiny but has the property that motivated
this interface: wt and one single mutant are encoded by BOTH a spiked sequence
and the bulk sub-library, so they are neither purely spiked nor purely bulk.
"""

import numpy as np
import pandas as pd
import pytest

from tfscreen.genetics import (
    library_composition_table,
    read_library_composition,
    write_library_composition,
)
from tfscreen.genetics.library_design import LIBRARY_COMPOSITION_COLUMNS


@pytest.fixture
def library_config():
    # Codons: atg gca aaa ccg gaa tgc.  Site 2 (gca, Ala) is tile 1 and site 5
    # (gaa, Glu) is tile 2, both NNT.  NNT encodes Ala (gct) but not Glu, so wt
    # is reachable in tile 1 only.  Spikes: wt and A2L.
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


class TestLibraryCompositionTable:

    def test_columns_and_uniqueness(self, library_config):
        table = library_composition_table(library_config)
        assert list(table.columns) == LIBRARY_COMPOSITION_COLUMNS
        assert not table["genotype"].duplicated().any()
        assert table["pool_fraction"].sum() == pytest.approx(1.0)

    def test_wt_is_flagged(self, library_config):
        table = library_composition_table(library_config).set_index("genotype")
        assert table.loc["wt", "is_wt"]
        assert not table["is_wt"].drop("wt").any()

    def test_spiked_genotypes_are_also_in_the_bulk(self, library_config):
        """The case --spiked cannot express: spiked *and* bulk."""
        table = library_composition_table(library_config).set_index("genotype")

        for genotype in ("wt", "A2L"):
            assert table.loc[genotype, "in_spiked_origin"]
            # Encoded by a spiked sequence and by the bulk single-1 library,
            # so most of its cells are congression-affected bulk cells.
            assert 0 < table.loc[genotype, "bulk_fraction"] < 1
            assert "spiked" in table.loc[genotype, "origins"]
            assert "single-1" in table.loc[genotype, "origins"]

    def test_bulk_only_genotype(self, library_config):
        table = library_composition_table(library_config).set_index("genotype")
        assert not table.loc["A2C", "in_spiked_origin"]
        assert table.loc["A2C", "bulk_fraction"] == 1.0

    def test_spiked_share_scales_with_mixture(self, library_config):
        """More spike in the pool -> less of wt's mass is bulk."""
        low = library_composition_table(library_config).set_index("genotype")

        library_config["library_mixture"]["spiked"] = 100
        high = library_composition_table(library_config).set_index("genotype")

        assert (high.loc["wt", "bulk_fraction"]
                < low.loc["wt", "bulk_fraction"])

    def test_no_spiked_seqs(self, library_config):
        library_config.pop("spiked_seqs")
        library_config["library_mixture"].pop("spiked")
        table = library_composition_table(library_config)
        assert not table["in_spiked_origin"].any()
        assert (table["bulk_fraction"] == 1.0).all()

    def test_missing_library_mixture_raises(self, library_config):
        library_config.pop("library_mixture")
        with pytest.raises(ValueError, match="library_mixture"):
            library_composition_table(library_config)

    def test_mixture_key_mismatch_raises(self, library_config):
        library_config["library_mixture"].pop("spiked")
        with pytest.raises(ValueError, match="library_mixture"):
            library_composition_table(library_config)


class TestReadWriteRoundTrip:

    def test_round_trip(self, library_config, tmp_path):
        table = library_composition_table(library_config)
        path = str(tmp_path / "tfs_configure_library.csv")
        write_library_composition(table, path)

        back = read_library_composition(path)
        assert list(back.columns) == LIBRARY_COMPOSITION_COLUMNS
        assert back["is_wt"].dtype == bool
        assert back["in_spiked_origin"].dtype == bool
        pd.testing.assert_frame_equal(back, table, check_dtype=False)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_library_composition(str(tmp_path / "nope.csv"))

    def test_missing_column(self, tmp_path):
        path = str(tmp_path / "bad.csv")
        pd.DataFrame({"genotype": ["wt"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            read_library_composition(path)

    def test_duplicate_genotype(self, library_config, tmp_path):
        table = library_composition_table(library_config)
        path = str(tmp_path / "dup.csv")
        pd.concat([table, table.head(1)]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="duplicate genotypes"):
            read_library_composition(path)

    def test_write_requires_columns(self, tmp_path):
        with pytest.raises(ValueError, match="missing columns"):
            write_library_composition(pd.DataFrame({"genotype": ["wt"]}),
                                      str(tmp_path / "x.csv"))
