"""Tests for tfscreen.genetics.library_subsample (library_genotypes and
library_subsample in the library YAML)."""

import numpy as np
import pandas as pd
import pytest
import yaml

from tfscreen.genetics import LibraryManager, library_composition_table
from tfscreen.genetics.library_subsample import (
    build_library,
    format_subsample_summary,
    subsample_design,
    subsample_library_df,
    subsample_record,
)


@pytest.fixture
def config():
    # Two NNT sites per tile; tile 1 can encode wt (Ala, gct), tile 2 cannot
    # (Glu). Spikes: wt and A2L (also bulk members of single-1).
    return {
        "reading_frame": 0,
        "first_amplicon_residue": 1,
        "wt_seq":      "atggcaaaaccggaatgcggc",
        "degen_sites": "...nntnnt...nnt......",
        "tiles":       "...111111...222......",
        "tile_combos": ["single-1", "single-2", "double-1-2"],
        "spiked_seqs": [".....................",
                        "...ctt..............."],
        "library_mixture": {"single-1": 10, "single-2": 10,
                            "double-1-2": 100, "spiked": 1},
        "transform_sizes": {"single-1": 1000, "single-2": 1000,
                            "double-1-2": 100000, "spiked": 10},
        "total_num_reads": 1_000_000,
        "pcr_template_molecules": 50_000,
    }


def _full(config):
    df = LibraryManager(config).build_library_df()
    df["genotype"] = df["genotype"].astype(str)
    return df


def _kept(df, origin):
    return set(df.loc[df["library_origin"] == origin, "genotype"].astype(str))


def test_no_subsample_returns_input(config):
    df = _full(config)
    out, summary = subsample_library_df(df, config)
    assert out is df and summary is None
    assert subsample_design(config, summary) == {}
    assert subsample_record(config) is None


# ---------------------------------------------------------------------------
# library_genotypes
# ---------------------------------------------------------------------------

def test_list_keeps_listed_plus_wt_and_spikes(config, tmp_path):
    full = _full(config)
    doubles = sorted(g for g in _kept(full, "double-1-2") if g.count("/") == 1)
    singles = sorted(g for g in _kept(full, "single-2"))[:3]
    listed = doubles[:5] + singles
    # unsorted mutations and comments are fine
    lines = ["# kept genotypes"] + ["/".join(reversed(listed[0].split("/")))] \
        + listed[1:]
    (tmp_path / "keep.txt").write_text("\n".join(lines) + "\n")
    config["library_genotypes"] = "keep.txt"
    out, summary = subsample_library_df(full, config, base_dir=str(tmp_path))

    protected = {"wt", "A2L"}
    assert set(out["genotype"].astype(str)) == set(listed) | protected
    # A2L is kept in every origin that encodes it, spiked and bulk
    assert set(out.loc[out["genotype"] == "A2L", "library_origin"]) == \
        {"single-1", "spiked"}
    # weights renormalized within each origin
    np.testing.assert_allclose(
        out.groupby("library_origin", observed=True)["weight"].sum(), 1.0)
    s = summary.set_index("library_origin")
    assert s.loc["spiked", "num_genotypes_kept"] == 2
    assert s.loc["single-2", "num_genotypes_kept"] == 3


def test_list_path_relative_to_yaml(config, tmp_path):
    sub = tmp_path / "cfg"
    sub.mkdir()
    full = _full(config)
    keep = sorted(_kept(full, "single-2"))[:2]
    (sub / "keep.txt").write_text("\n".join(keep))
    config["library_genotypes"] = "keep.txt"
    path = sub / "library.yaml"
    path.write_text(yaml.safe_dump(config))
    df, summary = build_library(str(path))
    assert set(keep) <= set(df["genotype"].astype(str))
    assert summary is not None


def test_list_genotype_outside_library_is_an_error(config, tmp_path):
    (tmp_path / "keep.txt").write_text("A2L\nQ99W\n")
    config["library_genotypes"] = str(tmp_path / "keep.txt")
    with pytest.raises(ValueError, match="not in the described library"):
        subsample_library_df(_full(config), config)


def test_list_missing_or_empty_file(config, tmp_path):
    config["library_genotypes"] = str(tmp_path / "nope.txt")
    with pytest.raises(FileNotFoundError):
        subsample_library_df(_full(config), config)
    (tmp_path / "empty.txt").write_text("# nothing\n")
    config["library_genotypes"] = str(tmp_path / "empty.txt")
    with pytest.raises(ValueError, match="lists no genotypes"):
        subsample_library_df(_full(config), config)


def test_list_that_empties_an_origin_is_an_error(config, tmp_path):
    full = _full(config)
    # only tile-1 singles: single-2 has no wt and nothing listed
    (tmp_path / "keep.txt").write_text("\n".join(_kept(full, "single-1")))
    config["library_genotypes"] = str(tmp_path / "keep.txt")
    with pytest.raises(ValueError, match="no genotypes"):
        subsample_library_df(full, config)


# ---------------------------------------------------------------------------
# library_subsample
# ---------------------------------------------------------------------------

def test_subsample_keeps_n_per_origin(config):
    full = _full(config)
    config.update(library_subsample={"double-1-2": 20},
                  library_subsample_seed=1)
    out, summary = subsample_library_df(full, config)
    dbl = _kept(out, "double-1-2")
    assert len(dbl - {"wt", "A2L"}) == 20
    # other origins untouched
    for o in ("single-1", "single-2", "spiked"):
        assert _kept(out, o) == _kept(full, o)
    s = summary.set_index("library_origin")
    assert s.loc["double-1-2", "num_genotypes"] == \
        len(_kept(full, "double-1-2"))


def test_subsample_seed_reproducible_and_nested(config):
    full = _full(config)

    def draw(n, seed, extra=None):
        cf = dict(config, library_subsample={"double-1-2": n, **(extra or {})},
                  library_subsample_seed=seed)
        return _kept(subsample_library_df(full, cf)[0], "double-1-2")

    assert draw(30, 5) == draw(30, 5)
    assert draw(30, 5) != draw(30, 6)
    # a smaller N keeps a subset of a larger N's genotypes
    assert draw(10, 5) <= draw(30, 5)
    # subsampling a second origin leaves the first origin's draw alone
    assert draw(30, 5) == draw(30, 5, {"single-2": 4})


@pytest.mark.parametrize("update,match", [
    ({"library_subsample": {"double-1-2": 5}}, "library_subsample_seed"),
    ({"library_subsample": {"double-1-2": 5},
      "library_subsample_seed": -1}, "non-negative integer"),
    ({"library_subsample": {"double-1-2": 5},
      "library_subsample_seed": 1.5}, "non-negative integer"),
    ({"library_subsample": {"double-9-9": 5},
      "library_subsample_seed": 1}, "not in the library"),
    ({"library_subsample": {"spiked": 1},
      "library_subsample_seed": 1}, "spiked origin"),
    ({"library_subsample": {"double-1-2": 0},
      "library_subsample_seed": 1}, "integer >= 1"),
    ({"library_subsample": {"double-1-2": True},
      "library_subsample_seed": 1}, "integer >= 1"),
    ({"library_subsample": {"double-1-2": 10_000},
      "library_subsample_seed": 1}, "only"),
    ({"library_subsample": [5], "library_subsample_seed": 1},
     "must map library origins"),
    ({"library_subsample": {"double-1-2": 5}, "library_subsample_seed": 1,
      "library_subsample_depth": "per_genotype"}, "library_subsample_depth"),
])
def test_subsample_errors(config, update, match):
    config.update(update)
    with pytest.raises(ValueError, match=match):
        subsample_library_df(_full(config), config)


def test_list_then_subsample(config, tmp_path):
    full = _full(config)
    doubles = sorted(g for g in _kept(full, "double-1-2") if "/" in g)[:40]
    keep = doubles + sorted(_kept(full, "single-1") | _kept(full, "single-2"))
    (tmp_path / "keep.txt").write_text("\n".join(keep))
    config.update(library_genotypes=str(tmp_path / "keep.txt"),
                  library_subsample={"double-1-2": 10},
                  library_subsample_seed=2)
    out, _ = subsample_library_df(full, config)
    dbl = _kept(out, "double-1-2") - {"wt", "A2L"}
    # (double-1-2 also encodes some singles, which the list keeps)
    assert len(dbl) == 10 and dbl <= set(keep)


# ---------------------------------------------------------------------------
# Depth (subsample_design) and the fit-side composition
# ---------------------------------------------------------------------------

def test_full_library_depth_keeps_per_sequence_quantities(config):
    full = _full(config)
    config.update(library_subsample={"double-1-2": 20},
                  library_subsample_seed=1)
    out, summary = subsample_library_df(full, config)
    design = subsample_design(config, summary)

    s = summary.set_index("library_origin")
    frac = s["num_sequences_kept"] / s["num_sequences"]
    mix = config["library_mixture"]
    for o in mix:
        assert design["library_mixture"][o] == pytest.approx(mix[o] * frac[o])
        assert design["transform_sizes"][o] == \
            max(1, round(config["transform_sizes"][o] * frac[o]))
    share = sum(design["library_mixture"].values()) / sum(mix.values())
    assert design["kept_pool_share"] == pytest.approx(share)
    assert design["total_num_reads"] == round(1_000_000 * share)
    assert design["pcr_template_molecules"] == round(50_000 * share)
    assert "cfu0" not in design
    # a kept double's share of the pool relative to a single is the same as
    # in the full library
    text = format_subsample_summary(config, summary, design)
    assert "double-1-2" in text and "cfu0 unchanged" in text


def test_mixture_missing_an_origin_is_an_error(config):
    full = _full(config)
    config.update(library_subsample={"double-1-2": 20},
                  library_subsample_seed=1)
    _, summary = subsample_library_df(full, config)
    del config["library_mixture"]["single-2"]
    with pytest.raises(ValueError,
                       match=r"library_mixture has no value.*single-2"):
        subsample_design(config, summary)


def test_as_written_depth_changes_nothing(config):
    config.update(library_subsample={"double-1-2": 20},
                  library_subsample_seed=1,
                  library_subsample_depth="as_written")
    _, summary = subsample_library_df(_full(config), config)
    assert subsample_design(config, summary) == {}


def test_composition_of_a_subsample(config):
    full_table = library_composition_table(config).set_index("genotype")
    config.update(library_subsample={"double-1-2": 20},
                  library_subsample_seed=1)
    sub = library_composition_table(config).set_index("genotype")
    sub_df, _ = build_library(config)
    assert set(sub.index) == set(sub_df["genotype"].astype(str))
    assert sub["pool_fraction"].sum() == pytest.approx(1.0)
    # full_library depth: a kept genotype all of whose sequences are kept
    # (here those encoded by one origin) keeps its abundance relative to
    # every other such genotype, and every kept spike its bulk fraction
    one = [g for g in sub.index if "|" not in full_table.loc[g, "origins"]]
    assert any("/" in g for g in one) and any("/" not in g for g in one)
    ratio = (sub.loc[one, "pool_fraction"]
             / full_table.loc[one, "pool_fraction"])
    np.testing.assert_allclose(ratio, ratio.iloc[0], rtol=1e-10)
    np.testing.assert_allclose(sub.loc[["wt", "A2L"], "bulk_fraction"],
                               full_table.loc[["wt", "A2L"], "bulk_fraction"],
                               rtol=1e-10)

    config["library_subsample_depth"] = "as_written"
    written = library_composition_table(config).set_index("genotype")
    ratio = (written.loc[one, "pool_fraction"]
             / full_table.loc[one, "pool_fraction"])
    assert not np.allclose(ratio, ratio.iloc[0])


def test_subsample_record(config, tmp_path):
    (tmp_path / "keep.txt").write_text("A2L\n")
    config.update(library_genotypes=str(tmp_path / "keep.txt"),
                  library_subsample={"double-1-2": 3},
                  library_subsample_seed=4)
    rec = subsample_record(config)
    assert rec == {"library_genotypes": "keep.txt",
                   "library_subsample": {"double-1-2": 3},
                   "library_subsample_seed": 4,
                   "library_subsample_depth": "full_library"}


def test_summary_columns(config):
    config.update(library_subsample={"double-1-2": 3}, library_subsample_seed=4)
    _, summary = subsample_library_df(_full(config), config)
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == ["library_origin", "num_genotypes",
                                     "num_genotypes_kept", "num_sequences",
                                     "num_sequences_kept"]
