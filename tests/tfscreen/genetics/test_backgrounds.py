"""
Genetic backgrounds (genetics/backgrounds.py, planning/dimer-thermo-model.md
step 3d): absolute genotype names, per-background composition, and the
configure checks.
"""
import os

import pandas as pd
import pytest
import yaml

from tfscreen.genetics import backgrounds as B
from tfscreen.genetics.library_design import (
    library_composition_table,
    read_library_composition,
    write_library_composition,
)
from tfscreen.genetics.library_manager import LibraryManager
from tfscreen.process_raw.fastq_to_counts import FastqToCounts
from tfscreen.tfmodel.scripts.configure_model_cli import (
    check_backgrounds_in_data,
)

_SMOKE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "smoke-tests", "test_data")
_LIBRARY_YAML = os.path.join(_SMOKE, "library-smoke.yaml")
FIRST = 27


def _base():
    with open(_LIBRARY_YAML) as fh:
        return yaml.safe_load(fh)


def _with_codon(seq, residue, codon):
    i = (residue - FIRST) * 3
    return seq[:i] + codon + seq[i + 3:]


def _config():
    cf = _base()
    cf["expected_5p"], cf["expected_3p"] = "cgcgtggtgaaccag", "gtagaacgaagcggc"
    ref = cf["wt_seq"]
    cf["backgrounds"] = {
        # L90A: residue 90 lies in no tile, inside the amplicon (27-98)
        "bgA": {"wt_seq": _with_codon(ref, 90, "gcg")},
        "bgFar": {"outside_mutations": ["M1A"],
                  "library_mixture": {"single-1": 1, "single-2": 1,
                                      "double-1-2": 10}},
    }
    return cf


def test_reference_residue_90_is_leucine():
    lm = LibraryManager(_base())
    assert lm.aa_seq[90 - FIRST] == "L"


# ---------------------------------------------------------------------------
# read_backgrounds / background_config
# ---------------------------------------------------------------------------

def test_names_are_absolute():
    cf = _config()
    assert B.background_names(cf) == ["reference", "bgA", "bgFar"]
    assert B.background_parent(cf, "reference") == "wt"
    assert B.background_parent(cf, "bgA") == "L90A"
    assert B.background_parent(cf, "bgFar") == "M1A"
    lib = LibraryManager(B.background_config(cf, "bgA")).build_library_df()
    genos = set(lib["genotype"].astype(str))
    assert all("L90A" in g for g in genos)
    assert "M42I/L90A" in genos


def test_background_defaults_drop_the_spikes():
    cf = _config()
    bg = B.background_config(cf, "bgA")
    assert "spiked_seqs" not in bg and "spiked" not in bg["library_mixture"]
    assert "backgrounds" not in bg
    far = B.background_config(cf, "bgFar")
    assert far["library_mixture"] == {"single-1": 1, "single-2": 1,
                                      "double-1-2": 10}
    ref = B.background_config(cf, "reference")
    assert ref["wt_seq"] == cf["wt_seq"] and "spiked_seqs" in ref


def _bad(**spec):
    cf = _base()
    cf["backgrounds"] = {"bg": spec}
    return cf


@pytest.mark.parametrize("cf,match", [
    (_bad(color="red"), "unknown keys"),
    (_bad(), "is the reference"),
    (_bad(wt_seq="acgt"), "bases"),
    (_bad(outside_mutations=["K84"]), "not a mutation"),
    (_bad(outside_mutations=["K84K"]), "not a mutation"),
    (_bad(outside_mutations=["M1A", "M1V"]), "two mutations at one site"),
])
def test_read_backgrounds_errors(cf, match):
    with pytest.raises(ValueError, match=match):
        B.read_backgrounds(cf)


def test_reference_name_is_reserved():
    cf = _base()
    cf["backgrounds"] = {"reference": {"outside_mutations": ["M1A"]}}
    with pytest.raises(ValueError, match="invalid name"):
        B.read_backgrounds(cf)


def test_design_names_strip_the_parent():
    parent = frozenset({"L90A"})
    assert B.design_names(["L90A", "M42I/L90A", "L90V", "wt"], parent) == [
        "wt", "M42I", "L90V", "wt"]
    assert B.parent_mutations(B.background_config(_config(), "bgA")) == {"L90A"}
    assert B.parent_mutations(B.background_config(_config(), "bgFar")) == {
        "M1A"}
    assert B.parent_mutations(_base()) == frozenset()


def test_subsample_keeps_the_same_members_in_every_background():
    cf = _config()
    cf["library_subsample"] = {"single-1": 3, "double-1-2": 4}
    cf["library_subsample_seed"] = 11
    lib, summaries = B.build_background_library(cf)
    members = {}
    for bg, block in lib.groupby("background"):
        parent = B.parent_mutations(B.background_config(cf, bg))
        members[bg] = set(B.design_names(block["genotype"], parent))
    assert members["bgA"] == members["bgFar"]
    # the same draw as the reference's, which also keeps its spikes
    spikes = B.reference_spiked_members(B.background_config(cf, "bgA"))
    assert spikes
    assert members["bgA"] - spikes == members["reference"] - spikes
    assert len(members["bgA"] - spikes) > 3 + 4
    assert set(summaries) == {"reference", "bgA", "bgFar"}
    # the composition table subsamples each background the same way
    table = library_composition_table(cf)
    for bg, block in table.groupby("background"):
        parent = B.parent_mutations(B.background_config(cf, bg))
        assert set(B.design_names(block["genotype"], parent)) == members[bg]


def test_outside_mutation_inside_the_amplicon_is_an_error():
    cf = _base()
    cf["backgrounds"] = {"bg": {"outside_mutations": ["L90A"]}}
    with pytest.raises(ValueError, match="inside the amplicon"):
        LibraryManager(B.background_config(cf, "bg"))


def test_unknown_background():
    with pytest.raises(ValueError, match="not declared"):
        B.background_config(_config(), "bgZ")


# ---------------------------------------------------------------------------
# Reads: the expected library of a background
# ---------------------------------------------------------------------------

def test_fastq_expects_the_background_library():
    cf = _config()
    ftc = FastqToCounts(LibraryManager(B.background_config(cf, "bgA")))
    expected = set(ftc.all_expected_genotypes)
    assert "L90A" in expected and "wt" not in expected
    ref = FastqToCounts(LibraryManager(B.background_config(cf, "reference")))
    assert "wt" in set(ref.all_expected_genotypes)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_composition_has_one_block_per_background(tmp_path):
    cf = _config()
    table = library_composition_table(cf)
    assert list(table.columns[:2]) == ["background", "genotype"]
    assert list(dict.fromkeys(table["background"])) == ["reference", "bgA",
                                                         "bgFar"]
    for _, block in table.groupby("background"):
        assert block["pool_fraction"].sum() == pytest.approx(1.0)
    ref = table[table["background"] == "reference"].drop(columns="background")
    plain = library_composition_table(_base())
    pd.testing.assert_frame_equal(ref.reset_index(drop=True), plain)
    bg = table[table["background"] == "bgA"]
    assert not bg["in_spiked_origin"].any()
    assert (bg["bulk_fraction"] == 1.0).all()

    path = tmp_path / "library.csv"
    write_library_composition(table, path)
    back = read_library_composition(str(path))
    assert list(back.columns[:2]) == ["background", "genotype"]
    assert len(back) == len(table)


def test_composition_without_backgrounds_is_unchanged():
    table = library_composition_table(_base())
    assert "background" not in table.columns


# ---------------------------------------------------------------------------
# Configure checks
# ---------------------------------------------------------------------------

def _growth(genotypes, library, background):
    return pd.DataFrame({"genotype": genotypes, "library": library,
                         "background": background})


def test_check_backgrounds_in_data():
    table = library_composition_table(_config())
    good = pd.concat([_growth(["wt", "M42I"], "kanR", "reference"),
                      _growth(["L90A", "L90A/M42I"], "pheS", "bgA")])
    check_backgrounds_in_data(table, good)

    wrong = _growth(["M42I"], "pheS", "bgA")     # needs L90A/M42I
    with pytest.raises(ValueError, match="background 'bgA'"):
        check_backgrounds_in_data(table, wrong)
    mixed = pd.concat([_growth(["wt"], "kanR", "reference"),
                       _growth(["L90A"], "kanR", "bgA")])
    with pytest.raises(ValueError, match="more than one background"):
        check_backgrounds_in_data(table, mixed)
    with pytest.raises(ValueError, match="does not declare"):
        check_backgrounds_in_data(table, _growth(["wt"], "kanR", "bgQ"))
    # no declared backgrounds: nothing to check here (the orchestrator
    # refuses a background column without them)
    check_backgrounds_in_data(library_composition_table(_base()),
                              _growth(["wt"], "kanR", "bgA"))
