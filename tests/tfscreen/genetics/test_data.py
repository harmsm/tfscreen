"""
Unit tests for tfscreen.genetics.data.

These tests verify the correctness and completeness of the codon/amino-acid
lookup tables and the degenerate-base specifier dictionary.
"""

import pytest

from tfscreen.genetics.data import (
    CODON_TO_AA,
    DEGEN_BASE_SPECIFIER,
    COMPLEMENT_DICT,
)


# ---------------------------------------------------------------------------
# CODON_TO_AA
# ---------------------------------------------------------------------------

class TestCodonToAa:
    def test_has_64_entries(self):
        """All 64 sense/stop codons must be present."""
        assert len(CODON_TO_AA) == 64

    def test_all_keys_are_three_lowercase_nt(self):
        valid_bases = set("acgt")
        for codon in CODON_TO_AA:
            assert len(codon) == 3, f"Codon length != 3: {codon!r}"
            assert set(codon) <= valid_bases, f"Non-ACGT base in codon: {codon!r}"

    def test_all_values_are_single_char(self):
        for codon, aa in CODON_TO_AA.items():
            assert isinstance(aa, str) and len(aa) == 1, (
                f"Unexpected value for {codon}: {aa!r}"
            )

    def test_standard_start_codon(self):
        assert CODON_TO_AA["atg"] == "M"

    def test_stop_codons(self):
        for stop in ("taa", "tag", "tga"):
            assert CODON_TO_AA[stop] == "*", f"Stop codon {stop!r} should map to '*'"

    def test_known_amino_acids(self):
        known = {
            "ttt": "F", "ttc": "F",     # Phe
            "tta": "L", "ttg": "L",     # Leu
            "att": "I", "atc": "I",     # Ile
            "gtt": "V", "gtc": "V",     # Val
            "tct": "S", "tcc": "S",     # Ser
            "cct": "P", "ccc": "P",     # Pro
            "act": "T", "acc": "T",     # Thr
            "gct": "A", "gcc": "A",     # Ala
            "tat": "Y", "tac": "Y",     # Tyr
            "cat": "H", "cac": "H",     # His
            "caa": "Q", "cag": "Q",     # Gln
            "aat": "N", "aac": "N",     # Asn
            "aaa": "K", "aag": "K",     # Lys
            "gat": "D", "gac": "D",     # Asp
            "gaa": "E", "gag": "E",     # Glu
            "tgt": "C", "tgc": "C",     # Cys
            "tgg": "W",                  # Trp
            "cgt": "R", "cgc": "R",     # Arg
            "ggt": "G", "ggc": "G",     # Gly
        }
        for codon, expected_aa in known.items():
            assert CODON_TO_AA[codon] == expected_aa, (
                f"CODON_TO_AA[{codon!r}] = {CODON_TO_AA[codon]!r}, expected {expected_aa!r}"
            )

    def test_all_standard_amino_acids_represented(self):
        standard_aas = set("ACDEFGHIKLMNPQRSTVWY*")
        observed_aas = set(CODON_TO_AA.values())
        assert observed_aas == standard_aas


# ---------------------------------------------------------------------------
# DEGEN_BASE_SPECIFIER
# ---------------------------------------------------------------------------

class TestDegenBaseSpecifier:
    def test_canonical_bases_map_to_themselves(self):
        for base in "acgt":
            assert DEGEN_BASE_SPECIFIER[base] == base

    def test_n_maps_to_all_bases(self):
        assert DEGEN_BASE_SPECIFIER["n"] == "acgt"

    def test_r_is_purine(self):
        assert set(DEGEN_BASE_SPECIFIER["r"]) == set("ag")

    def test_y_is_pyrimidine(self):
        assert set(DEGEN_BASE_SPECIFIER["y"]) == set("ct")

    def test_all_values_contain_only_acgt(self):
        for code, expansion in DEGEN_BASE_SPECIFIER.items():
            assert set(expansion) <= set("acgt"), (
                f"DEGEN_BASE_SPECIFIER[{code!r}] contains non-ACGT chars: {expansion!r}"
            )

    def test_all_expansions_are_sorted(self):
        """Expansions should be in alphabetical order (acgt)."""
        for code, expansion in DEGEN_BASE_SPECIFIER.items():
            assert list(expansion) == sorted(expansion), (
                f"DEGEN_BASE_SPECIFIER[{code!r}] is not sorted: {expansion!r}"
            )

    def test_has_all_iupac_codes(self):
        expected_keys = set("acgtrymswbvdhn")
        assert set(DEGEN_BASE_SPECIFIER.keys()) == expected_keys


# ---------------------------------------------------------------------------
# COMPLEMENT_DICT
# ---------------------------------------------------------------------------

class TestComplementDict:
    def test_a_complements_t(self):
        assert COMPLEMENT_DICT["a"] == "t"

    def test_t_complements_a(self):
        assert COMPLEMENT_DICT["t"] == "a"

    def test_g_complements_c(self):
        assert COMPLEMENT_DICT["g"] == "c"

    def test_c_complements_g(self):
        assert COMPLEMENT_DICT["c"] == "g"

    def test_self_complementary_entries(self):
        for base in ("n", "-"):
            assert COMPLEMENT_DICT[base] == base

    def test_complement_is_involution(self):
        """Complementing twice should return the original base."""
        for base in "acgt":
            comp = COMPLEMENT_DICT[base]
            assert COMPLEMENT_DICT[comp] == base, (
                f"Complement of complement({base!r}) != {base!r}"
            )
