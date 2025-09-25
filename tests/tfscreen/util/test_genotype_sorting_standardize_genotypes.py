# test_genotype_sorting.py

import pytest
from tfscreen.util.genotype_sorting import standardize_genotypes

# -------------------------------------------------------------------------
# Tests for valid genotypes that should be standardized correctly
# -------------------------------------------------------------------------

def test_simple_case():
    """
    Tests basic functionality: a standard list with wt, single, and
    double mutations. Also confirms output order is preserved.
    """
    genotypes = ["wt", "A1T", "Q50L/A1T", "A1T"]
    expected = ["wt", "A1T", "A1T/Q50L", "A1T"]
    assert standardize_genotypes(genotypes) == expected

def test_wildtype_variations():
    """
    Tests that various wildtype representations (case-insensitive)
    are all converted to 'wt'.
    """
    genotypes = ["wt", "WT", "wildtype", "WILDTYPE", "WildType"]
    expected = ["wt", "wt", "wt", "wt", "wt"]
    assert standardize_genotypes(genotypes) == expected

def test_self_mutation_to_wildtype():
    """
    Tests that genotypes with only self->self mutations (e.g., A47A)
    are correctly standardized to 'wt'.
    """
    genotypes = ["A1A", "C100C", "D5D/E6E"]
    expected = ["wt", "wt", "wt"]
    assert standardize_genotypes(genotypes) == expected

def test_self_mutation_is_dropped():
    """
    Tests that self->self mutations are dropped from a multi-mutation
    genotype string.
    """
    genotypes = ["A1T/C50C", "V10V/L20F/V30V"]
    expected = ["A1T", "L20F"]
    assert standardize_genotypes(genotypes) == expected

def test_duplicate_mutations_are_dropped():
    """
    Tests that identical mutations in a genotype are dropped, keeping one.
    """
    genotypes = ["A1T/A1T", "A1T/B2C/A1T"]
    expected = ["A1T", "A1T/B2C"]
    assert standardize_genotypes(genotypes) == expected

def test_sorting_of_mutations():
    """
    Tests that mutations are sorted by site number, regardless of
    their original order in the string.
    """
    genotypes = ["Q50L/A1T", "C100A/Z5B/D50E"]
    expected = ["A1T/Q50L", "Z5B/D50E/C100A"]
    assert standardize_genotypes(genotypes) == expected

def test_empty_input():
    """
    Tests that an empty input list returns an empty list.
    """
    assert standardize_genotypes([]) == []

def test_large_site_numbers():
    """
    Tests that site numbers with multiple digits are parsed correctly.
    """
    genotypes = ["A1234T", "Y999C/X10000Z"]
    expected = ["A1234T", "Y999C/X10000Z"]
    assert standardize_genotypes(genotypes) == expected

def test_all_features_combined():
    """
    A complex test combining multiple features: duplicate genotypes in the
    input, case variations, self-mutations, duplicate mutations, and sorting.
    """
    genotypes = [
        "Q50L/A1T",       # Needs sorting
        "WT",             # Wildtype variation
        "A1T/A1T",        # Duplicate mutation
        "G100G",          # Self-mutation to wt
        "V10F/C5A/G300G", # Mixed with self-mutation and needs sorting
        "Q50L/A1T"        # Duplicate of first entry
    ]
    expected = [
        "A1T/Q50L",
        "wt",
        "A1T",
        "wt",
        "C5A/V10F",
        "A1T/Q50L"
    ]
    assert standardize_genotypes(genotypes) == expected

# -------------------------------------------------------------------------
# Tests for invalid genotypes that should raise a ValueError
# -------------------------------------------------------------------------

def test_error_unparsable_mutation():
    """
    Tests that genotypes that don't fit the 'XsiteY' pattern raise an error.
    """
    with pytest.raises(ValueError, match="could not parse mutation 'A1'"):
        standardize_genotypes(["A1"])

    with pytest.raises(ValueError, match="could not get site number from mutation 'Fifty'"):
        standardize_genotypes(["Fifty"])

    with pytest.raises(ValueError, match="could not get site number from mutation 'A!T'"):
        standardize_genotypes(["A!T"])

def test_error_unparsable_site_number():
    """
    Tests that a ValueError is raised if the site number is not an integer.
    """
    with pytest.raises(ValueError, match="could not get site number from mutation 'AoneT'"):
        standardize_genotypes(["AoneT"])
    
    with pytest.raises(ValueError, match="could not get site number from mutation 'A1.0T'"):
        standardize_genotypes(["A1.0T"])

def test_error_multiple_mutations_at_same_site():
    """
    Tests that a ValueError is raised for conflicting mutations at one site.
    """
    with pytest.raises(ValueError, match="genotype 'A1T/A1V' has multiple mutations at the same site"):
        standardize_genotypes(["A1T/A1V"])

    # Should fail even if not adjacent
    with pytest.raises(ValueError, match="genotype 'A1T/Q50L/A1V' has multiple mutations at the same site"):
        standardize_genotypes(["A1T/Q50L/A1V"])