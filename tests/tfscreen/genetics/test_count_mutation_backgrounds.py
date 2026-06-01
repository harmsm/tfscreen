import pytest
import pandas as pd
import numpy as np

from tfscreen.genetics import count_mutation_backgrounds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def example_genotypes():
    """The worked example from the docstring."""
    return ["wt", "M42I", "H74A", "M42I/H74A", "M42I/K84L", "H74A/K84L"]


@pytest.fixture
def example_df(example_genotypes):
    return pd.DataFrame({"genotype": example_genotypes, "value": range(6)})


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------

class TestCountMutationBackgrounds:

    def test_docstring_example_from_list(self, example_genotypes):
        """
        GIVEN the worked docstring example as a plain list
        WHEN count_mutation_backgrounds is called
        THEN the two background columns should match the expected values exactly.
        """
        result = count_mutation_backgrounds(example_genotypes)

        expected_b1 = [0, 3, 3, 3, 3, 3]
        expected_b2 = [0, 0, 0, 3, 2, 2]

        assert result["mut_backgrounds_1"].tolist() == expected_b1
        assert result["mut_backgrounds_2"].tolist() == expected_b2

    def test_docstring_example_from_dataframe(self, example_df):
        """
        GIVEN the worked example as a DataFrame
        WHEN count_mutation_backgrounds is called
        THEN values match and extra columns are preserved.
        """
        result = count_mutation_backgrounds(example_df)

        assert result["mut_backgrounds_1"].tolist() == [0, 3, 3, 3, 3, 3]
        assert result["mut_backgrounds_2"].tolist() == [0, 0, 0, 3, 2, 2]
        assert "value" in result.columns

    def test_column_count_equals_max_mutations(self):
        """
        GIVEN a dataset whose maximum genotype has 3 mutations
        WHEN count_mutation_backgrounds is called
        THEN exactly 3 mut_backgrounds_* columns are created.
        """
        genotypes = ["wt", "A1B", "A1B/C2D/E3F"]
        result = count_mutation_backgrounds(genotypes)

        bg_cols = [c for c in result.columns if c.startswith("mut_backgrounds_")]
        assert len(bg_cols) == 3

    def test_only_singles_creates_one_column(self):
        """
        GIVEN a dataset with only singles and wt
        WHEN count_mutation_backgrounds is called
        THEN exactly 1 mut_backgrounds_* column is created.
        """
        genotypes = ["wt", "A1B", "C2D", "E3F"]
        result = count_mutation_backgrounds(genotypes)

        bg_cols = [c for c in result.columns if c.startswith("mut_backgrounds_")]
        assert len(bg_cols) == 1

    def test_only_wt_returns_no_new_columns(self):
        """
        GIVEN a dataset containing only 'wt'
        WHEN count_mutation_backgrounds is called
        THEN no mut_backgrounds_* columns are added.
        """
        result = count_mutation_backgrounds(["wt", "wt"])
        bg_cols = [c for c in result.columns if c.startswith("mut_backgrounds_")]
        assert len(bg_cols) == 0

    def test_wt_rows_are_all_zeros(self, example_genotypes):
        """
        GIVEN a list with wt entries
        WHEN count_mutation_backgrounds is called
        THEN wt rows have 0 in all background columns.
        """
        result = count_mutation_backgrounds(example_genotypes)
        wt_rows = result[result["genotype"] == "wt"]
        bg_cols = [c for c in result.columns if c.startswith("mut_backgrounds_")]
        assert (wt_rows[bg_cols].values == 0).all()

    def test_mutation_appearing_once_has_count_one(self):
        """
        GIVEN a mutation that appears in exactly one genotype
        WHEN count_mutation_backgrounds is called
        THEN its background count is 1.
        """
        genotypes = ["wt", "A1B", "C2D"]
        result = count_mutation_backgrounds(genotypes)
        # A1B appears only in row 1; C2D only in row 2
        assert result.loc[1, "mut_backgrounds_1"] == 1
        assert result.loc[2, "mut_backgrounds_1"] == 1

    def test_mutation_count_includes_both_singles_and_doubles(self):
        """
        GIVEN a mutation that appears as a single and in a double
        WHEN count_mutation_backgrounds is called
        THEN the count reflects both occurrences.
        """
        genotypes = ["A1B", "A1B/C2D"]
        result = count_mutation_backgrounds(genotypes)
        # A1B appears in both rows → count 2
        assert result.loc[0, "mut_backgrounds_1"] == 2
        assert result.loc[1, "mut_backgrounds_1"] == 2

    # -----------------------------------------------------------------------
    # Input type handling
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("input_type", [list, pd.Series, np.array])
    def test_input_types(self, input_type, example_genotypes):
        """
        GIVEN different iterable input types
        WHEN count_mutation_backgrounds is called
        THEN it produces a valid DataFrame with correct values.
        """
        result = count_mutation_backgrounds(input_type(example_genotypes))
        assert isinstance(result, pd.DataFrame)
        assert result["mut_backgrounds_1"].tolist() == [0, 3, 3, 3, 3, 3]

    def test_dataframe_input_missing_genotype_column_raises(self):
        """
        GIVEN a DataFrame without a 'genotype' column
        WHEN count_mutation_backgrounds is called
        THEN a ValueError is raised.
        """
        df = pd.DataFrame({"value": [1, 2]})
        with pytest.raises(ValueError):
            count_mutation_backgrounds(df)

    # -----------------------------------------------------------------------
    # Column placement
    # -----------------------------------------------------------------------

    def test_new_columns_inserted_after_genotype(self, example_df):
        """
        GIVEN a DataFrame with columns before and after 'genotype'
        WHEN count_mutation_backgrounds is called
        THEN mut_backgrounds_* columns appear immediately after 'genotype'.
        """
        df = pd.DataFrame({
            "id": range(6),
            "genotype": ["wt", "M42I", "H74A", "M42I/H74A", "M42I/K84L", "H74A/K84L"],
            "score": range(6),
        })
        result = count_mutation_backgrounds(df)
        cols = list(result.columns)
        geno_idx = cols.index("genotype")
        assert cols[geno_idx + 1] == "mut_backgrounds_1"
        assert cols[geno_idx + 2] == "mut_backgrounds_2"
        # original columns still present
        assert "id" in cols
        assert "score" in cols

    def test_does_not_modify_original_dataframe(self, example_df):
        """
        GIVEN a DataFrame input
        WHEN count_mutation_backgrounds is called
        THEN the original DataFrame is not modified.
        """
        original_cols = list(example_df.columns)
        count_mutation_backgrounds(example_df)
        assert list(example_df.columns) == original_cols

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------

    def test_empty_input(self):
        """
        GIVEN an empty list
        WHEN count_mutation_backgrounds is called
        THEN an empty DataFrame is returned without error.
        """
        result = count_mutation_backgrounds([])
        assert isinstance(result, pd.DataFrame)

    def test_triple_mutant(self):
        """
        GIVEN a dataset with a triple mutant
        WHEN count_mutation_backgrounds is called
        THEN three background columns are created with correct values.
        """
        genotypes = ["A1B", "A1B/C2D", "A1B/C2D/E3F"]
        result = count_mutation_backgrounds(genotypes)

        bg_cols = [c for c in result.columns if c.startswith("mut_backgrounds_")]
        assert len(bg_cols) == 3

        # A1B in all 3, C2D in 2, E3F in 1
        assert result.loc[2, "mut_backgrounds_1"] == 3
        assert result.loc[2, "mut_backgrounds_2"] == 2
        assert result.loc[2, "mut_backgrounds_3"] == 1
